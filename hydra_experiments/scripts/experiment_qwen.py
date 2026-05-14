import sys
import os
import argparse
import base64
import pandas as pd
import json
import re
import asyncio
import io
import csv
from PIL import Image
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TEST_FILE, OUTPUT_DIR, get_effective_value, get_formatted_artwork_date, get_formatted_artist_date
from evaluate import evaluate_and_save

MODEL_NAME = os.getenv("MODEL_NAME_OVERRIDE", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ") 
PORT = os.getenv("PORT", "8000")
OUTPUT_FILENAME = "qwen2.5_72b_test" 


IMAGE_DIRS = [f"/dataset/chunk_{i:02d}" for i in range(14)] + ["/dataset/chunk_AIC"]

MAX_CONCURRENT_REQUESTS = 50 
MAX_IMAGE_SIDE = 800
MAX_TOTAL_PIXELS = 600000

# Setup client
client = AsyncOpenAI(
    base_url=f"http://localhost:{PORT}/v1",
    api_key="EMPTY",
)

def load_data(n_samples=None):
    print(f"Loading Test Data from {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE)
    df['object_ID'] = df['object_ID'].astype(str)
    
    if n_samples:
        print(f"Subsampling first {n_samples} rows for testing...")
        df = df.head(n_samples)
    print(f"Loaded {len(df)} total rows.")
    return df

def setup_checkpoint(modality, n_samples):
    base_name = f"{OUTPUT_FILENAME}_{modality}"
    if n_samples: base_name += "_debug"
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{base_name}_checkpoint.csv")
    
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        chk_df = pd.read_csv(checkpoint_path)
        processed_ids = set(chk_df['object_ID'].astype(str))
        print(f"Found checkpoint! Skipping {len(processed_ids)} already processed artworks.")
    else:
        # Create file with headers
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(checkpoint_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['object_ID', 'qwen_prompt', 'qwen_raw_response', 'qwen_predicted_has_error', 'qwen_predicted_error_type', 'qwen_reasoning'])
            
    return checkpoint_path, processed_ids

def get_local_image_path(row):
    target_id = row.get('image_object_id_error')
    if pd.isna(target_id) or str(target_id).strip() == "":
        target_id = row['object_ID']
    
    img_filename = f"{target_id}.jpg"
    for chunk_dir in IMAGE_DIRS:
        img_path = os.path.join(chunk_dir, img_filename)
        if os.path.exists(img_path):
            return img_path
    return None

def resize_and_encode_image(image_path, max_side=1280):
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            width, height = img.size
            if max(width, height) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            
            MAX_TOTAL_PIXELS = 1600000 
            while img.width * img.height > MAX_TOTAL_PIXELS:
                img = img.resize((int(img.width * 0.9), int(img.height * 0.9)))
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return None

def extract_json(text):
    try: return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return {"has_error": None, "error_type": "Parse Error", "reasoning": text}

def parse_metadata_list(val):
    if pd.isna(val) or val == "" or val == "[]":
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            try:
                return json.loads(cleaned.replace("''", '"'))
            except:
                try:
                    return ast.literal_eval(cleaned)
                except:
                    pass
        return [cleaned]
    return []

def construct_messages(row, modality):    
    def get_val(col_name):
        val = get_effective_value(row, col_name)
        return val if val.strip() != "" else None

    db_val = get_val('date_begin')
    de_val = get_val('date_end')
    db_bce = row.get('date_begin_bce') == True
    de_bce = row.get('date_end_bce') == True
    
    start = f"{int(abs(float(db_val)))} BCE" if db_bce and db_val else (int(float(db_val)) if db_val else "?")
    end = f"{int(abs(float(de_val)))} BCE" if de_bce and de_val else (int(float(de_val)) if de_val else "?")
    date_str = f"{start} - {end}"

    artist = get_val('artist_name') or "Unknown Artist"
    role = get_val('artist_role')
    nat = get_val('artist_nationality')
    adb = get_val('artist_date_begin')
    ade = get_val('artist_date_end')
    
    artist_str = f"{artist}"
    if role: artist_str += f" ({role})"
    
    artist_info = []
    if nat: artist_info.append(str(nat))
    if adb or ade:
        b = str(int(float(adb))) if adb else "?"
        e = str(int(float(ade))) if ade else "?"
        artist_info.append(f"{b}-{e}")
        
    if artist_info:
        artist_str += f", {', '.join(artist_info)}"

    mats = parse_metadata_list(get_val('materials'))
    techs = parse_metadata_list(get_val('techniques'))
    medium_val = ", ".join(sorted(list(set(mats + techs)))) if (mats or techs) else "N/A"
    
    period = get_val('period')
    dynasty = get_val('dynasty')
    reign = get_val('reign')
    hist_ctx = []
    if period: hist_ctx.append(f"Period: {period}")
    if dynasty: hist_ctx.append(f"Dynasty: {dynasty}")
    if reign: hist_ctx.append(f"Reign: {reign}")
    hist_str = " | ".join(hist_ctx) if hist_ctx else "N/A"

    formatted_metadata = f"""Title: {row.get('title', 'Untitled')}
Object Name: {get_val('object_name') or 'N/A'}
Artist: {artist_str}
Date: {date_str}
Historical Context: {hist_str}
Culture: {get_val('culture') or 'N/A'}
Location: {get_val('location') or 'N/A'}
Dimensions: {get_val('dimensions_json') or 'N/A'}
Medium: {medium_val}
Subjects: {get_val('subjects') or 'N/A'}
Inscriptions: {get_val('inscriptions') or 'N/A'}
Description: {row.get('description', 'N/A')}"""

    content = []

    if modality in ['table', 'both']:
        if modality == 'both':
            instruction = "Analyze the following artwork metadata and the attached image."
            evidence = "Compare the visual evidence in the image (style, period, subject, materials) with the provided text."
        else:
            instruction = "Analyze the following artwork metadata. No image is available."
            evidence = "Check for internal consistency and factual plausibility within the provided text."

        text_prompt = f"""You are an Art Historian. {instruction}
        
{formatted_metadata}
        
Question: Does the artwork metadata contain errors? 
{evidence}
Important: NaN or missing values (N/A, ?) are NOT errors. They are acceptable.
        
Return a JSON object with:
- "has_error": boolean (true if there is a factual contradiction, false otherwise)
- "error_type": string (if has_error is true, specify 'date', 'dimensions', 'medium', 'artist', 'culture', 'image', etc)
- "reasoning": string explanation of why you think there is or isn't an error.
"""
        content.append({"type": "text", "text": text_prompt})
        
    elif modality == 'image':
        text_prompt = """You are an Art Historian. Analyze the attached artwork image. 
        
Question: Does the image contain any obvious visual errors, anomalies, or corrupted pixels? 
        
Return a JSON object with:
- "has_error": boolean (true/false)
- "error_type": string (e.g. 'image')
- "reasoning": string explanation
"""
        content.append({"type": "text", "text": text_prompt})

    if modality in ['image', 'both']:
        img_path = get_local_image_path(row)
        if img_path:
            base64_img = resize_and_encode_image(img_path, max_side=MAX_IMAGE_SIDE)
            if base64_img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
    
    return [{"role": "user", "content": content}]

async def process_row(sem, row, modality, lock, checkpoint_path):
    async with sem:
        try:
            messages = construct_messages(row, modality)
            
            text_prompt = ""
            for item in messages[0]["content"]:
                if item["type"] == "text":
                    text_prompt = item["text"]
                    break

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.0 
            )
            raw_content = response.choices[0].message.content
            parsed = extract_json(raw_content)
            
            has_error = parsed.get('has_error')
            error_type = parsed.get('error_type')
            reasoning = parsed.get('reasoning')

        except Exception as e:
            text_prompt = text_prompt if 'text_prompt' in locals() else "Error generating prompt"
            raw_content = f"API Error: {str(e)}"
            has_error = None
            error_type = "API Error"
            reasoning = ""

        async with lock:
            with open(checkpoint_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([row['object_ID'], text_prompt, raw_content, has_error, error_type, reasoning])

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None, help="Run only on first N samples")
    parser.add_argument('--modality', type=str, choices=['table', 'image', 'both'], default='both')
    args = parser.parse_args()

    print(f"--- Starting Qwen 2.5 Run | Modality: {args.modality.upper()} | Samples: {args.n_samples if args.n_samples else 'ALL'} ---")
    
    df = load_data(args.n_samples)
    checkpoint_path, processed_ids = setup_checkpoint(args.modality, args.n_samples)
    

    df_todo = df[~df['object_ID'].astype(str).isin(processed_ids)]
    
    if len(df_todo) > 0:
        print(f"Queueing {len(df_todo)} remaining requests to vLLM...")
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        lock = asyncio.Lock()
        
        tasks = [process_row(sem, row, args.modality, lock, checkpoint_path) for _, row in df_todo.iterrows()]
        await tqdm_asyncio.gather(*tasks)
    else:
        print("All items already processed in checkpoint!")

    print("\nMerging checkpoint results for final evaluation...")
    
    results_df = pd.read_csv(checkpoint_path)
    results_df['object_ID'] = results_df['object_ID'].astype(str)
    
    full_df = df.merge(results_df, on='object_ID', how='left')
    
    full_df['predicted_is_error'] = full_df['qwen_predicted_has_error'].fillna(False).astype(bool)
    
    y_true = full_df['error_type'].notna().values
    y_pred = full_df['predicted_is_error'].values
    
    fname = f"{OUTPUT_FILENAME}_{args.modality}"
    if args.n_samples: fname += "_debug"
        
    evaluate_and_save(full_df, y_true, y_pred, fname, OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())