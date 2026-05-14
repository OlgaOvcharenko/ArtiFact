import sys
import os
import argparse
import base64
import pandas as pd
import json
import re
import asyncio
import io
import ast
from PIL import Image
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TEST_FILE, OUTPUT_DIR

MODEL_NAME = os.getenv("MODEL_NAME_OVERRIDE", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ") 
PORT = os.getenv("PORT", "8000")
OUTPUT_FILENAME = "qwen2.5_72b_image_only_SYSTEM" 

IMAGE_DIRS = [f"/dataset/chunk_{i:02d}" for i in range(14)] + ["/dataset/chunk_AIC"]

MAX_CONCURRENT_REQUESTS = 50 
MAX_IMAGE_SIDE = 800
MAX_TOTAL_PIXELS = 600000

client = AsyncOpenAI(
    base_url=f"http://localhost:{PORT}/v1",
    api_key="EMPTY",
)

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

def resize_and_encode_image(image_path, max_side=MAX_IMAGE_SIDE):
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            width, height = img.size
            if max(width, height) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            
            while img.width * img.height > MAX_TOTAL_PIXELS:
                img = img.resize((int(img.width * 0.9), int(img.height * 0.9)))
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        return None

def parse_metadata_list(val):
    if pd.isna(val) or val == "" or val == "[]": return []
    if isinstance(val, list): return val
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            try: return json.loads(cleaned.replace("''", '"'))
            except: 
                try: return ast.literal_eval(cleaned)
                except: pass
        return [cleaned]
    return []

def extract_json(text):
    try: return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return {"has_error": False, "reasoning": "Parse Error: " + text}

def construct_messages(row, target_label, claim_text):
    text_prompt = f"""You are an Art Historian. Analyze the attached artwork image.

Specific Claim to Verify:
{target_label}: {claim_text}

Question: Based ONLY on the visual evidence in the image (style, period, material properties, physical appearance), is this specific claim correct?

Important:
- Only flag as has_error: true if the image CLEARLY contradicts the claim.
- Missing values (N/A, ?) are NOT errors.
- You are blind to all other metadata; judge only this specific claim.

Return a JSON object with:
- "has_error": boolean
- "reasoning": string explanation.
"""
    content = [{"type": "text", "text": text_prompt}]
    img_path = get_local_image_path(row)
    if img_path:
        image_data = resize_and_encode_image(img_path)
        if image_data:
            content.append({"type": "image_url", "image_url": {"url": image_data}})
        
    return [{"role": "user", "content": content}]

async def process_row(sem, row, t_key, t_info, claim_text, lock, target_cache, target_cache_file):
    async with sem:
        try:
            messages = construct_messages(row, t_info['label'], claim_text)
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.0 
            )
            raw_content = response.choices[0].message.content
            parsed = extract_json(raw_content)
            
        except Exception as e:
            parsed = {"has_error": False, "reasoning": f"API Error: {str(e)}"}

        async with lock:
            target_cache[str(row['object_ID'])] = parsed

async def main():
    parser = argparse.ArgumentParser(description='Qwen Image-Only Strict Attribute Audit')
    parser.add_argument('--limit', type=int, default=39000, help='Total rows to evaluate (default 39k)')
    parser.add_argument('--target', type=str, default='all', help='Specific target to evaluate (e.g. object_name, dimensions_json, or all)')
    args = parser.parse_args()

    targets = {
        'artist': {'label': 'Artist', 'gt': 'artist_name_error'},
        'date': {'label': 'Date', 'gt': ['date_begin_error', 'date_end_error']},
        'medium': {'label': 'Medium (Materials/Techniques)', 'gt': ['materials_error', 'techniques_error']},
        'culture': {'label': 'Culture', 'gt': 'culture_error'},
        'location': {'label': 'Location', 'gt': 'location_error'},
        'dimensions_json': {'label': 'Dimensions', 'gt': 'dimensions_json_error'},
        'object_name': {'label': 'Object Type', 'gt': 'image_url_error'}
    }
    
    if args.target != 'all':
        if args.target in targets:
            targets = {args.target: targets[args.target]}
            global OUTPUT_FILENAME
            OUTPUT_FILENAME = f"{OUTPUT_FILENAME}_{args.target}"
        else:
            print(f"Error: Invalid target '{args.target}'. Choose from {list(targets.keys())} or 'all'.")
            sys.exit(1)

    print(f"--- Loading Dataset: {TEST_FILE} ---")
    full_df = pd.read_csv(TEST_FILE)
    
    gt_cols = ['artist_name_error', 'date_begin_error', 'date_end_error', 'materials_error', 'techniques_error', 
               'culture_error', 'location_error', 'dimensions_json_error', 'image_url_error']
    full_df['_any_error'] = full_df[gt_cols].notna().any(axis=1)
    
    print(f"Applying balanced sampling for limit of {args.limit}...")
    dirty_subset = full_df[full_df['_any_error'] == True]
    clean_subset = full_df[full_df['_any_error'] == False]
    n_dirty = min(len(dirty_subset), args.limit // 2)
    n_clean = args.limit - n_dirty
    
    df = pd.concat([
        dirty_subset.sample(n=n_dirty, random_state=42),
        clean_subset.sample(n=n_clean, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sampled {len(df)} artworks. Starting 7-pass isolated audit...")

    final_results = {} # target -> {obj_id -> res}

    for t_key, t_info in targets.items():
        print(f"\n>>> PASS: {t_info['label'].upper()} <<<")
        
        target_cache_file = os.path.join(OUTPUT_DIR, f"qwen_cache_image_only_strict_{t_key}.json")
        if os.path.exists(target_cache_file):
            with open(target_cache_file, 'r') as f: target_cache = json.load(f)
        else: target_cache = {}

        def get_claim(row):
            def gv(c): return row.get(f"{c}_error") if pd.notna(row.get(f"{c}_error")) else row.get(c)
            if t_key == 'date':
                db, de = gv('date_begin'), gv('date_end')
                start = f"{int(abs(db))} BCE" if row.get('date_begin_bce') and pd.notna(db) else (int(db) if pd.notna(db) else "?")
                end = f"{int(abs(de))} BCE" if row.get('date_end_bce') and pd.notna(de) else (int(de) if pd.notna(de) else "?")
                return f"{start} - {end}"
            elif t_key == 'artist':
                a, n = gv('artist_name'), gv('artist_nationality')
                return f"{a or 'Unknown Artist'} ({n if pd.notna(n) else '?'})"
            elif t_key == 'medium':
                m = sorted(list(set(parse_metadata_list(gv('materials')) + parse_metadata_list(gv('techniques')))))
                return ", ".join(m) if m else "N/A"
            else:
                return str(gv(t_key) or 'N/A')

        uncached = [row for _, row in df.iterrows() if str(row['object_ID']) not in target_cache]
        
        if uncached:
            print(f"[{t_key}] Processing {len(uncached)} remaining rows...")
            sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            lock = asyncio.Lock()
            
            CHUNK_SIZE = 500
            for chunk_start in range(0, len(uncached), CHUNK_SIZE):
                chunk_df = uncached[chunk_start:chunk_start+CHUNK_SIZE]
                print(f"  -> Chunk {chunk_start}/{len(uncached)}")
                tasks = [process_row(sem, row, t_key, t_info, get_claim(row), lock, target_cache, target_cache_file) for row in chunk_df]
                await tqdm_asyncio.gather(*tasks)
                
                with open(target_cache_file, 'w') as f: json.dump(target_cache, f, indent=2)

        final_results[t_key] = target_cache

    print("\nEvaluation complete! Calculating System-Level Metrics...")
    
    y_true_system = df['_any_error'].values
    y_pred_system = []
    
    for _, row in df.iterrows():
        oid = str(row['object_ID'])
        is_flagged_by_any_pass = False
        for t_key in targets:
            if final_results[t_key].get(oid, {}).get('has_error', False) is True:
                is_flagged_by_any_pass = True
                break
        y_pred_system.append(is_flagged_by_any_pass)
    
    y_pred_system = np.array(y_pred_system)
    
    metrics = {
        'precision': float(precision_score(y_true_system, y_pred_system, zero_division=0)),
        'recall': float(recall_score(y_true_system, y_pred_system, zero_division=0)),
        'f1': float(f1_score(y_true_system, y_pred_system, zero_division=0)),
        'accuracy': float(accuracy_score(y_true_system, y_pred_system)),
        'tp': int(np.sum((y_true_system == True) & (y_pred_system == True))),
        'fp': int(np.sum((y_true_system == False) & (y_pred_system == True))),
        'fn': int(np.sum((y_true_system == True) & (y_pred_system == False))),
        'tn': int(np.sum((y_true_system == False) & (y_pred_system == False)))
    }

    category_recall = {}
    for t_key, t_info in targets.items():
        if isinstance(t_info['gt'], list):
            gt_mask = df[t_info['gt']].notna().any(axis=1).values
        else:
            gt_mask = df[t_info['gt']].notna().values
            
        support = int(np.sum(gt_mask))
        if support > 0:
            tp = np.sum((gt_mask == True) & (y_pred_system == True))
            category_recall[t_info['label']] = {'recall': float(tp / support), 'count': support}

    print("\n" + "="*60)
    print("SYSTEM-LEVEL QWEN IMAGE-ONLY EVALUATION (7-Pass Audit)")
    print("="*60)
    print(f"Overall Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Overall Precision: {metrics['precision']:.2%}")
    print(f"Overall Recall:    {metrics['recall']:.2%}")
    print(f"Overall F1 Score:  {metrics['f1']:.2%}")
    print("-" * 60)
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    print("-" * 60)
    print("\nSensitivity breakdown (Recall by Error Category):")
    for label, info in category_recall.items():
        print(f"  {label:<30s}: {info['recall']:.2%} (n={info['count']})")
    print("="*60)
    
    report = {'overall': metrics, 'category_recall': category_recall, 'limit': args.limit}
    with open(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME}_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    final_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        oid = str(row['object_ID'])
        row_dict = row.to_dict()
        row_dict['predicted_has_error'] = y_pred_system[i]
        for t_key in targets:
            res = final_results[t_key].get(oid, {})
            row_dict[f'qwen_image_only_{t_key}_flag'] = res.get('has_error', False)
            row_dict[f'qwen_image_only_{t_key}_reasoning'] = res.get('reasoning', 'N/A')
        final_rows.append(row_dict)
    
    pd.DataFrame(final_rows).to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME}_report.csv'), index=False)

if __name__ == "__main__":
    asyncio.run(main())