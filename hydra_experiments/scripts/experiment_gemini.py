import sys
import os
import argparse
import pandas as pd
import json
import requests
import base64
import ast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from io import BytesIO
from PIL import Image

sys.path.append(os.getcwd())
import LLM_parsing.llm as llm

BASE_DIR = os.getcwd()
TEST_FILE = os.path.join(BASE_DIR, 'benchmark_unified_full_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')
CACHE_DIR = os.path.join(BASE_DIR, '.llm_cache')
MODEL_NAME = "v25lite"

os.makedirs(CACHE_DIR, exist_ok=True)


def download_image_as_base64(url):
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return None

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
            
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        return None

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

def construct_messages(row, include_image=True):
    def get_val(col_name):
        err_col = f"{col_name}_error"
        if pd.notna(row.get(err_col)):
            return row.get(err_col)
        return row.get(col_name)

    db_val = get_val('date_begin')
    de_val = get_val('date_end')
    db_bce = row.get('date_begin_bce') == True
    de_bce = row.get('date_end_bce') == True
    
    start = f"{int(abs(db_val))} BCE" if db_bce and pd.notna(db_val) else (int(db_val) if pd.notna(db_val) else "?")
    end = f"{int(abs(de_val))} BCE" if de_bce and pd.notna(de_val) else (int(de_val) if pd.notna(de_val) else "?")
    date_str = f"{start} - {end}"

    artist = get_val('artist_name') or "Unknown Artist"
    role = get_val('artist_role')
    nat = get_val('artist_nationality')
    adb = get_val('artist_date_begin')
    ade = get_val('artist_date_end')
    
    artist_str = f"{artist}"
    if pd.notna(role): artist_str += f" ({role})"
    
    artist_info = []
    if pd.notna(nat): artist_info.append(str(nat))
    if pd.notna(adb) or pd.notna(ade):
        b = str(int(adb)) if pd.notna(adb) else "?"
        e = str(int(ade)) if pd.notna(ade) else "?"
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
    if pd.notna(period): hist_ctx.append(f"Period: {period}")
    if pd.notna(dynasty): hist_ctx.append(f"Dynasty: {dynasty}")
    if pd.notna(reign): hist_ctx.append(f"Reign: {reign}")
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

    if include_image:
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
    
    content = [{"type": "text", "text": text_prompt}]
    
    # Image handling
    if include_image:
        img_url = get_val('image_url')
        if pd.notna(img_url):
            image_data = download_image_as_base64(img_url)
            if image_data:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": image_data}
                })
        
    return [{"role": "user", "content": content}]

def compute_metrics(y_true, y_pred):
    if len(y_true) == 0: return {}
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'tp': int(np.sum((y_true == True) & (y_pred == True))),
        'fp': int(np.sum((y_true == False) & (y_pred == True))),
        'fn': int(np.sum((y_true == True) & (y_pred == False))),
        'tn': int(np.sum((y_true == False) & (y_pred == False)))
    }

def main():
    parser = argparse.ArgumentParser(description='Stage 8: Benchmark Evaluation with Gemini')
    parser.add_argument('--limit', type=int, default=None, help='Number of rows to evaluate')
    parser.add_argument('--no-image', action='store_true', help='Run evaluation without images')
    parser.add_argument('--output-suffix', type=str, default=None, help='Suffix for output files')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading test set from {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
    
    mode_suffix = "text" if args.no_image else "multimodal"
    output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    row_cache_file = os.path.join(OUTPUT_DIR, f"row_level_eval_cache_{mode_suffix}{output_suffix}.json")
    if os.path.exists(row_cache_file):
        with open(row_cache_file, 'r') as f:
            row_cache = json.load(f)
    else:
        row_cache = {}

    print(f"Processing {'text-only' if args.no_image else 'multimodal'} evaluation in chunks...")
    import concurrent.futures

    def process_row(i, row):
        return i, construct_messages(row, include_image=(not args.no_image))

    CHUNK_SIZE = 500
    for chunk_start in range(0, len(df), CHUNK_SIZE):
        chunk_df = df.iloc[chunk_start:chunk_start+CHUNK_SIZE]
        
        uncached_rows = []
        uncached_indices = []
        
        for idx, row in chunk_df.iterrows():
            obj_id = str(row.get('object_ID', idx))
            if obj_id in row_cache and row_cache[obj_id].get('has_error') is not None:
                continue
            uncached_rows.append(row)
            uncached_indices.append(obj_id)
            
        if not uncached_rows:
            continue
            
        print(f"\n[Chunk {chunk_start}/{len(df)}] Building prompts and downloading images for {len(uncached_rows)} uncached rows...")
        conversations = [None] * len(uncached_rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(process_row, i, row) for i, row in enumerate(uncached_rows)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preparing prompts"):
                i, conv = future.result()
                conversations[i] = conv

        print(f"[Chunk {chunk_start}/{len(df)}] Calling Gemini for {len(conversations)} rows...")
        results = llm.generate_json_batch(
            message_list=conversations,
            model=MODEL_NAME,
            chunk_size=50,
            max_workers=30,
            rpm_limit=500,
            cache_dir=CACHE_DIR,
            fallback_model=None
        )
        
        for obj_id, res_tuple in zip(uncached_indices, results):
            res = res_tuple[0] if isinstance(res_tuple, tuple) else res_tuple
            if isinstance(res, dict) and res:
                row_cache[obj_id] = res
                
        with open(row_cache_file, 'w') as f:
            json.dump(row_cache, f, indent=2)

    print("\nEvaluation complete! Processing Final Results...")
    predictions = []
    reasonings = []
    for idx, row in df.iterrows():
        obj_id = str(row.get('object_ID', idx))
        cached_res = row_cache.get(obj_id, {})
        
        predictions.append(cached_res.get('has_error', False))
        reasonings.append(cached_res.get('reasoning', 'N/A'))
        
    df['predicted_has_error'] = predictions
    df['llm_reasoning'] = reasonings
    
    y_true = df['error_type'].notna()
    y_pred = pd.Series(predictions).fillna(False).astype(bool)
    
    metrics = compute_metrics(y_true, y_pred)
    
    subtype_metrics = {}
    error_df = df[df['error_subtype'].notna()]
    for subtype in sorted(error_df['error_subtype'].unique()):
        mask = df['error_subtype'] == subtype
        subset_preds = y_pred[mask]
        subtype_metrics[subtype] = {
            'recall': float(subset_preds.mean()),
            'count': int(mask.sum())
        }

    report = {
        'overall': metrics,
        'by_subtype': subtype_metrics
    }
    
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    report_path = os.path.join(OUTPUT_DIR, f'evaluation_report{suffix}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    df.to_csv(os.path.join(OUTPUT_DIR, f'evaluation_results{suffix}.csv'), index=False)
    
    print("\n" + "="*50)
    print("STAGE 8: EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics.get('accuracy', 0):.2%}")
    print(f"Precision: {metrics.get('precision', 0):.2%}")
    print(f"Recall:    {metrics.get('recall', 0):.2%}")
    print(f"F1 Score:  {metrics.get('f1', 0):.2%}")
    print("\nRecall breakdown by Subtype:")
    for sub, m in subtype_metrics.items():
        print(f"  {sub:25s}: {m['recall']:.2%} (n={m['count']})")
    print("="*50)

if __name__ == "__main__":
    main()
