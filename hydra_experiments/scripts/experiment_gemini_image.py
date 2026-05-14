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
import concurrent.futures

sys.path.append(os.getcwd())
import LLM_parsing.llm as llm

BASE_DIR = os.getcwd()
TEST_FILE = os.path.join(BASE_DIR, 'benchmark_unified_full_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')
CACHE_DIR = os.path.join(BASE_DIR, '.llm_cache')
MODEL_NAME = "v25lite"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_content_cache = {}

def download_image_as_base64(url):
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return None
    
    if url in image_content_cache:
        return image_content_cache[url]

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
        base64_str = f"data:image/jpeg;base64,{encoded_string}"
        image_content_cache[url] = base64_str
        return base64_str
    except Exception:
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
    img_url = row.get('image_url')
    if pd.notna(img_url):
        image_data = download_image_as_base64(img_url)
        if image_data:
            content.append({"type": "image_url", "image_url": {"url": image_data}})
        
    return [{"role": "user", "content": content}]

def main():
    parser = argparse.ArgumentParser(description='Gemini Image-Only Strict Attribute Audit')
    parser.add_argument('--limit', type=int, default=39000, help='Total rows to evaluate (default 39k)')
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

    final_results = {} 

    for t_key, t_info in targets.items():
        print(f"\n>>> PASS: {t_info['label'].upper()} <<<")
        
        target_cache_file = os.path.join(OUTPUT_DIR, f"cache_image_only_strict_{t_key}.json")
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

        CHUNK_SIZE = 500
        for chunk_start in range(0, len(df), CHUNK_SIZE):
            chunk_df = df.iloc[chunk_start:chunk_start+CHUNK_SIZE]
            uncached = [row for _, row in chunk_df.iterrows() if str(row['object_ID']) not in target_cache]
            if not uncached: continue

            print(f"[{t_key}] Processing {len(uncached)} rows (Chunk {chunk_start})...")
            
            convs = [None] * len(uncached)
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(lambda i, r: (i, construct_messages(r, t_key, t_info['label'], get_claim(r))), idx, row) 
                           for idx, row in enumerate(uncached)]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preparing"):
                    idx, conv = future.result()
                    convs[idx] = conv

            results = llm.generate_json_batch(convs, model=MODEL_NAME, max_workers=30, rpm_limit=1000, cache_dir=CACHE_DIR)
            
            for row, res in zip(uncached, results):
                obj_id = str(row['object_ID'])
                target_cache[obj_id] = res[0] if isinstance(res, tuple) else res
            
            with open(target_cache_file, 'w') as f: json.dump(target_cache, f, indent=2)

        final_results[t_key] = target_cache

    print("\nEvaluation complete! Calculating System-Level Metrics...")
    
    y_true_system = df['_any_error'].values
    
    y_pred_system = []
    
    for _, row in df.iterrows():
        oid = str(row['object_ID'])
        is_flagged_by_any_pass = False
        for t_key in targets:
            if final_results[t_key].get(oid, {}).get('has_error', False):
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
    print("SYSTEM-LEVEL IMAGE-ONLY EVALUATION (7-Pass Audit)")
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
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report_image_only_SYSTEM.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    final_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        oid = str(row['object_ID'])
        row_dict = row.to_dict()
        row_dict['predicted_has_error'] = y_pred_system[i]
        for t_key in targets:
            res = final_results[t_key].get(oid, {})
            row_dict[f'gemini_image_only_{t_key}_flag'] = res.get('has_error', False)
            row_dict[f'gemini_image_only_{t_key}_reasoning'] = res.get('reasoning', 'N/A')
        final_rows.append(row_dict)
    
    pd.DataFrame(final_rows).to_csv(os.path.join(OUTPUT_DIR, 'evaluation_results_image_only_SYSTEM.csv'), index=False)

if __name__ == "__main__":
    main()
