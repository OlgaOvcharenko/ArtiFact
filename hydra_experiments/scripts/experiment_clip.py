import sys
import os
import time
import ast
import glob
import pandas as pd
import numpy as np
import torch
import warnings
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TEST_FILE, CLIP_MODEL_NAME, CLIP_THRESHOLD, get_effective_value, get_formatted_artwork_date, get_formatted_artist_date
from evaluate import evaluate_and_save

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = 'clip_zeroshot'
BATCH_SIZE = 512 
IMAGE_DIRS = [f"/dataset/chunk_{i:02d}" for i in range(14)] + ["/dataset/chunk_AIC"]
EMBEDDINGS_DIR = "/results/image_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/results"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_checkpoint.csv")

def load_all_embeddings_to_dict(embeddings_dir=EMBEDDINGS_DIR):
    npz_files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
    emb_dict = {}
    if not npz_files:
        print(f"⚠️ No embedding files found in {embeddings_dir}. Falling back to image loading.")
        return emb_dict
        
    for f in tqdm(npz_files, desc="Loading Image Embeddings"):
        try:
            data = np.load(f)
            ids = data['object_ids']
            embs = data['embeddings']
            for obj_id, emb in zip(ids, embs):
                emb_dict[str(obj_id)] = emb
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return emb_dict

def get_local_image(row):
    target_id = row.get('image_object_id_error')
    if pd.isna(target_id) or str(target_id).strip() == "":
        target_id = row['object_ID']
        
    img_filename = f"{target_id}.jpg"
    for chunk_dir in IMAGE_DIRS:
        img_path = os.path.join(chunk_dir, img_filename)
        if os.path.exists(img_path):
            try:
                return Image.open(img_path).convert("RGB")
            except Exception:
                return None 
    return None

def parse_dimensions(dim_str):
    if pd.isna(dim_str) or not dim_str:
        return None
    try:
        d = ast.literal_eval(str(dim_str))
        if not isinstance(d, dict): return None
        unit = d.get('unit', '')
        parts = []
        for k in ['height', 'width', 'depth', 'length', 'diameter']:
            if k in d and pd.notna(d[k]) and d[k] != '':
                parts.append(f"{k} {d[k]}")
        if not parts: return None
        res = " x ".join(parts)
        if unit: res += f" {unit}"
        return res
    except Exception:
        return None

def build_text_description(row):
    parts = []
    title = get_effective_value(row, 'title')
    if title: parts.append(f'"{title}"')
    obj = get_effective_value(row, 'object_name')
    if obj: parts.append(f"a {obj}")
    
    artist = get_effective_value(row, 'artist_name')
    if artist:
        nat = get_effective_value(row, 'artist_nationality')
        role = get_effective_value(row, 'artist_role')
        dates = get_formatted_artist_date(row)
        artist_details = []
        if nat: artist_details.append(nat)
        if role: artist_details.append(role)
        if dates: artist_details.append(dates)
        if artist_details: parts.append(f"by {artist} ({', '.join(artist_details)})")
        else: parts.append(f"by {artist}")

    artwork_date = get_formatted_artwork_date(row)
    if artwork_date: parts.append(f"date: {artwork_date}")

    mat = get_effective_value(row, 'materials')
    if mat and mat not in ('[]', 'nan'): parts.append(f"made with {mat.strip('[]\"')}")

    tech = get_effective_value(row, 'techniques')
    if tech and tech not in ('[]', 'nan'): parts.append(f"technique: {tech.strip('[]\"')}")

    dim_json = get_effective_value(row, 'dimensions_json')
    dim_str = parse_dimensions(dim_json)
    if dim_str: parts.append(f"dimensions: {dim_str}")

    for meta in ['culture', 'location', 'period', 'dynasty', 'reign']:
        val = get_effective_value(row, meta)
        if val: parts.append(f"{meta.capitalize()}: {val}")

    subj = get_effective_value(row, 'subjects')
    if subj and subj not in ('[]', 'nan'): parts.append(f"subjects: {subj.strip('[]\"')}")
        
    desc = get_effective_value(row, 'description')
    if desc:
        if len(desc) > 70: desc = desc[:70] + '...' 
        parts.append(desc)

    text = ", ".join(parts) if parts else "an artwork"
    return text

def compute_clip_similarities(df, model, processor, device, emb_dict, batch_size=512):
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        chk_df = pd.read_csv(CHECKPOINT_FILE)
        processed_ids = set(chk_df['object_ID'].astype(str))
        print(f"Found checkpoint! Skipping {len(processed_ids)} previously processed items.")
    else:
        pd.DataFrame(columns=['object_ID', 'clip_similarity']).to_csv(CHECKPOINT_FILE, index=False)

    df_todo = df[~df['object_ID'].astype(str).isin(processed_ids)].copy()
    if len(df_todo) == 0:
        print("All items already processed!")
        return

    rows_list = list(df_todo.iterrows())
    n_batches = (len(rows_list) + batch_size - 1) // batch_size
    failed_count = 0

    print(f"Processing {len(df_todo)} items in {n_batches} batches...")
    
    for batch_idx in tqdm(range(n_batches), desc="Running CLIP (Zero-Shot)"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(rows_list))
        batch_rows = rows_list[start:end]

        valid_ids = []
        valid_images = []
        valid_image_embeds = []
        valid_texts = []

        for _, row in batch_rows:
            target_id = row.get('image_object_id_error')
            if pd.isna(target_id) or str(target_id).strip() == "":
                target_id = str(row['object_ID'])
            
            # get embedding from cache
            if str(target_id) in emb_dict:
                valid_image_embeds.append(torch.from_numpy(emb_dict[str(target_id)]).to(device))
                valid_ids.append(row['object_ID'])
                valid_texts.append(build_text_description(row))
            else:
                # or image loading
                img = get_local_image(row)
                if img is not None:
                    valid_images.append(img)
                    valid_ids.append(row['object_ID'])
                    valid_texts.append(build_text_description(row))
                else:
                    failed_count += 1

        if not valid_ids: continue

        try:
            text_inputs = processor(text=valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                text_outputs = model.get_text_features(**text_inputs)
                text_embeds = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

                if valid_images:
                    img_inputs = processor(images=valid_images, return_tensors="pt").to(device)
                    img_outputs = model.get_image_features(**img_inputs)
                    img_embeds = img_outputs / img_outputs.norm(dim=-1, keepdim=True)
                    
                    if valid_image_embeds:
                        all_img_embeds = torch.cat([torch.stack(valid_image_embeds), img_embeds], dim=0)
                    else:
                        all_img_embeds = img_embeds
                else:
                    all_img_embeds = torch.stack(valid_image_embeds)

                # calculate similarity
                sims = (all_img_embeds * text_embeds).sum(dim=-1).cpu().numpy()

            batch_results = pd.DataFrame({'object_ID': valid_ids, 'clip_similarity': sims})
            batch_results.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")

    print(f"Failed items (missing image/embedding): {failed_count}")

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE)
    df['object_ID'] = df['object_ID'].astype(str)
    print(f"Dataset: {len(df)} rows")

    print("Loading Pre-computed Embeddings...")
    emb_dict = load_all_embeddings_to_dict()

    print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    compute_clip_similarities(df, model, processor, DEVICE, emb_dict, BATCH_SIZE)

    print("Merging results...")
    if os.path.exists(CHECKPOINT_FILE):
        final_results = pd.read_csv(CHECKPOINT_FILE)
        final_results['object_ID'] = final_results['object_ID'].astype(str)
        final_results = final_results.drop_duplicates(subset=['object_ID'], keep='last')
        df = df.merge(final_results, on='object_ID', how='left')
    else:
        df['clip_similarity'] = np.nan

    y_pred = np.where(df['clip_similarity'].isna(), False, df['clip_similarity'] < CLIP_THRESHOLD)
    df['predicted_has_error'] = y_pred
    y_true = df['error_type'].notna().values
    
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    evaluate_and_save(df, y_true, y_pred, EXPERIMENT_NAME, OUTPUT_DIR)

if __name__ == "__main__":
    main()
