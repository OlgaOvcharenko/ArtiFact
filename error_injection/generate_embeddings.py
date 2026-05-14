import pandas as pd
import numpy as np
import os
import requests
import glob
import argparse
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoProcessor, AutoModel
import yaml

BASE_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset'
DEFAULT_CONFIG = os.path.join(BASE_DIR, 'error_injection/benchmark_config.yaml')
EMB_DIR = os.path.join(BASE_DIR, 'error_injection/embeddings')
CHUNKS_DIR = os.path.join(EMB_DIR, 'chunks')
os.makedirs(CHUNKS_DIR, exist_ok=True)

MODEL_NAME = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 32
NUM_WORKERS = 16
BATCH_SIZE = 32
SAVE_EVERY_N_BATCHES = 400

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_all_images(config, override_limit=None):
    target_size = override_limit if override_limit is not None else config['dataset'].get('target_size', 0)
    seed = config['dataset'].get('random_seed', 42)
    files = config['dataset']['input_files']
    
    if target_size == 0:
        print("Target size is 0: Loading ALL images for full dataset index:")
        dfs = []
        for file_path in files:
            full_path = os.path.join(BASE_DIR, file_path)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path, usecols=['object_ID', 'image_url'], low_memory=False)
                dfs.append(df)
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset='object_ID').dropna(subset=['image_url'])
        n_before = len(merged)
        merged = merged[~merged['image_url'].str.contains('artic.edu', na=False)]
        print(f"Skipping AIC images: {n_before - len(merged)} removed, {len(merged)} remaining.")
        return merged

    print(f"Target size is {target_size}: Performing proportional sampling for test index:")
    file_metadata = []
    total_raw_rows = 0
    for file_path in files:
        full_path = os.path.join(BASE_DIR, file_path)
        if os.path.exists(full_path):
            count = sum(1 for _ in open(full_path)) - 1
            file_metadata.append({'path': file_path, 'count': count})
            total_raw_rows += count

    sampled_dfs = []
    for meta in file_metadata:
        weight = meta['count'] / total_raw_rows
        file_target = max(1, int(target_size * weight))
        df = pd.read_csv(os.path.join(BASE_DIR, meta['path']), usecols=['object_ID', 'image_url'], low_memory=False)
        df = df.dropna(subset=['image_url'])
        df = df[~df['image_url'].str.contains('artic.edu', na=False)]
        if len(df) <= file_target:
            sampled_dfs.append(df)
        else:
            sampled_dfs.append(df.sample(n=file_target, random_state=seed))
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    return final_df.drop_duplicates(subset='object_ID')

def rewrite_url_for_small(url):
    if not isinstance(url, str):
        return url
    if 'iiif.micr.io' in url or ('iiif' in url and '/full/max/' in url):
        url = url.replace('/full/max/0/', '/full/400,/0/')
        url = url.replace('/full/full/0/', '/full/400,/0/')
    elif 'metmuseum.org' in url and '/original/' in url:
        url = url.replace('/original/', '/web-large/')
    return url

def load_image(url):
    try:
        url = rewrite_url_for_small(url)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, timeout=12, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        return img
    except:
        return None

def save_chunk(embeddings, ids, chunk_idx):
    if not embeddings: return
    emb_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:06d}_embs.npy')
    ids_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:06d}_ids.npy')
    np.save(emb_path, np.vstack(embeddings))
    np.save(ids_path, np.array(ids))

def main():
    parser = argparse.ArgumentParser(description='Generate visual embeddings using fast CLIP.')
    parser.add_argument('config_path', nargs='?', default=DEFAULT_CONFIG, help='Path to config YAML')
    parser.add_argument('--limit', type=int, default=None, help='Override target size (total rows)')
    args = parser.parse_args()
    
    config_path = args.config_path
    print(f"Loading config from {config_path}:")
    config = load_config(config_path)
    
    df = load_all_images(config, override_limit=args.limit)
    if df.empty:
        print("No images found in input files.")
        return

    print(f"Loading {MODEL_NAME}:")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print(f"Processing {len(df)} unique images:")
    processed_set = set()
    chunk_files = glob.glob(os.path.join(CHUNKS_DIR, '*_ids.npy'))
    if chunk_files:
        print(f"Found {len(chunk_files)} existing chunks. Scanning IDs:")
        for f in chunk_files:
            ids = np.load(f)
            processed_set.update(str(oid) for oid in ids)
        print(f" Already processed: {len(processed_set)} images.")

    main_ids_path = os.path.join(EMB_DIR, 'object_ids.npy')
    if os.path.exists(main_ids_path):
        old_ids = np.load(main_ids_path)
        processed_set.update(str(oid) for oid in old_ids)
        print(f" Legacy IDs loaded: {len(old_ids)}")

    df_to_process = df[~df['object_ID'].astype(str).isin(processed_set)]
    
    if df_to_process.empty:
        print("All images in the config are already processed! Done.")
        return

    print(f"Resuming: {len(df_to_process)} images left to encode.")
    rows = list(df_to_process.iterrows())

    def download_worker(row_tuple):
        _, r = row_tuple
        url = r.get('image_url')
        oid = r.get('object_ID')
        if pd.isna(url):
            return None
        img = load_image(url)
        if img:
            return (oid, img)
        return None

    def download_batch(batch_rows):
        """Download a batch of images in parallel."""
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            try:
                results = list(ex.map(download_worker, batch_rows, timeout=45))
            except Exception:
                results = [None] * len(batch_rows)
        return results

    current_chunk_embs = []
    current_chunk_ids = []
    chunk_count = len(chunk_files)

    print(f"Processing {len(rows)} images in batches of {BATCH_SIZE} with {NUM_WORKERS} download workers:")

    total_success = 0
    total_fail = 0
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    n_batches = len(batches)

    next_batch_future = None
    prefetch_executor = ThreadPoolExecutor(max_workers=1)

    try:
        with tqdm(total=n_batches) as pbar:
            for b_idx, batch_rows in enumerate(batches):

                if b_idx + 1 < n_batches:
                    next_batch_future = prefetch_executor.submit(download_batch, batches[b_idx + 1])

                if b_idx == 0:
                    results = download_batch(batch_rows)
                else:
                    results = current_results

                valid = [(oid, img) for r in results if r for oid, img in [r]]

                batch_success = len(valid)
                batch_fail = len(batch_rows) - batch_success
                total_success += batch_success
                total_fail += batch_fail

                if valid:
                    batch_ids = [v[0] for v in valid]
                    batch_imgs = [v[1] for v in valid]

                    try:
                        with torch.no_grad():
                            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
                            image_features = model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        embs = image_features.cpu().float().numpy()
                        current_chunk_embs.append(embs)
                        current_chunk_ids.extend(batch_ids)
                    except Exception as e:
                        print(f"\n  [ERROR] Batch encoding failed at batch {b_idx}: {e}")

                if (b_idx + 1) % SAVE_EVERY_N_BATCHES == 0:
                    if current_chunk_embs:
                        save_chunk(current_chunk_embs, current_chunk_ids, chunk_count)
                        current_chunk_embs = []
                        current_chunk_ids = []
                        chunk_count += 1

                if next_batch_future is not None:
                    current_results = next_batch_future.result()
                    next_batch_future = None

                pbar.update(1)
                pbar.set_postfix({'ok': total_success, 'fail': total_fail})
    finally:
        prefetch_executor.shutdown(wait=False)

    if current_chunk_embs:
        save_chunk(current_chunk_embs, current_chunk_ids, chunk_count)

    print("\nMerging all chunks into final index:")
    all_chunks_embs = []
    all_chunks_ids = []
    
    if os.path.exists(os.path.join(EMB_DIR, 'clip_embeddings.npy')):
        all_chunks_embs.append(np.load(os.path.join(EMB_DIR, 'clip_embeddings.npy')))
        all_chunks_ids.append(np.load(os.path.join(EMB_DIR, 'object_ids.npy')))

    emb_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, '*_embs.npy')))
    id_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, '*_ids.npy')))
    
    for e_f, i_f in zip(emb_files, id_files):
        all_chunks_embs.append(np.load(e_f))
        all_chunks_ids.append(np.load(i_f))

    if all_chunks_embs:
        final_embs = np.vstack(all_chunks_embs)
        final_ids = np.concatenate(all_chunks_ids)
        
        np.save(os.path.join(EMB_DIR, 'clip_embeddings.npy'), final_embs)
        np.save(os.path.join(EMB_DIR, 'object_ids.npy'), final_ids)
        print(f"Final index saved: {final_embs.shape}")
    
    print("Process complete.")

if __name__ == "__main__":
    main()
