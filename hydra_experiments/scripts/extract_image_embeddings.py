import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


MODEL_PATH = "/models/clip-vit-l-14"
IMAGE_DIRS = [f"/dataset/chunk_{i:02d}" for i in range(14)] + ["/dataset/chunk_AIC"]
OUTPUT_DIR = "/results/image_embeddings"
BATCH_SIZE = 256
CHUNK_SIZE = 10000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_all_image_paths():
    print("Scanning folders for images...")
    all_paths = []
    for d in IMAGE_DIRS:
        if os.path.exists(d):
            paths = glob.glob(os.path.join(d, "*.jpg"))
            all_paths.extend(paths)
    
    all_paths.sort()
    return all_paths

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # IMAGES
    all_image_paths = get_all_image_paths()
    total_images = len(all_image_paths)
    print(f"Found a total of {total_images} images across all chunks.")

    if total_images == 0:
        print("No images found. Check your container mounts.")
        return

    # load model
    print(f"Loading CLIP from {MODEL_PATH} onto {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    model.eval()

    num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in range(num_chunks):
        chunk_file = os.path.join(OUTPUT_DIR, f"embeddings_chunk_{chunk_idx:04d}.npz")
        
        if os.path.exists(chunk_file):
            print(f"Chunk {chunk_idx} already exists. Skipping...")
            continue
            
        print(f"\nProcessing Chunk {chunk_idx} / {num_chunks - 1}")
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_images)
        chunk_paths = all_image_paths[start_idx:end_idx]
        
        chunk_object_ids = []
        chunk_embeddings = []
        
        # process chunk in batches
        n_batches = (len(chunk_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for b in tqdm(range(n_batches), desc=f"Chunk {chunk_idx} Batches"):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, len(chunk_paths))
            batch_paths = chunk_paths[b_start:b_end]
            
            valid_images = []
            valid_ids = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    valid_images.append(img)
                    obj_id = os.path.basename(path).replace(".jpg", "")
                    valid_ids.append(obj_id)
                except Exception:
                    continue
            
            if not valid_images:
                continue
                
            try:
                inputs = processor(images=valid_images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats_np = feats.cpu().numpy()
                    
                chunk_embeddings.append(feats_np)
                chunk_object_ids.extend(valid_ids)
                
            except Exception as e:
                print(f"Error in batch: {e}")
                
        if chunk_embeddings:
            final_embeddings = np.concatenate(chunk_embeddings, axis=0)
            final_ids = np.array(chunk_object_ids)
            
            np.savez(chunk_file, object_ids=final_ids, embeddings=final_embeddings)
            print(f"Saved {len(final_ids)} embeddings to {chunk_file}")

    print("\nAll image embeddings successfully extracted and saved!")

if __name__ == "__main__":
    main()
