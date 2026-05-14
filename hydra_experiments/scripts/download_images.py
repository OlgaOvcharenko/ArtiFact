import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor

TRAIN_FILE = "/data/benchmark_20k_train.csv"
TEST_FILE = "/data/benchmark_20k_test.csv"
OUTPUT_DIR = "/tmp/raw_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_effective_url(row):
    """
    Determines which URL to download.
    Priority: 'image_error' (if exists) > 'image'
    """
    if "image_error" in row and pd.notna(row["image_error"]):
        return row["image_error"]
    return row["image"]

def load_all_data():
    """Loads and combines train/test to get a full download list."""
    tasks = []
    
    # 1. Load Train
    if os.path.exists(TRAIN_FILE):
        print(f"Loading {TRAIN_FILE}...")
        df_train = pd.read_csv(TRAIN_FILE)
        for _, row in df_train.iterrows():
            tasks.append((str(row['object_ID']), get_effective_url(row)))
    else:
        print(f"Warning: {TRAIN_FILE} not found.")

    # 2. Load Test
    if os.path.exists(TEST_FILE):
        print(f"Loading {TEST_FILE}...")
        df_test = pd.read_csv(TEST_FILE)
        for _, row in df_test.iterrows():
            tasks.append((str(row['object_ID']), get_effective_url(row)))
    else:
        print(f"Warning: {TEST_FILE} not found.")
        
    return tasks

def download_image(task):
    object_id, url = task
    
    # Filename is simply the Object ID
    filename = os.path.join(OUTPUT_DIR, f"{object_id}.jpg")
    
    if os.path.exists(filename):
        return # Skip if exists

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download ID {object_id}: {e}")

# --- MAIN ---
all_tasks = load_all_data()
print(f"Found {len(all_tasks)} images to download.")

print("Starting download with 16 threads...")
with ThreadPoolExecutor(max_workers=16) as executor:
    executor.map(download_image, all_tasks)

print("Download complete.")
