import os
import sys
import glob
import json
import re
import pandas as pd
import argparse

sys.path.append(os.getcwd())
from LLM_parsing.llm import generate_json_batch
from prompts.artist import artist_prompts

INPUT_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset/parsed_pipeline'
MODEL = 'g25lite'

def is_empty(val):
    return pd.isna(val) or str(val).strip() in ('', 'nan', '{}', '[]', 'None')

def process_file(fpath, model):
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath, low_memory=False, dtype=str)

    artist_col = None
    possible_cols = ['artist_information', 'artist_ulan', 'principal_maker']
    for col in possible_cols:
        if col in df.columns:
            artist_col = col
            break
            
    if not artist_col or 'parsed_artist' not in df.columns:
        return

    mask = df[artist_col].notna() & df[artist_col].apply(
        lambda x: not is_empty(x)
    ) & df['parsed_artist'].apply(is_empty)

    candidates = df[mask].copy()
    if len(candidates) == 0:
        return

    print(f"  Found {len(candidates)} candidates for direct parsing in {fname} using column '{artist_col}'")

    prompt_template = artist_prompts["artist"]
    message_batch = []
    indices = candidates.index.tolist()
    
    for _, row in candidates.iterrows():
        info = str(row.get(artist_col, ''))
        
        if artist_col in ['artist_information', 'principal_maker'] and ('\n' in info or '|' in info):
            info = re.split(r'[\n|]', info)[0].strip()
            
        titles = str(row.get('artist_titles', '[]'))
        p = prompt_template.replace("{artist_information}", info).replace("{artist_titles}", titles)
        message_batch.append([{"role": "user", "content": p}])

    print(f"  Sending {len(message_batch)} records to {model}...")
    
    results = generate_json_batch(
        message_list=message_batch,
        model=model,
        chunk_size=20,
        rpm_limit=1000,
        ignore_cache=False
    )

    updated = 0
    for idx, (result_tuple, _) in zip(indices, results):
        if result_tuple and isinstance(result_tuple, dict):
            # Parse it strictly as the artist pipeline expects
            if 'scratchpad' in result_tuple:
                del result_tuple['scratchpad']
            result_tuple = [result_tuple]
            
            df.at[idx, 'parsed_artist'] = json.dumps(result_tuple)
            updated += 1

    if updated > 0:
        df.to_csv(fpath, index=False)
        print(f"Updated {updated} records in {fname}")
    else:
        print(f"No valid responses for {fname}")

def main():
    parser = argparse.ArgumentParser(description="Direct LLM mop-up for unparsed artists.")
    parser.add_argument('--input-dir', default=INPUT_DIR, help="Directory containing parsed CSVs")
    parser.add_argument('--model', default=MODEL, help="LLM model to use")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.csv')))
    files = [f for f in files if not os.path.basename(f).startswith('_')]

    print(f"Checking {len(files)} files for unparsed artists...")
    for f in files:
        process_file(f, args.model)

    print("\nDone.")

if __name__ == '__main__':
    main()
