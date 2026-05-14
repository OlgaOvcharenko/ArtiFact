import os
import sys
import glob
import json
import re
import pandas as pd
import argparse

sys.path.append(os.getcwd())
from LLM_parsing.llm import generate_json_batch

INPUT_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset/parsed_pipeline'
MODEL = 'g25lite'

SYSTEM_PROMPT = """You are an expert at parsing museum artwork dimension strings into structured JSON.

Given a raw dimension string, extract the measurements and return ONLY a JSON object in this exact format:
{
  "parts": [
    {
      "part": "<part name or empty string if single piece>",
      "measurements": [
        {"attribute": "<height|width|depth|length|diameter|weight|thickness>", "value": <float in cm or grams>, "note": "<optional context like 'with frame' or 'overall'>"}
      ]
    }
  ]
}

Rules:
- Convert everything to metric: cm for lengths, grams for weights
- Convert inches using: 1 in = 2.54 cm
- Convert oz using: 1 oz = 28.349 g, 1 lb = 453.592 g
- Mixed fractions like '2 1/4' = 2.25
- If multiple separate parts (a, b, c), create separate part entries
- Return ONLY the JSON, no explanation
"""

def build_prompt(raw_str):
    return f"Parse this dimension string:\n{raw_str}"

def is_empty(val):
    return pd.isna(val) or str(val).strip() in ('', 'nan', '{}', '[]', 'None')

def process_file(fpath, model):
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath, low_memory=False, dtype=str)

    if 'raw_dimensions' not in df.columns or 'parsed_dimensions' not in df.columns:
        return

    mask = df['raw_dimensions'].notna() & df['raw_dimensions'].apply(
        lambda x: not is_empty(x)
    ) & df['parsed_dimensions'].apply(is_empty)

    candidates = df[mask]
    if len(candidates) == 0:
        return

    print(f"  Found {len(candidates)} candidates for direct parsing in {fname}")

    raw_strings = candidates['raw_dimensions'].tolist()
    
    message_batch = []
    indices = candidates.index.tolist()
    
    for r in raw_strings:
        p = build_prompt(r)
        message_batch.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p}
        ])

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
        if result_tuple and isinstance(result_tuple, dict) and 'parts' in result_tuple:
            df.at[idx, 'parsed_dimensions'] = json.dumps(result_tuple)
            updated += 1

    if updated > 0:
        df.to_csv(fpath, index=False)
        print(f"Updated {updated} records in {fname}")
    else:
        print(f"No valid responses for {fname}")


def main():
    parser = argparse.ArgumentParser(description="Direct LLM mop-up for unparsed dimensions.")
    parser.add_argument('--input-dir', default=INPUT_DIR, help="Directory containing parsed CSVs")
    parser.add_argument('--model', default=MODEL, help="LLM model to use")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.csv')))
    files = [f for f in files if not os.path.basename(f).startswith('_')]

    print(f"Checking {len(files)} files for unparsed dimensions...")
    for f in files:
        process_file(f, args.model)

    print("\nDone.")

if __name__ == '__main__':
    main()
