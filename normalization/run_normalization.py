import os
import argparse
import pandas as pd
from tqdm import tqdm

from normalize_dates import normalize_date
from normalize_dimensions import normalize_dimensions

def normalize_file(input_path: str, output_dir: str):
    basename = os.path.basename(input_path)
    print(f"\nProcessing {basename}...")
    
    prefix = "MET"
    if basename.startswith("aic_"):
        prefix = "AIC"
    elif basename.startswith("rijks_"):
        prefix = "RIJKS"
        
    df = pd.read_csv(input_path, dtype=str)
    
    if 'date' in df.columns:
        df['raw_date'] = df['date']
    if 'dimensions' in df.columns:
        df['raw_dimensions'] = df['dimensions']
        
    normalized_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing rows"):
        if 'date' in row and pd.notna(row['date']) and str(row['date']).strip():
            raw_d = str(row['date'])
            norm_d = normalize_date(raw_d)
            if norm_d:
                df.at[idx, 'date'] = norm_d
            else:
                df.at[idx, 'date'] = None
                
        if 'dimensions' in row and pd.notna(row['dimensions']) and str(row['dimensions']).strip():
            raw_dim = str(row['dimensions'])
            norm_dim = normalize_dimensions(raw_dim)
            if norm_dim:
                df.at[idx, 'dimensions'] = norm_dim
            else:
                df.at[idx, 'dimensions'] = None

    if 'object_ID' in df.columns:
        df['object_ID'] = df['object_ID'].apply(
            lambda x: f"{prefix}_{str(x)}" if pd.notna(x) and not str(x).startswith(f"{prefix}_") else x
        )
        
    exceptions = {'object_ID', 'title', 'description', 'inscriptions', 'image_url', 'cho_uri', 'raw_date', 'raw_dimensions'}
    cols_to_lower = [col for col in df.columns if col not in exceptions]
    for col in cols_to_lower:
        df[col] = df[col].astype(str).str.lower().where(pd.notna(df[col]), None)

    out_path = os.path.join(output_dir, basename)
    df.to_csv(out_path, index=False)
    print(f"Saved normalized dataset to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Normalize artwork dates and dimensions.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw extraction CSVs")
    parser.add_argument("--output-dir", default="normalized_pipeline", help="Directory for normalized CSVs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
        
    print(f"Found {len(csv_files)} files to normalize.")
    for f in csv_files:
        normalize_file(os.path.join(args.input_dir, f), args.output_dir)

if __name__ == "__main__":
    main()
