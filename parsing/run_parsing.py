import os
import argparse
import pandas as pd
import json
from tqdm import tqdm

from parse_dates import parse_normalized_date
from parse_dimensions import parse_normalized_dimensions
from parse_medium import parse_medium as parse_medium_func, ensure_resources, term_dictionary
from parse_artist import parse_artist_info_string
import re

def safe_json_dumps(obj):
    if not obj:
        return None
    return json.dumps(obj, ensure_ascii=False)

def extract_rijks_medium(row):
    ensure_resources()
    
    raw_materials = []
    if 'material' in row and pd.notna(row['material']):
        raw_materials = [re.sub(r'\(.*?\)', '', m).strip().lower() for m in str(row['material']).split(',') if m.strip()]
        
    raw_techniques = []
    if 'technique' in row and pd.notna(row['technique']):
        raw_techniques = [re.sub(r'\(.*?\)', '', t).strip().lower() for t in str(row['technique']).split(',') if t.strip()]
        
    final_materials = set()
    final_techniques = set()
    
    for m in raw_materials:
        if not m: continue
        if m in term_dictionary and term_dictionary[m].get('type') == 'technique':
            final_techniques.add(m)
        else:
            final_materials.add(m)
            
    for t in raw_techniques:
        if not t: continue
        if t in term_dictionary and term_dictionary[t].get('type') == 'material':
            final_materials.add(t)
        else:
            final_techniques.add(t)
            
    if not final_materials and not final_techniques:
        return None
        
    return {
        "materials": sorted(list(final_materials)),
        "techniques": sorted(list(final_techniques))
    }

def dedupe_rijks_artist(raw):
    if pd.isna(raw) or not str(raw).strip():
        return raw
    text = str(raw).strip().lower()
    text = re.sub(r"[\[\]\"']", '', text)
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) <= 1:
        return text.strip()
    for candidate in sorted(parts, key=len, reverse=True):
        if all(p in candidate for p in parts):
            return candidate
    return max(parts, key=len)

def combine_medium_string(row) -> str:
    parts = []
    for col in ['medium', 'material', 'technique']:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() != 'nan':
                parts.append(val)
    return " , ".join(parts) if parts else ""

def parse_file(input_path: str, output_dir: str) -> dict:
    basename = os.path.basename(input_path)
    print(f"\nProcessing {basename}...")
    
    is_aic = basename.startswith("aic_")
    is_rijks = basename.startswith("rijks_")
    
    df = pd.read_csv(input_path, dtype=str)
    
    parsed_dates = []
    parsed_dims = []
    parsed_mediums = []
    parsed_artists = []
    
    stats = {
        'total': len(df),
        'date_attempted': 0, 'date_success': 0,
        'dim_attempted': 0, 'dim_success': 0,
        'med_attempted': 0, 'med_success': 0,
        'art_attempted': 0, 'art_success': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing rows"):
        p_date = None
        raw_date_str = str(row['raw_date']).strip() if 'raw_date' in row and pd.notna(row['raw_date']) else ""
        norm_date_str = str(row['date']).strip() if 'date' in row and pd.notna(row['date']) else ""
        
        if raw_date_str and raw_date_str.lower() != 'nan':
            stats['date_attempted'] += 1
            
            if norm_date_str and norm_date_str.lower() != 'nan':
                p_date = parse_normalized_date(norm_date_str)
                
            if not p_date:
                p_date = parse_normalized_date(raw_date_str)
                
            if p_date: stats['date_success'] += 1
        parsed_dates.append(safe_json_dumps(p_date))
        
        p_dims = None
        raw_dim_str = str(row['raw_dimensions']).strip() if 'raw_dimensions' in row and pd.notna(row['raw_dimensions']) else ""
        norm_dim_str = str(row['dimensions']).strip() if 'dimensions' in row and pd.notna(row['dimensions']) else ""
        
        if raw_dim_str and raw_dim_str.lower() != 'nan':
            stats['dim_attempted'] += 1
            
            if norm_dim_str and norm_dim_str.lower() != 'nan':
                p_dims = parse_normalized_dimensions(norm_dim_str)
                
            if not p_dims:
                p_dims = parse_normalized_dimensions(raw_dim_str)
                
            if p_dims: stats['dim_success'] += 1
        parsed_dims.append(safe_json_dumps(p_dims))
        
        p_med = None
        if is_rijks:
            has_mat = 'material' in row and pd.notna(row['material']) and str(row['material']).strip() and str(row['material']).strip().lower() != 'nan'
            has_tech = 'technique' in row and pd.notna(row['technique']) and str(row['technique']).strip() and str(row['technique']).strip().lower() != 'nan'
            if has_mat or has_tech:
                 stats['med_attempted'] += 1
            p_med = extract_rijks_medium(row)
            if p_med: stats['med_success'] += 1
        else:
            cmb_med = combine_medium_string(row)
            if cmb_med and cmb_med.lower() != 'nan':
                stats['med_attempted'] += 1
                p_med = parse_medium_func(cmb_med)
                if p_med: stats['med_success'] += 1
        parsed_mediums.append(safe_json_dumps(p_med))
        
        p_art = None
        if is_aic and 'artist_information' in row and pd.notna(row['artist_information']):
            art_str = str(row['artist_information']).strip()
            if art_str and art_str.lower() != 'nan':
                stats['art_attempted'] += 1
                p_art = parse_artist_info_string(art_str)
                if p_art: stats['art_success'] += 1
        parsed_artists.append(safe_json_dumps(p_art))

    df['parsed_date'] = parsed_dates
    df['parsed_dimensions'] = parsed_dims
    df['parsed_medium'] = parsed_mediums

    if is_aic:
        df['parsed_artist'] = parsed_artists
    elif 'parsed_artist' not in df.columns:
        df['parsed_artist'] = None

    if is_rijks and 'artist_name' in df.columns:
        df['artist_name'] = df['artist_name'].apply(dedupe_rijks_artist)

    out_path = os.path.join(output_dir, basename)
    df.to_csv(out_path, index=False)
    
    print(f"Saved parsed dataset to: {out_path}")
    print("File Parse Success Rates:")
    if stats['date_attempted'] > 0:
        print(f"Date: {stats['date_success']}/{stats['date_attempted']} ({stats['date_success']/stats['date_attempted']:.1%})")
    if stats['dim_attempted'] > 0:
        print(f"dimensions: {stats['dim_success']}/{stats['dim_attempted']} ({stats['dim_success']/stats['dim_attempted']:.1%})")
    if stats['med_attempted'] > 0:
        print(f"Medium:     {stats['med_success']}/{stats['med_attempted']} ({stats['med_success']/stats['med_attempted']:.1%})")
    if is_aic and stats['art_attempted'] > 0:
        print(f"Artist: {stats['art_success']}/{stats['art_attempted']} ({stats['art_success']/stats['art_attempted']:.1%})")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Parse artwork metadata into JSON fields.")
    parser.add_argument("--input-dir", required=True, help="Directory containing normalized CSVs")
    parser.add_argument("--output-dir", default="parsed_pipeline", help="Directory for final parsed CSVs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
        
    print(f"Found {len(csv_files)} files to parse.")
    
    global_stats = {
        'total': 0,
        'date_attempted': 0, 'date_success': 0,
        'dim_attempted': 0, 'dim_success': 0,
        'med_attempted': 0, 'med_success': 0,
        'art_attempted': 0, 'art_success': 0
    }
    
    for f in csv_files:
        file_stats = parse_file(os.path.join(args.input_dir, f), args.output_dir)
        for k in global_stats:
            global_stats[k] += file_stats[k]
            
    print("\n" + "="*50)
    print("FULL PIPELINE PARSING SUMMARY")
    print("="*50)
    print(f"Total rows processed: {global_stats['total']}")
    
    summary_data = []
    
    if global_stats['date_attempted'] > 0:
        pct = global_stats['date_success'] / global_stats['date_attempted']
        print(f"Dates successfully parsed: {global_stats['date_success']:>8} / {global_stats['date_attempted']:>8}  ({pct:6.1%})")
        summary_data.append({"Column": "date", "Attempted": global_stats['date_attempted'], "Success": global_stats['date_success'], "Success_Rate": f"{pct:.1%}"})
        
    if global_stats['dim_attempted'] > 0:
        pct = global_stats['dim_success'] / global_stats['dim_attempted']
        print(f"Dimensions successfully parsed: {global_stats['dim_success']:>8} / {global_stats['dim_attempted']:>8}  ({pct:6.1%})")
        summary_data.append({"Column": "dimensions", "Attempted": global_stats['dim_attempted'], "Success": global_stats['dim_success'], "Success_Rate": f"{pct:.1%}"})
        
    if global_stats['med_attempted'] > 0:
        pct = global_stats['med_success'] / global_stats['med_attempted']
        print(f"Mediums successfully parsed: {global_stats['med_success']:>8} / {global_stats['med_attempted']:>8}  ({pct:6.1%})")
        summary_data.append({"Column": "medium", "Attempted": global_stats['med_attempted'], "Success": global_stats['med_success'], "Success_Rate": f"{pct:.1%}"})
        
    if global_stats['art_attempted'] > 0:
        pct = global_stats['art_success'] / global_stats['art_attempted']
        print(f"Artists successfully parsed: {global_stats['art_success']:>8} / {global_stats['art_attempted']:>8}  ({pct:6.1%})  [AIC Only]")
        summary_data.append({"Column": "artist", "Attempted": global_stats['art_attempted'], "Success": global_stats['art_success'], "Success_Rate": f"{pct:.1%}"})
        
    print("="*50 + "\n")
    
    if summary_data:
        stats_df = pd.DataFrame(summary_data)
        stats_path = os.path.join(args.output_dir, "_parsing_summary.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Global parsing statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
