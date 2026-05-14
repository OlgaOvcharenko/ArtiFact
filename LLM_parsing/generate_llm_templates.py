import pandas as pd
import glob
from collections import Counter
import re
import os
import csv
import argparse

def generate_dimensions_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\s*[×x]\s*', ' x ', s)
    
    # numbers and fractions
    s = re.sub(r'\d+[-\s]+\d+/\d+', '<NUM>', s)
    s = re.sub(r'\d+/\d+', '<NUM>', s)
    s = re.sub(r'\d+(?:\.\d+)?', '<NUM>', s)
    
    # units
    s = re.sub(r'\b(?:cm|mm)\b\.?', '<CM>', s)
    s = re.sub(r'\b(?:in|inches)\b\.?', '<IN>', s)
    s = re.sub(r'\b(?:g|kg|oz|lb)\b\.?', '<WT>', s)
    
    # labels
    labels = {
        r'\b(?:h|height(?:s)?|hght)\b\.?': '<H>',
        r'\b(?:w|width(?:s)?|wdth)\b\.?': '<W>',
        r'\b(?:d|depth|dpth)\b\.?': '<D>',
        r'\b(?:l|length|len)\b\.?': '<L>',
        r'\b(?:diam|diameter|dia)\b\.?': '<DIAM>',
        r'\b(?:wt|weight)\b\.?': '<WEIGHT>',
        r'\b(?:th|thickness|thick)\b\.?': '<THICK>'
    }
    
    for pattern, token in labels.items():
        s = re.sub(pattern, token, s)
        
    # prefixes/modifiers
    s = re.sub(r'\b(?:max|min|overall|sheet|image|plate|paper|mount|mounts|frame)\b\.?:?', '<MOD>', s)

    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_date_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\d{1,4}', '<NUM>', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_artist_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\d{1,4}', '<NUM>', s)
    s = re.sub(r'[a-z]+', '<STR>', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def main():
    parser = argparse.ArgumentParser(description="Identify unique attribute templates from unparsed records.")
    parser.add_argument("--task", required=True, choices=['dimensions', 'date', 'artist'], help="Attribute type to scan.")
    parser.add_argument("--input-dir", default="parsed_pipeline", help="Directory containing parsed CSVs to scan.")
    args = parser.parse_args()


    col_mapping = {
        'dimensions': ('raw_dimensions', 'parsed_dimensions'),
        'date': ('raw_date', 'parsed_date'),
        'artist': ('artist_information', 'parsed_artist')
    }
    
    raw_col, parsed_col = col_mapping[args.task]
    template_func = {
        'dimensions': generate_dimensions_template,
        'date': generate_date_template,
        'artist': generate_artist_template
    }[args.task]

    print(f"Collecting unparsed {args.task} from {args.input_dir}...")
    failed_items = []
    
    search_pattern = os.path.join(args.input_dir, '*.csv')
    for file in glob.glob(search_pattern):
        if os.path.basename(file).startswith('_'):
            continue
        
        if args.task == 'artist' and ('dept_' in os.path.basename(file) or 'rijks_' in os.path.basename(file)):
            continue
            
        print(f"Processing {file}...")
        actual_header = pd.read_csv(file, nrows=0).columns.tolist()
        stripped_to_actual = {c.strip(): c for c in actual_header}
        
        if raw_col not in stripped_to_actual or parsed_col not in stripped_to_actual:
            continue
            
        actual_raw_col = stripped_to_actual[raw_col]
        actual_parsed_col = stripped_to_actual[parsed_col]
        
        try:
            df = pd.read_csv(file, usecols=[actual_raw_col, actual_parsed_col], dtype=str)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if raw_col not in df.columns:
            print(f"CRITICAL: {raw_col} not in df.columns for {file}. Actual columns: {df.columns.tolist()}")
            continue

        valid_raw = df.loc[df[raw_col].notna()].copy()
        
        def is_empty(val):
            return pd.isna(val) or val == "" or str(val).strip() == "{}" or str(val).strip() == "[]"
            
        failed_mask = valid_raw[parsed_col].apply(is_empty)
        failed = valid_raw.loc[failed_mask]
        
        print(f"DEBUG: failed.shape={failed.shape}, failed.columns={failed.columns.tolist()}")
        if not failed.empty:
            failed_items.extend(failed[raw_col].tolist())

    print(f"Total unparsed {args.task} found: {len(failed_items)}")
    
    print("Generating templates...")
    template_map = {}
    
    for d in failed_items:
        t = template_func(d)
        if t not in template_map:
            template_map[t] = []
        template_map[t].append(d)
        
    print(f"Total Unique Templates: {len(template_map)}")
    
    sorted_templates = sorted(template_map.items(), key=lambda item: len(item[1]), reverse=True)
    
    output_file = f'LLM_parsing/llm_unparsed_{args.task}_templates.csv'
    os.makedirs('LLM_parsing', exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'frequency', 'example_raw_string'])
        for template, strings in sorted_templates:

            stable_strings = sorted(list(set(strings)))
            writer.writerow([template, len(strings), stable_strings[0]])
            
    print(f"Successfully exported templates to {output_file}")

if __name__ == "__main__":
    main()
