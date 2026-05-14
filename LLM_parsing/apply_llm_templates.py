import os
import sys
import pandas as pd
import json
import re
import glob
import argparse

def get_numbers(s):
    if str(s) == 'nan' or not s:
        return []
    s_str = str(s)

    s_cleaned = re.sub(r'(\d+)\s+(\d+)/(\d+)', lambda m: str(float(m.group(1)) + float(m.group(2))/float(m.group(3))), s_str)
    s_cleaned = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1))/float(m.group(2))), s_cleaned)
    nums = re.findall(r'\d+\.\d+|\d+', s_cleaned)
    return [float(n) for n in nums]

def get_strings(s):
    if str(s) == 'nan' or not s:
        return []
    return re.findall(r'[a-z]+', str(s).lower())

def generate_dimensions_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\s*[×x]\s*', ' x ', s)
    s = re.sub(r'\d+[-\s]+\d+/\d+', '<NUM>', s)
    s = re.sub(r'\d+/\d+', '<NUM>', s)
    s = re.sub(r'\d+(?:\.\d+)?', '<NUM>', s)
    s = re.sub(r'\b(?:cm|mm)\b\.?', '<CM>', s)
    s = re.sub(r'\b(?:in|inches)\b\.?', '<IN>', s)
    s = re.sub(r'\b(?:g|kg|oz|lb)\b\.?', '<WT>', s)
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
    s = re.sub(r'\b(?:max|min|overall|sheet|image|plate|paper|mount|mounts|frame)\b\.?:?', '<MOD>', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_date_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\d{1,4}', '<NUM>', s)
    return re.sub(r'\s+', ' ', s).strip()

def generate_artist_template(s):
    s = str(s).strip().lower()
    s = re.sub(r'\d{1,4}', '<NUM>', s)
    s = re.sub(r'[a-z]+', '<STR>', s)
    return re.sub(r'\s+', ' ', s).strip()

def project_relative_value(val, llm_nums, target_nums, task):
    if not isinstance(val, (int, float)):
        return val

    for i, r_num in enumerate(llm_nums):
        conversions = [1.0, 2.54, 0.1, 100.0, 30.48, 1000.0, 453.592, 28.349]
        for multiplier in conversions:
            if abs((r_num * multiplier) - val) < 0.1:
                return round(target_nums[i] * multiplier, 2)

    if task == 'date' and val > 100:
        best_anchor_idx = -1
        min_dist = float('inf')
        
        for i, r_num in enumerate(llm_nums):
            if r_num > 10:
                dist = abs(r_num - val)
                if dist < min_dist:
                    min_dist = dist
                    best_anchor_idx = i
        
        if best_anchor_idx != -1:
            anchor_llm = llm_nums[best_anchor_idx]
            anchor_target = target_nums[best_anchor_idx]
            
            if 10 <= anchor_llm <= 21 and (anchor_llm - 1) * 100 < val <= anchor_llm * 100:
                year_in_century = val - (anchor_llm - 1) * 100
                if 0 < anchor_target <= 50:
                    return (anchor_target - 1) * 100 + year_in_century
            
            offset = val - anchor_llm
            if abs(offset) <= 100:
                return anchor_target + offset

    return val

def project_recursive(obj, llm_nums, target_nums, task, llm_strs=None, target_strs=None):
    if isinstance(obj, dict):
        return {k: project_recursive(v, llm_nums, target_nums, task, llm_strs, target_strs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [project_recursive(v, llm_nums, target_nums, task, llm_strs, target_strs) for v in obj]
    elif isinstance(obj, str) and task == 'artist' and llm_strs and target_strs:

        new_val = obj
        for s_llm, s_tgt in zip(llm_strs, target_strs):
            if s_llm in new_val.lower():
                if s_llm.capitalize() in new_val:
                    new_val = new_val.replace(s_llm.capitalize(), s_tgt.capitalize())
                else:
                    new_val = new_val.replace(s_llm, s_tgt)

        embedded_nums = re.findall(r'\b\d{1,4}\b', new_val)
        for num_str in embedded_nums:
            num_val = float(num_str)
            projected = project_relative_value(num_val, llm_nums, target_nums, task)
            if projected != num_val:
                new_val = new_val.replace(num_str, str(int(projected)))
        return new_val
    else:
        return project_relative_value(obj, llm_nums, target_nums, task)

def main():
    parser = argparse.ArgumentParser(description="Apply LLM attribute templates to parsed CSVs.")
    parser.add_argument("--task", required=True, choices=['dimensions', 'date', 'artist'], help="Attribute type to project.")
    parser.add_argument("--input-dir", default="parsed_pipeline", help="Directory containing parsed CSVs to project onto.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parsed JSONs even if not empty.")
    args = parser.parse_args()

    mapping_file = f'LLM_parsing/llm_{args.task}_mapping.json'
    templates_csv = f'LLM_parsing/llm_unparsed_{args.task}_templates.csv'
    
    if not os.path.exists(mapping_file) or not os.path.exists(templates_csv):
        print(f"Missing required LLM outputs for {args.task}. Please run generate_llm_templates.py and run_llm_templates.py first.")
        return
        
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    df_templates = pd.read_csv(templates_csv)

    example_maps = dict(zip(df_templates['template'], df_templates['example_raw_string']))

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

    print(f"Applying templated LLM projections for {args.task} in {args.input_dir} (overwrite={args.overwrite})...")
    total_projected = 0
    
    for file in glob.glob(os.path.join(args.input_dir, '*.csv')):
        if os.path.basename(file).startswith('_'): continue
        df = pd.read_csv(file, dtype=str)
        
        raw_col_current = raw_col
        if args.task == 'artist':
            for c in ['artist_information', 'artist_ulan', 'principal_maker']:
                if c in df.columns:
                    raw_col_current = c
                    break
                    
        if raw_col_current not in df.columns or parsed_col not in df.columns: continue
        
        def is_empty(val):
            return pd.isna(val) or val == "" or str(val).strip() == "{}" or str(val).strip() == "[]"
        
        if args.overwrite:
            mask = df[raw_col_current].notna()
        else:
            mask = df[raw_col_current].notna() & df[parsed_col].apply(is_empty)
            
        indices = df[mask].index
        if len(indices) == 0: continue
        
        applied = 0
        for idx in indices:
            raw_str = df.at[idx, raw_col_current]
            
            if args.task == 'artist' and ('\n' in str(raw_str) or '|' in str(raw_str)):
                continue
                
            t_str = template_func(raw_str)
            
            if t_str in mapping and t_str in example_maps:
                llm_json = mapping[t_str]
                llm_example_str = example_maps[t_str]
                
                llm_nums = get_numbers(llm_example_str)
                target_nums = get_numbers(raw_str)
                
                llm_strs = get_strings(llm_example_str) if args.task == 'artist' else None
                target_strs = get_strings(raw_str) if args.task == 'artist' else None
                
                if len(llm_nums) == len(target_nums):
                    projected = project_recursive(llm_json, llm_nums, target_nums, args.task, llm_strs, target_strs)
                    if projected:
                        if args.task == 'artist':
                            if not isinstance(projected, list):
                                projected = [projected]
                            valid = True
                            for p in projected:
                                nat = str(p.get('artist_nationality', '')).lower()
                                if len(nat) > 30 or any(w in nat for w in ['published by', 'printed by', ' by ', 'after ']):
                                    valid = False
                                    break
                                for v in p.values():
                                    if isinstance(v, str) and v.count('(') != v.count(')'):
                                        valid = False
                                        break
                            if not valid:
                                continue
                                
                        df.at[idx, parsed_col] = json.dumps(projected)
                        applied += 1
                        total_projected += 1
        
        if applied > 0:
            df.to_csv(file, index=False)
            print(f"  {file}: Projected {applied} {args.task}s.")

    print(f"Projection complete. Total {args.task}s projected: {total_projected}")

if __name__ == '__main__':
    main()
