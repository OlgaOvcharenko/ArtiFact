import os
import sys
import pandas as pd
import json
import glob

sys.path.append(os.getcwd())

from LLM_parsing.llm import generate_json_batch
from prompts.date import date_prompts

def process_csv(file_path, model="g25lite"):
    print(f"Checking {file_path} for dates to mop up...")
    df = pd.read_csv(file_path, low_memory=False)
    
    def is_candidate(row):
        raw = str(row.get('raw_date', ''))
        if not raw or raw == 'nan' or raw.strip() == '':
            return False
            
        parsed = str(row.get('parsed_date', ''))
        if not parsed or parsed == 'nan' or parsed.strip() in ('{}', '', 'nan'):
            return True
            
        return False

    mask = df.apply(is_candidate, axis=1)
    candidates = df[mask].copy()
    
    if candidates.empty:
        print(f"  No candidates in {os.path.basename(file_path)}")
        return
        
    print(f" Found {len(candidates)} candidates for direct parsing in {os.path.basename(file_path)}")
    
    prompt_template = date_prompts.get("artist")
    
    message_batch = []
    indices = candidates.index.tolist()
    
    for _, row in candidates.iterrows():
        raw_date = row['raw_date']
        prompt = prompt_template.format(date=raw_date)
        message_batch.append([{"role": "user", "content": prompt}])
    
    print(f"  Sending {len(message_batch)} records to {model}...")
    results = generate_json_batch(
        message_list=message_batch,
        model=model,
        chunk_size=20,
        rpm_limit=1000,
        ignore_cache=False
    )
    
    for idx, (result_tuple, _) in zip(indices, results):
        if result_tuple:
            df.at[idx, 'parsed_date'] = json.dumps(result_tuple)
            
    df.to_csv(file_path, index=False)
    print(f"Updated {len(candidates)} records in {os.path.basename(file_path)}")

def main():
    input_dir = 'parsed_pipeline'
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('_')]
    
    for f in csv_files:
        process_csv(f)

if __name__ == "__main__":
    main()
