import os
import pandas as pd
import json
import glob
import argparse

def is_anomalous(p_dict):
    try:
        b = p_dict.get('date_begin')
        e = p_dict.get('date_end')
        b_bce = p_dict.get('date_begin_bce', False)
        e_bce = p_dict.get('date_end_bce', False)

        if b is None or e is None:
            return False
            
        if not b_bce and not e_bce:
            if b > e: return True 
            if b > 2026 or e > 2026: return True
            
        if b_bce and e_bce:
            if b < e: return True
            
        if not b_bce and e_bce: 
            return True
            
        
        if b_bce and not e_bce:
            span = b + e
        else:
            span = abs(e - b)
            
        if span > 3000:
            return True
            
        return False
    except Exception as e:
        return True

def sweep_file(file_path):
    print(f"Sweeping {os.path.basename(file_path)} for anomalies...")
    df = pd.read_csv(file_path, low_memory=False)
    
    anomalies = 0
    for idx, row in df.iterrows():
        parsed_str = str(row.get('parsed_date', ''))
        
        if parsed_str and parsed_str.strip() not in ['{}', '[]', 'nan']:
            try:
                p_dict = json.loads(parsed_str)
                if is_anomalous(p_dict):
                    anomalies += 1
                    df.at[idx, 'parsed_date'] = "{}"
            except:
                anomalies += 1
                df.at[idx, 'parsed_date'] = "{}"
                
    if anomalies > 0:
        df.to_csv(file_path, index=False)
        print(f"Detected and clesned {anomalies} anomalies.")
    else:
        print("Clean.")
        
    return anomalies

def main():
    parser = argparse.ArgumentParser(description="Sweep parsed CSVs for date anomalies.")
    parser.add_argument("--input-dir", required=True, help="Directory containing parsed CSVs")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.input_dir, '*.csv'))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    
    total = sum(sweep_file(f) for f in files)
    print("=" * 50)
    print(f"Total anomalies found and cleaned: {total}")
    print("=" * 50)

if __name__ == '__main__':
    main()
