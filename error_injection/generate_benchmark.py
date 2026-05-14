import pandas as pd
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from injector_v2 import ErrorInjectorV2 as ErrorInjector

BASE_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset'
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'error_injection/knowledge_v4')
DEFAULT_CONFIG = os.path.join(BASE_DIR, 'error_injection/benchmark_config.yaml')
INDEX_FILE = os.path.join(BASE_DIR, 'error_injection/error_eligibility_index.csv')
PAIRS_FILE = os.path.join(BASE_DIR, 'error_injection/swap_pairs.json')

ERROR_COLS = [
    'date_begin', 'date_end', 'date_begin_bce', 'date_end_bce',
    'materials', 'techniques', 
    'dimensions_json', 
    'artist_name', 'artist_role', 'artist_nationality', 
    'artist_date_begin', 'artist_date_end',
    'image_url', 'image_object_id', 'culture', 'location'
]

ESSENTIAL_COLS = [
    'object_ID', 'title', 'artist_name', 'artist_role', 'artist_nationality', 
    'artist_date_begin', 'artist_date_end', 'culture', 'location', 
    'date_begin', 'date_end', 'date_begin_bce', 'date_end_bce', 
    'materials', 'techniques', 'dimensions_json', 'image_url',
    'subjects', 'description', 'inscriptions'
]

SUBTYPE_TO_TYPE = {
    'century_shift': 'date',
    'scale_error': 'dimension',
    'aspect_swap': 'dimension',
    'material_anachronism': 'material',
    'material_interchange': 'material',
    'technique_anachronism': 'technique',
    'technique_interchange': 'technique',
    'artist_tier_1_easy': 'artist',
    'artist_tier_2_medium': 'artist',
    'artist_tier_3_hard': 'artist',
    'artist_tier_4_hardest': 'artist',
    'culture_tight_swap': 'culture',
    'culture_continent_swap': 'culture',
    'country_level_swap': 'place',
    'city_level_swap': 'place',
    'embedding_swap': 'image',
    'image_tier_1_easy': 'image',
    'image_tier_2_medium': 'image',
    'image_tier_3_hard': 'image',
}

SWAP_ERRORS = {
    'artist_tier_1_easy', 'artist_tier_2_medium', 'artist_tier_3_hard', 'artist_tier_4_hardest',
    'image_tier_1_easy', 'image_tier_2_medium', 'image_tier_3_hard',
    'embedding_swap',
    'culture_tight_swap', 'culture_continent_swap',
    'country_level_swap', 'city_level_swap'
}

_injector = None

def init_worker(knowledge_dir, ref_df_json):
    global _injector
    if _injector is None:
        _injector = ErrorInjector(knowledge_dir)
        
        ref_df = pd.read_json(ref_df_json, orient='records')
        _injector.precompute_lookups(ref_df)

def process_mutation(row_dict, target_subtype):
    row = pd.Series(row_dict)
    result_row = row_dict.copy()
    
    for col in ['error_type', 'error_subtype'] + [f"{c}_error" for c in ERROR_COLS]:
        result_row[col] = None

    error_type = SUBTYPE_TO_TYPE.get(target_subtype)
    res = None
    if error_type == 'date': res = _injector.inject_date_error(row)
    elif error_type == 'dimension': res = _injector.inject_dimension_error(row)
    elif error_type == 'material': res = _injector.inject_material_error(row, target_subtype)
    elif error_type == 'technique': res = _injector.inject_technique_error(row, target_subtype)
    elif error_type == 'culture': res = _injector.inject_culture_error(row, target_subtype)
    elif error_type == 'place': res = _injector.inject_place_error(row, target_subtype)
    elif error_type == 'image': res = _injector.inject_image_error(row, subtype=target_subtype)
    
    if res:
        corrupted_row, meta = res
        result_row = populate_error_columns(result_row, row, error_type, meta, corrupted_row)
        return [result_row], 1
        
    return [result_row], 0

def process_swap(row_a_dict, row_b_dict, target_subtype):
    row_a = pd.Series(row_a_dict)
    row_b = pd.Series(row_b_dict)
    
    res_a = row_a_dict.copy()
    res_b = row_b_dict.copy()
    for col in ['error_type', 'error_subtype'] + [f"{c}_error" for c in ERROR_COLS]:
        res_a[col] = None
        res_b[col] = None
        
    res = _injector.execute_swap(row_a, row_b, target_subtype)
    
    if res and res[0] is not None:
        (corrupted_a, meta_a), (corrupted_b, meta_b) = res
        error_type = SUBTYPE_TO_TYPE.get(target_subtype)
        res_a = populate_error_columns(res_a, row_a, error_type, meta_a, corrupted_a)
        res_b = populate_error_columns(res_b, row_b, error_type, meta_b, corrupted_b)
        return [res_a, res_b], 2
        
    return [res_a, res_b], 0

def populate_error_columns(result_row, row, etype, meta, corrupted_row):
    if etype == 'date':
        result_row['date_begin_error'] = meta['new']['start']
        result_row['date_end_error'] = meta['new']['end']
        result_row['date_begin_bce_error'] = meta['new'].get('start_bce')
        result_row['date_end_bce_error'] = meta['new'].get('end_bce')
        
    elif etype == 'dimension': 
        result_row['dimensions_json_error'] = meta['new']
        
    elif etype == 'material': 
        result_row['materials_error'] = json.dumps(meta['new']) if isinstance(meta['new'], list) else meta['new']
        
    elif etype == 'technique': 
        result_row['techniques_error'] = json.dumps(meta['new']) if isinstance(meta['new'], list) else meta['new']
        
    elif etype == 'artist':
        for f in ['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end', 'culture', 'location']:
            val_fake = corrupted_row.get(f)
            result_row[f'{f}_error'] = val_fake
            
            if f in ['culture', 'location']:
                if pd.isna(val_fake) or str(val_fake).strip() == "" or str(val_fake).lower() == 'nan':
                    result_row[f] = None
        
        fake_row = _injector._propagate_text(row.to_dict(), [
            (str(row.get('artist_name')), str(corrupted_row.get('artist_name'))),
            (str(row.get('artist_nationality')), str(corrupted_row.get('artist_nationality')))
        ])
        result_row['title_error'] = fake_row.get('title')
        result_row['description_error'] = fake_row.get('description')
            
    elif etype == 'image': 
        result_row['image_url_error'] = meta['new']
        result_row['image_object_id_error'] = meta.get('new_id')
        
    elif etype == 'culture' or etype == 'place': 
        for f in ['culture', 'location']:
            val_fake = corrupted_row.get(f)
            result_row[f'{f}_error'] = val_fake
            
            if pd.isna(val_fake) or str(val_fake).strip() == "" or str(val_fake).lower() == 'nan':
                result_row[f] = None
        
        fake_row = _injector._propagate_text(row.to_dict(), [
            (str(row.get('culture')), str(corrupted_row.get('culture'))),
            (str(row.get('location')), str(corrupted_row.get('location')))
        ])
        result_row['title_error'] = fake_row.get('title')
        result_row['description_error'] = fake_row.get('description')
            
    result_row['error_type'] = meta.get('error_type', f'{etype}_error')
    result_row['error_subtype'] = meta.get('subtype', etype)
    return result_row

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('--limit', type=str, default=None) 
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    raw_limit = args.limit if args.limit is not None else config['dataset']['target_size']
    if str(raw_limit).lower() == 'full':
        target_size = None 
    else:
        target_size = int(raw_limit)

    seed = config['dataset'].get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n Stage 1: Loading Dataset Index (Memory Efficient) ")
    files = glob.glob(os.path.join(BASE_DIR, '6_unified', '*.csv'))
    dfs = []
    for f in tqdm(files, desc="Reading file indices"):
        try: 

            dfs.append(pd.read_csv(f, usecols=['object_ID'], low_memory=False))
        except: pass
    
    all_ids_df = pd.concat(dfs, ignore_index=True)
    available_ids = set(all_ids_df['object_ID'].tolist())
    total_available = len(available_ids)
    
    if target_size is None or target_size > total_available:
        target_size = total_available
        
    n_train = int(target_size * 0.6)
    n_test = target_size - n_train
    n_test_errors = int(n_test * (config['dataset'].get('error_rate', 50) / 100))
    
    print(f"\nTarget Benchmark Size: {target_size:,} rows")
    print(f"Train (Clean): {n_train:,} rows")
    print(f"Test: {n_test:,} rows ({n_test_errors:,} errors, {n_test - n_test_errors:,} clean)")

    print("Loading Eligibility Index and Swap Pairs...")
    index_df = pd.read_csv(INDEX_FILE)
    with open(PAIRS_FILE, 'r') as f: swap_pairs = json.load(f)

    dist = config['error_distribution']
    total_weight = sum(dist.values())
    quotas = {k: int(round((v / total_weight) * n_test_errors)) for k, v in dist.items()}
    
    diff = n_test_errors - sum(quotas.values())
    if diff != 0:
        sorted_keys = sorted(quotas.keys(), key=lambda k: quotas[k], reverse=True)
        idx = 0
        while diff > 0: quotas[sorted_keys[idx % len(sorted_keys)]] += 1; diff -= 1; idx += 1
        while diff < 0: quotas[sorted_keys[idx % len(sorted_keys)]] -= 1; diff += 1; idx += 1

    print("\n Stage 2: Smart Assembly (The Pair Hunt) ")
    selected_mutations = [] 
    selected_swaps = [] 
    
    priority_order = ['culture_tight_swap', 'culture_continent_swap', 'country_level_swap', 'city_level_swap']
    remaining_subtypes = [s for s in quotas.keys() if s not in priority_order]
    ordered_subtypes = priority_order + sorted(remaining_subtypes, key=lambda k: quotas[k])
    
    for subtype in ordered_subtypes:
        needed = quotas[subtype]
        if needed == 0: continue
        
        if subtype in SWAP_ERRORS:
            pairs_needed = needed // 2
            found_pairs = 0
            
            available_pairs = swap_pairs.get(subtype, [])
            random.shuffle(available_pairs)
            
            for p1, p2 in available_pairs:
                if found_pairs >= pairs_needed: break
                if p1 in available_ids and p2 in available_ids:
                    selected_swaps.append((p1, p2, subtype))
                    available_ids.remove(p1)
                    available_ids.remove(p2)
                    found_pairs += 1
            
            print(f"Filled {found_pairs*2:,}/{needed:,} quota for {subtype}")
            if needed % 2 != 0: quotas['scale_error'] += 1
        else:
            if subtype in index_df.columns:
                eligible_mask = index_df[subtype] == True
                eligible_ids = set(index_df[eligible_mask]['object_ID'].tolist())
                valid_candidates = list(eligible_ids.intersection(available_ids))
            else:
                valid_candidates = list(available_ids)
            
            picked = random.sample(valid_candidates, min(len(valid_candidates), needed))
            for oid in picked:
                selected_mutations.append((oid, subtype))
                available_ids.remove(oid)
            print(f"Filled {len(picked):,}/{needed:,} quota for {subtype}")

    current_errors = (len(selected_swaps) * 2) + len(selected_mutations)
    shortfall = n_test_errors - current_errors
    if shortfall > 0:
        pool = list(available_ids)
        easy_picked = random.sample(pool, min(len(pool), shortfall))
        for oid in easy_picked:
            selected_mutations.append((oid, 'scale_error'))
            available_ids.remove(oid)

    needed_clean = n_train + (n_test - n_test_errors)
    pool = list(available_ids)
    selected_clean = random.sample(pool, min(len(pool), needed_clean))
    
    for oid in selected_clean:
        if oid in available_ids:
            available_ids.remove(oid)
    
    test_clean_ids = set(selected_clean[:(n_test - n_test_errors)])
    train_clean_ids = set(selected_clean[(n_test - n_test_errors):])
    
    all_required_ids = test_clean_ids.union(train_clean_ids)
    for p1, p2, _ in selected_swaps:
        all_required_ids.add(p1); all_required_ids.add(p2)
    for oid, _ in selected_mutations:
        all_required_ids.add(oid)

    print("\n Stage 3: Loading Full Row Data for selected items ")
    full_rows = {}
    all_required_ids_str = {str(oid) for oid in all_required_ids}
    
    REF_COLS = ['object_ID', 'object_name', 'artist_name', 'materials', 'techniques', 'artist_nationality', 'date_begin', 'image_url', 'artist_role', 'artist_date_begin', 'artist_date_end', 'culture', 'location', 'dimensions_json', 'subjects', 'description', 'inscriptions']
    ref_rows = []
    
    for f in tqdm(files, desc="Fetching row content"):
        try:
            df_file = pd.read_csv(f, usecols=lambda c: c in ESSENTIAL_COLS or c in REF_COLS, low_memory=False, dtype={'object_ID': str})
            
            df_ref = df_file.dropna(subset=['image_url'], how='all').copy()
            if 'country' not in df_ref.columns and 'location' in df_ref.columns:
                df_ref['country'] = df_ref['location']
            ref_rows.append(df_ref[[c for c in REF_COLS + ['country'] if c in df_ref.columns]])
            
            df_needed = df_file[df_file['object_ID'].isin(all_required_ids_str)]
            
            for _, row in df_needed.iterrows():
                full_rows[row['object_ID']] = row.to_dict()
        except Exception as e:
            print(f"Warning: Failed to process {f}: {e}")

    if not full_rows:
        print("No row data was loaded.")
        return
    else:
        print(f"Successfully loaded {len(full_rows):,} rows into memory.")

    print("\n Stage 3.5: Preparing Reference Cache ")
    ref_df = pd.concat(ref_rows, ignore_index=True)
    ref_df_json = ref_df.to_json(orient='records')
    print(f"Reference cache ready with {len(ref_df):,} rows.")

    print("\nStage 4: Injection (Multiprocessing with Max 4 Workers)")
    test_results = []
    success_count = 0
    
    for oid in test_clean_ids:
        if oid in full_rows:
            row_dict = full_rows[oid].copy()
            for col in ['error_type', 'error_subtype'] + [f"{c}_error" for c in ERROR_COLS]:
                row_dict[col] = None
            test_results.append(row_dict)

    max_workers = min(4, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(KNOWLEDGE_DIR, ref_df_json)) as executor:

        swap_tasks = []
        for p1, p2, sub in selected_swaps:
            if p1 in full_rows and p2 in full_rows:
                swap_tasks.append((full_rows[p1], full_rows[p2], sub))
        
        if swap_tasks:
            swap_futures = [executor.submit(process_swap, r1, r2, sub) for r1, r2, sub in swap_tasks]
            for future in tqdm(as_completed(swap_futures), total=len(swap_tasks), desc="Executing Swaps"):
                res_rows, num_success = future.result()
                test_results.extend(res_rows)
                success_count += num_success


        available_pool = list(available_ids - train_clean_ids - test_clean_ids)
        random.shuffle(available_pool)
        
        for subtype in ordered_subtypes:
            if subtype in SWAP_ERRORS: continue
            
            needed = quotas[subtype]
            if needed <= 0: continue
            
            current_candidates = [oid for oid, s in selected_mutations if s == subtype]
            current_candidates = [oid for oid in current_candidates if oid not in train_clean_ids]
            
            subtype_success = 0
            
            pbar = tqdm(total=needed, desc=f"Injecting {subtype}")
            
            def try_batch(ids_to_try):
                tasks = [(full_rows[oid], subtype) for oid in ids_to_try if oid in full_rows]
                futures = [executor.submit(process_mutation, r, s) for r, s in tasks]
                batch_results = []
                batch_success = 0
                for f in as_completed(futures):
                    res_rows, n = f.result()
                    if n > 0:
                        batch_results.extend(res_rows)
                        batch_success += n
                        pbar.update(n)
                return batch_results, batch_success

            res, n = try_batch(current_candidates)
            test_results.extend(res)
            subtype_success += n
            success_count += n
            
            while subtype_success < needed and available_pool:
                refill_count = max(500, (needed - subtype_success) * 5)
                new_batch_ids = []
                while len(new_batch_ids) < refill_count and available_pool:
                    new_batch_ids.append(available_pool.pop())
                
                new_full_rows = {}
                for f in files:
                    df_f = pd.read_csv(f, usecols=ESSENTIAL_COLS, low_memory=False, dtype={'object_ID': str})
                    df_match = df_f[df_f['object_ID'].isin(new_batch_ids)]
                    for _, row in df_match.iterrows():
                        new_full_rows[row['object_ID']] = row.to_dict()
                    if len(new_full_rows) >= len(new_batch_ids): break
                
                tasks = [(new_full_rows[oid], subtype) for oid in new_batch_ids if oid in new_full_rows]
                futures = [executor.submit(process_mutation, r, s) for r, s in tasks]
                for f in as_completed(futures):
                    if subtype_success >= needed: break
                    res_rows, n = f.result()
                    if n > 0:
                        test_results.extend(res_rows)
                        subtype_success += n
                        success_count += n
                        pbar.update(n)
            
            pbar.close()

    # Final Output
    test_df = pd.DataFrame(test_results)
    train_ids_list = [oid for oid in train_clean_ids if oid in full_rows]
    df_train = pd.DataFrame([full_rows[oid] for oid in train_ids_list])
    
    base_output = config['dataset']['output_file']
    out_name = os.path.splitext(base_output)[0]
    df_train.to_csv(os.path.join(BASE_DIR, f"{out_name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(BASE_DIR, f"{out_name}_test.csv"), index=False)
    
    print("\n Summary ")
    print(f"Train dataset: {len(df_train):,} rows")
    print(f"Test dataset: {len(test_df):,} rows")
    error_count = len(test_df[test_df['error_type'].notna()])
    print(f"- Errors: {error_count:,} rows ({(error_count/len(test_df))*100:.1f}%)")
    print(f"Injection Success: {success_count:,}/{n_test_errors:,}")

if __name__ == "__main__":
    main()
