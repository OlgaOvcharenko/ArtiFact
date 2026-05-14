import pandas as pd
import numpy as np
import json
import os
import re
import time
from collections import defaultdict

BASE_DIR = '/Users/lulo/Documents/DEEM/Thesis/artwork-dataset'
INPUT_DIR = os.path.join(BASE_DIR, '6_unified')
OUTPUT_SINGLE_FILE = os.path.join(BASE_DIR, 'error_injection/error_eligibility_index.csv')
OUTPUT_PAIRS_FILE = os.path.join(BASE_DIR, 'error_injection/swap_pairs.json')
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'error_injection/knowledge_v4')

def load_json(filename):
    path = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f: return json.load(f)
    return {}

def build_index():
    start_time = time.time()
    print(f"Loading unified dataset from {INPUT_DIR} (673k+ rows)...")
    
    import glob
    files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except: pass
    df = pd.concat(dfs, ignore_index=True)
    
    print("Pre-processing columns...")
    df['title_clean'] = df['title'].fillna('').astype(str).str.lower().str.strip()
    df['art'] = df['artist_name'].fillna('').astype(str).str.lower().str.strip()
    df['obj'] = df['object_name'].fillna('').astype(str).str.lower().str.strip()
    df['nat'] = df['artist_nationality'].fillna('').astype(str).str.lower().str.strip()
    df['cult'] = df['culture'].fillna('').astype(str).str.lower().str.strip()
    df['tec'] = df['techniques'].fillna('').astype(str).str.lower().str.strip()
    df['mat'] = df['materials'].fillna('').astype(str).str.lower().str.strip()
    df['date'] = pd.to_numeric(df['date_begin'], errors='coerce')
    df['location_clean'] = df['location'].fillna('').astype(str).str.lower().str.strip()
    df['img'] = df['image_url'].fillna('').astype(str)
    
    def is_anon(x):
        return not x or x == 'nan' or 'unknown' in x or 'anonymous' in x
    df['is_anon'] = df['art'].apply(is_anon)

    idx_df = pd.DataFrame({'object_ID': df['object_ID']})
    
    idx_df['century_shift'] = df['is_anon'] & df['date'].notna()
    
    dim_str = df['dimensions_json'].fillna('')
    idx_df['aspect_swap'] = dim_str.str.contains('height', case=False) & dim_str.str.contains('width', case=False)
    idx_df['scale_error'] = dim_str.str.contains('value', case=False)
    
    idx_df['material_interchange'] = df['mat'] != ''
    idx_df['material_anachronism'] = df['mat'] != ''
    idx_df['technique_interchange'] = df['tec'] != ''
    idx_df['technique_anachronism'] = df['tec'] != ''
    
    idx_df.to_csv(OUTPUT_SINGLE_FILE, index=False)
    
    swap_pairs = defaultdict(list)
    
    def is_too_similar(a, b):
        a, b = str(a).lower().strip(), str(b).lower().strip()
        if not a or not b: return True
        if a == b: return True
        if a in b or b in a: return True
        
        a_words = set(re.findall(r'\w+', a))
        b_words = set(re.findall(r'\w+', b))
        if a_words.intersection(b_words): return True
        
        return False
    
    def get_sliding_pairs(d, sort_cols, mask_condition):
        d = d.sort_values(sort_cols).copy()
        d['next_id'] = d['object_ID'].shift(-1)
        for col in d.columns:
            if col not in ['object_ID', 'next_id']:
                d[f'next_{col}'] = d[col].shift(-1)
        valid = d[mask_condition(d)]
        return valid[['object_ID', 'next_id']].values.tolist()

    base_art = df[~df['is_anon'] & df['date'].notna() & (df['obj'] != '')]
    
    t4_df = base_art[(base_art['tec'] != '') & (base_art['nat'] != '')]
    swap_pairs['artist_tier_4_hardest'] = get_sliding_pairs(
        t4_df, ['obj', 'tec', 'nat', 'date'],
        lambda d: (d['obj'] == d['next_obj']) & (d['tec'] == d['next_tec']) & 
                  (d['nat'] == d['next_nat']) & (d['art'] != d['next_art']) & 
                  (np.abs(d['date'] - d['next_date']) <= 50)
    )
    
    t3_df = base_art[(base_art['tec'] != '') & (base_art['nat'] != '')]
    swap_pairs['artist_tier_3_hard'] = get_sliding_pairs(
        t3_df, ['obj', 'tec', 'date'],
        lambda d: (d['obj'] == d['next_obj']) & (d['tec'] == d['next_tec']) & 
                  (d['nat'] != d['next_nat']) & (d['art'] != d['next_art']) & 
                  (np.abs(d['date'] - d['next_date']) <= 50)
    )
    
    swap_pairs['artist_tier_2_medium'] = get_sliding_pairs(
        base_art, ['obj', 'date'],
        lambda d: (d['obj'] == d['next_obj']) & (d['art'] != d['next_art']) & 
                  (np.abs(d['date'] - d['next_date']) <= 50)
    )
    
    t1_df = base_art.sample(frac=1, random_state=42).copy()
    t1_df['next_id'] = t1_df['object_ID'].shift(-1)
    t1_df['next_art'] = t1_df['art'].shift(-1)
    valid_t1 = t1_df[t1_df['art'] != t1_df['next_art']]
    swap_pairs['artist_tier_1_easy'] = valid_t1[['object_ID', 'next_id']].values.tolist()

    has_img = df[(df['img'] != '')]
    
    known_img = has_img[~has_img['is_anon']]
    swap_pairs['image_tier_3_hard'] = get_sliding_pairs(
        known_img, ['art', 'img'],
        lambda d: (d['art'] == d['next_art']) & (d['img'] != d['next_img']) & (d['title_clean'] != d['next_title_clean'])
    )
    
    valid_obj_img = has_img[has_img['obj'] != '']
    swap_pairs['image_tier_2_medium'] = get_sliding_pairs(
        valid_obj_img, ['obj', 'img'],
        lambda d: (d['obj'] == d['next_obj']) & (d['img'] != d['next_img']) & (d['title_clean'] != d['next_title_clean'])
    )
    

    img_t1_df = valid_obj_img.sample(frac=1, random_state=42).copy()
    img_t1_df['next_id'] = img_t1_df['object_ID'].shift(-1)
    img_t1_df['next_obj'] = img_t1_df['obj'].shift(-1)
    img_t1_df['next_title_clean'] = img_t1_df['title_clean'].shift(-1)

    valid_img_t1 = img_t1_df[(img_t1_df['obj'] != img_t1_df['next_obj']) & (img_t1_df['title_clean'] != img_t1_df['next_title_clean'])]
    swap_pairs['image_tier_1_easy'] = valid_img_t1[['object_ID', 'next_id']].values.tolist()

    emb_path = os.path.join(BASE_DIR, 'error_injection/embeddings/clip_embeddings.npy')
    ids_path = os.path.join(BASE_DIR, 'error_injection/embeddings/object_ids.npy')
    
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        embs = np.load(emb_path)
        emb_ids = np.load(ids_path, allow_pickle=True)
        
        id_to_emb_idx = {str(val): i for i, val in enumerate(emb_ids)}
    
        df_emb = df[df['object_ID'].astype(str).isin(id_to_emb_idx)].copy()
        query_pool = df_emb.sample(min(100000, len(df_emb)), random_state=42)
        
        df_emb_set = df_emb.set_index('object_ID')
        art_map = df_emb_set['art'].to_dict()
        obj_map = df_emb_set['obj'].to_dict()
        title_map = df_emb_set['title_clean'].to_dict()
        
        emb_pairs = []
        batch_size = 1000
        used_ids = set()
        
        for i in range(0, len(query_pool), batch_size):
            batch = query_pool.iloc[i:i+batch_size]
            q_indices = [id_to_emb_idx[str(oid)] for oid in batch['object_ID']]
            q_embs = embs[q_indices]
            
            scores = np.dot(q_embs, embs.T)
            
            top_k = np.argpartition(scores, -20, axis=1)[:, -20:]
            
            for b_idx, match_indices in enumerate(top_k):
                q_oid = str(batch.iloc[b_idx]['object_ID'])
                if q_oid in used_ids: continue
                
                match_indices_sorted = match_indices[np.argsort(-scores[b_idx, match_indices])]
                
                q_art = art_map.get(q_oid, "")
                q_obj = obj_map.get(q_oid, "")
                q_title = title_map.get(q_oid, "")
                
                for m_idx in match_indices_sorted:
                    if scores[b_idx, m_idx] > 0.98: continue 
                    if scores[b_idx, m_idx] < 0.75: break
                    
                    m_oid = str(emb_ids[m_idx])
                    if m_oid not in art_map or m_oid in used_ids: continue
                    
                    m_art = art_map[m_oid]
                    m_obj = obj_map[m_oid]
                    m_title = title_map[m_oid]
                    diff_art = (q_art != m_art) and q_art and m_art
                    diff_semantic = (q_obj != m_obj) or (q_title != m_title)
                    
                    if diff_art and diff_semantic:
                        emb_pairs.append([q_oid, m_oid])
                        used_ids.add(q_oid)
                        used_ids.add(m_oid)
                        break
                        
        swap_pairs['embedding_swap'] = emb_pairs
        print(f"Found {len(emb_pairs):,} perfect embedding pairs.")

    
    anon_cult = df[df['is_anon'] & (df['cult'] != '')]
    cult_dict = anon_cult.groupby('cult')['object_ID'].apply(list).to_dict()
    
    cult_adj = load_json('culture_adjacency.json')
    t_cult_pairs = []
    for c1, ids1 in cult_dict.items():
        for c2 in cult_adj.get(c1, []):
            if c1 in c2 or c2 in c1: continue
            if c2 in cult_dict:
                ids2 = cult_dict[c2]
                for i1, i2 in zip(ids1[:1000], ids2[:1000]): t_cult_pairs.append([i1, i2])
    swap_pairs['culture_tight_swap'] = t_cult_pairs
    
    cult_cont = load_json('culture_to_continent.json')
    cont_to_cults = defaultdict(list)
    for k, v in cult_cont.items(): cont_to_cults[v].append(k)
    
    c_cult_pairs = []
    for cont, cults in cont_to_cults.items():
        for i in range(len(cults)):
            for j in range(i+1, len(cults)):
                c1, c2 = cults[i], cults[j]
                if c1 in c2 or c2 in c1: continue
                if c1 in cult_dict and c2 in cult_dict:
                    for id1, id2 in zip(cult_dict[c1][:1000], cult_dict[c2][:1000]):
                        c_cult_pairs.append([id1, id2])
    swap_pairs['culture_continent_swap'] = c_cult_pairs
    

    place_k = load_json('place_knowledge.json')
    valid_location = df[(df['location_clean'] != '') & (df['location_clean'] != 'unknown')]
    c_dict = valid_location.groupby('location_clean')['object_ID'].apply(list).to_dict()
    
    country_pairs = []
    country_adj = place_k.get('country_neighbors', {})
    for c1, ids1 in c_dict.items():
        for c2 in country_adj.get(c1, []):
            if is_too_similar(c1, c2): continue
            if c2 in c_dict:
                for id1, id2 in zip(ids1[:1000], c_dict[c2][:1000]): country_pairs.append([id1, id2])
    swap_pairs['country_level_swap'] = country_pairs
    
    city_pairs = []
    city_adj = place_k.get('city_neighbors', {})
    for c1, ids1 in c_dict.items():
        for c2 in city_adj.get(c1, []):
            if is_too_similar(c1, c2): continue
            if c2 in c_dict:
                for id1, id2 in zip(ids1[:1000], c_dict[c2][:1000]): city_pairs.append([id1, id2])
    swap_pairs['city_level_swap'] = city_pairs

    with open(OUTPUT_PAIRS_FILE, 'w') as f:
        json.dump(swap_pairs, f)

    print(f"Mutations Index saved to {OUTPUT_SINGLE_FILE}")
    print(f"Swap Pairs saved to {OUTPUT_PAIRS_FILE}")
    
    print("\nPair Counts Discovered:")
    for k, v in swap_pairs.items():
        print(f"  {k}: {len(v):,} pairs")

if __name__ == "__main__":
    build_index()
