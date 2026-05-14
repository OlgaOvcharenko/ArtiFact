import pandas as pd
import json
import os
import re
import glob
import ast
from collections import defaultdict

INPUT_DIR = "consolidated_pipeline"
OUTPUT_DIR = "6_unified"
MAPPING_DIR = "semantic_unification/vocab_output_v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mappings():
    with open(os.path.join(MAPPING_DIR, "vocab_canonical_mapping_mt.json")) as f:
        mt = json.load(f)
    mat_map = {k.lower(): v for k, v in mt.get("material", {}).items()}
    tec_map = {k.lower(): v for k, v in mt.get("technique", {}).items()}

    with open(os.path.join(MAPPING_DIR, "vocab_canonical_mapping_culture.json")) as f:
        culture_raw = json.load(f)
    if 'culture' in culture_raw and isinstance(culture_raw['culture'], dict):
        culture_raw = culture_raw['culture']
    culture_map = {k.lower(): v for k, v in culture_raw.items()}

    country_path = os.path.join(MAPPING_DIR, "vocab_canonical_mapping_country_llm.json")
    if os.path.exists(country_path):
        with open(country_path) as f:
            country_raw = json.load(f)
        if 'country' in country_raw and isinstance(country_raw['country'], dict):
            country_raw = country_raw['country']
        country_map = {k.lower(): v for k, v in country_raw.items()}
    else:
        country_path = os.path.join(MAPPING_DIR, "vocab_canonical_mapping_country.json")
        if os.path.exists(country_path):
            with open(country_path) as f:
                country_raw = json.load(f)
            if 'country' in country_raw and isinstance(country_raw['country'], dict):
                country_raw = country_raw['country']
            country_map = {k.lower(): v for k, v in country_raw.items()}
        else:
            country_map = {}

    with open(os.path.join(MAPPING_DIR, "vocab_canonical_mapping_object_name_master.json")) as f:
        obj_map = {k.lower(): v for k, v in json.load(f).items()}

    entity_map_path = os.path.join(MAPPING_DIR, "vocab_artist_entity_mapping.json")
    if os.path.exists(entity_map_path):
        with open(entity_map_path) as f:
            artist_entity_map = json.load(f)
    else:
        artist_entity_map = {}

    suffix_path = "semantic_unification/culture_suffix_mapping.json"
    if os.path.exists(suffix_path):
        with open(suffix_path) as f:
            suffix_map = json.load(f)
    else:
        suffix_map = {}

    demonym_map = {}
    
    DEMONYM_PATCH = {
        "dutch": "netherlands",
        "flemish": "belgium",
        "british": "united kingdom",
        "english": "united kingdom",
        "american": "united states",
        "french": "france",
        "japanese": "japan",
        "chinese": "china",
        "spanish": "spain",
        "italian": "italy",
        "german": "germany",
        "russian": "russia",
        "korean": "korea",
        "persian": "iran",
        "iranian": "iran",
        "syrian": "syria",
        "egyptian": "egypt",
        "greek": "greece",
        "indian": "india",
        "mexican": "mexico",
        "peruvian": "peru",
        "portuguese": "portugal",
        "turkish": "turkey",
        "norwegian": "norway",
        "swedish": "sweden",
        "danish": "denmark",
        "austrian": "austria",
        "swiss": "switzerland",
        "belgian": "belgium",
    }
    demonym_map.update(DEMONYM_PATCH)

    return mat_map, tec_map, culture_map, country_map, obj_map, artist_entity_map, suffix_map, demonym_map

def unify_list_field(raw, mapping):
    if pd.isna(raw) or not str(raw).strip():
        return None
    try:
        items = json.loads(raw)
    except Exception:
        try:
            items = ast.literal_eval(str(raw))
        except Exception:
            items = [str(raw)]
    unified = []
    seen = set()
    for item in items:
        if not item:
            continue
        key = str(item).lower().strip()
        mapped = mapping.get(key, item)

        if mapped == "discard":
            continue

        if mapped not in seen:
            unified.append(mapped)
            seen.add(mapped)

    if not unified:
        return None
    return json.dumps(unified, ensure_ascii=False)

def unify_string_field(raw, mapping):
    if pd.isna(raw) or not str(raw).strip():
        return None
    key = str(raw).lower().strip()
    return mapping.get(key, raw)


def process_file(filepath, mat_map, tec_map, culture_map, country_map, obj_map, artist_map, suffix_map, demonym_map):
    df = pd.read_csv(filepath, low_memory=False)
    return process_df(df, mat_map, tec_map, culture_map, country_map, obj_map, artist_map, suffix_map, demonym_map)

def process_df(df, mat_map, tec_map, culture_map, country_map, obj_map, artist_map, suffix_map, demonym_map):
    n = len(df)

    if 'materials' in df.columns:
        df['materials'] = df['materials'].apply(lambda x: unify_list_field(x, mat_map))
    if 'techniques' in df.columns:
        df['techniques'] = df['techniques'].apply(lambda x: unify_list_field(x, tec_map))
    
    if 'culture' in df.columns:
        if 'location' not in df.columns:
            df['location'] = None
            
        def route_culture(row):
            cult = row.get('culture')
            loc = row.get('location')
            
            if pd.isna(cult) or not isinstance(cult, str):
                return cult, loc

            key = str(cult).lower().strip()
            if key in culture_map:
                return culture_map[key], loc
            
            cleaned_cult = re.sub(r'\(', ', ', cult).replace(')', '')
            cleaned_cult = re.sub(r'\s*,\s*', ', ', cleaned_cult).strip()
            
            if ',' not in cleaned_cult:
                key_clean = cleaned_cult.lower()
                if key_clean in culture_map:
                    return culture_map[key_clean], loc
                return cult, loc 
                
            action = suffix_map.get(str(cult).lower().strip(), 'discard')
            if action == 'culture':
                return cult, loc
                
            parts = [p.strip() for p in cleaned_cult.split(',', 1)]
            new_cult = parts[0]
            suffix = parts[1] if len(parts) > 1 else ''
            
            if action == 'city':
                if pd.isna(loc) or str(loc).strip() == '':
                    loc = suffix
                return new_cult, loc
            elif action == 'discard':
                return new_cult, loc
                
            return cult, loc
            
        res = df.apply(route_culture, axis=1)
        df['culture'] = [r[0] for r in res]
        df['location'] = [r[1] for r in res]

        df['culture'] = df['culture'].apply(lambda x: unify_string_field(x, culture_map))

    if 'location' in df.columns:
        def unify_location(x):
            if pd.isna(x) or not str(x).strip():
                return x
            key = str(x).lower().strip()
            mapped = country_map.get(key)
            if mapped:
                return mapped
            mapped = demonym_map.get(key)
            if mapped:
                return mapped
            return x

        df['location'] = df['location'].apply(unify_location)
        df['location'] = df['location'].apply(lambda x: re.sub(r'\s*\(\s*\)', '', str(x)).strip() if pd.notna(x) else x)

    if 'object_name' in df.columns:
        df['object_name'] = df['object_name'].apply(lambda x: unify_string_field(x, obj_map))
    def unify_artist_entity(row):
        def clean_year(val):
            if pd.isna(val) or not str(val).strip():
                return ""
            s = str(val).strip().lower()
            if s == 'nan' or s == 'none' or s == '{}':
                return ""
            match = re.search(r'(\d{4})', s)
            if match:
                return match.group(1)
            match = re.search(r'(\d{1,3})', s)
            if match:
                return match.group(1)
            return s

        def clean_role(val):
            if pd.isna(val) or not str(val).strip():
                return ""
            s = str(val).strip()
            if s in ('{}', '[]', 'None', 'nan', 'null'):
                return ""
            return s

        def get_str(val):
            if pd.isna(val) or str(val).lower() == 'nan': return ""
            return str(val).strip()
        
        key = "|".join([
            get_str(row.get('artist_name')),
            clean_role(row.get('artist_role')),
            get_str(row.get('artist_nationality')),
            clean_year(row.get('artist_date_begin')),
            clean_year(row.get('artist_date_end'))
        ])
        
        if key in artist_map:
            canon = artist_map[key]
            res = pd.Series(canon, index=['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end'])
            res['artist_role'] = clean_role(res['artist_role'])
            res['artist_date_begin'] = clean_year(res['artist_date_begin'])
            res['artist_date_end'] = clean_year(res['artist_date_end'])
            return res
            
        res = row[['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end']].copy()
        res['artist_role'] = clean_role(res['artist_role'])
        res['artist_date_begin'] = clean_year(res['artist_date_begin'])
        res['artist_date_end'] = clean_year(res['artist_date_end'])
        return res

    if 'artist_name' in df.columns:
        cols = ['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end']
        subset = [c for c in cols if c in df.columns]
        if len(subset) == 5:
            df[subset] = df.apply(unify_artist_entity, axis=1)
        else:
            df['artist_name'] = df['artist_name'].apply(lambda x: unify_string_field(x, artist_map))

    return df

def compute_coverage(df_before, df_after, field, unified_field):
    both = df_before[field].notna() & df_after[unified_field].notna()
    if both.sum() == 0:
        return 0.0, 0.0
    changed = (df_before.loc[both, field].astype(str).str.lower().str.strip() !=
               df_after.loc[both, unified_field].astype(str).str.lower().str.strip())
    coverage = both.sum() / len(df_before) * 100
    change_rate = changed.sum() / both.sum() * 100
    return round(coverage, 1), round(change_rate, 1)


def main():
    mat_map, tec_map, culture_map, country_map, obj_map, artist_map, suffix_map, demonym_map = load_mappings()
    print(f"Materials: {len(mat_map):,} entries")
    print(f"Techniques: {len(tec_map):,} entries")
    print(f"Culture: {len(culture_map):,} entries")
    print(f"Culture Suffix Routing: {len(suffix_map):,} entries")
    print(f"Country: {len(country_map):,} entries")
    print(f"Demonym Patches: {len(demonym_map):,} entries")
    print(f"Object Name: {len(obj_map):,} entries")
    print(f"Artist: {len(artist_map):,} entries")
    print()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    print(f"Processing {len(files)} CSV files from {INPUT_DIR}/\n")

    total_rows = 0
    global_stats = defaultdict(lambda: {'coverage': [], 'change_rate': []})

    for filepath in files:
        fname = os.path.basename(filepath)
        df_orig = pd.read_csv(filepath, low_memory=False)
        df_out  = process_file(filepath, mat_map, tec_map, culture_map, country_map, obj_map, artist_map, suffix_map, demonym_map)

        rows = len(df_out)
        total_rows += rows

        stats_line = []
        for field in ['materials', 'techniques', 'culture', 'location', 'object_name', 'artist_name']:
            uf = field
            if field in df_orig.columns and uf in df_out.columns:
                cov, chg = compute_coverage(df_orig, df_out, field, uf)
                global_stats[field]['coverage'].append(cov)
                global_stats[field]['change_rate'].append(chg)
                stats_line.append(f"{field}→{chg:.0f}%chg")

        out_path = os.path.join(OUTPUT_DIR, fname)
        df_out.to_csv(out_path, index=False)
        print(f"{fname} ({rows:,} rows) | {' | '.join(stats_line)}")

    print(f"\n{'='*70}")
    print(f"Total rows processed: {total_rows:,}")
    print(f"{'='*70}")
    print(f"{'Field':<20} {'Avg Coverage':>14} {'Avg Change Rate':>16}")
    print(f"{'-'*52}")
    for field, data in global_stats.items():
        cov  = sum(data['coverage']) / len(data['coverage'])
        chg  = sum(data['change_rate']) / len(data['change_rate'])
        print(f"  {field:<18} {cov:>12.1f}%  {chg:>14.1f}%")
    print(f"\nUnified files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
