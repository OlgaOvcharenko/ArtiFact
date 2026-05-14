import pandas as pd
import json
import os
import glob
import ast
import re

INPUT_DIR  = "parsed_pipeline"
OUTPUT_DIR = "consolidated_pipeline"

COLS_TO_DROP = {
    'date',
    'raw_date',
    'medium',
    'material',
    'technique',
    'raw_dimensions',
    'parsed_date',
    'parsed_medium',
    'parsed_artist',
    'artist_display_bio',
    'artist_information',
    'artist_titles',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_json(raw):
    if pd.isna(raw) or not str(raw).strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(str(raw))
        except Exception:
            return None


def unwrap_date(row):
    d = safe_json(row.get('parsed_date'))
    if d and isinstance(d, dict):
        return {
            'date_begin': d.get('date_begin'),
            'date_end': d.get('date_end'),
            'date_begin_bce': d.get('date_begin_bce', False),
            'date_end_bce': d.get('date_end_bce', False),
        }
    
    begin = row.get('date_begin')
    end = row.get('date_end')
    begin_bce = row.get('date_begin_bce', False)
    end_bce = row.get('date_end_bce', False)

    try:
        if begin is not None and float(begin) < 0:
            begin = abs(float(begin))
            begin_bce = True
        if end is not None and float(end) < 0:
            end = abs(float(end))
            end_bce = True
    except (ValueError, TypeError):
        pass

    return {
        'date_begin': begin,
        'date_end': end,
        'date_begin_bce': begin_bce,
        'date_end_bce': end_bce,
    }


def unwrap_medium(row):
    d = safe_json(row.get('parsed_medium'))
    if d and isinstance(d, dict):
        mats = [m for m in d.get('materials', []) if m]
        tecs = [t for t in d.get('techniques', []) if t]
        return {
            'materials': json.dumps(mats) if mats else None,
            'techniques': json.dumps(tecs) if tecs else None,
        }
    return {'materials': None, 'techniques': None}


def unwrap_dimensions(row):
    raw = row.get('parsed_dimensions')
    d = safe_json(raw)
    if d and isinstance(d, dict):
        d.pop('scratchpad', None)
        return json.dumps(d)
    return raw


def unwrap_artist(row):
    parsed = safe_json(row.get('parsed_artist'))
    
    def clean_year(val):
        if pd.isna(val) or not str(val).strip():
            return None
        s = str(val).strip().lower()
        if s == 'nan' or s == 'none' or s == '{}':
            return None

        match = re.search(r'(\d{4})', s)
        if match:
            return match.group(1)

        match = re.search(r'(\d{1,3})', s)
        if match:
            return match.group(1)
        return s

    def clean_role(val):
        if pd.isna(val) or not str(val).strip():
            return None
        s = str(val).strip()

        if s in ('{}', '[]', 'None', 'nan', 'null'):
            return None
        return s

    out = {
        'artist_name': row.get('artist_name'),
        'artist_role': clean_role(row.get('artist_role')),
        'artist_nationality': row.get('artist_nationality'),
        'artist_date_begin': clean_year(row.get('artist_date_begin')),
        'artist_date_end': clean_year(row.get('artist_date_end')),
        'culture': None,
        'country': None,
        'region': None,
        'city': None,
    }

    if parsed and isinstance(parsed, list) and len(parsed) > 0:
        a = parsed[0]
        
        name = str(a.get('artist_name', '') or '').strip()
        if name:
            out['artist_name'] = name
            out['artist_role'] = clean_role(a.get('role') or row.get('artist_role'))
            out['artist_nationality'] = str(a.get('artist_nationality', '') or row.get('artist_nationality', '') or '').strip() or None
            out['artist_date_begin'] = clean_year(a.get('artist_date_begin') or row.get('artist_date_begin'))
            out['artist_date_end'] = clean_year(a.get('artist_date_end') or row.get('artist_date_end'))

        out['culture'] = a.get('culture') or None
        out['country'] = a.get('country') or None
        out['region'] = a.get('region')  or None
        out['city'] = a.get('city')    or None

    return out


CANONICAL_COLS = [
    'object_ID', 'title', 'object_name',
    'date_begin', 'date_end', 'date_begin_bce', 'date_end_bce',
    'materials', 'techniques', 'dimensions_json',
    'culture', 'location', 'period', 'dynasty', 'reign',
    'artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end',
    'subjects', 'description', 'inscriptions', 'image_url',
]

def process_file(filepath):
    df = pd.read_csv(filepath, low_memory=False)

    dates = df.apply(lambda row: unwrap_date(row), axis=1, result_type='expand')
    medium = df.apply(lambda row: unwrap_medium(row), axis=1, result_type='expand')
    dimensions = df.apply(lambda row: unwrap_dimensions(row), axis=1)
    artist_data = df.apply(lambda row: unwrap_artist(row), axis=1, result_type='expand')
    out = pd.DataFrame()
    for col in CANONICAL_COLS:
        if col in ['date_begin', 'date_end', 'date_begin_bce', 'date_end_bce']:
            out[col] = dates[col]
        elif col in ['materials', 'techniques']:
            out[col] = medium[col]
        elif col == 'dimensions_json':
            out[col] = dimensions
        elif col in ['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end']:
            out[col] = artist_data[col]
        elif col == 'culture':
            out[col] = df[col] if col in df.columns else None
            if col in artist_data.columns:
                out[col] = out[col].fillna(artist_data[col])
        elif col == 'location':
            c = df['country'] if 'country' in df.columns else pd.Series(index=df.index, dtype=object)
            r = df['region'] if 'region' in df.columns else pd.Series(index=df.index, dtype=object)
            ci = df['city'] if 'city' in df.columns else pd.Series(index=df.index, dtype=object)
            out['location'] = c.fillna(r).fillna(ci)
        else:
            out[col] = df[col] if col in df.columns else None

    return out

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    print(f"Stage 5: Schema Consolidation")
    print(f"Input:  {INPUT_DIR}/  ({len(files)} files)")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Schema: {len(CANONICAL_COLS)} canonical columns\n")

    total_rows = 0
    for filepath in files:
        fname = os.path.basename(filepath)
        df_out = process_file(filepath)
        out_path = os.path.join(OUTPUT_DIR, fname)
        df_out.to_csv(out_path, index=False)
        total_rows += len(df_out)

        mat_cov  = df_out['materials'].notna().sum()
        tec_cov  = df_out['techniques'].notna().sum()
        cult_cov = df_out['culture'].notna().sum()
        art_cov  = df_out['artist_name'].notna().sum()
        n = len(df_out)
        print(f"{fname} ({n:,} rows) | "
              f"materials={mat_cov:,} | techniques={tec_cov:,} | "
              f"culture={cult_cov:,} | artist={art_cov:,}")

    print(f"\nTotal rows consolidated: {total_rows:,}")
    print(f"Output saved to: {OUTPUT_DIR}/")
    print(f"\nReady for Stage 6: apply_unification.py")


if __name__ == "__main__":
    main()
