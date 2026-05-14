import os
import pandas as pd

DATA_DIR = '/data'
OUTPUT_DIR = '/results'

DATASET_VERSION = 'full' 

TRAIN_FILE = os.path.join(DATA_DIR, f'benchmark_unified_{DATASET_VERSION}_train.csv')
TEST_FILE = os.path.join(DATA_DIR, f'benchmark_unified_{DATASET_VERSION}_test.csv')

# CLIP configuration
CLIP_MODEL_NAME = '/models/clip-vit-l-14' 
CLIP_THRESHOLD = 0.25 

ERROR_COLS_MAPPING = {
    'date_begin': 'date_begin_error',
    'date_end': 'date_end_error',
    'date_begin_bce': 'date_begin_bce_error',
    'date_end_bce': 'date_end_bce_error',
    'materials': 'materials_error',
    'techniques': 'techniques_error',
    'dimensions_json': 'dimensions_json_error',
    'artist_name': 'artist_name_error',
    'artist_role': 'artist_role_error',
    'artist_nationality': 'artist_nationality_error',
    'artist_date_begin': 'artist_date_begin_error',
    'artist_date_end': 'artist_date_end_error',
    'image_url': 'image_url_error',
    'culture': 'culture_error',
    'location': 'location_error'
}

def get_effective_value(row, col):
    err_col = ERROR_COLS_MAPPING.get(col)
    if err_col and err_col in row and pd.notna(row[err_col]):
        return str(row[err_col])
    
    val = row.get(col)
    if pd.notna(val):
        return str(val)
    return ""

def format_date_string(begin, end, begin_bce, end_bce):
    def format_single(date_val, bce_flag):
        if not date_val or str(date_val).lower() == 'nan': return ""

        date_str = str(date_val).strip()
        if date_str.endswith(".0"): date_str = date_str[:-2]
        is_bce = str(bce_flag).lower() in ['true', '1', 'yes']
        return f"{date_str} BCE" if is_bce else f"{date_str}"

    b_str = format_single(begin, begin_bce)
    e_str = format_single(end, end_bce)
    
    if b_str and e_str and b_str != e_str:
        return f"{b_str} - {e_str}"
    elif b_str:
        return b_str
    elif e_str:
        return e_str
    return ""

def get_formatted_artwork_date(row):
    begin = get_effective_value(row, 'date_begin')
    end = get_effective_value(row, 'date_end')
    begin_bce = get_effective_value(row, 'date_begin_bce')
    end_bce = get_effective_value(row, 'date_end_bce')
    return format_date_string(begin, end, begin_bce, end_bce)

def get_formatted_artist_date(row):
    begin = get_effective_value(row, 'artist_date_begin')
    end = get_effective_value(row, 'artist_date_end')
    return format_date_string(begin, end, False, False)
