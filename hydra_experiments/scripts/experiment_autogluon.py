import sys
import os
import argparse
import ast
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except:
    pass

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_FILE, TEST_FILE, OUTPUT_DIR, get_effective_value, ERROR_COLS_MAPPING
from evaluate import evaluate_and_save
from autogluon.multimodal import MultiModalPredictor
from cleanlab.filter import find_label_issues

EXPERIMENT_NAME = 'autogluon_cleanlab'
IMAGE_DIRS = [f"/dataset/chunk_{i:02d}" for i in range(14)] + ["/dataset/chunk_AIC"]
MODEL_DIR = os.path.expanduser("~/autogluon_models") 
os.makedirs(MODEL_DIR, exist_ok=True)  

def get_img_path(row):
    target_id = row.get('image_object_id_error')
    if pd.isna(target_id) or str(target_id).strip() == "":
        target_id = row['object_ID']
        
    img_filename = f"{target_id}.jpg"
    for chunk_dir in IMAGE_DIRS:
        path = os.path.join(chunk_dir, img_filename)
        if os.path.exists(path): return path
    return None

def parse_list_col(val):
    if pd.isna(val) or val in ('[]', '', 'nan'): return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list): return [str(x).lower().strip() for x in parsed]
        return [str(parsed).lower().strip()]
    except: return [str(val).lower().strip()]

def extract_numeric_dimensions(df):
    """dimensions parsing. Keep main part only."""
    heights, widths, depths, diameters, thicknesses, lengths = [], [], [], [], [], []
    for _, row in df.iterrows():
        dim_str = row.get('dimensions_json', '')
        h, w, d, dia, t, l = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        try:
            if pd.notna(dim_str) and dim_str != '':
                data = json.loads(dim_str) if isinstance(dim_str, str) else dim_str
                if isinstance(data, dict) and 'parts' in data and len(data['parts']) > 0:
                    first_part = data['parts'][0]
                    for meas in first_part.get('measurements', []):
                        attr = meas.get('attribute', '').lower()
                        val = meas.get('value')
                        try:
                            val = float(val)
                            if attr == 'height' and pd.isna(h): h = val
                            elif attr == 'width' and pd.isna(w): w = val
                            elif attr == 'depth' and pd.isna(d): d = val
                            elif attr == 'diameter' and pd.isna(dia): dia = val
                            elif attr == 'thickness' and pd.isna(t): t = val
                            elif attr == 'length' and pd.isna(l): l = val
                        except: pass
        except: pass
        heights.append(h); widths.append(w); depths.append(d); diameters.append(dia); thicknesses.append(t); lengths.append(l)
    df['dim_height'] = heights; df['dim_width'] = widths; df['dim_depth'] = depths; df['dim_diameter'] = diameters; df['dim_thickness'] = thicknesses; df['dim_length'] = lengths
    return df.drop(columns=['dimensions_json'])

def load_data(filepath, n_samples=None, modality='both', is_test=False, seed=42):
    df = pd.read_csv(filepath)
    df['object_ID'] = df['object_ID'].astype(str)
    df['_source'] = df['object_ID'].str.split('_').str[0]
    
    if is_test:
        for col in ERROR_COLS_MAPPING.keys():
            if col in df.columns:
                df[col] = df.apply(lambda r: get_effective_value(r, col), axis=1)

    df = extract_numeric_dimensions(df)

    def calculate_midpoint(row):
        try:
            b = float(row.get('date_begin', np.nan))
            e = float(row.get('date_end', b))
            if pd.isna(b): return np.nan
            if str(row.get('date_begin_bce', '')).lower() in ['true', '1', 'yes']: b = -b
            if str(row.get('date_end_bce', '')).lower() in ['true', '1', 'yes']: e = -e
            return (b + e) / 2.0
        except: return np.nan

    df['date_midpoint'] = df.apply(calculate_midpoint, axis=1)
    df['image_path'] = df.apply(get_img_path, axis=1)
    
    if modality in ['image', 'both']:
        df = df.dropna(subset=['image_path'])
    
    if n_samples:
        print(f"Performing High-Fidelity Stratified Subsampling (n={n_samples})...")
        df['stratify_col'] = df['_source'] + "_" + df['error_type'].fillna('clean').astype(str)
        
        counts = df['stratify_col'].value_counts(normalize=True)
        samples = []
        for strata, weight in counts.items():
            strata_n = max(1, int(weight * n_samples))
            strata_df = df[df['stratify_col'] == strata]
            samples.append(strata_df.sample(min(len(strata_df), strata_n), random_state=seed))
        
        df = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)
        df = df.drop(columns=['stratify_col'])
        
    return df

def get_feature_columns(train_data, target_col, modality):
    exclude = ['error_type', 'error_subtype', 'object_ID', 'image', 'image_error', 'image_url', target_col]
    exclude.extend(list(ERROR_COLS_MAPPING.values()))
    
    if 'date' in target_col:
        exclude.extend(['date_begin', 'date_end', 'date_begin_bce', 'date_end_bce', 'period', 'dynasty', 'reign', 'date_midpoint'])
    elif 'dim_' in target_col:
        exclude.extend(['dim_height', 'dim_width', 'dim_depth', 'dim_diameter', 'dim_thickness', 'dim_length'])
    elif 'artist' in target_col:
        exclude.extend(['artist_name', 'artist_role', 'artist_nationality', 'artist_date_begin', 'artist_date_end'])
    elif 'culture' in target_col:
        exclude.extend(['culture', 'location'])
    elif 'location' in target_col:
        exclude.extend(['location', 'culture'])

    if modality == 'table':
        exclude.append('image_path')
        return [c for c in train_data.columns if c not in exclude]
    elif modality == 'image':
        return ['image_path']
    else:
        return [c for c in train_data.columns if c not in exclude]

def get_checkpoint_path(target_col, modality, n_samples, seed):
    safe_name = target_col.replace(" ", "_").replace("/", "_")
    base_dir = os.path.join(OUTPUT_DIR, f"ag_checkpoints_{modality}")
    if n_samples: base_dir += "_debug"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"target_{safe_name}_seed_{seed}.json")

def load_checkpoint(target_col, modality, n_samples, seed):
    path = get_checkpoint_path(target_col, modality, n_samples, seed)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return set([int(k) for k in data['flagged']]), {int(k): v for k, v in data['corrections'].items()}
    return None, None

def save_checkpoint(target_col, modality, n_samples, seed, flagged_indices, corrections):
    path = get_checkpoint_path(target_col, modality, n_samples, seed)
    data = {'flagged': list(flagged_indices), 'corrections': corrections}
    with open(path, 'w') as f: json.dump(data, f)

def process_multiclass_target(target_col, train_df, test_df, time_limit, modality, n_samples, seed):
    print(f"\n--- Multiclass Target: {target_col} | Seed: {seed} ---")
    flagged, corrections = load_checkpoint(target_col, modality, n_samples, seed)
    if flagged is not None:
        print(f"Loaded from checkpoint. Flagged {len(flagged)} issues.")
        return flagged, corrections

    train_data = train_df.copy()
    test_data = test_df.copy()
    train_data[target_col] = train_data[target_col].fillna('unknown').astype(str)
    test_data[target_col] = test_data[target_col].fillna('unknown').astype(str)
    
    class_counts = train_data[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    train_data = train_data[train_data[target_col].isin(valid_classes)]
    
    if len(train_data) == 0: 
        save_checkpoint(target_col, modality, n_samples, seed, set(), {})
        return set(), {}

    feature_cols = get_feature_columns(train_data, target_col, modality)
    model_path = os.path.join(MODEL_DIR, f"ag_{target_col}_{modality}_seed_{seed}")

    predictor = None
    if os.path.exists(model_path):
        try:
            print(f"Existing model found at {model_path}. Attempting to load...")
            predictor = MultiModalPredictor.load(model_path)
            print("Model loaded successfully. Skipping training phase.")
        except Exception as e:
            print(f"Could not load existing model: {e}. Will retrain.")
            shutil.rmtree(model_path)

    if predictor is None:
        print(f"Starting training phase (Time Limit: {time_limit}s)...")
        predictor = MultiModalPredictor(label=target_col, path=model_path, verbosity=0)
        predictor.fit(
            train_data=train_data[feature_cols + [target_col]], 
            time_limit=time_limit, 
            seed=seed, 
            hyperparameters={"env.num_gpus": -1}
        )    
    test_subset = test_data[test_data[target_col].isin(valid_classes)]
    if len(test_subset) == 0: 
        save_checkpoint(target_col, modality, n_samples, seed, set(), {})
        return set(), {}

    probs = predictor.predict_proba(test_subset[feature_cols])
    labels = test_subset[target_col].values
    possible_labels = probs.columns.values
    label_to_idx = {l: i for i, l in enumerate(possible_labels)}
    valid_mask = [l in label_to_idx for l in labels]
    probs_clean = probs.values[valid_mask]
    int_labels = np.array([label_to_idx[l] for l in labels[valid_mask]])
    indices_clean = test_subset.index[valid_mask]
    
    issues_idx = find_label_issues(labels=int_labels, pred_probs=probs_clean, return_indices_ranked_by='self_confidence')
    flagged_indices = set(indices_clean[issues_idx])
    corrections = {int(indices_clean[idx]): possible_labels[p] for idx, p in zip(issues_idx, probs_clean[issues_idx].argmax(axis=1))}
    
    print(f"  Flagged {len(issues_idx)} issues.")
    del predictor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if os.path.exists(model_path): shutil.rmtree(model_path)
    save_checkpoint(target_col, modality, n_samples, seed, flagged_indices, corrections)
    return flagged_indices, corrections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--time_limit', type=int, default=36000)
    parser.add_argument('--modality', type=str, choices=['table', 'image', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target', type=str, default=None, choices=['object_name', 'artist_name', 'culture', 'location'], help="Run only one specific target model.")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- AutoGluon Master | Modality: {args.modality.upper()} | Time Limit: {args.time_limit}s/model | Seed: {args.seed} ---")
    
    print("Loading Data...")
    df_train = load_data(TRAIN_FILE, args.n_samples, args.modality, is_test=False, seed=args.seed)
    df_test = load_data(TEST_FILE, args.n_samples, args.modality, is_test=True, seed=args.seed)    

    print("\nExhaustive Binary Feature Expansion (n >= 1)...")
    for list_col in ['materials', 'techniques']:
        df_train[list_col] = df_train[list_col].apply(parse_list_col)
        df_test[list_col] = df_test[list_col].apply(parse_list_col)
        
        mlb = MultiLabelBinarizer()
        train_bin_matrix = mlb.fit_transform(df_train[list_col])
        test_bin_matrix = mlb.transform(df_test[list_col])
        
        feat_names = [f"feat_{list_col}_{cls.replace(' ', '_').replace('/', '_')}" for cls in mlb.classes_]
        
        train_bin_df = pd.DataFrame(train_bin_matrix, columns=feat_names, index=df_train.index)
        test_bin_df = pd.DataFrame(test_bin_matrix, columns=feat_names, index=df_test.index)
        
        df_train = pd.concat([df_train, train_bin_df], axis=1)
        df_test = pd.concat([df_test, test_bin_df], axis=1)
        
        df_train = df_train.drop(columns=[list_col])
        df_test = df_test.drop(columns=[list_col])
    
    print("Starting...")
    flag_reasons = defaultdict(list)
    suggested_corrections = defaultdict(dict) 

    if args.target:
        targets = [args.target]
    else:
        targets = ['object_name', 'artist_name', 'culture', 'location']

    for target in targets:
        flagged, corrections = process_multiclass_target(target, df_train, df_test, args.time_limit, args.modality, args.n_samples, args.seed)
        for idx in flagged:
            flag_reasons[idx].append(f"ag:{target}")
            suggested_corrections[idx][target] = corrections[idx]

    if not args.target:
        df_test['predicted_has_error'] = False
        for idx in flag_reasons:
            df_test.loc[idx, 'predicted_has_error'] = True
            
        y_true = df_test['error_type'].notna().values
        y_pred = df_test['predicted_has_error'].values
        fname = f"{EXPERIMENT_NAME}_{args.modality}_seed_{args.seed}{'_debug' if args.n_samples else ''}"
        evaluate_and_save(df_test, y_true, y_pred, fname, OUTPUT_DIR)
    
    print("Done.")

if __name__ == "__main__":
    main()
