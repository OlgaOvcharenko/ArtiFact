import sys
import os
import argparse
import ast
import warnings
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from cleanlab.filter import find_label_issues

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_FILE, TEST_FILE, OUTPUT_DIR, get_effective_value
from evaluate import evaluate_and_save

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = 'clip_cleanlab'
BATCH_SIZE = 64 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_CLASS_SAMPLES = 5        

def load_data(filepath, n_samples=None, seed=42):
    df = pd.read_csv(filepath)
    print(f"DEBUG: Loaded {len(df)} rows from {filepath}")
    df['object_ID'] = df['object_ID'].astype(str)
    if n_samples:
        clean = df[df['error_type'].isna()]
        error = df[df['error_type'].notna()]
        nc = min(n_samples // 2, len(clean))
        ne = min(n_samples // 2, len(error))
        df = pd.concat([clean.sample(nc, random_state=seed), error.sample(ne, random_state=seed)]).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"DEBUG: Subsampled to {len(df)} rows")
    return df

def parse_list_col(val):
    if pd.isna(val) or val in ('[]', '', 'nan'): return []
    try:
        parsed = ast.literal_eval(str(val).strip())
        if isinstance(parsed, list): return [str(x).strip().lower() for x in parsed if str(x).strip()]
        return [str(parsed).strip().lower()]
    except: return [str(val).strip().lower()]

def load_all_embeddings_to_dict(embeddings_dir="/results/image_embeddings"):
    npz_files = glob.glob(os.path.join(embeddings_dir, "*.npz"))
    print(f"DEBUG: Found {len(npz_files)} .npz files in {embeddings_dir}")
    emb_dict = {}
    for f in tqdm(npz_files, desc="Reading Embeddings"):
        data = np.load(f)
        ids = data['object_ids']
        embs = data['embeddings']
        for obj_id, emb in zip(ids, embs): emb_dict[str(obj_id)] = emb
    print(f"DEBUG: Total unique IDs in emb_dict: {len(emb_dict)}")
    if emb_dict:
        print(f"DEBUG: Sample ID from emb_dict: '{list(emb_dict.keys())[0]}'")
    return emb_dict

def map_embeddings_to_df(df, emb_dict):
    emb_dim = 768
    for v in emb_dict.values():
        emb_dim = v.shape[0]
        break
    embeddings = np.zeros((len(df), emb_dim), dtype=np.float32)
    valid_mask = np.zeros(len(df), dtype=bool)
    match_count = 0
    for i, row in df.iterrows():
        target_id = row.get('image_object_id_error')
        if pd.isna(target_id) or str(target_id).strip() == "":
            target_id = str(row['object_ID'])
        
        target_id_str = str(target_id)
        if target_id_str in emb_dict:
            embeddings[i] = emb_dict[target_id_str]
            valid_mask[i] = True
            match_count += 1
        elif i < 5:
            print(f"DEBUG: Missed match for ID: '{target_id_str}'")
            
    print(f"DEBUG: Successfully mapped {match_count} embeddings out of {len(df)} rows")
    return embeddings, valid_mask

def cleanlab_binary_train_and_flag_test(X_train, y_train, X_test, y_test, test_valid_indices, seed=42):
    if len(np.unique(y_train)) < 2: return set()
    try:
        clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        issues = find_label_issues(labels=y_test, pred_probs=probs, return_indices_ranked_by='self_confidence')
        return set(test_valid_indices[issues])
    except: return set()

def cleanlab_multiclass_train_and_flag_test(X_train_all, y_train_raw, X_test_all, y_test_raw, test_valid_indices, seed=42):
    class_counts = pd.Series(y_train_raw).value_counts()
    valid_classes = set(class_counts[class_counts >= MIN_CLASS_SAMPLES].index)

    train_mask = np.array([y in valid_classes for y in y_train_raw])
    X_train_v, y_train_v = X_train_all[train_mask], y_train_raw[train_mask]
    if len(np.unique(y_train_v)) < 2: 
        return set()
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_v.astype(str))
    num_classes = len(le.classes_)
    test_mask = np.array([y in set(le.classes_) for y in y_test_raw])
    if test_mask.sum() == 0: 
        return set()
    
    X_test_v, y_test_encoded = X_test_all[test_mask], le.transform(y_test_raw[test_mask].astype(str))
    test_indices_v = test_valid_indices[test_mask]
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_probs_ensemble = np.zeros((len(X_test_v), num_classes), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_v, y_train_encoded)):
        X_tr, y_tr = X_train_v[train_idx], y_train_encoded[train_idx]
        model = nn.Linear(X_tr.shape[1], num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=2048, shuffle=True)
        model.train()
        for _ in range(10):
            for b_x, b_y in loader:
                optimizer.zero_grad(); out = model(b_x.to(device)); loss = criterion(out, b_y.to(device)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(torch.from_numpy(X_test_v), batch_size=2048)
            fold_test_probs = [torch.softmax(model(b_x.to(device)), dim=1).cpu().numpy() for b_x in test_loader]
            test_probs_ensemble += np.vstack(fold_test_probs) / n_folds
    
    issues = find_label_issues(labels=y_test_encoded, pred_probs=test_probs_ensemble, return_indices_ranked_by='self_confidence')
    return set(test_indices_v[issues])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    df_train = load_data(TRAIN_FILE, n_samples=args.n_samples, seed=args.seed)
    df_test = load_data(TEST_FILE, n_samples=args.n_samples, seed=args.seed)

    print("Loading Embeddings...")
    emb_dict = load_all_embeddings_to_dict()
    train_embs, train_vm = map_embeddings_to_df(df_train, emb_dict)
    test_embs, test_vm = map_embeddings_to_df(df_test, emb_dict)
    del emb_dict

    print("Expanding Binary Features (n >= 1)...")
    mlb = MultiLabelBinarizer()
    
    train_mats = df_train['materials'].apply(parse_list_col)
    train_techs = df_train['techniques'].apply(parse_list_col)
    train_bin = mlb.fit_transform(train_mats + train_techs)
    
    test_mats = df_test.apply(lambda r: get_effective_value(r, 'materials'), axis=1).apply(parse_list_col)
    test_techs = df_test.apply(lambda r: get_effective_value(r, 'techniques'), axis=1).apply(parse_list_col)
    test_bin = mlb.transform(test_mats + test_techs)
    
    X_train = np.hstack([train_embs[train_vm], train_bin[train_vm]]).astype(np.float32)
    X_test = np.hstack([test_embs[test_vm], test_bin[test_vm]]).astype(np.float32)
    test_valid_idx = np.where(test_vm)[0]

    print(f"Feature Space: {X_train.shape[1]} dimensions.")
    flag_reasons = defaultdict(list)

    for target_col in ['object_name', 'artist_name', 'culture', 'location']:
        print(f"Target: {target_col}")
        y_train_raw = df_train[target_col].fillna('unknown').astype(str).values[train_vm]
        y_test_raw = df_test.apply(lambda r: get_effective_value(r, target_col), axis=1).fillna('unknown').astype(str).values[test_vm]
        
        flagged = cleanlab_multiclass_train_and_flag_test(X_train, y_train_raw, X_test, y_test_raw, test_valid_idx, seed=args.seed)
        for idx in flagged: flag_reasons[idx].append(f"clip_cl:{target_col}")

    y_pred = np.zeros(len(df_test), dtype=bool)
    for idx in flag_reasons: y_pred[idx] = True
    df_test['predicted_has_error'] = y_pred
    
    y_true = df_test['error_type'].notna().values
    fname = f"{EXPERIMENT_NAME}_seed_{args.seed}"
    evaluate_and_save(df_test, y_true, y_pred, fname, OUTPUT_DIR)

if __name__ == "__main__":
    main()
