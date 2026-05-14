import os
import argparse
import random
import sys
import pandas as pd
from typing import Optional, List
from tqdm import tqdm
import requests
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.core.schema import ArtworkRecord
from extraction.core.utils import slugify
from extraction.core.network import SessionManager
from extraction.core.stats import ExtractionStats

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
CACHE_DIR = "../notebooks/cache/met"
OUTPUT_DIR = os.path.join(CACHE_DIR, "met_samples_by_department")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION_MGR = SessionManager()

DESCRIPTIONS = {}

def load_descriptions():
    """Load descriptions from BetterMetObjects.csv into a global dictionary."""
    global DESCRIPTIONS
    csv_path = "data/BetterMetObjects.csv"
    if os.path.exists(csv_path):
        print(f"Loading descriptions from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, usecols=["object_id", "description"], dtype={"object_id": int, "description": str})
            df = df.dropna(subset=["description"])
            DESCRIPTIONS = dict(zip(df["object_id"], df["description"]))
            print(f"Loaded {len(DESCRIPTIONS)} descriptions.")
        except Exception as e:
            print(f"Error loading descriptions: {e}")
    else:
        print(f"Warning: {csv_path} not found. Descriptions will be empty.")


def _cache_path(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return os.path.join(CACHE_DIR, f"{safe}.json")


def fetch_json(url: str, params: Optional[dict] = None, cache_key: str = None, force: bool = False) -> dict:
    cache_file = _cache_path(cache_key)

    if not force and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    r = SESSION_MGR.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def met_object(object_id: int) -> Optional[dict]:
    url = f"{BASE_URL}/objects/{int(object_id)}"
    try:
        return fetch_json(url, cache_key=f"object_{object_id}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def normalize_object(o: dict, description: Optional[str] = None) -> ArtworkRecord:
    constituents = o.get("constituents") or []
    first_artist = constituents[0] if constituents else {}
    tags = o.get("tags") or []
    tag_terms = [t.get("term") for t in tags if isinstance(t, dict) and t.get("term")]

    return ArtworkRecord(
        object_ID=str(o.get("objectID")),
        title=o.get("title"),
        object_name=o.get("objectName"),
        classification=o.get("classification"),
        department=o.get("department"),
        
        date=str(o.get("objectDate")) if o.get("objectDate") is not None else None,
        date_begin=str(o.get("objectBeginDate")) if o.get("objectBeginDate") is not None else None,
        date_end=str(o.get("objectEndDate")) if o.get("objectEndDate") is not None else None,
        
        medium=o.get("medium"),
        dimensions=o.get("dimensions"),
        
        culture=o.get("culture"),
        country=o.get("country"),
        region=o.get("region"),
        city=o.get("city"),
        period=o.get("period"),
        dynasty=o.get("dynasty"),
        reign=o.get("reign"),
        
        artist_name=o.get("artistDisplayName") or first_artist.get("name"),
        artist_role=o.get("artistRole") or first_artist.get("role"),
        artist_nationality=o.get("artistNationality"),
        artist_date_begin=str(o.get("artistBeginDate")) if o.get("artistBeginDate") is not None else None,
        artist_date_end=str(o.get("artistEndDate")) if o.get("artistEndDate") is not None else None,
        artist_display_bio=o.get("artistDisplayBio"),
        
        subjects=",".join(tag_terms) if tag_terms else None,
        description=description,
        
        image_url=o.get("primaryImage"),
    )


def met_objects_ids_from_dept(department_ids: List[int] | int) -> List[int]:
    if isinstance(department_ids, (list, tuple, set)):
        dep_str = "|".join(str(d) for d in department_ids)
    else:
        dep_str = str(department_ids)
    url = f"{BASE_URL}/objects"
    params = {"departmentIds": dep_str}
    data = fetch_json(url, params=params, cache_key=f"objects_{dep_str}")
    return data.get("objectIDs") or []


def met_departments() -> List[dict]:
    url = f"{BASE_URL}/departments"
    data = fetch_json(url, cache_key="departments")
    return data.get("departments", [])


def run_samples_by_department(sample_size: int = 10, max_scan_per_dept: int = 10, only_descriptions: bool = False) -> None:
    departments = met_departments()
    print(f"[all-depts] Found {len(departments)} departments.")

    summary_rows = []
    
    global_stats = ExtractionStats("MET (Overall Extraction)")

    for d in departments:
        dep_id = d["departmentId"]
        dep_name = d["displayName"]
        
        dept_stats = ExtractionStats(f"MET Department: {dep_name}")

        ids = met_objects_ids_from_dept(dep_id)
        print(f"\n[dept {dep_id}] {dep_name} — {len(ids)} candidate IDs (pre-filter). Scanning...")

        random.shuffle(ids)
        scan_limit = min(len(ids), max_scan_per_dept)
        
        pbar = tqdm(ids[:scan_limit], desc=f"Scanning {dep_name}")
        
        for oid in pbar:
            dept_stats.record_seen()
            global_stats.record_seen()
            
            desc = DESCRIPTIONS.get(oid)
            
            if only_descriptions and not desc:
                dept_stats.record_skipped("missing_description")
                global_stats.record_skipped("missing_description")
                continue

            o = met_object(oid)
            if not o:
                dept_stats.record_skipped("fetch_failed_or_404")
                global_stats.record_skipped("fetch_failed_or_404")
                continue
                
            if not o.get("isPublicDomain"):
                dept_stats.record_skipped("not_public_domain")
                global_stats.record_skipped("not_public_domain")
                continue
                
            if not (o.get("primaryImage") or o.get("primaryImageSmall")):
                dept_stats.record_skipped("no_image")
                global_stats.record_skipped("no_image")
                continue

            record = normalize_object(o, description=desc)
            
            if not record.image_url:
                dept_stats.record_skipped("empty_image_url")
                global_stats.record_skipped("empty_image_url")
                continue

            dept_stats.record_collected(record.to_dict())
            global_stats.record_collected(record.to_dict())
            
            pbar.set_postfix({"collected": dept_stats.collected})
            
            if dept_stats.collected >= sample_size:
                break

        if dept_stats.collected == 0:
            print(f"  -> No qualifying objects found for {dep_name}.")
            summary_rows.append({
                "departmentId": dep_id,
                "department": dep_name,
                "candidates": len(ids),
                "scanned": scan_limit,
                "kept": 0,
                "output_csv": None
            })
            continue

        df = dept_stats.generate_dataframe()
        suffix = "_desc" if only_descriptions else ""
        out_name = f"dept_{dep_id}_{slugify(dep_name)}_sample_{len(df)}{suffix}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f"  -> {dep_name} Kept {len(df)} objects. Saved: {out_path}")

        filled = int(df.notna().sum().sum())
        missing = int(df.isna().sum().sum())
        
        summary_rows.append({
            "departmentId": dep_id,
            "department": dep_name,
            "candidates": len(ids),
            "scanned": scan_limit,
            "kept": len(df),
            "cells_filled": filled,
            "cells_missing": missing,
            "output_csv": out_path
        })

    global_stats.print_summary()

    overview_df = pd.DataFrame(summary_rows)
    overview_path = os.path.join(OUTPUT_DIR, "overview_by_department.csv")
    overview_df.to_csv(overview_path, index=False)
    print(f"\n[all-depts] Overview saved: {overview_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MET sampling script.")
    p.add_argument("--run-dept-search", action="store_true",
                   help="Run search-based sample for a department (default Egyptian Art).")
    p.add_argument("--run-all-departments", action="store_true",
                   help="Run the full department sweep (can be heavy).")

    p.add_argument("--dept-id", type=int, default=10, help="Department ID to use where applicable.")
    p.add_argument("--dept-search-n", type=int, default=10, help="Rows to keep for search in dept sample.")

    p.add_argument("--all-depts-n", type=int, default=100000, help="Target rows per department in full sweep.")
    p.add_argument("--all-depts-max-scan", type=int, default=200000, help="Max objects scanned per department.")
    p.add_argument("--out", default="csv_pipeline", help="Output directory (default: csv_pipeline).")
    p.add_argument("--descriptions", action="store_true", help="Only extract records that have a description.")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    global OUTPUT_DIR
    args = parse_args(argv)
    
    if args.out:
        OUTPUT_DIR = args.out
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not args.run_dept_search:
        load_descriptions()
        run_samples_by_department(sample_size=args.all_depts_n,
                                  max_scan_per_dept=args.all_depts_max_scan,
                                  only_descriptions=args.descriptions)
    
    if args.run_dept_search:
        print("Dept search to be implemented.")

if __name__ == "__main__":
    main()