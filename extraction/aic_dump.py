import os
import json
import argparse
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.core.schema import ArtworkRecord
from extraction.core.utils import html_to_text, list_to_str
from extraction.core.stats import ExtractionStats

ARTWORKS_DIR = "../notebooks/artic-api-data/json/artworks"

def build_iiif_url(payload: dict, size: int = 843) -> str | None:
    data = payload.get("data", payload)
    img_id = data.get("image_id")
    if not img_id:
        alt_ids = data.get("alt_image_ids") or []
        img_id = alt_ids[0] if alt_ids else None

    if not img_id:
        return None

    iiif_base = "https://www.artic.edu/iiif/2".rstrip("/")
    return f"{iiif_base}/{img_id}/full/{size},/0/default.jpg"

def flatten_one(payload: dict) -> ArtworkRecord:
    d = payload.get("data", payload) 
    
    date_begin = d.get("date_start")
    date_end = d.get("date_end")
    date_display = d.get("date_display")
    if not date_display and (date_begin or date_end):
        date_display = f"{date_begin}-{date_end}"

    return ArtworkRecord(
        object_ID=str(d.get("id")),
        title=d.get("title"),
        object_name=d.get("artwork_type_title"),
        classification=list_to_str(d.get("classification_titles")),
        
        date=date_display,
        date_begin=str(date_begin) if date_begin else None,
        date_end=str(date_end) if date_end else None,
        
        medium=d.get("medium_display"),
        material=list_to_str(d.get("material_titles")),
        technique=list_to_str(d.get("technique_titles")),
        dimensions=d.get("dimensions"),
        
        country=d.get("place_of_origin"),
        
        artist_information=html_to_text(d.get("artist_display")),
        artist_titles=list_to_str(d.get("artist_titles")),
        
        subjects=list_to_str(d.get("subject_titles")),
        inscriptions=d.get("inscriptions"),
        description=d.get("description"),
        
        image_url=build_iiif_url(payload)
    )

def parse_args():
    p = argparse.ArgumentParser(description="AIC dump flattener")
    p.add_argument("--descriptions", action="store_true", help="Only extract records that have a description.")
    p.add_argument("--out", default="csv_pipeline", help="Output directory folder (default: csv_pipeline)")
    p.add_argument("--limit", type=int, default=0, help="Max records to collect (0 for all).")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    filename = "aic_artworks_dump_desc.csv" if args.descriptions else "aic_artworks_dump.csv"
    output_path = os.path.join(args.out, filename)
    
    stats = ExtractionStats("AIC (Art Institute of Chicago)")
    
    files = [f for f in os.listdir(ARTWORKS_DIR) if f.lower().endswith(".json")]
    
    for fname in tqdm(files, desc="Parsing AIC Data"):
        if args.limit > 0 and stats.collected >= args.limit:
            break
            
        stats.record_seen()
        fpath = os.path.join(ARTWORKS_DIR, fname)
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            stats.record_skipped("load_error")
            stats.record_failure()
            continue

        d = payload.get("data", payload)

        if not d.get("is_public_domain", False):
            stats.record_skipped("not_public_domain")
            continue

        has_image_id = bool(d.get("image_id"))
        has_alt_images = bool(d.get("alt_image_ids"))
        if not (has_image_id or has_alt_images):
            stats.record_skipped("no_image_metadata")
            continue

        try:
            record: ArtworkRecord = flatten_one(payload)
        except Exception:
            stats.record_skipped("flatten_error")
            stats.record_failure()
            continue

        if not record.image_url:
            stats.record_skipped("no_image_url_built")
            continue

        if args.descriptions and not record.description:
            stats.record_skipped("missing_description")
            continue

        stats.record_collected(record.to_dict())

    stats.print_summary()

    if stats.collected > 0:
        df = stats.generate_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Wrote data to {output_path}")
    else:
        print("No rows collected; not writing CSV.")

if __name__ == "__main__":
    main()