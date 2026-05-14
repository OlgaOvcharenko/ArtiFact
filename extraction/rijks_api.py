import os
import json
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Tuple
import time
import sys
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.core.schema import ArtworkRecord
from extraction.core.utils import resolve_language, list_to_str
from extraction.core.network import SessionManager
from extraction.core.stats import ExtractionStats

from rijks_oai_utils import fill_row_from_rdf, NS

BASE = "https://data.rijksmuseum.nl/oai"

DEFAULT_SET_IDS = [
    "260245","2611","26113","26118","2612","26121","261223","261224",
    "261231","261233","26126","2613","26142","26147","2616","2618","26191"
]

CACHE_DIR = os.environ.get("RIJKS_OAI_CACHE", "../notebooks/cache/rijks-oai")
os.makedirs(CACHE_DIR, exist_ok=True)

SESSION_MGR = SessionManager(rotate_every=25)

def _oai_cache_path(params: Dict[str, Any]) -> str:
    key = json.dumps(params, sort_keys=True)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.xml")


def fetch_oai(params: Dict[str, Any], timeout: int = 30, force: bool = False) -> ET.Element:
    cache_file = _oai_cache_path(params)
    if not force and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return ET.fromstring(f.read())

    r = SESSION_MGR.get(BASE, params=params, timeout=timeout)
    r.raise_for_status()
    content = r.content
    with open(cache_file, "wb") as f:
        f.write(content)
    return ET.fromstring(content)


def extract_record(rdf: ET.Element) -> Tuple[ArtworkRecord, Optional[str]]:
    """Extract an XML/RDF record into the standard ArtworkRecord."""
    cho_uri = None
    agg = rdf.find(".//ore:Aggregation", NS)
    if agg is not None:
        res = agg.find("edm:aggregatedCHO", NS)
        if res is not None:
            cho_uri = res.attrib.get('{%s}resource' % NS['rdf'])
            
    if not cho_uri:
        prov_cho = rdf.find(".//edm:ProvidedCHO", NS)
        if prov_cho is not None:
            cho_uri = prov_cho.attrib.get('{%s}about' % NS['rdf'])

    from rijks_ld_utils import build_row
    row_dict = build_row(cho_uri)
    row_dict["cho_uri"] = cho_uri
    fill_row_from_rdf(row_dict, rdf) 
    
    for k, v in row_dict.items():
        row_dict[k] = resolve_language(v)

    title_val = row_dict.get("title")
    if isinstance(title_val, list) and len(title_val) > 0:
        title_val = title_val[0]

    record = ArtworkRecord(
        object_ID=str(row_dict.get("object_ID") or ""),
        cho_uri=cho_uri,
        title=title_val,
        object_name=list_to_str(row_dict.get("object_name")),
        department=list_to_str(row_dict.get("department")),
        
        date=list_to_str(row_dict.get("date")),
        
        medium=list_to_str(row_dict.get("medium")),
        material=list_to_str(row_dict.get("material")),
        technique=list_to_str(row_dict.get("technique")),
        dimensions=list_to_str(row_dict.get("dimensions")),
        
        country=list_to_str(row_dict.get("country")),
        
        artist_name=list_to_str(row_dict.get("artist_name")),
        artist_role=list_to_str(row_dict.get("artist_role")),
        artist_date_begin=list_to_str(row_dict.get("artist_date_begin")),
        artist_date_end=list_to_str(row_dict.get("artist_date_end")),
        
        subjects=list_to_str(row_dict.get("subjects")),
        inscriptions=list_to_str(row_dict.get("inscriptions")),
        description=list_to_str(row_dict.get("description")),
        
        image_url=row_dict.get("image")
    )
        
    return record, cho_uri

def harvest_set(set_id: str, limit: int, only_descriptions: bool = False) -> pd.DataFrame:
    stats = ExtractionStats(f"Rijks Set {set_id}")
    token = None
    
    pbar = tqdm(total=limit, desc=f"Harvesting Set {set_id}")

    while stats.collected < limit:
        args = {"verb": "ListRecords", "metadataPrefix": "edm"} if not token else {"verb": "ListRecords", "resumptionToken": token}
        args["set"] = set_id if not token else args.get("set", None)
        
        try:
            root = fetch_oai(args)
        except Exception:
            stats.record_skipped("fetch_oai_error")
            stats.record_failure()
            break

        records = root.findall(".//oai:record", NS)
        if not records:
            break
            
        for rec in records:
            if stats.collected >= limit:
                break
                
            stats.record_seen()
            md = rec.find(".//oai:metadata", NS)
            if md is None:
                stats.record_skipped("no_metadata")
                continue
                
            rdf = md.find(".//rdf:RDF", NS)
            if rdf is None:
                stats.record_skipped("no_rdf")
                continue
                
            try:
                record, _ = extract_record(rdf)
                if not record.image_url:
                    stats.record_skipped("no_image")
                    continue
                    
                if only_descriptions and not record.description:
                    stats.record_skipped("missing_description")
                    continue
                    
                stats.record_collected(record.to_dict())
                pbar.update(1)
                
            except Exception as e:
                stats.record_skipped("extract_record_error")
                stats.record_failure()

        rt = root.find(".//oai:resumptionToken", NS)
        token = (rt.text.strip() if rt is not None and rt.text else None)
        if not token:
            break
            
    pbar.close()
    stats.print_summary()
    return stats.generate_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Rijksmuseum OAI/JSON-LD harvester.")
    
    parser.add_argument("--out", default="csv_pipeline", help="Output directory folder (default: csv_pipeline)")
    parser.add_argument("--descriptions", action="store_true", help="Only extract records that have a description.")
    
    sub = parser.add_subparsers(dest="cmd", help="Command to run")

    p_harvest = sub.add_parser("harvest", help="Harvest specific set")
    p_harvest.add_argument("--set", default="260239", help="OAI set identifier (default: 260239)")
    p_harvest.add_argument("--limit", type=int, default=1000000, help="Max records to collect")

    p_multi = sub.add_parser("harvest-many", help="Harvest multiple sets")
    p_multi.add_argument("--sets", nargs="*", default=DEFAULT_SET_IDS, help="OAI set ids")
    p_multi.add_argument("-n", "--limit", type=int, default=1000000, help="Max per set")

    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    if args.cmd is None:
        args.cmd = "harvest"
        args.set = "260239"
        args.limit = 1000000

    suffix = "_desc" if args.descriptions else ""

    if args.cmd == "harvest":
        out_name = f"rijks_{args.set}{suffix}.csv"
        out_csv = os.path.join(args.out, out_name)
        df = harvest_set(args.set, args.limit, only_descriptions=args.descriptions)
        
        if not df.empty:
            df.to_csv(out_csv, index=False)
            print(f"Wrote {len(df)} rows to {out_csv}")
        else:
            print("No rows collected.")

    elif args.cmd == "harvest-many":
        for sid in args.sets:
            out_name = f"rijks_{sid}_{args.limit}{suffix}.csv"
            out_csv = os.path.join(args.out, out_name)
            df = harvest_set(sid, args.limit, only_descriptions=args.descriptions)
            
            if not df.empty:
                df.to_csv(out_csv, index=False)
                print(f"[{sid}] {len(df)} rows -> {out_csv}")
            else:
                print(f"[{sid}] No rows collected.")

if __name__ == "__main__":
    main()
