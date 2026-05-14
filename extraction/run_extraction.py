import argparse
import sys
import os

def parse_args():
    p = argparse.ArgumentParser(description="Unified Extraction Runner for Museum Datasets")
    p.add_argument("--source", choices=["aic", "met", "rijks", "all"], default="all",
                   help="Which museum source to extract from.")
    p.add_argument("--out", default="csv_pipeline", help="Output directory where CSVs will be saved.")
    p.add_argument("--descriptions-only", action="store_true", help="Only extract records that have a description.")
    
    p.add_argument("--met-limit", type=int, default=100000, help="Target rows per MET department.")
    
    p.add_argument("--aic-limit", type=int, default=100000, help="Max records for AIC dump.")
    
    p.add_argument("--rijks-limit", type=int, default=1000000, help="Max records for Rijks Set.")
    p.add_argument("--rijks-set", default="260239", help="Specific Rijks OAI set to harvest. Defaults to all paintings.")

    return p.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    original_argv = sys.argv.copy()

    try:
        if args.source in ["aic", "all"]:
            print("\n" + "="*50)
            print("Starting AIC Extraction")
            import aic_dump
            sys.argv = ["aic_dump.py", "--out", args.out, "--limit", str(args.aic_limit)]
            if args.descriptions_only:
                sys.argv.append("--descriptions")
            aic_dump.main()

        if args.source in ["rijks", "all"]:
            print("\n" + "="*50)
            print("Starting Rijksmuseum Extraction")
            import rijks_api
            sys.argv = ["rijks_api.py", "--out", args.out, "harvest", "--limit", str(args.rijks_limit), "--set", args.rijks_set]
            if args.descriptions_only:
                sys.argv.insert(3, "--descriptions")
            rijks_api.main()

        if args.source in ["met", "all"]:
            print("\n" + "="*50)
            print("🚀 Starting MET Extraction")
            import met_api
            sys.argv = ["met_api.py", "--out", args.out, "--run-all-departments", "--all-depts-n", str(args.met_limit)]
            if args.descriptions_only:
                sys.argv.append("--descriptions")
            met_api.main()

    finally:
        sys.argv = original_argv
        print("\nExtraction suite finished.")

if __name__ == "__main__":
    main()
