#!/bin/bash

set -e

MET_LIMIT=${1:-"1000000"}
AIC_LIMIT=${1:-"1000000"}
RIJKS_LIMIT=${1:-"1000000"}
OUTPUT_DIR="csv_pipeline"

# venv setup
if [ -d "venv" ]; then
    echo "Venv"
    source venv/bin/activate
fi

echo -e "\n1) extraction pipeline"
python3 extraction/run_extraction.py \
    --met-limit "$MET_LIMIT" \
    --aic-limit "$AIC_LIMIT" \
    --rijks-limit "$RIJKS_LIMIT" \
    --out "$OUTPUT_DIR"

echo "1) extraction complete"
