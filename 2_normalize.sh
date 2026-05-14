#!/bin/bash
# STAGE 2: NORMALIZATION

set -e

INPUT_DIR="csv_pipeline"
OUTPUT_DIR="normalized_pipeline"

if [ -d "venv" ]; then
    echo "venv"
    source venv/bin/activate
fi

echo -e "\n2 normalization"
python3 normalization/run_normalization.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "2 normalization complete"
