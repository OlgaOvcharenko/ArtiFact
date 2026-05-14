#!/bin/bash
# STAGE 3: BASE PARSING

set -e

INPUT_DIR="normalized_pipeline"
OUTPUT_DIR="parsed_pipeline"

if [ -d "venv" ]; then
    echo "venv"
    source venv/bin/activate
fi

echo -e "\n3) parsing pipeline"
python3 parsing/run_parsing.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

echo -e "\ncheck for date anomalies"
python3 parsing/date_anomaly_detector.py --input-dir "$OUTPUT_DIR"

echo "3) parsing complete."
