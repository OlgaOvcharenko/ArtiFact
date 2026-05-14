#!/bin/bash
# STGAE 4: LLM PARSING

set -e

PARSED_DIR="parsed_pipeline"

if [ -d "venv" ]; then
    echo "venv"
    source venv/bin/activate
fi

echo -e "\n4) templated projections"
python3 LLM_parsing/apply_llm_templates.py --task dimensions --input-dir "$PARSED_DIR"
# python3 LLM_parsing/apply_llm_templates.py --task artist --input-dir "$PARSED_DIR" artist template directly is too risky.
python3 LLM_parsing/apply_llm_templates.py --task date --input-dir "$PARSED_DIR"

echo -e "\nDirect llm parsing"
echo "  dates"
python3 LLM_parsing/run_llm_direct_dates.py --input-dir "$PARSED_DIR"
echo "  dimensions"
python3 LLM_parsing/run_llm_direct_dimensions.py --input-dir "$PARSED_DIR"
echo "  artists"
python3 LLM_parsing/run_llm_direct_artists.py --input-dir "$PARSED_DIR"

echo "4)LLM parsing complete"
