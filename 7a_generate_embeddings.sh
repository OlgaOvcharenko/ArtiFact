#!/bin/bash
# Stage 7a: Generate Fast CLIP Embeddings
# Generates visual embeddings for the benchmark dataset using a lightweight CLIP model
# This has to be run before 7_inject_errors.sh if one wants new embeddings for the image swap

set -e
source venv/bin/activate

LIMIT=$1

if [ -z "$LIMIT" ]; then
    python3 error_injection/generate_embeddings.py
else
    python3 error_injection/generate_embeddings.py --limit "$LIMIT"
fi
echo ""
echo "CLIP embeddings generated"
