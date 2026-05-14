#!/bin/bash
# Stage 5: Schema Consolidation

set -e
source venv/bin/activate

echo "\n5) mapping"
python3 semantic_unification/schema_consolidate.py
echo "mapping complete"
