#!/bin/bash
# Stage 6: Apply Vocabulary Unification

set -e
source venv/bin/activate

echo "6) semantic unification"
python3 semantic_unification/apply_unification.py
echo ""
echo "unification complete in 6_unified/"
