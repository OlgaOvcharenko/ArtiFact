#!/bin/bash
# Stage 7: World Knowledge

set -e
source venv/bin/activate

echo "7) world knowledge reconstruction"
python3 error_injection/build_knowledge.py
echo ""
echo "knowldge base generated"
