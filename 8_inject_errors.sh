#!/bin/bash
# Stage 8: Error Injection & Benchmark Generation
# Generates final train/test datasets with injected semantic errors.

set -e
source venv/bin/activate

LIMIT=$1

echo "\n8) error injection"
python3 error_injection/build_error_index.py

if [ -z "$LIMIT" ]; then
    python3 error_injection/generate_benchmark.py
else
    python3 error_injection/generate_benchmark.py --limit "$LIMIT"
fi
echo "\nBenchmark generates"
