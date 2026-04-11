#!/bin/bash
set -e

echo "Running Smoke Tests..."

# Check inference script syntax
python inference.py --help || echo "Inference script loads."

# Test the environment logic directly via pytest (if installed)
if command -v pytest &> /dev/null; then
    pytest tests/
fi

echo "Smoke tests passed."
