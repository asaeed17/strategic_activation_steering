#!/bin/bash
# run_extraction.sh — Extract steering vectors for all variants (negotiation + control)
# Usage: bash run_extraction.sh <model>

set -e

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env/bin/activate"

# Redirect HF cache to project dir (home quota is too small for model weights)
export HF_HOME="${SCRIPT_DIR}/.hf_cache"

# Log to file and stdout
LOGFILE="${SCRIPT_DIR}/extraction_log.txt"
exec > >(tee "$LOGFILE") 2>&1

if [ -z "$1" ]; then
    echo "Usage: bash run_extraction_all_variants.sh <model>"
    echo "Example: bash run_extraction_all_variants.sh llama-3b"
    exit 1
fi

MODEL="$1"
VARIANTS=(
    neg15dim_12pairs_raw
    neg15dim_12pairs_matched
    neg15dim_20pairs_matched
    neg15dim_80pairs_matched
    neg8dim_12pairs_raw
    neg8dim_12pairs_matched
    neg8dim_20pairs_matched
    neg8dim_80pairs_matched
)

echo "Starting extraction at $(date)"
echo "Model: $MODEL"
echo "Variants: ${VARIANTS[*]}"
echo "=========================================="

for v in "${VARIANTS[@]}"; do
    echo ""
    echo "=== [$v] Extracting negotiation vectors ($(date)) ==="
    python extract_vectors.py \
        --models $MODEL \
        --pairs_file steering_pairs/${v}/negotiation_steering_pairs.json \
        --output_dir vectors/${v}/negotiation

    echo "=== [$v] Extracting control vectors ($(date)) ==="
    python extract_vectors.py \
        --models $MODEL \
        --pairs_file steering_pairs/${v}/control_steering_pairs.json \
        --output_dir vectors/${v}/control
done

echo ""
echo "=========================================="
echo "Extraction complete at $(date)"
echo "=========================================="
