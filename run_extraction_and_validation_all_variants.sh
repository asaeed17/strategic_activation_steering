#!/bin/bash
# run_extraction_and_validation.sh — Extract steering vectors + run validation (stages 1-3)
# Usage: bash run_extraction_and_validation.sh <model> [variant]

set -e

# More robust way to get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Activate venv (fixing path from env to venv)
if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/env/bin/activate" ]; then
    source "${SCRIPT_DIR}/env/bin/activate"
fi

# Redirect HF cache to project dir (home quota is too small for model weights)
export HF_HOME="${SCRIPT_DIR}/.hf_cache"
mkdir -p "$HF_HOME"

# Log to file and stdout
LOGFILE="${SCRIPT_DIR}/extraction_and_validation_log.txt"
exec > >(tee "$LOGFILE") 2>&1

if [ -z "$1" ]; then
    echo "Usage: bash run_extraction_and_validation.sh <model> [variant]"
    echo "Example: bash run_extraction_and_validation.sh llama-3b neg8dim_12pairs_matched"
    exit 1
fi

MODEL="$1"

if [ -n "$2" ]; then
    VARIANTS=("$2")
    echo "Using specified variant: $2"
else
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
    echo "Using all default variants."
fi

echo "Starting extraction + validation at $(date)"
echo "Model: $MODEL"
echo "Variants: ${VARIANTS[*]}"
echo "=========================================="

# ── Phase 1: Extraction ──────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "PHASE 1: EXTRACTION"
echo "=========================================="

for v in "${VARIANTS[@]}"; do
    echo ""
    echo "=== [$v] Extracting negotiation vectors ($(date)) ==="
    python extract_vectors.py \
        --models $MODEL \
        --methods mean_diff \
        --pairs_file steering_pairs/${v}/negotiation_steering_pairs.json \
        --output_dir vectors/${v}/negotiation

    echo "=== [$v] Extracting control vectors ($(date)) ==="
    python extract_vectors.py \
        --models $MODEL \
        --methods mean_diff \
        --pairs_file steering_pairs/${v}/control_steering_pairs.json \
        --output_dir vectors/${v}/control
done

echo ""
echo "=========================================="
echo "Extraction complete at $(date)"
echo "=========================================="

# ── Phase 2: Validation (stages 1-3, skipping 4 & 5) ─────────────────────────

echo ""
echo "=========================================="
echo "PHASE 2: VALIDATION (stages 1-3)"
echo "=========================================="

python validation/run_full_validation.py \
    --models $MODEL \
    --variants "${VARIANTS[@]}" \
    --skip-stage 4 \
    --skip-stage 5

echo ""
echo "=========================================="
echo "Extraction + validation complete at $(date)"
echo "=========================================="
