#!/bin/bash
# run_validation.sh — Validate steering vectors for all variants
# Usage: bash run_validation.sh

set -e

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env/bin/activate"

# Redirect HF cache to project dir (home quota is too small for model weights)
export HF_HOME="${SCRIPT_DIR}/.hf_cache"

# Log to file and stdout
LOGFILE="${SCRIPT_DIR}/validation_log.txt"
exec > >(tee "$LOGFILE") 2>&1

MODEL="qwen2.5-3b"
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

echo "Starting validation at $(date)"
echo "Model: $MODEL"
echo "Variants: ${VARIANTS[*]}"
echo "=========================================="

for v in "${VARIANTS[@]}"; do
    echo ""
    echo "=== [$v] Validating ($(date)) ==="
    python validate_vectors.py \
        --model $MODEL \
        --full \
        --method mean_diff \
        --negotiation_pairs steering_pairs/${v}/negotiation_steering_pairs.json \
        --control_pairs steering_pairs/${v}/control_steering_pairs.json \
        --vectors_dir vectors/${v}/negotiation \
        --output_dir results/validation/${v}
done

echo ""
echo "=========================================="
echo "Validation complete at $(date)"
echo "=========================================="
