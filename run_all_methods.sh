#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env/bin/activate"
export HF_HOME="${SCRIPT_DIR}/.hf_cache"

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

# 1. Validate PCA vectors
echo "=== STEP 1: PCA Validation ($(date)) ==="
for v in "${VARIANTS[@]}"; do
    echo "--- Validating $v (pca) ---"
    python validate_vectors.py --model $MODEL --full --method pca \
        --negotiation_pairs steering_pairs/${v}/negotiation_steering_pairs.json \
        --control_pairs steering_pairs/${v}/control_steering_pairs.json \
        --vectors_dir vectors/${v}/negotiation \
        --output_dir results/validation/${v}
done

# 2. Orthogonal projection on PCA
echo "=== STEP 2: PCA Projection ($(date)) ==="
python orthogonal_projection.py --all-variants --method pca --probe

# 3. Extract logreg vectors
echo "=== STEP 3: Logreg Extraction ($(date)) ==="
for v in "${VARIANTS[@]}"; do
    echo "--- Extracting $v (negotiation) ---"
    python extract_vectors.py --models $MODEL \
        --pairs_file steering_pairs/${v}/negotiation_steering_pairs.json \
        --output_dir vectors/${v}/negotiation
    echo "--- Extracting $v (control) ---"
    python extract_vectors.py --models $MODEL \
        --pairs_file steering_pairs/${v}/control_steering_pairs.json \
        --output_dir vectors/${v}/control
done

# 4. Validate logreg vectors
echo "=== STEP 4: Logreg Validation ($(date)) ==="
for v in "${VARIANTS[@]}"; do
    echo "--- Validating $v (logreg) ---"
    python validate_vectors.py --model $MODEL --full --method logreg \
        --negotiation_pairs steering_pairs/${v}/negotiation_steering_pairs.json \
        --control_pairs steering_pairs/${v}/control_steering_pairs.json \
        --vectors_dir vectors/${v}/negotiation \
        --output_dir results/validation/${v}
done

# 5. Orthogonal projection on logreg
echo "=== STEP 5: Logreg Projection ($(date)) ==="
python orthogonal_projection.py --all-variants --method logreg --probe

echo "=== ALL DONE ($(date)) ==="
