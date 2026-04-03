#!/bin/bash
# Layer gradient sweep: firmness + empathy, layers 4-20 (step 2), UG proposer, alpha=7, n=15
# Machine: pochard-l, Qwen 2.5-7B bfloat16 (unquantized)

set -e

export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
source /cs/student/projects1/2022/moiimran/venv/bin/activate

PROJ=/cs/student/projects1/2022/moiimran/comp0087_snlp_cwk
cd "$PROJ"

VECTORS_DIR=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTPUT_DIR=results/ultimatum/layer_gradient

LAYERS="4 6 8 10 12 14 16 18 20"
DIMENSIONS="firmness empathy"

TOTAL=18
COUNT=0

echo "=========================================="
echo "Layer gradient sweep starting at $(date)"
echo "Machine: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null)"
echo "Configs: $TOTAL (2 dims x 9 layers)"
echo "=========================================="

for DIM in $DIMENSIONS; do
    for LAYER in $LAYERS; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] $DIM L${LAYER} -- started at $(date)"

        python ultimatum_game.py \
            --model qwen2.5-7b \
            --dimension "$DIM" \
            --vectors_dir "$VECTORS_DIR" \
            --layers "$LAYER" \
            --alpha 7 \
            --steered_role proposer \
            --n_games 15 \
            --variable_pools \
            --paired \
            --temperature 0.0 \
            --game ultimatum \
            --output_dir "$OUTPUT_DIR"

        echo "[$COUNT/$TOTAL] $DIM L${LAYER} -- finished at $(date)"
    done
done

echo ""
echo "=========================================="
echo "All $TOTAL configs completed at $(date)"
echo "=========================================="
