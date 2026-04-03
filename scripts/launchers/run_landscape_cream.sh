#!/bin/bash
# Landscape screen - Machine: cream (GPU 0)
# Dimensions: firmness, empathy, fairness_norm, narcissism, spite
# 5 dims x 7 layers x 2 alphas = 70 configs, n=15 paired games each

set -e

source /cs/student/projects1/2022/moiimran/venv/bin/activate
export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
export CUDA_VISIBLE_DEVICES=0

cd /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

DIMENSIONS=(firmness empathy fairness_norm narcissism spite)
LAYERS=(6 8 10 12 14 18 20)
ALPHAS=(-7 7)

TOTAL=$((${#DIMENSIONS[@]} * ${#LAYERS[@]} * ${#ALPHAS[@]}))
COUNT=0

echo "=========================================="
echo "LANDSCAPE SCREEN - cream (GPU 0)"
echo "Start: $(date)"
echo "Total configs: $TOTAL"
echo "=========================================="

for dim in "${DIMENSIONS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      COUNT=$((COUNT + 1))
      echo ""
      echo "[${COUNT}/${TOTAL}] dim=${dim} layer=${layer} alpha=${alpha} -- $(date)"

      python ultimatum_game.py --model qwen2.5-7b --dimension "$dim" \
        --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
        --layers "$layer" --alpha "$alpha" --steered_role proposer \
        --n_games 15 --variable_pools --paired --temperature 0.0 \
        --game ultimatum \
        --output_dir results/ultimatum/landscape_screen

      echo "[${COUNT}/${TOTAL}] DONE dim=${dim} layer=${layer} alpha=${alpha} -- $(date)"
    done
  done
done

echo ""
echo "=========================================="
echo "ALL DONE - cream"
echo "End: $(date)"
echo "=========================================="
