#!/bin/bash
# Landscape screen batch A: firmness, empathy, fairness_norm, narcissism, spite
# Loads model once, runs all 70 configs sequentially

source /cs/student/projects1/2022/moiimran/venv/bin/activate
export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
cd /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

echo "=========================================="
echo "LANDSCAPE BATCH A — Start: $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

python run_landscape_batch.py \
  --model qwen2.5-7b \
  --dimensions firmness empathy fairness_norm narcissism spite \
  --layers 6 8 10 12 14 18 20 \
  --alphas -7 7 \
  --n_games 15 \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --output_dir results/ultimatum/landscape_screen

echo "=========================================="
echo "LANDSCAPE BATCH A — End: $(date)"
echo "=========================================="
