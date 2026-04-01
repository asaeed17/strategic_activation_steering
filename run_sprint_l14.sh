#!/usr/bin/env bash
# Sprint: L14 adjudication — test teammate's d=-3.05 peak
# 2 configs: firmness L14 a=7, empathy L14 a=7, UG mode
# ~40 min on RTX 3090 Ti
set -euo pipefail

VECTORS=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTDIR=results/ultimatum/llm_vs_llm/l14
COMMON="--model qwen2.5-7b --vectors_dir $VECTORS --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 --game ultimatum"

echo "=== L14 adjudication (2 configs) ==="
echo "Start: $(date)"

echo "--- firmness UG L14 alpha=7 ---"
python ultimatum_game.py $COMMON \
  --dimension firmness --layers 14 --alpha 7 \
  --output_dir $OUTDIR

echo ""
echo "--- empathy UG L14 alpha=7 ---"
python ultimatum_game.py $COMMON \
  --dimension empathy --layers 14 --alpha 7 \
  --output_dir $OUTDIR

echo ""
echo "=== L14 adjudication complete ==="
echo "End: $(date)"
