#!/usr/bin/env bash
# Run top 5 UG configs (empathy proposer) for 200 games each.
# Requires extended POOL_SIZES (200 values) in ultimatum_game.py.
set -euo pipefail

VECTORS=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTDIR=results/ultimatum/top5_200games
COMMON="--model qwen2.5-7b --dimension empathy --steered_role proposer \
  --game ultimatum --n_games 200 --variable_pools --paired \
  --temperature 0.0 --dtype bfloat16 \
  --vectors_dir $VECTORS --output_dir $OUTDIR"

echo "=== Top 5 UG configs, n=200 ==="
echo "Start: $(date)"

echo "--- empathy L10 alpha=7 ---"
python ultimatum_game.py $COMMON --layers 10 --alpha 7

echo "--- empathy L10 alpha=10 ---"
python ultimatum_game.py $COMMON --layers 10 --alpha 10

echo "--- empathy L10 alpha=3 ---"
python ultimatum_game.py $COMMON --layers 10 --alpha 3

echo "--- empathy L12 alpha=10 ---"
python ultimatum_game.py $COMMON --layers 12 --alpha 10

echo "--- empathy L12 alpha=7 ---"
python ultimatum_game.py $COMMON --layers 12 --alpha 7

echo ""
echo "=== Done ==="
echo "End: $(date)"
