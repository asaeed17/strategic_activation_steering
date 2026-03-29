#!/usr/bin/env bash
# Sprint: Text-visibility control — test framing effect when responder sees full text
# 3 configs: baseline, firmness L12 a=7, empathy L10 a=7 — all with --text_visible
# ~1 hr on RTX 3090 Ti
set -euo pipefail

VECTORS=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTDIR=results/ultimatum/confirmatory_v2/text_visible
COMMON="--model qwen2.5-7b --vectors_dir $VECTORS \
  --n_games 100 --variable_pools --temperature 0.0 --game ultimatum --text_visible"

echo "=== Text-visibility control (3 configs) ==="
echo "Start: $(date)"

echo "--- baseline (text visible) ---"
python ultimatum_game.py $COMMON \
  --output_dir $OUTDIR

echo ""
echo "--- firmness L12 alpha=7 (text visible) ---"
python ultimatum_game.py $COMMON \
  --dimension firmness --layers 12 --alpha 7 \
  --steered_role proposer --paired \
  --output_dir $OUTDIR

echo ""
echo "--- empathy L10 alpha=7 (text visible) ---"
python ultimatum_game.py $COMMON \
  --dimension empathy --layers 10 --alpha 7 \
  --steered_role proposer --paired \
  --output_dir $OUTDIR

echo ""
echo "=== Text-visibility control complete ==="
echo "End: $(date)"
