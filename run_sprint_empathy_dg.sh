#!/usr/bin/env bash
# Sprint: Empathy DG thin cells — shores up "nullification" claim
# 4 configs: empathy DG at L{10,12} x alpha={3,10}
# Existing: L10 a=7, L12 a=7 (in llm_vs_llm/dg_empathy/)
# ~2 hrs on RTX 3090 Ti (fp16, no quantize needed)
set -euo pipefail

VECTORS=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTDIR=results/ultimatum/llm_vs_llm/dg_empathy
COMMON="--model qwen2.5-7b --vectors_dir $VECTORS --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 --game dictator"

echo "=== Empathy DG thin cells (4 configs) ==="
echo "Start: $(date)"

for LAYER in 10 12; do
  for ALPHA in 3 10; do
    echo ""
    echo "--- empathy DG L${LAYER} alpha=${ALPHA} ---"
    python ultimatum_game.py $COMMON \
      --dimension empathy --layers $LAYER --alpha $ALPHA \
      --output_dir $OUTDIR
  done
done

echo ""
echo "=== All empathy DG configs complete ==="
echo "End: $(date)"
