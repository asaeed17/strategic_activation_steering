#!/bin/bash
# Run 3B empathy: 5 layers x 4 alphas = 20 configs

LAYERS=(8 12 16 20 24)
ALPHAS=(-7 5 7 15)
VECTORS_DIR=vectors/ultimatum_10dim_20pairs_general_matched/negotiation
OUTPUT_DIR=results/ultimatum/multimodel_3b

TOTAL=$(( ${#LAYERS[@]} * ${#ALPHAS[@]} ))
COUNT=0

for L in "${LAYERS[@]}"; do
  for A in "${ALPHAS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=========================================="
    echo "[$COUNT/$TOTAL] 3B empathy L${L} a=${A}"
    echo "=========================================="
    python ultimatum_game.py --model qwen2.5-3b --dimension empathy \
        --vectors_dir "$VECTORS_DIR" \
        --layers "$L" --alpha "$A" --steered_role proposer --game ultimatum \
        --n_games 50 --variable_pools --paired --temperature 0.0 \
        --output_dir "$OUTPUT_DIR"
  done
done

echo ""
echo "Done. $COUNT configs completed."
