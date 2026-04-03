#!/bin/bash
# Empathy DG confirmatory_v3 — 6 configs, unquantized (bfloat16)
# Run on mallard-l, 2026-03-31
# Purpose: resolve the precision confound (quantized vs unquantized baseline)

set -e

export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
source /cs/student/projects1/2022/moiimran/venv/bin/activate
cd /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

OUTDIR=results/ultimatum/confirmatory_v3/dg_empathy
mkdir -p "$OUTDIR"

echo "=== Starting empathy DG v3 runs at $(date) ==="
echo "Machine: $(hostname)"
echo "No --quantize flag (bfloat16 mode)"
echo ""

# Config 1: L10 alpha=3
echo "[1/6] L10 alpha=3 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 10 --alpha 3 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[1/6] L10 alpha=3 done at $(date)"
echo ""

# Config 2: L10 alpha=7
echo "[2/6] L10 alpha=7 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 10 --alpha 7 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[2/6] L10 alpha=7 done at $(date)"
echo ""

# Config 3: L10 alpha=10
echo "[3/6] L10 alpha=10 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 10 --alpha 10 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[3/6] L10 alpha=10 done at $(date)"
echo ""

# Config 4: L12 alpha=3
echo "[4/6] L12 alpha=3 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 12 --alpha 3 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[4/6] L12 alpha=3 done at $(date)"
echo ""

# Config 5: L12 alpha=7
echo "[5/6] L12 alpha=7 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 12 --alpha 7 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[5/6] L12 alpha=7 done at $(date)"
echo ""

# Config 6: L12 alpha=10
echo "[6/6] L12 alpha=10 starting at $(date)"
python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
  --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
  --layers 12 --alpha 10 --steered_role proposer \
  --n_games 100 --variable_pools --paired --temperature 0.0 \
  --game dictator --output_dir "$OUTDIR"
echo "[6/6] L12 alpha=10 done at $(date)"
echo ""

echo "=== All 6 configs complete at $(date) ==="
