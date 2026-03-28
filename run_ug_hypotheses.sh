#!/bin/bash
# run_confirmatory.sh — Main confirmatory experiment (Phase 3)
#
# Design: 100 diverse pools, temp=0, one game per pool per condition.
# Honest n=100 (no pseudoreplication at temp=0 with 100 pools).
#
# UG: 2 dims × 2 layers × 3 alphas = 12 configs × 100 = 1,200 paired games
#   Firmness: alpha = {3, 7, 10}  (positive — demand UP)
#   Empathy:  alpha = {-3, -7, -10} (negative — demand DOWN, fixes sign)
# DG: firmness × 2 layers × 3 alphas = 6 configs × 100 = 600 games
# Robustness: empathy L12 a=-7 temp=0.3 = 100 games
# Total: 19 configs, ~1,900 paired games, ~3 hrs on g4dn.xlarge (4-bit)
#
# Usage:
#   tmux new -s confirmatory
#   source /opt/pytorch/bin/activate
#   nohup bash run_confirmatory.sh > confirmatory_log.txt 2>&1 &

set -euo pipefail
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"

MODEL="qwen2.5-7b"
VECTORS_DIR="vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
METHOD="mean_diff"
BASE_OUT="results/ultimatum/confirmatory"
SEED=42
N=100

# Detect GPU VRAM and set quantization flag
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -lt 20000 ]; then
    echo "GPU VRAM=${VRAM_MB}MB (<20GB) — using 4-bit quantization"
    QUANT_FLAG="--quantize --dtype float16"
else
    echo "GPU VRAM=${VRAM_MB}MB — using bf16"
    QUANT_FLAG=""
fi

echo "======================================================================"
echo "CONFIRMATORY EXPERIMENT — Phase 3"
echo "  100 diverse pools, temp=0, paired, n=$N per config"
echo "  Model: $MODEL   Vectors: general pairs   Method: $METHOD"
echo "  Quantize: ${QUANT_FLAG:-none}"
echo "  Start: $(date)"
echo "======================================================================"

run_config() {
    local DIM=$1
    local LAYER=$2
    local ALPHA=$3
    local GAME=${4:-ultimatum}
    local TEMP=${5:-0.0}
    local OUTDIR=$6

    echo ""
    echo "===== ${GAME^^}: $DIM L${LAYER} a=${ALPHA} temp=${TEMP} ====="
    echo "  Start: $(date)"
    python ultimatum_game.py \
        --model "$MODEL" \
        --vectors_dir "$VECTORS_DIR" \
        --dimension "$DIM" \
        --method "$METHOD" \
        --layers "$LAYER" \
        --alpha "$ALPHA" \
        --steered_role proposer \
        --game "$GAME" \
        --n_games "$N" \
        --variable_pools \
        --paired \
        --temperature "$TEMP" \
        --seed "$SEED" \
        $QUANT_FLAG \
        --output_dir "$OUTDIR"
    echo "  Done: $(date)"
}

# ===== UG: FIRMNESS (positive alpha — H2: demand UP) =====
echo ""
echo "############## UG FIRMNESS ##############"
for LAYER in 10 12; do
    for ALPHA in 3 7 10; do
        run_config firmness $LAYER $ALPHA ultimatum 0.0 "$BASE_OUT/ug"
    done
done

# ===== UG: EMPATHY (NEGATIVE alpha — H1: demand DOWN) =====
echo ""
echo "############## UG EMPATHY ##############"
for LAYER in 10 12; do
    for ALPHA in -3 -7 -10; do
        run_config empathy $LAYER $ALPHA ultimatum 0.0 "$BASE_OUT/ug"
    done
done

# ===== DG: FIRMNESS (positive alpha — H5: UG≈DG) =====
echo ""
echo "############## DG FIRMNESS ##############"
for LAYER in 10 12; do
    for ALPHA in 3 7 10; do
        run_config firmness $LAYER $ALPHA dictator 0.0 "$BASE_OUT/dg"
    done
done

# ===== ROBUSTNESS: EMPATHY L12 a=-7 temp=0.3 =====
echo ""
echo "############## ROBUSTNESS ##############"
run_config empathy 12 -7 ultimatum 0.3 "$BASE_OUT/robustness"

echo ""
echo "======================================================================"
echo "CONFIRMATORY EXPERIMENT COMPLETE: $(date)"
echo ""
echo "Pull results:"
echo "  rsync -avz -e 'ssh -i ~/.ssh/snlp-gpu-key.pem' ubuntu@\$(hostname -I | awk '{print \$1}'):~/snlp/$BASE_OUT/ local_confirmatory/"
echo ""
echo "Analysis:"
echo "  python3 analysis/analyse_ug_hypotheses.py \\"
echo "    --results_dir $BASE_OUT/ug \\"
echo "    --dg_dir $BASE_OUT/dg \\"
echo "    --empathy_alpha_list -3 -7 -10"
echo "======================================================================"
