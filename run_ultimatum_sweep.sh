#!/bin/bash
# run_ultimatum_sweep.sh — Run Ultimatum Game steering experiments on AWS GPU.
#
# Usage:
#   bash run_ultimatum_sweep.sh --stage 0   # Baseline only
#   bash run_ultimatum_sweep.sh --stage 1   # Quick scan (4 dims × 2 layers × 2 alphas)
#   bash run_ultimatum_sweep.sh --stage 2   # Deep dive (full sweep on specified dims)
#   bash run_ultimatum_sweep.sh --stage 3   # Cross-method (PCA comparison)
#   bash run_ultimatum_sweep.sh --stage all # Run 0 → 1 → pause for review
#
# Designed for tmux on AWS g5.xlarge. Each config saves incrementally.

set -euo pipefail

export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"

MODEL="qwen2.5-7b"
VECTORS_DIR="vectors/neg8dim_12pairs_matched/negotiation"
METHOD="mean_diff"
OUT_DIR="results/ultimatum"
SEED=42

STAGE="${1:---stage}"
if [[ "$STAGE" == "--stage" ]]; then
    STAGE="${2:-all}"
fi

echo "======================================================================"
echo "ULTIMATUM GAME STEERING SWEEP — Stage: $STAGE"
echo "  Model: $MODEL   Method: $METHOD"
echo "  Output: $OUT_DIR"
echo "  Start: $(date)"
echo "======================================================================"

run_config() {
    local dim="$1"
    local role="$2"
    local layer="$3"
    local alpha="$4"
    local n_games="$5"

    echo ""
    echo "--- Config: dim=$dim role=$role layer=$layer alpha=$alpha n=$n_games ---"
    python ultimatum_game.py \
        --model "$MODEL" \
        --vectors_dir "$VECTORS_DIR" \
        --dimension "$dim" \
        --method "$METHOD" \
        --layers "$layer" \
        --alpha "$alpha" \
        --steered_role "$role" \
        --n_games "$n_games" \
        --variable_pools \
        --paired \
        --seed "$SEED" \
        --output_dir "$OUT_DIR"
}

# ===== STAGE 0: Baseline =====
if [[ "$STAGE" == "0" || "$STAGE" == "all" ]]; then
    echo ""
    echo "========== STAGE 0: BASELINE (no steering) =========="
    python ultimatum_game.py \
        --model "$MODEL" \
        --n_games 100 \
        --variable_pools \
        --seed "$SEED" \
        --output_dir "$OUT_DIR"
    echo "Stage 0 complete: $(date)"
fi

# ===== STAGE 1: Quick scan =====
if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
    echo ""
    echo "========== STAGE 1: QUICK SCAN =========="
    echo "4 dims × 2 layers × 2 alphas = 16 proposer configs + 2 responder configs"

    N=50

    # Proposer steering
    for dim in firmness anchoring empathy batna_awareness; do
        for layer in 12 14; do
            for alpha in 6 10; do
                run_config "$dim" proposer "$layer" "$alpha" "$N"
            done
        done
    done

    # Responder steering (firmness only in quick scan)
    for layer in 12 14; do
        run_config firmness responder "$layer" 6 "$N"
    done

    echo ""
    echo "Stage 1 complete: $(date)"
    echo "Run: python analysis/analyse_ultimatum.py --results_dir $OUT_DIR --stage 1"

    if [[ "$STAGE" == "all" ]]; then
        echo ""
        echo "======================================================================"
        echo "STAGE 1 DONE. Review results before proceeding to Stage 2."
        echo "If signal found, run: bash run_ultimatum_sweep.sh --stage 2"
        echo "======================================================================"
    fi
fi

# ===== STAGE 2: Deep dive =====
# Edit DIMS below based on Stage 1 results (top 2-3 dimensions)
if [[ "$STAGE" == "2" ]]; then
    echo ""
    echo "========== STAGE 2: DEEP DIVE =========="

    # ---- EDIT THESE based on Stage 1 results ----
    DIMS="firmness anchoring"
    # ----------------------------------------------

    N=100

    for dim in $DIMS; do
        echo ""
        echo "=== Deep dive: $dim (proposer) ==="
        for layer in 10 12 14 16; do
            for alpha in 3 6 10 15; do
                run_config "$dim" proposer "$layer" "$alpha" "$N"
            done
        done

        echo ""
        echo "=== Deep dive: $dim (responder) ==="
        for layer in 12 14; do
            for alpha in 6 10; do
                run_config "$dim" responder "$layer" "$alpha" "$N"
            done
        done
    done

    echo ""
    echo "Stage 2 complete: $(date)"
fi

# ===== STAGE 3: Cross-method (PCA comparison) =====
if [[ "$STAGE" == "3" ]]; then
    echo ""
    echo "========== STAGE 3: CROSS-METHOD (PCA) =========="

    # Use best config from Stage 2 — edit these:
    DIM="firmness"
    LAYER="14"
    ALPHA="6"

    echo "--- PCA comparison: $DIM L$LAYER a$ALPHA ---"
    python ultimatum_game.py \
        --model "$MODEL" \
        --vectors_dir "$VECTORS_DIR" \
        --dimension "$DIM" \
        --method pca \
        --layers "$LAYER" \
        --alpha "$ALPHA" \
        --steered_role proposer \
        --n_games 100 \
        --variable_pools \
        --paired \
        --seed "$SEED" \
        --output_dir "$OUT_DIR"

    echo "Stage 3 complete: $(date)"
fi

echo ""
echo "======================================================================"
echo "SWEEP FINISHED: $(date)"
echo "Results in: $OUT_DIR"
echo "======================================================================"
