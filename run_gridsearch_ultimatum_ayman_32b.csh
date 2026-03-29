#!/bin/csh
# run_gridsearch_ultimatum_ayman_32b.csh
# Grid search over negotiation dimensions using the Ultimatum Game.
# Qwen 2.5-32B GPTQ-Int4, rule-based opponent, fully reproducible.
#
# Runs ONE layer per invocation. Edit FIXED_LAYERS below to choose which.
# The script loops over all 10 dimensions x 2 roles for that layer.
# Skips any dimension/role combo that already has a final_best.json.
#
# Usage:
#   csh run_gridsearch_ultimatum_ayman_32b.csh

# ═══════════════════════════════════════════════════════════════════════
# EDIT THIS: Choose which layer to run on this machine
# ═══════════════════════════════════════════════════════════════════════
set FIXED_LAYERS = ( 20 )
# ═══════════════════════════════════════════════════════════════════════

# set FIXED_POOL  = 100
set FIXED_POOL  = ""   # leave empty to use variable pool sizes
# ────────────────────────────────────────────────────────────────────────
# This is only in Ayman's script — everyone else just activates the environment first
source /cs/student/projects1/2022/aymakhan/.venv/bin/activate.csh
setenv HF_HOME .hf_cache
if ( $?HF_TOKEN ) then
    setenv HF_TOKEN "$HF_TOKEN"
endif
# ── Rule-based opponent mode ─────────────────────────────────────────────
set RULEBASED   = (--rulebased)
# set RULEBASED = ()
# ────────────────────────────────────────────────────────────────────────

# ── Fixed settings ──────────────────────────────────────────────────────
set MODEL       = "qwen2.5-32b-gptq"
set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
set SUFFIX      = "_ayman_32b_L${FIXED_LAYERS}"
# ────────────────────────────────────────────────────────────────────────

set OUT_DIR = "results/ultimatum/llm_vs_rulebased/${MODEL}${SUFFIX}"

set DIMS = ( \
    firmness \
    empathy \
    composure \
    anchoring \
    greed \
    fairness_norm \
    flattery \
    narcissism \
    spite \
    undecidedness \
)

foreach layer ( $FIXED_LAYERS )
    foreach role ( proposer responder )
        foreach dim ( $DIMS )

            set CURRENT_OUT_DIR = "${OUT_DIR}/L${layer}"

            if ( -f "${CURRENT_OUT_DIR}/${role}/${dim}/final_best.json" ) then
                echo "==> Skipping ${role}/${dim} on Layer ${layer} (already complete)"
            else
                echo "==> Running ${role}/${dim} on Layer ${layer}"

                set LAYERS_FLAG = ( --fixed_layers $layer )

                if ( "$FIXED_POOL" != "" ) then
                    set POOL_FLAG = ( --fixed_pool $FIXED_POOL )
                else
                    set POOL_FLAG = ()
                endif

                python lightweight_gridsearch_ultimatum.py \
                    --model         "${MODEL}" \
                    --dimension     "${dim}" \
                    --role          "${role}" \
                    --vectors_dir   "${VECTORS_DIR}" \
                    --coarse_alphas -5 5 15 \
                    --n_games       50 \
                    --output_suffix "${SUFFIX}" \
                    --output_dir    "${CURRENT_OUT_DIR}" \
                    $LAYERS_FLAG \
                    $POOL_FLAG \
                    $RULEBASED
            endif
        end
    end
end

echo "==> All dimensions complete for Layer ${FIXED_LAYERS}."
