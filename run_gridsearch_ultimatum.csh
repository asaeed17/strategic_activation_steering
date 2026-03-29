#!/bin/csh
# run_gridsearch_ultimatum.csh <model> <vectors_dir> [output_suffix]
# Grid search over negotiation dimensions using the Ultimatum Game.
# No API calls — rule-based opponent, fully reproducible.
#
# Usage:
#   ./run_gridsearch_ultimatum.csh qwen2.5-7b vectors/neg8dim_12pairs_matched/negotiation _7b_ult

# ── Set layers here ─────────────────────────────────────────────────────────
# qwen2.5-7b  (28 layers):  12 16 19  (~43%, 57%, 68%)
# qwen2.5-32b (64 layers):  28 36 44  (~44%, 56%, 69%)
set FIXED_LAYERS = ( 28 32 36 )
# set FIXED_POOL  = 100
set FIXED_POOL  = ""   # leave empty to use variable pool sizes
# ────────────────────────────────────────────────────────────────────────────
setenv HF_HOME hf_cache/
if ( $?HF_TOKEN ) then
    setenv HF_TOKEN "$HF_TOKEN"
endif
# ── Rule-based opponent mode ─────────────────────────────────────────────────
# Set RULEBASED = (--rulebased) to use deterministic opponents instead of LLM baseline.
#   Proposer role: rule-based responder accepts iff responder_share/pool >= 0.35
#   Responder role: sweeps offers at 10,20,...,90% of pool; LLM decides each
# Set RULEBASED = () for the original LLM-vs-LLM baseline.
set RULEBASED   = (--rulebased)
# set RULEBASED = ()
# ────────────────────────────────────────────────────────────────────────────

# ── Args (with fallback defaults) ───────────────────────────────────────────
set MODEL       = "qwen2.5-32b-gptq"
set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
set SUFFIX      = "_damon_variable_newprompt_l28_l32_l36"

if ( $#argv >= 1 ) then
    set MODEL = "$argv[1]"
endif
if ( $#argv >= 2 ) then
    set VECTORS_DIR = "$argv[2]"
endif
if ( $#argv >= 3 ) then
    set SUFFIX = "$argv[3]"
endif
if ( $#argv >= 4 ) then
    set RULEBASED = ($argv[4])
endif
# ────────────────────────────────────────────────────────────────────────────

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
                # echo "==> Waiting for GPU to be free..."
                # while ( 1 )
                #     set GPU_USED = `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`
                #     if ( $GPU_USED == 0 ) break
                #     echo "    GPU busy ($GPU_USED process). Waiting 10s..."
                #     sleep 10
                # end

                echo "==> Running ${role}/${dim} on Layer ${layer}"

                set LAYERS_FLAG = ( --fixed_layers $layer )

                if ( ${%FIXED_POOL} > 0 ) then
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

echo "==> All dimensions and layers complete."
