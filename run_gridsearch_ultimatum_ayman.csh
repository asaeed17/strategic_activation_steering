#!/bin/csh
# run_gridsearch_ultimatum_ayman.csh <layer>
# Grid search over negotiation dimensions using the Ultimatum Game.
# No API calls — rule-based opponent, fully reproducible.
#
# Usage:
#   csh run_gridsearch_ultimatum_ayman.csh 18
#   csh run_gridsearch_ultimatum_ayman.csh 20

# ── Layer argument (required) ───────────────────────────────────────────────
if ( $#argv < 1 ) then
    echo "Usage: csh run_gridsearch_ultimatum_ayman.csh <layer>"
    exit 1
endif
set FIXED_LAYERS = ( $argv[1] )
# ────────────────────────────────────────────────────────────────────────────

# set FIXED_POOL  = 100
set FIXED_POOL  = ""   # leave empty to use variable pool sizes
# ────────────────────────────────────────────────────────────────────────────
source /cs/student/projects1/2022/aymakhan/.venv/bin/activate.csh
setenv HF_HOME /cs/student/projects1/2022/aymakhan/comp0087_snlp_cwk/.hf_cache
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

# ── Fixed settings ──────────────────────────────────────────────────────────
set MODEL       = "qwen2.5-7b"
set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
set SUFFIX      = "_ayman_aim_prompt_L${FIXED_LAYERS}"
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
                #     echo "    GPU busy ($GPU_USED process). Waiting 30s..."
                #     sleep 30
                # end

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

echo "==> All dimensions and layers complete."
