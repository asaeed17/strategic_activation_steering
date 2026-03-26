#!/bin/csh
# run_gridsearch_ultimatum.csh <model> <vectors_dir> [output_suffix]
# Grid search over negotiation dimensions using the Ultimatum Game.

# ── Set layers here ─────────────────────────────────────────────────────────
set FIXED_LAYERS = ( 10 14 )
set FIXED_POOL  = 100
# set FIXED_POOL  = ""   # leave empty to use variable pool sizes
# ────────────────────────────────────────────────────────────────────────────

# ── Args (with fallback defaults) ───────────────────────────────────────────
set MODEL       = "qwen2.5-7b"
set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_matched"
set SUFFIX      = ""

if ( $#argv >= 1 ) then
    set MODEL = "$argv[1]"
endif
if ( $#argv >= 2 ) then
    set VECTORS_DIR = "$argv[2]"
endif
if ( $#argv >= 3 ) then
    set SUFFIX = "$argv[3]"
endif
# ────────────────────────────────────────────────────────────────────────────

set OUT_DIR = "results/ultimatum/temp03_mindims_v4"

set DIMS = ( \
    firmness \
    empathy \
    composure \
    anchoring \
    batna_awareness \
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
                echo "==> Waiting for GPU to be free..."
                while ( 1 )
                    set GPU_USED = `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`
                    if ( $GPU_USED == 0 ) break
                    echo "    GPU busy ($GPU_USED process). Waiting 10s..."
                    sleep 10
                end

                echo "==> Running ${role}/${dim} on Layer ${layer}"

                set LAYERS_FLAG = ( --fixed_layers $layer )

                if ( "$FIXED_POOL" != "" ) then
                    set POOL_FLAG = ( --fixed_pool $FIXED_POOL )
                else
                    set POOL_FLAG = ()
                endif

                # Note: 0 is explicitly passed first to generate the baseline
                python lightweight_gridsearch_ultimatum.py \
                    --model         "${MODEL}" \
                    --dimension     "${dim}" \
                    --role          "${role}" \
                    --vectors_dir   "${VECTORS_DIR}" \
                    --coarse_alphas 0 -5 5 15 \
                    --n_games       50 \
                    --output_suffix "${SUFFIX}" \
                    --output_dir    "${CURRENT_OUT_DIR}" \
                    $LAYERS_FLAG \
                    $POOL_FLAG
            endif
        end
    end
end

echo "==> All dimensions and layers complete."