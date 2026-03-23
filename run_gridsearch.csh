#!/bin/csh
# run_gridsearch.csh <model> <vectors_dir> [output_suffix]
# Grid search over all 15 negotiation dimensions.
# Saves results per dimension — safe to relaunch if the server dies.
#
# Usage:
#   ./run_gridsearch.csh qwen2.5-3b vectors/neg15dim_12pairs_matched/negotiation
#   ./run_gridsearch.csh qwen2.5-3b vectors/neg15dim_12pairs_matched/negotiation _v2
#   ./run_gridsearch.csh llama3-3b  vectors/neg15dim_20pairs_matched/negotiation

if ($#argv < 2) then
    echo "Usage: $0 <model> <vectors_dir> [output_suffix]"
    echo "  e.g. $0 qwen2.5-3b vectors/neg15dim_12pairs_matched/negotiation _v2"
    exit 1
endif

set MODEL       = $argv[1]
set VECTORS_DIR = $argv[2]
set SUFFIX      = ""
if ($#argv >= 3) set SUFFIX = $argv[3]
set VEC_SET    = `echo $VECTORS_DIR | awk -F'/' '{print $(NF-1)}'`

set DIMS = ( \
    firmness \
    empathy \
    active_listening \
    assertiveness \
    interest_based_reasoning \
    emotional_regulation \
    strategic_concession_making \
    anchoring \
    rapport_building \
    batna_awareness \
    reframing \
    patience \
    value_creation \
    information_gathering \
    clarity_and_directness \
)

foreach dim ($DIMS)
    set OUT = "hyperparameter_results/gridsearch_${VEC_SET}${SUFFIX}/${MODEL}/${dim}"

    if ( -f "${OUT}/final_best.json" ) then
        echo "==> Skipping ${dim} (already complete)"
    else
        echo "==> Running dimension: ${dim}"
        python lightweight_gridsearch.py \
            --model        "${MODEL}" \
            --dimension    "${dim}" \
            --vectors_dir  "${VECTORS_DIR}" \
            --fixed_layers 18 20 22 24 \
            --two_pass \
            --coarse_alphas 5 15 \
            --s2_games     20 \
            --min_span     200 \
            --dataset_split train \
            --output_suffix "${SUFFIX}" \
            --use_craigslist
    endif
end

echo "==> All dimensions complete."
