#!/bin/csh
# run_gridsearch.csh
# Grid search over all 15 negotiation dimensions for neg15dim_12pairs_matched.
# Saves results per dimension — safe to relaunch if the server dies.

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
    set OUT = "results/gridsearch_neg15dim_12pairs_matched/${dim}"

    if ( -f "${OUT}/final_best.json" ) then
        echo "==> Skipping ${dim} (already complete)"
    else
        echo "==> Running dimension: ${dim}"
        python lightweight_gridsearch.py \
            --model qwen2.5-3b \
            --dimension "${dim}" \
            --vectors_dir vectors/neg15dim_12pairs_matched/negotiation \
            --presets late middle_late \
            --alphas -30 -20 -15 -10 -5 5 10 15 20 25 30 \
            --use_craigslist \
            --output_dir "${OUT}"
    endif
end

echo "==> All dimensions complete."
