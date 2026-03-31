#!/bin/csh
# run_resource_exchange_32b_multiple_gpu.csh
# Resource Exchange Game with activation steering across multiple GPU machines.
# Qwen 2.5-32B GPTQ-Int4, LLM-vs-LLM.
#
# Two modes:
#   Dispatch:  csh run_resource_exchange_32b_multiple_gpu.csh <username>
#              SSHes into free GPU machines and launches one layer per machine.
#
#   Local:     csh run_resource_exchange_32b_multiple_gpu.csh <layer_number>
#              Runs all dims x alphas for that layer on the current machine.

# ═══════════════════════════════════════════════════════════════════════
# EDIT THESE
# ═══════════════════════════════════════════════════════════════════════
set DIMS = ( \
    firmness \
    empathy \
    spite \
    narcissism \
)

set MACHINES = ( \
    aylesbury-l \
    barnacle-l \
    brent-l \
    bufflehead-l \
    cackling-l \
    canada-l \
    crested-l \
    eider-l \
    gadwall-l \
    goosander-l \
    gressingham-l \
    harlequin-l \
    mallard-l \
    mandarin-l \
    pintail-l \
    ruddy-l \
    scaup-l \
    scoter-l \
    shelduck-l \
    shoveler-l \
    smew-l \
    wigeon-l \
)

set LAYERS       = ( 23 28 41 )
set ALPHAS       = ( 20 50 )
set N_GAMES      = 20
set MODEL        = "qwen2.5-32b-gptq"
set VECTORS_DIR  = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
# ═══════════════════════════════════════════════════════════════════════

if ( $#argv < 1 ) then
    echo "Usage:"
    echo "  Dispatch: csh $0 <username>"
    echo "  Local:    csh $0 <layer_number>"
    exit 1
endif

# Detect mode: if arg is a number, run locally. Otherwise, dispatch.
echo "$argv[1]" | grep -q '^[0-9][0-9]*$'
if ( $status == 0 ) then
    goto local_run
else
    goto dispatch
endif

# ═══════════════════════════════════════════════════════════════════════
# DISPATCH MODE — one layer per free machine
# ═══════════════════════════════════════════════════════════════════════
dispatch:
    set UCL_USER    = "$argv[1]"
    set JUMP_HOST   = "${UCL_USER}@knuckles.cs.ucl.ac.uk"
    set DOMAIN      = "cs.ucl.ac.uk"
    set PROJECT_DIR = "/cs/student/projects1/2022/${UCL_USER}/comp0087_snlp_cwk"
    set SSH_OPTS    = "-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

    echo "============================================================"
    echo "Resource Exchange Dispatch: ${#LAYERS} layers across ${#MACHINES} machines"
    echo "User: ${UCL_USER}"
    echo "Layers: ${LAYERS}"
    echo "Dims: ${DIMS}"
    echo "Alphas: ${ALPHAS}  N_games: ${N_GAMES}"
    echo "============================================================"
    echo ""

    set machine_idx = 1

    foreach layer ( $LAYERS )
        set found = 0

        while ( $machine_idx <= $#MACHINES )
            set machine = $MACHINES[$machine_idx]

            set raw = `ssh $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l" |& grep '^[0-9]' | tr -d ' '`

            if ( "$raw" == "0" ) then
                echo "==> Layer ${layer} -> ${machine} (GPU free)"

                set LOG = "${PROJECT_DIR}/logs/resource_exchange_32b_${UCL_USER}_L${layer}.log"

                ssh -f $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} \
                    "/bin/bash -c 'cd ${PROJECT_DIR} && nohup csh run_resource_exchange_32b_multiple_gpu.csh ${layer} > ${LOG} 2>&1 &'"

                set found = 1
                @ machine_idx++
                sleep 2
                break
            else
                echo "    ${machine}: GPU busy (${raw} procs), skipping"
                @ machine_idx++
                sleep 1
            endif
        end

        if ( $found == 0 ) then
            echo "==> WARNING: No free machine for Layer ${layer}"
        endif
    end

    echo ""
    echo "============================================================"
    echo "Dispatch complete. Check logs at:"
    echo "  ${PROJECT_DIR}/logs/resource_exchange_32b_${UCL_USER}_L*.log"
    echo "============================================================"
    exit 0

# ═══════════════════════════════════════════════════════════════════════
# LOCAL MODE — Run all dims x alphas for a single layer
# ═══════════════════════════════════════════════════════════════════════
local_run:
    set LAYER = "$argv[1]"

    # This is only in Ayman's script — everyone else just activates the environment first
    source /cs/student/projects1/2022/aymakhan/.venv/bin/activate.csh
    setenv HF_HOME .hf_cache

    set OUT_BASE = "results/resource_exchange/${MODEL}"
    set BASELINE_FILE = "${OUT_BASE}/baseline_L${LAYER}.json"

    # Step 1: Compute baseline once (if not already saved)
    if ( ! -f "${BASELINE_FILE}" ) then
        echo "==> Computing baseline for Layer ${LAYER} (once for all dims/alphas)"

        python resource_exchange_game.py \
            --model         "${MODEL}" \
            --layers        $LAYER \
            --alpha         0 \
            --steered_player 1 \
            --n_games       $N_GAMES \
            --paired \
            --vectors_dir   "${VECTORS_DIR}" \
            --save_baseline "${BASELINE_FILE}" \
            --output_dir    "${OUT_BASE}/baseline_L${LAYER}"
    else
        echo "==> Baseline already computed: ${BASELINE_FILE}"
    endif

    # Step 2: Run steered configs, reusing baseline
    foreach dim ( $DIMS )
        foreach alpha ( $ALPHAS )
            set OUT_DIR = "${OUT_BASE}/${dim}_P1_L${LAYER}_a${alpha}"

            if ( -f "${OUT_DIR}/results.json" ) then
                echo "==> Skipping ${dim} L${LAYER} a${alpha} (already complete)"
            else
                echo "==> Running ${dim} L${LAYER} alpha=${alpha}"

                python resource_exchange_game.py \
                    --model         "${MODEL}" \
                    --dimension     "${dim}" \
                    --layers        $LAYER \
                    --alpha         $alpha \
                    --steered_player 1 \
                    --n_games       $N_GAMES \
                    --paired \
                    --vectors_dir   "${VECTORS_DIR}" \
                    --baseline_file "${BASELINE_FILE}" \
                    --output_dir    "${OUT_DIR}"
            endif
        end
    end

    echo "==> All configs complete for Layer ${LAYER}."
    exit 0
