#!/bin/csh
# run_resource_exchange_32b_multiple_gpu.csh
# Resource Exchange Game with activation steering across multiple GPU machines.
# Qwen 2.5-32B GPTQ-Int4, rule-based opponent.
#
# Two modes:
#   Dispatch:  csh run_resource_exchange_32b_multiple_gpu.csh <username>
#              SSHes into free GPU machines and launches one dimension per machine.
#
#   Local:     csh run_resource_exchange_32b_multiple_gpu.csh <dimension>
#              Runs the resource exchange game for that dimension on the current machine.

# ═══════════════════════════════════════════════════════════════════════
# EDIT THESE
# ═══════════════════════════════════════════════════════════════════════
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

set LAYERS       = ( 10 )
set ALPHAS       = ( 3 7 10 )
set N_GAMES      = 50
set MODEL        = "qwen2.5-32b-gptq"
set VECTORS_DIR  = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
# ═══════════════════════════════════════════════════════════════════════

if ( $#argv < 1 ) then
    echo "Usage:"
    echo "  Dispatch: csh $0 <username>"
    echo "  Local:    csh $0 <dimension>"
    exit 1
endif

# Detect mode: if arg matches a known dimension, run locally. Otherwise, dispatch.
echo "$argv[1]" | grep -q '^[a-z_]*$'
if ( $status == 0 ) then
    # Could be a dimension or a username — check if it's a known dimension
    set is_dim = 0
    foreach d ( $DIMS )
        if ( "$argv[1]" == "$d" ) then
            set is_dim = 1
            break
        endif
    end
    if ( $is_dim == 1 ) then
        goto local_run
    endif
endif
goto dispatch

# ═══════════════════════════════════════════════════════════════════════
# DISPATCH MODE
# ═══════════════════════════════════════════════════════════════════════
dispatch:
    set UCL_USER    = "$argv[1]"
    set JUMP_HOST   = "${UCL_USER}@knuckles.cs.ucl.ac.uk"
    set DOMAIN      = "cs.ucl.ac.uk"
    set PROJECT_DIR = "/cs/student/projects1/2022/${UCL_USER}/comp0087_snlp_cwk"
    set SSH_OPTS    = "-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

    echo "============================================================"
    echo "Resource Exchange Dispatch: ${#DIMS} dims across ${#MACHINES} machines"
    echo "User: ${UCL_USER}"
    echo "Dims: ${DIMS}"
    echo "Layers: ${LAYERS}  Alphas: ${ALPHAS}"
    echo "============================================================"
    echo ""

    set machine_idx = 1

    foreach dim ( $DIMS )
        set found = 0

        while ( $machine_idx <= $#MACHINES )
            set machine = $MACHINES[$machine_idx]

            set raw = `ssh $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l" |& grep '^[0-9]' | tr -d ' '`

            if ( "$raw" == "0" ) then
                echo "==> ${dim} -> ${machine} (GPU free)"

                set LOG = "${PROJECT_DIR}/logs/resource_exchange_32b_${UCL_USER}_${dim}.log"

                ssh -f $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} \
                    "cd ${PROJECT_DIR} && nohup /bin/bash -c 'csh run_resource_exchange_32b_multiple_gpu.csh ${dim}' > ${LOG} 2>&1 &"

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
            echo "==> WARNING: No free machine for ${dim}"
        endif
    end

    echo ""
    echo "============================================================"
    echo "Dispatch complete. Check logs at:"
    echo "  ${PROJECT_DIR}/logs/resource_exchange_32b_${UCL_USER}_*.log"
    echo "============================================================"
    exit 0

# ═══════════════════════════════════════════════════════════════════════
# LOCAL MODE — Run resource exchange for a single dimension
# ═══════════════════════════════════════════════════════════════════════
local_run:
    set DIM = "$argv[1]"

    # This is only in Ayman's script — everyone else just activates the environment first
    source /cs/student/projects1/2022/aymakhan/.venv/bin/activate.csh
    setenv HF_HOME .hf_cache

    set OUT_BASE = "results/resource_exchange/${MODEL}"

    foreach layer ( $LAYERS )
        foreach alpha ( $ALPHAS )
            set OUT_DIR = "${OUT_BASE}/${DIM}_P1_L${layer}_a${alpha}"

            if ( -f "${OUT_DIR}/results.json" ) then
                echo "==> Skipping ${DIM} L${layer} a${alpha} (already complete)"
            else
                echo "==> Running ${DIM} L${layer} alpha=${alpha}"

                python resource_exchange_game.py \
                    --model         "${MODEL}" \
                    --dimension     "${DIM}" \
                    --layers        $layer \
                    --alpha         $alpha \
                    --steered_player 1 \
                    --n_games       $N_GAMES \
                    --paired \
                    --rulebased \
                    --quantize \
                    --vectors_dir   "${VECTORS_DIR}" \
                    --output_dir    "${OUT_DIR}"
            endif
        end
    end

    echo "==> All configs complete for ${DIM}."
    exit 0
