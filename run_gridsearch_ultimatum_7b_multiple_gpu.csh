#!/bin/csh
# run_gridsearch_ultimatum_7b_multiple_gpu.csh
# Grid search over negotiation dimensions using the Ultimatum Game.
# Qwen 2.5-7B, rule-based opponent, fully reproducible.
#
# Two modes:
#   Dispatch:  csh run_gridsearch_ultimatum_7b_multiple_gpu.csh <username>
#              SSHes into free GPU machines and launches one layer per machine.
#
#   Local:     csh run_gridsearch_ultimatum_7b_multiple_gpu.csh <layer_number>
#              Runs the gridsearch for that layer on the current machine.

# ═══════════════════════════════════════════════════════════════════════
# EDIT THESE
# ═══════════════════════════════════════════════════════════════════════
set LAYERS = ( 10 12 14 16 )

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
# DISPATCH MODE — SSH into free machines, launch one layer per machine
# ═══════════════════════════════════════════════════════════════════════
dispatch:
    set UCL_USER    = "$argv[1]"
    set JUMP_HOST   = "${UCL_USER}@knuckles.cs.ucl.ac.uk"
    set DOMAIN      = "cs.ucl.ac.uk"
    set PROJECT_DIR = "/cs/student/projects3/2022/${UCL_USER}/comp0087_snlp_cwk"
    set VENV_ACTIVATE = "/cs/student/projects3/2022/${UCL_USER}/.venv/bin/activate.csh"
    set SSH_OPTS    = "-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

    echo "============================================================"
    echo "Dispatch mode: ${#LAYERS} layers across ${#MACHINES} machines"
    echo "User: ${UCL_USER}"
    echo "Layers: ${LAYERS}"
    echo "============================================================"
    echo ""

    # Track which machine index to start from (so we don't reuse machines)
    set machine_idx = 1

    foreach layer ( $LAYERS )
        set found = 0

        while ( $machine_idx <= $#MACHINES )
            set machine = $MACHINES[$machine_idx]

            # Check GPU availability
            set raw = `ssh $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l" |& grep '^[0-9]' | tr -d ' '`

            if ( "$raw" == "0" ) then
                echo "==> Layer ${layer} -> ${machine} (GPU free)"

                set LOG = "${PROJECT_DIR}/logs/gridsearch_7b_${UCL_USER}_L${layer}.log"

                ssh $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} \
                    "cd ${PROJECT_DIR} && source ${VENV_ACTIVATE} && nohup csh run_gridsearch_ultimatum_7b_multiple_gpu.csh ${layer} >& ${LOG} &"

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
    echo "  ${PROJECT_DIR}/logs/gridsearch_7b_${UCL_USER}_L*.log"
    echo "============================================================"
    exit 0

# ═══════════════════════════════════════════════════════════════════════
# LOCAL MODE — Run gridsearch for a single layer on this machine
# ═══════════════════════════════════════════════════════════════════════
local_run:
    set FIXED_LAYERS = ( $argv[1] )

    set FIXED_POOL  = ""   # leave empty to use variable pool sizes

    setenv HF_HOME .hf_cache
    if ( $?HF_TOKEN ) then
        setenv HF_TOKEN "$HF_TOKEN"
    endif

    set RULEBASED   = (--rulebased)

    set MODEL       = "qwen2.5-7b"
    set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
    set SUFFIX      = "_7b_L${FIXED_LAYERS}"

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
                        --coarse_alphas -15 -5 5 15 \
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
    exit 0
