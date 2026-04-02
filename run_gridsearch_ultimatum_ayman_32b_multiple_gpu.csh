#!/bin/csh
# run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh
# Grid search over negotiation dimensions using the Ultimatum Game.
# Qwen 2.5-32B GPTQ-Int4, rule-based opponent, fully reproducible.
#
# Two modes:
#   Dispatch:  csh run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh <username>
#              SSHes into free GPU machines and launches one (layer, dim_group) per machine.
#              7 layers × 3 dim groups = 21 jobs across available machines.
#
#   Local:     csh run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh <layer_number> <group_number>
#              Runs the gridsearch for that layer + dimension group on the current machine.
#              Each group runs both proposer and responder roles.

# ═══════════════════════════════════════════════════════════════════════
# EDIT THESE
# ═══════════════════════════════════════════════════════════════════════
set LAYERS = ( 32 )

# Dimension groups (3 groups: 4 + 3 + 3 = 10)
# Group 1: firmness empathy spite narcissism
# Group 2: greed fairness_norm flattery
# Group 3: composure anchoring undecidedness

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
    echo "  Local:    csh $0 <layer_number> <group_number>"
    echo ""
    echo "  Groups: 1 = firmness"
    echo "          2 = empathy"
    echo "          3 = spite"
    echo "          4 = narcissism"
    exit 1
endif

# Detect mode: if first arg is a number, run locally. Otherwise, dispatch.
echo "$argv[1]" | grep -q '^[0-9][0-9]*$'
if ( $status == 0 ) then
    goto local_run
else
    goto dispatch
endif

# ═══════════════════════════════════════════════════════════════════════
# DISPATCH MODE — SSH into free machines, one (layer, dim_group) per machine
# ═══════════════════════════════════════════════════════════════════════
dispatch:
    set UCL_USER    = "$argv[1]"
    set DOMAIN      = "cs.ucl.ac.uk"
    set PROJECT_DIR = "/cs/student/projects3/2022/${UCL_USER}/comp0087_snlp_cwk"
    set SSH_OPTS    = "-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
    set NUM_GROUPS  = 4

    set total_jobs = 0
    @ total_jobs = $#LAYERS * $NUM_GROUPS

    echo "============================================================"
    echo "Dispatch mode: ${#LAYERS} layers x ${NUM_GROUPS} dim groups = ${total_jobs} jobs"
    echo "User: ${UCL_USER}"
    echo "Layers: ${LAYERS}"
    echo "Groups: 1=firmness  2=empathy  3=spite  4=narcissism"
    echo "============================================================"
    echo ""

    set machine_idx = 1

    foreach layer ( $LAYERS )
        foreach group ( 1 2 3 4 )
            set found = 0

            while ( $machine_idx <= $#MACHINES )
                set machine = $MACHINES[$machine_idx]

                set raw = `ssh $SSH_OPTS ${machine}.${DOMAIN} '/bin/bash -c "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l"' |& grep '^[0-9]' | tr -d ' '`

                if ( "$raw" == "0" ) then
                    echo "==> Layer ${layer}, Group ${group} -> ${machine} (GPU free)"

                    set LOG = "${PROJECT_DIR}/logs/gridsearch_32b_${UCL_USER}_L${layer}_G${group}.log"

        #            ssh $SSH_OPTS ${machine}.${DOMAIN} \
        #                "nohup /bin/bash -c 'cd ${PROJECT_DIR} && csh run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh ${layer} ${group} > ${LOG} 2>&1' &"
                ssh -f $SSH_OPTS -l $UCL_USER -J $JUMP_HOST ${machine}.${DOMAIN} \
                    "/bin/bash -c 'cd ${PROJECT_DIR} && nohup csh run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh ${layer} > ${LOG} 2>&1 &'"

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
                echo "==> WARNING: No free machine for Layer ${layer}, Group ${group}"
            endif
        end
    end

    echo ""
    echo "============================================================"
    echo "Dispatch complete. Check logs at:"
    echo "  ${PROJECT_DIR}/logs/gridsearch_32b_${UCL_USER}_L*_G*.log"
    echo "============================================================"
    exit 0

# ═══════════════════════════════════════════════════════════════════════
# LOCAL MODE — Run gridsearch for a single layer + dimension group
# ═══════════════════════════════════════════════════════════════════════
local_run:
    if ( $#argv < 2 ) then
        echo "Local mode requires two arguments: <layer_number> <group_number>"
        echo "  Groups: 1 = firmness  2 = empathy  3 = spite  4 = narcissism"
        exit 1
    endif

    set FIXED_LAYERS = ( $argv[1] )
    set GROUP_NUM    = $argv[2]

    # Map group number to dimensions
    if ( "$GROUP_NUM" == "1" ) then
        set DIMS = ( firmness )
    else if ( "$GROUP_NUM" == "2" ) then
        set DIMS = ( empathy )
    else if ( "$GROUP_NUM" == "3" ) then
        set DIMS = ( spite )
    else if ( "$GROUP_NUM" == "4" ) then
        set DIMS = ( narcissism )
    else
        echo "Invalid group number: ${GROUP_NUM} (must be 1-4)"
        exit 1
    endif

    set FIXED_POOL  = ""   # leave empty to use variable pool sizes
    # Set prompt to avoid "Undefined variable" in non-interactive shells
    if ( ! $?prompt ) set prompt = ""
    source /cs/student/projects3/2022/asaeed/comp0087_snlp_cwk/env/bin/activate.csh
    setenv HF_HOME .hf_cache
    if ( $?HF_TOKEN ) then
        setenv HF_TOKEN "$HF_TOKEN"
    endif

    set RULEBASED   = (--rulebased)

    set MODEL       = "qwen2.5-32b-gptq"
    set VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
    set SUFFIX      = "_ayman_32b_L${FIXED_LAYERS}"

    set OUT_DIR = "results/ultimatum/llm_vs_rulebased/${MODEL}${SUFFIX}"

    echo "==> Layer ${FIXED_LAYERS}, Group ${GROUP_NUM}: ${DIMS}"

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

    echo "==> All dimensions in Group ${GROUP_NUM} complete for Layer ${FIXED_LAYERS}."
    exit 0
