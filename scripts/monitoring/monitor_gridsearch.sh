#!/bin/bash
# monitor_gridsearch.sh — Watches dispatched gridsearch jobs and auto-restarts dead ones.
#
# Usage:  bash monitor_gridsearch.sh <username> [check_interval_minutes]
#         Default check interval: 5 minutes
#
# What it does:
#   1. Finds all incomplete (layer, group) combos (no final_best.json for all roles/dims)
#   2. Checks if a python gridsearch process is running on any machine for each combo
#   3. If dead → finds a free 24GB machine and relaunches
#   4. Repeats until all combos are complete
#
# Run in background:  nohup bash monitor_gridsearch.sh asaeed > logs/monitor.log 2>&1 &

set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────
UCL_USER="${1:?Usage: bash monitor_gridsearch.sh <username> [check_interval_min]}"
CHECK_INTERVAL_MIN="${2:-5}"
DOMAIN="cs.ucl.ac.uk"
PROJECT_DIR="/cs/student/projects3/2022/${UCL_USER}/comp0087_snlp_cwk"
SCRIPT_NAME="run_gridsearch_ultimatum_ayman_32b_multiple_gpu.csh"
SSH_OPTS="-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes"
MODEL="qwen2.5-32b-gptq"
SUFFIX_PREFIX="_ayman_32b"
MIN_VRAM_MB=20000  # Only use machines with >= 20GB free VRAM

# Layers and group→dim mapping (must match the csh script)
LAYERS=( 32 )
declare -A GROUP_DIMS=( [1]="firmness" [2]="empathy" [3]="spite" [4]="narcissism" )
ROLES=( proposer responder )

# All 24GB duck machines (no hotspot — only 16GB)
MACHINES=(
    aylesbury-l barnacle-l brent-l bufflehead-l cackling-l canada-l
    crested-l eider-l gadwall-l goosander-l gressingham-l harlequin-l
    mallard-l mandarin-l pintail-l ruddy-l scaup-l scoter-l
    shelduck-l shoveler-l smew-l wigeon-l
)

# Track which machine each (layer,group) is running on
declare -A JOB_MACHINE  # key: "L${layer}_G${group}" → machine name

# ─── Helpers ──────────────────────────────────────────────────────────
log() { echo "$(date '+%H:%M:%S')  $*"; }

is_combo_complete() {
    local layer=$1 group=$2
    local dim="${GROUP_DIMS[$group]}"
    local base="${PROJECT_DIR}/results/ultimatum/llm_vs_rulebased/${MODEL}${SUFFIX_PREFIX}_L${layer}/L${layer}"
    for role in "${ROLES[@]}"; do
        if [[ ! -f "${base}/${role}/${dim}/final_best.json" ]]; then
            return 1
        fi
    done
    return 0
}

# Check if a gridsearch python process for this layer+group is running on a machine
find_running_machine() {
    local layer=$1 group=$2
    local dim="${GROUP_DIMS[$group]}"

    # First check the machine we last dispatched to
    local key="L${layer}_G${group}"
    if [[ -n "${JOB_MACHINE[$key]:-}" ]]; then
        local machine="${JOB_MACHINE[$key]}"
        local count
        count=$(ssh $SSH_OPTS "${machine}.${DOMAIN}" \
            "ps -u ${UCL_USER} -o pid,args | grep 'python.*lightweight_gridsearch_ultimatum.*--dimension ${dim}' | grep -vc grep" 2>/dev/null | tail -1 | tr -d '[:space:]')
        count="${count:-0}"
        if [[ "$count" -gt 0 ]]; then
            echo "$machine"
            return 0
        fi
    fi

    # Fallback: scan all machines
    for machine in "${MACHINES[@]}"; do
        local count
        count=$(ssh $SSH_OPTS "${machine}.${DOMAIN}" \
            "ps -u ${UCL_USER} -o pid,args | grep 'python.*lightweight_gridsearch_ultimatum.*--dimension ${dim}' | grep -vc grep" 2>/dev/null | tail -1 | tr -d '[:space:]')
        count="${count:-0}"
        if [[ "$count" -gt 0 ]]; then
            JOB_MACHINE[$key]="$machine"
            echo "$machine"
            return 0
        fi
    done

    return 1
}

# Find a free machine with enough VRAM
find_free_machine() {
    for machine in "${MACHINES[@]}"; do
        # Skip machines already running our jobs
        local in_use=false
        for key in "${!JOB_MACHINE[@]}"; do
            if [[ "${JOB_MACHINE[$key]}" == "$machine" ]]; then
                in_use=true
                break
            fi
        done
        if $in_use; then continue; fi

        local free_mb
        free_mb=$(ssh $SSH_OPTS "${machine}.${DOMAIN}" \
            'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1' 2>/dev/null || echo "0")
        free_mb=$(echo "$free_mb" | tr -d ' ')

        if [[ "$free_mb" =~ ^[0-9]+$ ]] && [[ "$free_mb" -ge "$MIN_VRAM_MB" ]]; then
            echo "$machine"
            return 0
        fi
    done
    return 1
}

launch_on_machine() {
    local machine=$1 layer=$2 group=$3
    local key="L${layer}_G${group}"
    local log_file="${PROJECT_DIR}/logs/gridsearch_32b_${UCL_USER}_${key}_auto.log"

    ssh $SSH_OPTS "${machine}.${DOMAIN}" \
        "nohup /bin/bash -c 'cd ${PROJECT_DIR} && csh ${SCRIPT_NAME} ${layer} ${group} > ${log_file} 2>&1' &" 2>/dev/null

    JOB_MACHINE[$key]="$machine"
    log "LAUNCHED  ${key} (${GROUP_DIMS[$group]}) on ${machine} → ${log_file}"
}

# ─── Main loop ────────────────────────────────────────────────────────
log "Monitor started. User=${UCL_USER}, layers=(${LAYERS[*]}), check every ${CHECK_INTERVAL_MIN}m"
log "Dimensions: $(for g in "${!GROUP_DIMS[@]}"; do echo -n "G${g}=${GROUP_DIMS[$g]} "; done)"

# Initial discovery: find where jobs are already running
for layer in "${LAYERS[@]}"; do
    for group in "${!GROUP_DIMS[@]}"; do
        if is_combo_complete "$layer" "$group"; then
            continue
        fi
        if machine=$(find_running_machine "$layer" "$group"); then
            log "FOUND     L${layer}_G${group} (${GROUP_DIMS[$group]}) running on ${machine}"
        fi
    done
done

while true; do
    all_done=true

    for layer in "${LAYERS[@]}"; do
        for group in $(echo "${!GROUP_DIMS[@]}" | tr ' ' '\n' | sort -n); do
            local_key="L${layer}_G${group}"
            dim="${GROUP_DIMS[$group]}"

            # Skip if complete
            if is_combo_complete "$layer" "$group"; then
                continue
            fi

            all_done=false

            # Check if running
            if machine=$(find_running_machine "$layer" "$group"); then
                log "OK        ${local_key} (${dim}) alive on ${machine}"
            else
                log "DEAD      ${local_key} (${dim}) — no process found"
                unset "JOB_MACHINE[$local_key]"

                # Find a free machine and relaunch
                if free_machine=$(find_free_machine); then
                    launch_on_machine "$free_machine" "$layer" "$group"
                else
                    log "WARNING   No free machine for ${local_key} — will retry next cycle"
                fi
            fi
        done
    done

    if $all_done; then
        log "ALL COMPLETE — all (layer, group) combos have final_best.json. Exiting."
        exit 0
    fi

    log "Sleeping ${CHECK_INTERVAL_MIN}m..."
    sleep $(( CHECK_INTERVAL_MIN * 60 ))
done
