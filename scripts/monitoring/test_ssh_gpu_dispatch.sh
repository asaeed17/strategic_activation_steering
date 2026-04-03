#!/usr/bin/env bash
# test_ssh_gpu_dispatch.sh
#
# Test script to verify the SSH + GPU dispatch concept before building
# the real distributed gridsearch runner.
#
# Tests:
#   1. SSH connectivity to each machine via jump host
#   2. GPU detection (nvidia-smi exists and returns output)
#   3. GPU availability check (no running processes)
#   4. Background command execution via nohup
#   5. Shared filesystem verification (can see project dir)
#
# Usage:
#   bash test_ssh_gpu_dispatch.sh
#   bash test_ssh_gpu_dispatch.sh --machines "wigeon-l teal-l gadwall-l"
#   bash test_ssh_gpu_dispatch.sh --dry-run   # just print what would happen

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
UCL_USER="aymakhan"
JUMP_HOST="${UCL_USER}@knuckles.cs.ucl.ac.uk"
DOMAIN="cs.ucl.ac.uk"
PROJECT_DIR="/cs/student/projects1/2022/aymakhan/comp0087_snlp_cwk"
SSH_TIMEOUT=10

# Default machine list — edit or override with --machines
MACHINES=(
    aylesbury-l
    barnacle-l
    brent-l
    bufflehead-l
    cackling-l
    canada-l
    crested-l
    eider-l
    gadwall-l
    goosander-l
    gressingham-l
    harlequin-l
    mallard-l
    mandarin-l
    pintail-l
    pocher-l
    ruddy-l
    scaup-l
    scoter-l
    shelduck-l
    shoveler-l
    smew-l
    wigeon-l
)

DRY_RUN=false

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --machines)
            IFS=' ' read -ra MACHINES <<< "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --user)
            UCL_USER="$2"
            JUMP_HOST="${UCL_USER}@knuckles.cs.ucl.ac.uk"
            shift 2
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────
ssh_cmd() {
    local machine="$1"
    shift
    ssh -o ConnectTimeout="$SSH_TIMEOUT" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -l "$UCL_USER" \
        -J "$JUMP_HOST" \
        "${machine}.${DOMAIN}" \
        "$@"
}

PASS=0
FAIL=0
FREE_MACHINES=()

log_pass() { echo "  [PASS] $1"; ((PASS++)); }
log_fail() { echo "  [FAIL] $1"; ((FAIL++)); }
log_skip() { echo "  [SKIP] $1"; }

# ── Main ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "SSH + GPU Dispatch Test"
echo "============================================================"
echo "User:        $UCL_USER"
echo "Jump host:   $JUMP_HOST"
echo "Machines:    ${#MACHINES[@]}"
echo "Project dir: $PROJECT_DIR"
echo "Dry run:     $DRY_RUN"
echo "============================================================"
echo ""

for machine in "${MACHINES[@]}"; do
    echo "── $machine ──────────────────────────────────────"

    # Test 1: SSH connectivity
    if $DRY_RUN; then
        log_skip "SSH connectivity (dry run)"
        continue
    fi

    if ssh_cmd "$machine" "echo ok" >/dev/null 2>&1; then
        log_pass "SSH connectivity"
    else
        log_fail "SSH connectivity — cannot reach $machine"
        echo ""
        continue
    fi

    # Test 2: nvidia-smi exists
    if ssh_cmd "$machine" "which nvidia-smi" >/dev/null 2>&1; then
        log_pass "nvidia-smi found"
    else
        log_fail "nvidia-smi not found — no GPU on this machine"
        echo ""
        continue
    fi

    # Test 3: GPU info
    gpu_info=$(ssh_cmd "$machine" "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader" 2>/dev/null || echo "ERROR")
    if [[ "$gpu_info" != "ERROR" ]]; then
        log_pass "GPU detected: $gpu_info"
    else
        log_fail "nvidia-smi query failed"
        echo ""
        continue
    fi

    # Test 4: GPU availability (no running processes)
    gpu_procs=$(ssh_cmd "$machine" "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" 2>/dev/null || echo "-1")
    gpu_procs=$(echo "$gpu_procs" | tr -d '[:space:]')
    if [[ "$gpu_procs" == "0" ]]; then
        log_pass "GPU is FREE (0 processes)"
        FREE_MACHINES+=("$machine")
    elif [[ "$gpu_procs" == "-1" ]]; then
        log_fail "Could not query GPU processes"
    else
        log_fail "GPU is BUSY ($gpu_procs processes running)"
        # Show who's using it
        ssh_cmd "$machine" "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader" 2>/dev/null | while read -r line; do
            echo "         -> $line"
        done
    fi

    # Test 5: Shared filesystem
    if ssh_cmd "$machine" "test -d '$PROJECT_DIR'" 2>/dev/null; then
        log_pass "Project dir exists on shared filesystem"
    else
        log_fail "Project dir NOT found at $PROJECT_DIR"
    fi

    # Test 6: Background execution (run a trivial command via nohup)
    test_marker="/tmp/gpu_dispatch_test_${UCL_USER}_$$"
    ssh_cmd "$machine" "nohup bash -c 'echo test_ok > $test_marker' >/dev/null 2>&1 &" 2>/dev/null
    sleep 1
    if ssh_cmd "$machine" "cat $test_marker 2>/dev/null" 2>/dev/null | grep -q "test_ok"; then
        log_pass "Background execution (nohup) works"
        ssh_cmd "$machine" "rm -f $test_marker" 2>/dev/null
    else
        log_fail "Background execution failed"
    fi

    echo ""
done

# ── Summary ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Machines tested:  ${#MACHINES[@]}"
echo "Tests passed:     $PASS"
echo "Tests failed:     $FAIL"
echo ""

if [[ ${#FREE_MACHINES[@]} -gt 0 ]]; then
    echo "FREE GPU machines (${#FREE_MACHINES[@]}):"
    for m in "${FREE_MACHINES[@]}"; do
        echo "  - $m"
    done
else
    echo "No free GPU machines found."
fi

echo ""
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    echo "Some tests failed. Review output above before building the real dispatcher."
    exit 1
else
    echo "All tests passed. Safe to build the dispatcher."
    exit 0
fi
