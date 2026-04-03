#!/bin/bash
# Monitor final 7B grid experiment across all 3 machines.
# Shows: file count, last log lines, GPU usage.
#
# Usage: bash monitor_final_grid.sh

USER=moiimran
JUMP=moiimran@knuckles.cs.ucl.ac.uk
REMOTE_DIR=/cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

HOSTS="scaup-l scoter-l shoveler-l"
# NFS shared filesystem — all machines write to same directory
# Total expected depends on current batch alphas
EXPECTED_TOTAL=180  # 10 dims x 9 layers x 2 alphas (BATCH 1)

echo "=================================================================="
echo "FINAL 7B GRID — MONITOR"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# File count (shared NFS, check from any machine)
echo ""
echo "--- Results (shared NFS) ---"
TOTAL_FILES=$(ssh -o ConnectTimeout=10 -J $JUMP -l $USER scaup-l.cs.ucl.ac.uk \
    "find $REMOTE_DIR/results/ultimatum/final_7b_grid -name '*.json' | wc -l" 2>/dev/null) || TOTAL_FILES="?"
echo "  Total files: $TOTAL_FILES / $EXPECTED_TOTAL"

# Per-dimension breakdown
echo "  Per-dimension:"
for dim in firmness empathy anchoring greed narcissism fairness_norm composure flattery spite undecidedness; do
    DIM_COUNT=$(ssh -o ConnectTimeout=10 -J $JUMP -l $USER scaup-l.cs.ucl.ac.uk \
        "find $REMOTE_DIR/results/ultimatum/final_7b_grid -name '${dim}_*.json' | wc -l" 2>/dev/null) || DIM_COUNT="?"
    echo "    $dim: $DIM_COUNT / 18"
done

# Per-machine process and GPU status
for host in $HOSTS; do
    echo ""
    echo "--- $host ---"

    # Process check
    PROC=$(ssh -o ConnectTimeout=10 -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "pgrep -c -f final_grid_${host%%-l}" 2>/dev/null) || PROC="0"
    if [ "$PROC" -gt 0 ]; then
        echo "  Process: RUNNING"
    else
        echo "  Process: NOT RUNNING"
    fi

    # GPU usage
    GPU_INFO=$(ssh -o ConnectTimeout=10 -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null) || GPU_INFO="?"
    echo "  GPU: $GPU_INFO"

    # Last few INFO lines from log (grep to skip progress bars)
    echo "  Recent log:"
    ssh -o ConnectTimeout=10 -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "grep -E '(DONE|START|SKIP|FAIL|ALL DONE)' $REMOTE_DIR/final_grid_${host%%-l}.log 2>/dev/null | tail -3" 2>/dev/null | sed 's/^/    /'

done

echo ""
echo "=================================================================="
echo "TOTAL FILES: $TOTAL_FILES / $EXPECTED_TOTAL"
if [ "$TOTAL_FILES" = "$EXPECTED_TOTAL" ]; then
    echo "STATUS: COMPLETE"
elif [ "$TOTAL_FILES" = "?" ]; then
    echo "STATUS: UNKNOWN (connection failed)"
else
    REMAINING=$((EXPECTED_TOTAL - TOTAL_FILES))
    echo "STATUS: IN PROGRESS ($REMAINING remaining)"
fi
echo "=================================================================="
