#!/bin/bash
# Pull final 7B grid results from all 3 machines.
#
# Usage: bash pull_final_grid_results.sh

USER=moiimran
JUMP=moiimran@knuckles.cs.ucl.ac.uk
REMOTE_DIR=/cs/student/projects1/2022/moiimran/comp0087_snlp_cwk
LOCAL_DIR=/Users/moiz/Documents/code/comp0087_snlp_cwk

HOSTS="scaup-l scoter-l shoveler-l"

echo "=================================================================="
echo "PULLING FINAL 7B GRID RESULTS"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# Create local output directory
mkdir -p "$LOCAL_DIR/results/ultimatum/final_7b_grid"

for host in $HOSTS; do
    echo ""
    echo "--- Pulling from $host ---"
    rsync -avz \
        -e "ssh -J $JUMP -l $USER" \
        "${host}.cs.ucl.ac.uk:$REMOTE_DIR/results/ultimatum/final_7b_grid/" \
        "$LOCAL_DIR/results/ultimatum/final_7b_grid/" \
        | tail -5
    echo "  Done."
done

# Also pull log files
echo ""
echo "--- Pulling log files ---"
for host in $HOSTS; do
    LOGNAME="final_grid_${host%%-l}"
    rsync -avz \
        -e "ssh -J $JUMP -l $USER" \
        "${host}.cs.ucl.ac.uk:$REMOTE_DIR/${LOGNAME}*.log" \
        "$LOCAL_DIR/" 2>/dev/null \
        | tail -3
done

# Count results
echo ""
echo "=================================================================="
TOTAL=$(ls "$LOCAL_DIR/results/ultimatum/final_7b_grid/"*.json 2>/dev/null | wc -l | tr -d ' ')
echo "TOTAL FILES PULLED: $TOTAL / 360"

if [ "$TOTAL" -eq 360 ]; then
    echo "STATUS: COMPLETE — all 360 configs collected"
else
    echo "STATUS: INCOMPLETE — $((360 - TOTAL)) files missing"
    echo ""
    echo "Missing dimensions/configs:"
    for dim in firmness empathy anchoring greed narcissism fairness_norm composure flattery spite undecidedness; do
        DIM_COUNT=$(ls "$LOCAL_DIR/results/ultimatum/final_7b_grid/${dim}_"*.json 2>/dev/null | wc -l | tr -d ' ')
        if [ "$DIM_COUNT" -lt 36 ]; then
            echo "  $dim: $DIM_COUNT / 36"
        fi
    done
fi
echo "=================================================================="
