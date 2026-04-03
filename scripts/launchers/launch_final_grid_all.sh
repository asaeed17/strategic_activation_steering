#!/bin/bash
# Master launcher for final 7B grid experiment.
# Syncs code to all 3 machines and launches batch runners.
#
# Usage: bash launch_final_grid_all.sh
#
# Machine assignment:
#   scaup-l:    firmness, empathy, anchoring        (108 configs)
#   scoter-l:   greed, narcissism, fairness_norm     (108 configs)
#   shoveler-l: composure, flattery, spite, undecidedness (144 configs)
#
# Total: 10 dims x 9 layers x 4 alphas x n=50 = 360 configs, 18,000 paired games

set -e

USER=moiimran
JUMP=moiimran@knuckles.cs.ucl.ac.uk
LOCAL_DIR=/Users/moiz/Documents/code/comp0087_snlp_cwk
REMOTE_DIR=/cs/student/projects1/2022/moiimran/comp0087_snlp_cwk
VENV=/cs/student/projects1/2022/moiimran/venv/bin

HOSTS="scaup-l scoter-l shoveler-l"

echo "=================================================================="
echo "FINAL 7B GRID — MASTER LAUNCHER"
echo "  360 configs across 3 machines"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================================="

# ------------------------------------------------------------------
# Step 1: Check all machines are reachable and GPUs are free
# ------------------------------------------------------------------
echo ""
echo "--- Step 1: Checking machines ---"
ALL_OK=true
for host in $HOSTS; do
    echo -n "  $host: "
    GPU_INFO=$(ssh -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null) || {
        echo "UNREACHABLE"
        ALL_OK=false
        continue
    }
    echo "$GPU_INFO"
    USED_MB=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
    if [ "$USED_MB" -gt 500 ]; then
        echo "    WARNING: $USED_MB MiB in use — another process may be running"
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Not all machines reachable. Continue anyway? (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        echo "Aborting."
        exit 1
    fi
fi

# ------------------------------------------------------------------
# Step 2: Sync code to all machines
# ------------------------------------------------------------------
echo ""
echo "--- Step 2: Syncing code to all machines ---"
for host in $HOSTS; do
    echo "  Syncing to $host..."
    rsync -avz --delete \
        --exclude '.hf_cache' \
        --exclude '__pycache__' \
        --exclude 'venv' \
        --exclude '.git' \
        --exclude 'results/' \
        --exclude '*.log' \
        --exclude '.mypy_cache' \
        -e "ssh -J $JUMP -l $USER" \
        "$LOCAL_DIR/" \
        "${host}.cs.ucl.ac.uk:$REMOTE_DIR/" \
        | tail -3
    echo "    Done."
done

# ------------------------------------------------------------------
# Step 3: Create output directory on all machines
# ------------------------------------------------------------------
echo ""
echo "--- Step 3: Creating output directories ---"
for host in $HOSTS; do
    ssh -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "mkdir -p $REMOTE_DIR/results/ultimatum/final_7b_grid"
    echo "  $host: OK"
done

# ------------------------------------------------------------------
# Step 4: Launch batch runners
# ------------------------------------------------------------------
echo ""
echo "--- Step 4: Launching batch runners ---"

# scaup-l: firmness, empathy, anchoring
echo "  Launching on scaup-l (firmness, empathy, anchoring — 108 configs)..."
ssh -J $JUMP -l $USER scaup-l.cs.ucl.ac.uk \
    "/bin/bash -c 'cd $REMOTE_DIR && bash launch_final_grid_scaup.sh'"
echo ""

# scoter-l: greed, narcissism, fairness_norm
echo "  Launching on scoter-l (greed, narcissism, fairness_norm — 108 configs)..."
ssh -J $JUMP -l $USER scoter-l.cs.ucl.ac.uk \
    "/bin/bash -c 'cd $REMOTE_DIR && bash launch_final_grid_scoter.sh'"
echo ""

# shoveler-l: composure, flattery, spite, undecidedness
echo "  Launching on shoveler-l (composure, flattery, spite, undecidedness — 144 configs)..."
ssh -J $JUMP -l $USER shoveler-l.cs.ucl.ac.uk \
    "/bin/bash -c 'cd $REMOTE_DIR && bash launch_final_grid_shoveler.sh'"
echo ""

# ------------------------------------------------------------------
# Step 5: Verify launches
# ------------------------------------------------------------------
echo "--- Step 5: Verifying processes ---"
sleep 5
for host in $HOSTS; do
    echo -n "  $host: "
    PIDS=$(ssh -J $JUMP -l $USER ${host}.cs.ucl.ac.uk \
        "pgrep -f 'final_grid_' -a 2>/dev/null | head -3" 2>/dev/null) || PIDS=""
    if [ -n "$PIDS" ]; then
        echo "RUNNING"
        echo "    $PIDS"
    else
        echo "NOT FOUND — check log files"
    fi
done

echo ""
echo "=================================================================="
echo "ALL LAUNCHED"
echo ""
echo "Monitor with:"
echo "  bash monitor_final_grid.sh"
echo ""
echo "Pull results with:"
echo "  bash pull_final_grid_results.sh"
echo "=================================================================="
