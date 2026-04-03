#!/bin/bash
# Launch final 7B grid on scoter-l
# Dimensions: greed, narcissism, fairness_norm (108 configs)
set -e

cd /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

VENV=/cs/student/projects1/2022/moiimran/venv/bin
export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
export PATH=$VENV:$PATH

# GPU check
CC=$($VENV/python -c "import torch; print(torch.cuda.get_device_capability()[0])")
if [ "$CC" -lt 8 ]; then
    echo "ABORT: GPU compute capability $CC < 8. Need Ampere. Aborting."
    exit 1
fi
echo "GPU check passed: CC=$CC"

# Check GPU memory usage
$VENV/python -c "
import torch
total = torch.cuda.get_device_properties(0).total_memory / 1e9
name = torch.cuda.get_device_name()
print(f'GPU: {name}, Total: {total:.1f} GB')
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
used_mb = int(result.stdout.strip())
print(f'GPU memory in use: {used_mb} MiB')
if used_mb > 500:
    print(f'WARNING: GPU has {used_mb} MiB in use. Another process may be running.')
"

# Kill any existing final_grid process for this machine
if pgrep -f "final_grid_scoter.py" > /dev/null 2>&1; then
    echo "WARNING: final_grid_scoter.py already running. Killing old process."
    pkill -f "final_grid_scoter.py"
    sleep 2
fi

LOGFILE=/cs/student/projects1/2022/moiimran/comp0087_snlp_cwk/final_grid_scoter.log

nohup $VENV/python final_grid_scoter.py > "$LOGFILE" 2>&1 &

echo "Launched PID: $!"
echo "Log: $LOGFILE"
sleep 3
echo "--- First lines of log ---"
head -15 "$LOGFILE" 2>/dev/null || echo "(log not yet created)"
