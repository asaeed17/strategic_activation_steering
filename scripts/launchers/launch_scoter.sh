#!/bin/bash
# Launch script for scoter-l — copy to machine and run
set -e

cd /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk

VENV=/cs/student/projects1/2022/moiimran/venv/bin
export HF_HOME=/cs/student/projects1/2022/moiimran/.hf_cache
export PATH=$VENV:$PATH

# GPU check
CC=$($VENV/python -c "import torch; print(torch.cuda.get_device_capability()[0])")
if [ "$CC" -lt 8 ]; then
    echo "ERROR: GPU compute capability $CC < 8. Need Ampere (RTX 3090 Ti). Aborting."
    exit 1
fi
echo "GPU check passed: CC=$CC"

# Check for existing process
if pgrep -f "overnight_scoter.py" > /dev/null 2>&1; then
    echo "WARNING: overnight_scoter.py already running. Killing old process."
    pkill -f "overnight_scoter.py"
    sleep 2
fi

nohup $VENV/python overnight_scoter.py \
  > /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk/overnight_scoter.log 2>&1 &

echo "Launched PID: $!"
sleep 3
echo "--- First lines of log ---"
head -10 /cs/student/projects1/2022/moiimran/comp0087_snlp_cwk/overnight_scoter.log 2>/dev/null || echo "(log not yet created)"
