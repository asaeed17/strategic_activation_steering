# GPU Setup Guide for Steering Experiments on zander-l

## Quick Start

### 1. SSH to the GPU server
```bash
ssh -l <your_username> -J <your_username>@knuckles.cs.ucl.ac.uk zander-l.cs.ucl.ac.uk
```

### 2. Create workspace in /tmp (297GB available)
```bash
mkdir -p /tmp/<your_username>_steering
cd /tmp/<your_username>_steering
```

### 3. Create Python virtual environment
```bash
python3 -m venv steering_env
```

### 4. Create and run setup script
```bash
# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash
source steering_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy tqdm scikit-learn scipy
EOF

# Run it
bash setup.sh
```

### 5. Clone or copy your code
```bash
# Option A: Copy from local
# On your local machine:
rsync -avz --exclude='vectors_gpu' --exclude='*.pyc' --exclude='__pycache__' \
    /path/to/comp0087_snlp_cwk/ \
    <username>@knuckles.cs.ucl.ac.uk:/tmp/<username>_steering/comp0087_snlp_cwk/

# Option B: Clone from GitHub (if public)
git clone https://github.com/asaeed17/comp0087_snlp_cwk.git
```

### 6. Run experiments
```bash
# Activate environment
bash -c 'source steering_env/bin/activate && cd comp0087_snlp_cwk && python extract_vectors.py --models qwen2.5-3b'
```

## System Specs
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER (16.7GB VRAM)
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **Python**: 3.9.25
- **Available space in /tmp**: 297GB

## Running Steering Experiments

### Extract vectors (if not already done)
```bash
bash -c 'source steering_env/bin/activate && cd comp0087_snlp_cwk && \
    python extract_vectors.py --models qwen2.5-3b --dimensions firmness strategic_concession_making'
```

### Run dose-response validation
```bash
bash -c 'source steering_env/bin/activate && cd comp0087_snlp_cwk && \
    python dose_response_validation.py --stage generate --model qwen2.5-3b'
```

### Run negotiation games with proper controls
```bash
# Single-role experiment (recommended)
bash -c 'source steering_env/bin/activate && cd comp0087_snlp_cwk && \
    python apply_steering.py --model qwen2.5-3b --dimension firmness \
    --alpha 5 --layers 16 --steered_role buyer --num_samples 50'
```

## Important Notes

1. **Shell**: The server uses csh/tcsh by default. Use `bash` for running scripts.

2. **VirtualBox errors**: Ignore VBoxManage errors - they don't affect PyTorch.

3. **Clean up**: /tmp is shared space. Clean up large files after experiments:
   ```bash
   rm -rf /tmp/<your_username>_steering
   ```

4. **Long-running jobs**: Use `nohup` for jobs that might take hours:
   ```bash
   nohup bash -c 'source steering_env/bin/activate && python your_script.py' > output.log 2>&1 &
   ```

5. **Check GPU usage**: Monitor GPU memory during runs:
   ```bash
   nvidia-smi
   ```

## Troubleshooting

### Module not found errors
Make sure to activate the virtual environment:
```bash
bash -c 'source steering_env/bin/activate && python your_script.py'
```

### CUDA out of memory
Reduce batch size or use smaller models (qwen2.5-1.5b instead of 3b/7b).

### Permission denied in /tmp
Create your own subdirectory: `/tmp/<username>_<project>`

## Example Test Script

Create `test_gpu.py`:
```python
import torch
import sys

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    # Test computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f'GPU computation successful!')
```

Run it:
```bash
bash -c 'source steering_env/bin/activate && python test_gpu.py'
```