#!/bin/bash
set -euo pipefail

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "INITIAL DIRECTORY STRUCTURE"
echo "=========================================="
tree --dirsfirst -I "__pycache__|.git|.idea|.DS_Store"

echo ""
echo "=========================================="
echo "MACHINE SPECIFICATIONS"
echo "=========================================="
date
lscpu | grep -E "Architecture|Model name|CPU\(s\)|MHz"
echo ""
cat /etc/os-release
echo ""
free -h
df -h /

echo ""
echo "=========================================="
echo "MICROMAMBA VERSION — micromamba --version"
echo "=========================================="
micromamba --version

echo ""
echo "=========================================="
echo "VERIFYING MIXUP IMPLEMENTATION — grep utils/training_strategies.py"
echo "=========================================="
grep -n "def mixup_data\|def mixup_step\|def mixup_smoothing_step\|Beta" utils/training_strategies.py

echo ""
echo "=========================================="
echo "VERIFYING LABEL SMOOTHING IMPLEMENTATION — grep utils/training_strategies.py"
echo "=========================================="
grep -n "def label_smoothing_loss\|def smoothing_step\|scatter_\|log_softmax" utils/training_strategies.py

echo ""
echo "=========================================="
echo "VERIFYING EARLY STOPPING IMPLEMENTATION — grep utils/early_stopping.py"
echo "=========================================="
grep -n "class EarlyStopping\|deepcopy\|patience" utils/early_stopping.py

echo ""
echo "=========================================="
echo "VERIFYING CNN ARCHITECTURE — grep utils/network.py"
echo "=========================================="
grep -n "class CNN\|class ResidualBlock\|class SEBlock\|nn.Conv2d\|nn.Linear" utils/network.py

echo ""
echo "=========================================="
echo "CREATING FRESH ENVIRONMENT — micromamba create --name comp0197-pt python=3.12 -y"
echo "=========================================="
micromamba env remove -n comp0197-pt -y || true
micromamba create --name comp0197-pt python=3.12 -y

echo ""
echo "=========================================="
echo "INSTALLING PACKAGES — pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cu128"
echo "=========================================="
micromamba run -n comp0197-pt pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=========================================="
echo "PYTORCH / CUDA INFORMATION"
echo "=========================================="

micromamba run -n comp0197-pt python -u - << 'EOF'
import torch

print("Torch version:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda, flush=True)
    print("GPU count:", torch.cuda.device_count(), flush=True)
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)
else:
    print("Running on CPU")
EOF

echo ""
echo "=========================================="
echo "TASK 1 TRAINING — micromamba run -n comp0197-pt python -u task1/train.py"
echo "=========================================="
micromamba run -n comp0197-pt python -u task1/train.py

echo ""
echo "=========================================="
echo "TASK 1 GENERATED FILES"
echo "=========================================="
find task1 -type f | sort

echo ""
echo "=========================================="
echo "TASK 2 TRAINING — micromamba run -n comp0197-pt python -u task2/train.py"
echo "=========================================="
micromamba run -n comp0197-pt python -u task2/train.py

echo ""
echo "=========================================="
echo "TASK 2 GENERATED FILES"
echo "=========================================="
find task2 -type f | sort

echo ""
echo "=========================================="
echo "TASK 1 EVALUATION — micromamba run -n comp0197-pt python -u task1/task.py"
echo "=========================================="
micromamba run -n comp0197-pt python -u task1/task.py

echo ""
echo "=========================================="
echo "TASK 2 EVALUATION — micromamba run -n comp0197-pt python -u task2/task.py"
echo "=========================================="
micromamba run -n comp0197-pt python -u task2/task.py

echo ""
echo "=========================================="
echo "PNG VERIFICATION — ls -lah task1/*.png && find task2 -name '*.png'"
echo "=========================================="
ls -lah task1/*.png
find task2 -name "*.png"

echo ""
echo "=========================================="
echo "EXACT OUTPUT FILENAMES — find task1/task2 -name '*gap*.png' / 'robustness_demo.png'"
echo "=========================================="
find task1 -name "*gap*.png"
find task2 -name "robustness_demo.png"

echo ""
echo "=========================================="
echo "FINAL DIRECTORY STRUCTURE"
echo "=========================================="
tree --dirsfirst -I "__pycache__|.git|.idea|.DS_Store"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE — $(date)"
echo "=========================================="
date


END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "TOTAL RUNTIME: ${ELAPSED} seconds"
