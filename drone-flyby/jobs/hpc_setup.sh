#!/bin/bash
# One-time environment setup on the DTU HPC LOGIN node (not submitted as a job).
# Run this once before submitting train.lsf.
#
#   ssh <user>@login.hpc.dtu.dk
#   git clone https://github.com/XinmingDavidBai/Nordic-AI-Cup-2026.git
#   cd Nordic-AI-Cup-2026
#   git checkout Eriks-challenge-2-drone-flyby
#   bash drone-flyby/jobs/hpc_setup.sh
#   bsub < drone-flyby/jobs/train.lsf

set -euo pipefail
cd /work3/s234812/nordic_cup/drone-flyby

# Load a recent Python (DTU HPC default may be too old)
module load python3/3.11.9

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install CUDA torch FIRST so requirements.txt does not overwrite it with a CPU wheel.
# gpua100 nodes run CUDA 12.x; cu124 covers 12.4+.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

echo "Setup done. Submit the training job with: bsub < jobs/train.lsf"
