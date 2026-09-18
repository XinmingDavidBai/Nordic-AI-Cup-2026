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
cd "$(dirname "$0")/.."   # -> drone-flyby/

# Adjust if DTU HPC's default `python3` isn't new enough. Check available
# versions with `module avail python` and load one, e.g.:
#   module load python3/3.11.9
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# CUDA torch: the requirements.txt torch entry may resolve to a CPU wheel
# depending on the index pip uses on the login node. If `python -c "import
# torch; print(torch.cuda.is_available())"` prints False after this, reinstall
# explicitly, e.g. for CUDA 12.1:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# LSF needs the -o/-e directories to exist before the job is dispatched.
mkdir -p logs

# Compute nodes (gpua100) have no internet access, so ultralytics can't
# lazily download the base checkpoint mid-job. Fetch it now, on the login
# node, into drone-flyby/ (where train.lsf's cwd will find it). Keep this
# in sync with MODEL= in train.lsf if you change which checkpoint it trains.
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

echo "Setup done. Submit the training job with: bsub < jobs/train.lsf"
