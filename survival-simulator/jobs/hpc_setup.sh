#!/bin/bash
# One-time environment setup on the DTU HPC LOGIN node (not submitted as a job).
# Run this once before submitting train.lsf.
#
#   ssh <user>@login.hpc.dtu.dk
#   git clone https://github.com/XinmingDavidBai/Nordic-AI-Cup-2026.git
#   cd Nordic-AI-Cup-2026
#   git checkout RL_murphynation
#   bash survival-simulator/jobs/hpc_setup.sh
#   bsub < survival-simulator/jobs/train.lsf

set -euo pipefail
cd /work3/<your_dtu_id>/nordic_cup/survival-simulator

# Load a recent Python (DTU HPC default may be too old)
module load python3/3.11.9

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# CPU-only: evolve.py evaluates episodes in a multiprocessing.Pool, no GPU/torch involved.
pip install -r requirements.txt

python -c "from src.core import SimulationCore; SimulationCore(seed=0); print('simulator imports OK')"

echo "Setup done. Submit the training job with: bsub < jobs/train.lsf"
