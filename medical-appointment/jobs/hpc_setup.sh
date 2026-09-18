#!/bin/bash
# One-time environment setup on the DTU HPC LOGIN node (not submitted as a job).
# Run this once before submitting jobs/finetune.lsf.
#
#   ssh <user>@login.hpc.dtu.dk
#   git clone https://github.com/XinmingDavidBai/Nordic-AI-Cup-2026.git   # or: rsync, see NEXT_STEPS.md
#   cd Nordic-AI-Cup-2026
#   git checkout 1-challenge-3-medical-appointment
#   bash medical-appointment/jobs/hpc_setup.sh
#   bsub < medical-appointment/jobs/finetune.lsf
#
# Pre-downloads the base model on the login node (which definitely has
# internet) so the training job on a compute node does not depend on it.

set -euo pipefail
cd /work3/s234812/nordic_cup/medical-appointment

# zhome (the default location for ~/.cache) has a small per-user quota that a
# ~6GB model download blows through. Redirect both the HF and pip caches to
# /work3, which has hundreds of TB free. Must be set before ANY pip install
# or model download, and finetune.lsf sets the same HF_HOME so the job finds
# what got cached here.
export HF_HOME=/work3/s234812/nordic_cup/medical-appointment/.hf_cache
export PIP_CACHE_DIR=/work3/s234812/nordic_cup/medical-appointment/.pip_cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

module load python3/3.11.9

rm -rf .venv  # avoid mixing python versions if an old venv is left from a prior run
python3 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH  # module load leaves a PYTHONPATH that shadows the venv's own packages
pip install --upgrade pip

# Install CUDA torch FIRST so the other installs do not overwrite it with a
# CPU wheel. gpua100/gpua10/gpul40s nodes run CUDA 12.x; cu124 covers 12.4+.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# bitsandbytes intentionally omitted: finetune_train.py loads the base model
# in plain bf16 (a 3B model easily fits an A100 without 4/8-bit quantization),
# and bitsandbytes' install can be finicky on clusters -- skip the extra risk.
# Pinned to <5: transformers 5.x is a major version bump that jobs/finetune_train.py
# and its trl/SFTTrainer API were never actually run against (only smoke-tested
# locally on 4.57.6) -- an untested API break here would only surface after
# however long gpua100's queue takes, so don't risk finding out the hard way.
pip install "transformers>=4.57,<5" accelerate peft trl datasets \
    sentencepiece protobuf huggingface_hub

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

mkdir -p logs checkpoints

echo "Pre-downloading unsloth/Llama-3.2-3B-Instruct (ungated mirror of the Meta weights)..."
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "unsloth/Llama-3.2-3B-Instruct"
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)
print("cached", name)
PY

echo "Setup done. Submit fine-tuning with: bsub < medical-appointment/jobs/finetune.lsf"
