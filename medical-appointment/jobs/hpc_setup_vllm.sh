#!/bin/bash
# One-time environment setup for E10 (synthetic data generation) on the DTU
# HPC LOGIN node. Separate venv from hpc_setup.sh's (.venv) because vLLM
# pins its own torch/transformers/xformers versions that can conflict with
# the fine-tuning stack.
#
#   ssh <user>@login.hpc.dtu.dk
#   cd /work3/s234812/nordic_cup/medical-appointment   # after hpc_setup.sh has been run once
#   bash jobs/hpc_setup_vllm.sh
#   bsub < jobs/synthetic_generate.lsf

set -euo pipefail
cd /work3/s234812/nordic_cup/medical-appointment

# zhome (the default ~/.cache location) has a small per-user quota that a
# 32B-model download blows straight through. Redirect to /work3 instead;
# synthetic_generate.lsf sets the same HF_HOME so the job finds this cache.
export HF_HOME=/work3/s234812/nordic_cup/medical-appointment/.hf_cache
export PIP_CACHE_DIR=/work3/s234812/nordic_cup/medical-appointment/.pip_cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

module load python3/3.11.9
rm -rf .venv-vllm  # avoid mixing python versions if an old venv is left from a prior run
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
unset PYTHONPATH  # module load leaves a PYTHONPATH that shadows the venv's own packages
pip install --upgrade pip

# vLLM ships its own CUDA-matched torch; let it manage that instead of
# pre-installing torch separately (unlike hpc_setup.sh).
pip install vllm openai

python -c "import vllm; print('vllm', vllm.__version__)"

echo "Setup done. Submit synthetic generation with: bsub < jobs/synthetic_generate.lsf"
