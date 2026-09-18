#!/bin/bash
# One-time setup for the RL-tIoU branch's own HPC workspace (login node, not a job).
# Idempotent. Run AFTER rsyncing this branch's medical-appointment/ to
# /work3/s234812/nordic_cup_rl/medical-appointment (see RL_TIOU_NOTES.md).
#
# Reuses the leaderboard branch's already-built venv and HF cache read-only
# (trl 1.13 / peft 0.21 / torch 2.6+cu124, base model cached) rather than
# spending 30+ min and 5GB rebuilding an identical venv; snapshots E9's fold
# adapters so the other session can't change them under a running RL job.
set -euo pipefail
SHARED=/work3/s234812/nordic_cup/medical-appointment
RL=/work3/s234812/nordic_cup_rl/medical-appointment
cd "$RL"
mkdir -p logs rl_results checkpoints_rl checkpoints_e9

for d in fold0 fold1 fold2 fold3 fold4 final; do
    if [ ! -f "checkpoints_e9/$d/adapter_model.safetensors" ]; then
        cp -r "$SHARED/checkpoints/$d" "checkpoints_e9/$d"
        rm -rf "checkpoints_e9/$d/trainer_state"
        echo "snapshotted E9 adapter $d"
    fi
done

module load python3/3.11.9
source $SHARED/.venv/bin/activate
unset PYTHONPATH
export HF_HOME=$SHARED/.hf_cache HF_HUB_OFFLINE=1
python - <<'PY'
import trl, peft, torch, transformers
print('trl', trl.__version__, 'peft', peft.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('unsloth/Llama-3.2-3B-Instruct')  # must resolve from cache, offline
print('base tokenizer resolves from the shared cache: ok')
import json, os
t = json.load(open('tools/rl_reward_table.json')); print('reward table entries:', len(t))
PY
echo "Setup done. Submit with: bsub < jobs/rl_grpo.lsf   (env RUN=..., FOLDS=..., EXTRA=... to vary)"
