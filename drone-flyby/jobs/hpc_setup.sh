#!/bin/bash
# One-time setup on the DTU HPC LOGIN node (not submitted as a job). Compute
# nodes have no internet, so everything the training job downloads is fetched
# here. Safe to re-run: finished steps are skipped or cheap.
#
#   ssh <user>@login.hpc.dtu.dk
#   mkdir -p /work3/<user> && cd /work3/<user>      # scratch: the datasets need ~10 GB, too much for $HOME
#   git clone https://github.com/XinmingDavidBai/Nordic-AI-Cup-2026.git
#   cd Nordic-AI-Cup-2026 && git checkout Eriks-challenge-2-drone-flyby
#   cd drone-flyby
#   bash jobs/hpc_setup.sh
#   bsub < jobs/train.lsf

set -euo pipefail
cd "$(dirname "$0")/.."      # drone-flyby/, wherever the clone is

# Load a recent Python (DTU HPC default may be too old); train.lsf loads the same.
module load python3/3.11.9

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install CUDA torch FIRST so requirements.txt does not overwrite it with a CPU wheel.
# gpua100 nodes run CUDA 12.x; cu124 covers 12.4+.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# LSF needs the -o/-e directory to exist before the job is dispatched.
mkdir -p logs

# Downloads ultralytics would otherwise attempt mid-job, on an offline node:
# the base checkpoint (keep in sync with MODEL= in train.lsf), the AMP-check
# checkpoint, the plot font, and SAM (only used if the committed cut-out bank
# is missing).
python - <<'EOF'
from ultralytics import SAM, YOLO
from ultralytics.utils import WEIGHTS_DIR
from ultralytics.utils.checks import check_font

YOLO('yolo11s.pt')
YOLO(WEIGHTS_DIR / 'yolo26n.pt')
check_font('Arial.ttf')
SAM('datasets/synth_assets/mobile_sam.pt')
EOF

# Synthetic backgrounds (~130 MB, not in git): exactly the pinned set the laptop
# composed synth_v2 from. Skip this by copying datasets/synth_assets/backgrounds
# over from the laptop first (see synth/README.md); images already present are kept.
python -m synth.fetch_backgrounds --pinned

echo "Setup done. Check with: PREFLIGHT_ONLY=1 bash jobs/train.lsf ; submit with: bsub < jobs/train.lsf"
