#!/bin/bash
# THE COMMAND (from drone-flyby/, after git pull):
#
#     bash Train/train_local.sh
#
# Fine-tune an earlier checkpoint instead of training from scratch:
#
#     INIT_WEIGHTS=weights/<checkpoint>.pt bash Train/train_local.sh
#
# Single-GPU training on a machine WITH internet (the teammate's Lightning T4
# studio): fresh clone -> candidate weights + scores, nothing else to run first.
# The script itself, each step skipped when already done:
#   1. creates .venv (CUDA torch + requirements.txt) if there is none;
#   2. downloads what is missing: base checkpoint, ultralytics' AMP-check
#      checkpoint and plot font, the pinned synthetic backgrounds (incl. the
#      residential ones), SAM only if the committed cut-out bank is missing;
#   3. composes datasets/$SYNTH, again if it was made by an older recipe;
#   4. trains on helsinki + $SYNTH/train, validating on $SYNTH/val;
#   5. copies the candidate to weights/candidate_<run>.pt and scores it on $SYNTH val.
# Everything is also written to logs/train_local_<time>.log. It runs for hours:
# in a plain terminal, start it inside tmux (or with nohup) so it survives a disconnect.
#
#   PREFLIGHT_ONLY=1 bash Train/train_local.sh   # print config + what would be fetched/composed; changes nothing
#
# Two ways to train (every setting in "run config" can be set on the command line):
#   from scratch: yolo11s.pt, 50 epochs;
#   fine-tune:    INIT_WEIGHTS replaces MODEL, 30 epochs.
#   EPOCHS=... overrides either. The epoch count is the planned schedule (learning
#   rate and the final no-mosaic epochs are laid out over it), so set it up front.
#
# Validation runs on the held-out synthetic set (background locations never trained
# on), not on helsinki frames, so best.pt and early stopping follow transfer to
# unseen ground. PATIENCE=10 ends a run that has not improved on it for 10 epochs;
# the candidate is that best.pt. Default data is synth_v3 (dark object silhouettes
# like the validation renderer, long cast shadows, backgrounds at any angle,
# red-roof residential hard negatives; synth/README.md); SYNTH=synth_v2 rebuilds
# the old set. Recorded validation views are never used (synth/guard.py).
#
# Tuned for a single T4 (16 GB): batch 8, workers 6, FP16 AMP, lr0 scaled from
# the default. If batch 8 OOMs mid-epoch, drop IMGSZ to 768 before reducing BATCH.
# (jobs/train.lsf is the DTU HPC variant; this script does not need jobs/hpc_setup.sh.)

set -euo pipefail

if [ ! -f train_detector.py ]; then
    echo "Run from the drone-flyby/ folder: cd <clone>/drone-flyby && bash Train/train_local.sh" >&2
    exit 1
fi
mkdir -p logs weights

# ---- run config (T4 16GB; each can be set from the command line) --------------
INIT_WEIGHTS=${INIT_WEIGHTS:-}   # set: fine-tune this checkpoint instead of training MODEL from scratch
MODEL=${MODEL:-yolo11s.pt}       # downloaded if missing
if [ -n "$INIT_WEIGHTS" ]; then
    MODEL=$INIT_WEIGHTS
    EPOCHS=${EPOCHS:-30}
fi
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}         # epochs without improvement on the synthetic val set; 0 = never stop early
IMGSZ=${IMGSZ:-960}              # drop to 768 only if batch 8 still OOMs
BATCH=${BATCH:-8}                # T4 16GB: start here; try 12-16 only with AMP + monitoring
WORKERS=${WORKERS:-6}            # set explicitly; T4 boxes usually have fewer cores
LR0=${LR0:-0.005}                # scaled from default 0.01 for batch 8 (0.01 * 8/16)
USE_SYNTH=${USE_SYNTH:-1}        # 0 = helsinki crops only (validates on helsinki, no early stop)
SYNTH=${SYNTH:-synth_v3}         # name, sizes, seed and recipe must match to reproduce a set exactly
SYNTH_TRAIN=6000
SYNTH_VAL=600
SYNTH_SEED=0
case "$SYNTH" in
    synth_v2) COMPOSE_ARGS=(--legacy-v2) ;;              # the old recipe, byte-identical
    *) COMPOSE_ARGS=() ;;                                # current recipe (synth/compose.py RECIPE)
esac
NAME=${NAME:-drone_detector_${SYNTH}_t4${INIT_WEIGHTS:+_ft}}
PYTHON=${PYTHON:-python3}        # only used to create .venv and for the stdlib checks
ALLOW_CPU=${ALLOW_CPU:-0}        # 1 = train even if torch sees no GPU (hours longer)
# ------------------------------------------------------------------------------

ASSETS=datasets/synth_assets
PREFLIGHT=${PREFLIGHT_ONLY:-0}
[ "$PREFLIGHT" = 1 ] || exec > >(tee -a "logs/train_local_$(date +%Y%m%d_%H%M%S).log") 2>&1

# Problems no download can fix.
problems=()
[ -z "$INIT_WEIGHTS" ] || [ -f "$INIT_WEIGHTS" ] || problems+=("INIT_WEIGHTS=$INIT_WEIGHTS does not exist (path relative to drone-flyby/)")
[ -d src/helsinki/images ] || problems+=("src/helsinki/images missing: incomplete clone?")
command -v "$PYTHON" >/dev/null || problems+=("no $PYTHON on PATH (set PYTHON=...)")
if [ ${#problems[@]} -gt 0 ]; then
    echo "NOT READY:" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    exit 1
fi

if [ "$USE_SYNTH" = 1 ]; then
    echo "Config: model $MODEL, $EPOCHS epochs (patience $PATIENCE), train helsinki + $SYNTH ${COMPOSE_ARGS[*]:-}, validate on $SYNTH val, run name $NAME"
else
    echo "Config: model $MODEL, $EPOCHS epochs, helsinki only (validates on helsinki frames, no early stop), run name $NAME"
fi

if [ "$PREFLIGHT" = 1 ]; then
    echo "Would do:"
    [ -f .venv/bin/activate ] || echo "  - create .venv and install CUDA torch + requirements.txt"
    [ -n "$INIT_WEIGHTS" ] || [ -f "$MODEL" ] || echo "  - download $MODEL"
    if [ "$USE_SYNTH" = 1 ]; then
        stale=$("$PYTHON" -m synth.preflight "$SYNTH" --check recipe || true)
        [ -z "$stale" ] || echo "  - delete the stale datasets/$SYNTH and compose it again ($stale)"
        if [ -n "$stale" ] || [ ! -f "datasets/$SYNTH/manifest.json" ]; then
            "$PYTHON" -m synth.preflight "$SYNTH" --check inputs | sed 's/^/  - fetch: /' || true
            echo "  - compose datasets/$SYNTH ($SYNTH_TRAIN + $SYNTH_VAL views, ~20-25 min with $WORKERS workers)"
        fi
    fi
    echo "  - train $EPOCHS epochs, then score weights/candidate_${NAME}_<time>.pt"
    echo "Preflight done. Run: ${INIT_WEIGHTS:+INIT_WEIGHTS=$INIT_WEIGHTS }bash Train/train_local.sh"
    exit 0
fi

echo "Job started $(date) on $(hostname) in $PWD"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- 1. environment -----------------------------------------------------------
if [ ! -f .venv/bin/activate ]; then
    echo "== Creating .venv ($(date))"
    "$PYTHON" -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    # On Linux x86-64 the PyPI torch wheel is the CUDA build. Installed FIRST so
    # requirements.txt keeps it instead of pulling another build.
    pip install torch torchvision
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [ "$ALLOW_CPU" != 1 ]; then
        echo "torch sees no GPU ($(python -c 'import torch; print(torch.__version__)')). Fix the torch build " \
             "(pip install --force-reinstall torch torchvision) or set ALLOW_CPU=1 to train on the CPU anyway." >&2
        exit 1
    fi
    echo "!!! No GPU: training on the CPU (ALLOW_CPU=1)"
fi

# ---- 2. downloads ---------------------------------------------------------------
echo "== Checking downloads ($(date))"
MODEL_TO_FETCH=$([ -z "$INIT_WEIGHTS" ] && echo "$MODEL" || true) python - <<'EOF'
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import WEIGHTS_DIR
from ultralytics.utils.checks import check_font

model = os.environ.get('MODEL_TO_FETCH')
if model and not Path(model).is_file():
    YOLO(model)                                  # base checkpoint, into drone-flyby/
if not (WEIGHTS_DIR / 'yolo26n.pt').is_file():
    YOLO(WEIGHTS_DIR / 'yolo26n.pt')             # ultralytics' AMP check loads it on the first GPU epoch
check_font('Arial.ttf')                          # training plots
EOF

EXTRA_ARGS=()
if [ "$USE_SYNTH" = 1 ]; then
    # ---- 3. synthetic data ------------------------------------------------------
    stale=$(python -m synth.preflight "$SYNTH" --check recipe || true)
    if [ -n "$stale" ]; then
        echo "== $stale; removing it ($(date))"
        rm -rf "datasets/$SYNTH"
    fi
    if [ ! -f "datasets/$SYNTH/manifest.json" ]; then
        echo "== Fetching missing pinned backgrounds ($(date))"
        python -m synth.fetch_backgrounds --pinned      # keeps what is there; fails if a pinned image is gone upstream
        if [ ! -f "$ASSETS/cutouts/cutouts.json" ]; then
            echo "== Building the cut-out bank ($(date))"
            python -m synth.build_cutouts --model "$ASSETS/mobile_sam.pt"   # downloads SAM on first use
        fi
        problems=()
        while IFS= read -r line; do problems+=("$line"); done < <(python -m synth.preflight "$SYNTH" --check inputs || true)
        if [ ${#problems[@]} -gt 0 ]; then
            printf 'NOT READY: %s\n' "${problems[@]}" >&2
            exit 1
        fi
        echo "== Composing datasets/$SYNTH ($(date))"
        resume=()
        [ -d "datasets/$SYNTH" ] && resume=(--resume)    # an earlier run was cut off mid-compose
        python -m synth.compose --name "$SYNTH" --train "$SYNTH_TRAIN" --val "$SYNTH_VAL" --seed "$SYNTH_SEED" \
            --workers "$WORKERS" ${COMPOSE_ARGS[@]+"${COMPOSE_ARGS[@]}"} ${resume[@]+"${resume[@]}"}
    fi
    # Train on helsinki (all frames) + synthetic train; validate on the synthetic
    # held-out locations, which picks best.pt and drives early stopping.
    EXTRA_ARGS=(--extra-dataset "datasets/$SYNTH/train" --val-dataset "datasets/$SYNTH/val" --val-frames none)
else
    PATIENCE=0   # helsinki val says nothing about new terrain: never stop on it
fi

# ---- 4. training ------------------------------------------------------------------
echo "== Training ($(date))"
python train_detector.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$([ "$ALLOW_CPU" = 1 ] && ! python -c 'import torch,sys; sys.exit(not torch.cuda.is_available())' && echo cpu || echo 0)" \
    --workers "$WORKERS" \
    --name "$NAME" \
    --patience "$PATIENCE" \
    --no-install \
    --amp \
    --lr0 "$LR0" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ---- 5. candidate + scores ------------------------------------------------------
TAG=${NAME}_$(date +%Y%m%d_%H%M%S)
if [ "$USE_SYNTH" = 1 ]; then
    # best.pt = best epoch on the synthetic held-out set: the candidate.
    cp "runs/$NAME/weights/best.pt" "weights/candidate_${TAG}.pt"
    cp "runs/$NAME/weights/last.pt" "weights/candidate_${TAG}_last.pt"
    echo "== Scoring on datasets/$SYNTH val ($(date))"
    python -m synth.evaluate "weights/candidate_${TAG}.pt" "weights/candidate_${TAG}_last.pt" \
        --synthetic "datasets/$SYNTH/data.yaml" --out "logs/eval_${TAG}.json"
else
    cp "runs/$NAME/weights/last.pt" "weights/candidate_${TAG}.pt"
fi

echo "Job finished $(date)."
echo "Candidate weights: $PWD/weights/candidate_${TAG}.pt"
[ "$USE_SYNTH" = 1 ] && echo "Scores:            $PWD/logs/eval_${TAG}.json (synthetic val; slightly optimistic, it also picked the epoch)"
echo "Check it against the recorded validation run on the laptop (evaluation only):"
echo "  python -m synth.evaluate weights/candidate_${TAG}.pt weights/detector.pt --recording <merged recording folder> --gallery eval_gallery"
