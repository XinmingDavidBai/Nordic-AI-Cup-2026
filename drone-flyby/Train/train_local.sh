#!/bin/bash
# Local T4 training job: synthetic data -> trained weights -> scores.
#
# Run from the drone-flyby/ folder:
#   cd <clone>/drone-flyby
#   bash jobs/hpc_setup.sh                  # once, on the login node (venv + cached assets)
#   PREFLIGHT_ONLY=1 bash jobs/train_local_t4.sh   # optional: checks inputs, no GPU
#   bash jobs/train_local_t4.sh                    # full run on the local T4
#
# Tuned for a single T4 (16 GB): batch 8, workers 6, FP16 AMP, lr0 scaled
# from the default. If batch 8 OOMs mid-epoch, drop IMGSZ to 768 before
# reducing BATCH further.

set -euo pipefail

if [ ! -f train_detector.py ]; then
    echo "Run from the drone-flyby/ folder: cd <clone>/drone-flyby && bash jobs/train_local_t4.sh" >&2
    exit 1
fi
mkdir -p logs

# ---- run config (T4 16GB) ---------------------------------------------------
MODEL=yolo11s.pt
EPOCHS=100
IMGSZ=960                 # drop to 768 only if batch 8 still OOMs
BATCH=8                   # T4 16GB: start here; try 12-16 only with AMP + monitoring
WORKERS=6                 # set explicitly; T4 boxes usually have fewer cores
LR0=0.005                 # scaled from default 0.01 for batch 8 (0.01 * 8/16)
USE_SYNTH=1
SYNTH=synth_v2
SYNTH_TRAIN=6000
SYNTH_VAL=600
SYNTH_SEED=0
NAME=drone_detector_synth_t4
# ------------------------------------------------------------------------------

ASSETS=datasets/synth_assets
problems=()
[ -f .venv/bin/activate ] || problems+=("no .venv: run 'bash jobs/hpc_setup.sh' on the login node")
[ -f "$MODEL" ] || problems+=("$MODEL not cached here: run 'bash jobs/hpc_setup.sh' on the login node")
[ -d src/helsinki/images ] || problems+=("src/helsinki/images missing: incomplete clone?")
if [ "$USE_SYNTH" = 1 ] && [ ! -f "datasets/$SYNTH/manifest.json" ]; then
    [ -f synth/backgrounds_pinned.json ] || problems+=("synth/backgrounds_pinned.json missing: git pull")
    [ -f "$ASSETS/backgrounds/backgrounds.json" ] || problems+=("no synthetic backgrounds in $ASSETS/backgrounds: run 'bash jobs/hpc_setup.sh' on the login node, or copy them from the laptop (synth/README.md)")
    if [ ! -f "$ASSETS/cutouts/cutouts.json" ] && [ ! -f "$ASSETS/mobile_sam.pt" ]; then
        problems+=("no cut-out bank ($ASSETS/cutouts, committed: git pull) and no SAM checkpoint to build one")
    fi
fi
if [ ${#problems[@]} -gt 0 ]; then
    echo "NOT READY:" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    exit 1
fi
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    echo "Preflight OK. Run with: bash jobs/train_local_t4.sh"
    exit 0
fi

echo "Job started $(date) on $(hostname) in $PWD"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load python3/3.11.9 2>/dev/null || true
source .venv/bin/activate

EXTRA_ARGS=()
if [ "$USE_SYNTH" = 1 ]; then
    if [ ! -f "$ASSETS/cutouts/cutouts.json" ]; then
        echo "== Building the cut-out bank ($(date))"
        python -m synth.build_cutouts --model "$ASSETS/mobile_sam.pt"
    fi
    if [ ! -f "datasets/$SYNTH/manifest.json" ]; then
        echo "== Composing datasets/$SYNTH ($(date))"
        resume=()
        [ -d "datasets/$SYNTH" ] && resume=(--resume)
        python -m synth.compose --name "$SYNTH" --train "$SYNTH_TRAIN" --val "$SYNTH_VAL" --seed "$SYNTH_SEED" \
            --workers "$WORKERS" ${resume[@]+"${resume[@]}"}
    fi
    EXTRA_ARGS=(--extra-dataset "datasets/$SYNTH/train")
fi

echo "== Training ($(date))"
python train_detector.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device 0 \
    --workers "$WORKERS" \
    --name "$NAME" \
    --patience 0 \
    --no-install \
    --amp \
    --lr0 "$LR0" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

TAG=${NAME}_$(date +%Y%m%d_%H%M%S)
cp "runs/$NAME/weights/last.pt" "weights/candidate_${TAG}.pt"
cp "runs/$NAME/weights/best.pt" "weights/candidate_${TAG}_helsinki_best.pt"

if [ "$USE_SYNTH" = 1 ]; then
    echo "== Scoring on datasets/$SYNTH val ($(date))"
    python -m synth.evaluate "weights/candidate_${TAG}.pt" "weights/candidate_${TAG}_helsinki_best.pt" \
        --synthetic "datasets/$SYNTH/data.yaml" --device 0 --out "logs/eval_${TAG}.json"
fi

echo "Job finished $(date). Candidate weights: $PWD/weights/candidate_${TAG}.pt"