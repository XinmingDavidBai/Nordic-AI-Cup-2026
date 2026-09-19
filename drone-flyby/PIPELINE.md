# Drone flyby pipeline: how to run it, test it, debug it

Everything here runs locally. **Nothing in this code submits anything or
contacts cases.nordicaicup.com.**

## 0. The API key

> **Put your team API key in `.env`:** copy `.env.example` to `.env` and fill
> in `NAIC_API_KEY=`. (There's also a marked placeholder in
> `pipeline/config.py`, but `.env` is gitignored and the config file isn't.)

The code doesn't use the key: the evaluation service calls **your** endpoint. You
paste the key and your URL (`http://<host>:9053/predict`) into the form at
cases.nordicaicup.com when *you* decide to verify, validate or submit.
Remember the evaluation run allows **one completed attempt only**.

## 1. Setup

Python 3.12 or 3.13 (the pipeline was developed on 3.13; AMD's Windows GPU
wheels need 3.12).

```cmd
cd drone-flyby
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is the single source of truth for dependencies: add new
libraries there, not ad hoc. Plain `pip install -r requirements.txt` gives you
the **CPU** build of torch on Windows, which is enough for everything except
serious training. To train on a GPU, install the matching torch build **first**
(section 1a), then run `pip install -r requirements.txt`. Never add `-U`/`--upgrade`
to that command: it can swap your GPU torch for the CPU one.

### 1a. GPU torch (do this before `pip install -r requirements.txt`)

Check it worked afterwards with
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
ROCm builds of torch reuse the `torch.cuda` API, so `True` means the AMD GPU is
usable and `--device 0` / auto device selection just work.

**AMD Radeon on Linux (ROCm), e.g. RX 7800 XT.** ROCm 7.2.1 officially supports
the RX 7800 XT (Ubuntu 22.04/24.04, RHEL 10.1). Install AMD's Radeon driver
stack for ROCm on the host first, following
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html
then, inside the venv:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

(That index has torch 2.14.0+rocm7.2, the same torch version the pipeline was
tested with, with wheels for Python 3.12 and 3.13. AMD's own tested alternative
is torch 2.9.1 from repo.radeon.com:
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)

**AMD Radeon on Windows (ROCm for Windows, preview).** Needs Windows 11,
Python **3.12** and AMD graphics driver 26.2.2 or newer. AMD lists the gfx1101
architecture as supported but names only the RX 7700, not the RX 7800 XT (same
chip family). Expect it to work, but verify with the check above. In the venv:

```cmd
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
pip install -r requirements.txt
```

Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
(these URLs are for ROCm 7.2.1; check that page if they have moved on).

**DirectML: don't.** `torch-directml` is a preview whose last release
(0.2.5, Sep 2024) pins torch 2.4.1, and ultralytics has no DirectML device, so
YOLO training would silently run on the CPU. Use ROCm (Linux or Windows) instead.

**NVIDIA:** install the CUDA build from https://pytorch.org/get-started/locally/.

## 2. Check the harness (no model needed)

```cmd
python local_evaluator.py --oracle                  # must print 1.000
python debug_replay.py --detector gt --viz          # pipeline with a perfect detector
python run_local.py --detector gt --realtime        # same, through api.py over HTTP
```

`--detector gt` reads the local ground truth and reports only what the camera
could actually see: the visible part of each box, if that part is at least
`GT_MIN_VIEW_PIXELS` long in the transmitted image. Objects cut by the view or
frame edge are reported as their visible part, like a real detector would
(`GT_MIN_VISIBLE_FRACTION=0.5` restores the old "at least half in view" rule).
It won't run on any sequence except the local evaluator's. It lets you tune
tracking and the camera policy before a detector exists.

Reference numbers (helsinki, 25 frames, GT detector, default settings, CPU laptop):

| run | mAP@0.50 |
|---|---|
| `debug_replay --detector gt` (greedy policy) | 0.774 |
| `--policy sweep_l1` | 0.969 |
| `--policy hold_l0` | 0.875 |
| `run_local --detector gt` (HTTP, 25/25 frames) | 0.774 |
| `GT_MIN_VIEW_PIXELS=14`: hold_l0 / sweep_l1 / greedy | 0.527 / 0.831 / 0.715 |

The last row matters most. How much zooming is worth depends on how small an
object your real detector can still find at L0. Once you have a model, check
this with the real detector rather than trusting the GT numbers.

`sweep_l1`'s fixed 6-waypoint L1 patrol is the minimum needed for full-frame
coverage under the per-frame movement cap, so any one cell only gets a close
look once every 6 frames -- an object that enters and leaves inside that
window is missed no matter how the waypoints are ordered (this is exactly what
tanked `medium_plane` to 0.000 AP in the helsinki reference run: it's only on
screen for 5 frames, timed just after the patrol's last pass through that
corner). `_SWEEP_GLANCE_EVERY` in `camera_policy.py` periodically detours to a
level-0 (whole-frame) glance between L1 stops -- always a legal single-frame
move from any L1 waypoint -- to catch short-lived objects anywhere, without
changing the L1 patrol's own coverage. That's what raised sweep_l1 from 0.889
to 0.969 above.

Greedy is sensitive to small changes: one different camera choice early on can
mirror its whole patrol. For example, greedy at `POLICY_CANDIDATE_STEP` 100 /
120 / 140 scores 0.782 / 0.774 / 0.935. Compare policies over several settings,
not one run.

## 3. Train the detector

```cmd
python train_detector.py --build-only        # look at datasets/drone_yolo first
python train_detector.py                     # yolo11n, 150 epochs, imgsz 960
python train_detector.py --model yolo11s.pt --epochs 300 --batch 16 --device 0   # GPU 0 (ROCm or CUDA)
```

Training images are rendered the same way the evaluator renders views
(crop + `INTER_AREA` resize) at L0, L1 and L2: 1 L0 view, 6 L1 views and 12 L2
views per frame, 75% of them placed near objects. The dataset is built from
`src/` (in git) with a fixed `--seed`, so every machine builds the same images;
no dataset needs to be copied around.

Every run archives both its `best.pt` and `last.pt` to
`models_weights/<name>_<timestamp>_{best,last}.pt` (gitignored, local only)
alongside a `manifest.json` tracking each archived run's timestamp and
ultralytics fitness score. Only the **last 2 runs** are kept -- archiving a
3rd deletes the oldest run's pair, so 4 files total (2 runs x best+last).
Of the 2 runs' `best.pt`, whichever scored higher gets copied to
`weights/detector.pt`, regardless of which one ran more recently -- that's
what `DETECTOR_BACKEND=auto` loads, and it **is** tracked in git
(`.gitignore` only excludes `models_weights/`), so `git pull` is how a
teammate picks up the latest validated weights. `last.pt` is kept purely for
reference/resuming and never competes for that slot. If you want anything
else out of `models_weights/` (the runner-up, or a `last.pt`), grab it
directly -- that folder isn't pushed, so ask whoever trained it or pull it
over `scp`.

Before a long GPU run, do the 1-epoch smoke run and check that the checkpoint
loads on the machine that will serve it:

```cmd
python train_detector.py --max-frames 3 --epochs 1 --imgsz 320 --l1-crops 1 --l2-crops 2 --device 0 --name smoke --no-install --dataset-dir datasets/smoke
python debug_replay.py --detector yolo --weights runs/smoke/weights/best.pt
```

A full run (yolo11n, 150 epochs, 960 px) takes roughly 8-9 hours on a laptop
CPU, so use a GPU if you can. When handing a checkpoint back, include
`runs/<name>/args.yaml` and `results.csv` and the git commit it was trained
from, and keep the same `ultralytics` version (pinned in requirements.txt) on
both machines.

Watch out: helsinki has one instance per class, and validation/evaluation use
different scenes. The default val split (every 6th frame) leaks because
neighbouring frames look almost the same. Ways to get more data:
- The synthetic set (`synth/README.md`): the helsinki objects composited onto
  varied open aerial imagery, added with `--extra-dataset datasets/synth_v2/train`.
- `python api.py` records every request of a **validation** run to
  `recordings/` by default. The rules would allow training on those views, but
  team rule: they are for **evaluation and decisions only**, never training
  input, labelled or not (`synth/guard.py` refuses them). Summarise them with
  `summarize_recordings.py` and score checkpoints on them with `synth/evaluate.py`.

Then:

```cmd
python debug_replay.py --detector yolo --viz
python run_local.py --detector yolo --realtime
```

## 4. Serve it

```cmd
python api.py                 # port 9053, or set PORT
```

Startup loads the model and runs warmup inferences before the first request.
Every request logs its per-stage timing (`decode / ego / detect / track /
policy / total`).

`api.py` refuses to start unless the detector resolves to a working `yolo`
(missing weights, `gt`, `none`, `edges` or a failing warmup all stop it), because
any of those answers every real frame with a valid, empty response. Set
`ALLOW_NON_YOLO_DETECTOR=1` to start one on purpose for local debugging
(`run_local.py --detector gt` does this for you). Before spending a validation
attempt, check what is deployed:

```cmd
curl http://<your-host>:9053/api
```

`detector.backend` must be `yolo` and `detector.weights_sha256` must match
the checkpoint you meant to deploy (`certutil -hashfile weights\detector.pt SHA256`,
first 12 characters). `detector.stats` counts frames where the detector raised
or found nothing. The server also logs a `!!! DETECTOR ...` error when either
happens on several frames in a row.

**Serve on a GPU.** The detector is most of the per-frame time on a CPU, and a
round trip over 333 ms loses frames (each one scores zero). `DETECTOR_DEVICE`
empty/`auto` picks a CUDA or ROCm GPU whenever torch can use one, else Apple
MPS, else the CPU; `cpu`, `0`, `cuda:N` or `mps` force one, and a forced GPU torch
cannot see stops startup. The startup log says which device and why, and
`/api` has it under `detector.device_info` (`kind`, `how` = auto/forced/fallback,
`reason`, `gpu_name`, `torch_build`, `gpu_hardware_unused`). An auto-picked GPU
that fails the warmup inference falls back to the CPU with a `!!!` error. GPU
hardware torch cannot use (CPU-only torch, missing driver, `CUDA_VISIBLE_DEVICES`,
a container without `--gpus all`) also logs `!!! DETECTOR ON CPU` and sets
`gpu_hardware_unused: true`. On a host that must have a GPU, set
`DETECTOR_REQUIRE_GPU=1` and the server refuses to start on the CPU.

**Run exactly one server process.** The tracker, ego-motion and camera patrol
live in that process's memory, per `sequence_id`. Never use uvicorn `--workers`,
several replicas, or a load balancer without sticky sessions. Call `/api` a few
times: `process.pid` must be the same every time. A retried request (same
`request_id`) gets its earlier answer back, and an out-of-order `frame_index` is
answered from the current tracker; neither touches the state, both log a
`!!! REPEATED` / `!!! OUT-OF-ORDER` warning and are counted in
`sequences.stats`. Up to `SEQUENCE_STATES_KEPT` (4) sequences are kept, so a
stray request for another sequence cannot wipe the run in progress.

Every request is recorded to `recordings/<sequence_id>/` (view PNG plus request,
response, arrival order and process id). Keep the recording of every validation
run: it is the only way to debug one afterwards. `RECORD_DIR=` (empty) turns it
off; `run_local.py` does that for local runs.

`run_local.py` uses port 9063 by default, so it never collides with a server you
left running on 9053.

## How the pipeline works (`pipeline/`)

| file | job |
|---|---|
| `config.py` | Every knob, plus the API key slot. Anything can be overridden by an env var or `.env`. |
| `detector.py` | Backends `yolo`, `gt` (debug), `edges` (organisers' baseline), `none`. They all return **source-pixel** boxes. |
| `egomotion.py` | Ground-motion model. **Affine**, not a single shift (see below). |
| `tracker.py` | Remembers every object in source coordinates and moves it with the flow each frame. Matches, refines, votes on class, counts misses, merges duplicates, and sets output confidence. |
| `camera_policy.py` | Picks the next legal view: `greedy` (coverage/detail map + uncertain-track bonus), `sweep_l1`, `hold_l0`. |
| `predictor.py` | Glue code, per-sequence state and timings. Every stage is wrapped in a try/except, so a valid response always goes out. |
| `recorder.py` | Optional background saving of requests/responses. |

**Why affine ego-motion:** the camera isn't pointing straight down. Fitted on
the helsinki annotations, objects move about 54 px/frame at the top of the frame
and about 80 px/frame at the bottom, and they drift outward in x. A
translation-only model got GT-detector greedy mAP to 0.50; the affine model
raised it to 0.84. The model is fitted online (ridge regression toward a prior)
from phase correlation of the overlap between consecutive views, tiled into up
to 3x3 patches, plus matched tracks. Starting from zero (`EGO_PRIOR=zero`), it
learned x-expansion 6.73 and y-gradient 11.57, against 6.69 and 11.36 fitted
from the ground truth.

**Camera legality:** candidate views come from `camera_constraints`, and each
one is re-checked with `utils.describe_camera_rejection` before it's sent. An
illegal proposal is replaced with "hold". No command was refused in any local run.

## Things to try first

1. Train a real detector, then compare `hold_l0 / sweep_l1 / greedy` with it.
2. Tune the policy: `POLICY_LEVEL_DETAIL_*`, `POLICY_DETAIL_DECAY`,
   `POLICY_TRACK_BONUS`. So far greedy only uses L2 now and then.
3. Output confidence: `TRACK_COAST_HALF_LIFE`, `TRACK_RUNNER_UP_MIN_SHARE`.
   AP is ranking-based, so how confident coasted tracks are relative to
   freshly seen ones matters.
4. Latency: the detector is about 45 ms on CPU with the (untrained) yolo11n at
   960. Test with `run_local.py --realtime --simulate-latency-ms 200` to see
   how much headroom you have.
5. Check that the flight direction/flow on validation matches helsinki. The
   debug dump (`--dump-json`) logs `flow_theta` every frame.
