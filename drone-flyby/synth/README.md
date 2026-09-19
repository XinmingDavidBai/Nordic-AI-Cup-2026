# Synthetic training data

The first full validation run (249 frames: a flight from fields over a marina
and industry into dense red-roofed suburbs) showed why this exists. It holds the
same 16 object models as helsinki, several instances per class, on quite
different ground. The helsinki-trained detector saw some of them but mostly
misnamed them, and 83 % of what the server answered was `ta-ta`. Helsinki's
`ta-ta` is a pale pink-beige blob, so the model had learnt "pinkish blob", and it
fired on orange roofs at up to 0.70 confidence. Both checkpoints we had score
**0.03 mAP50** on the held-out synthetic set (synth_v1 val), i.e. the same objects
on unseen ground.

A 1-epoch CPU fine-tune on synth_v1 (a wiring test, not a model) already reaches
0.20 mAP50 there. On the recording it finds the real hangars (0.95-0.99) and small
towers, and the roof false positives disappear. Its new systematic false positive
is `condor` on marina piers lined with boats, a hard negative to watch.

The pipeline cuts the 16 objects out of helsinki and composites them onto
varied, openly licensed aerial imagery, rendered exactly the way the evaluator
renders views.

## Data rule: validation views are never training data

Recorded validation views (`recordings/`, and any copy of them) may be used to
**evaluate** and to **inform decisions** (what terrain looks like, object
scale), never as training input, not even as unlabelled backgrounds. Enforced in
code: every synth script and `train_detector.py` (scenes and `--extra-dataset`,
checked before anything is built) pass their inputs through `synth/guard.py`,
which refuses anything named like a recording or containing recorder output
(`<n>_frame_<f>_idx_<i>.json/png`). Every manifest records
`"validation_recordings_used": false` and the checked inputs.

The same goes for the ground itself: nothing is fetched from Denmark/Oresund
(`fetch_backgrounds.EXCLUDE_BBOX`), where the recorded scene appears to be.
Public imagery of that place would be validation data by the back door, so
`compose.py` refuses such a background too.

## Steps (laptop, CPU)

```cmd
python -m synth.build_cutouts        :: SAM cut-outs of every object, ~3 min; review datasets/synth_assets/cutouts/review.jpg
python -m synth.fetch_backgrounds    :: OpenAerialMap mosaics (CC-BY 4.0), ~10 min per 40 locations, resumable
python -m synth.compose              :: datasets/synth_v2: 6000 train + 600 val views, ~10 min on 12 workers
python -m synth.compose --resume     :: finish an interrupted compose (same arguments; identical output)
```

- **Cut-outs** (`build_cutouts.py`): each object from up to 6 helsinki frames,
  segmented by MobileSAM from its box, stored at true source-pixel size. Masks
  far off their class's median coverage are dropped (SAM grabbed ground).
- **Backgrounds** (`fetch_backgrounds.py`): one tile mosaic per location at
  0.12-0.38 m/px: cities, villages, rail yards, industry, fields, forest,
  rivers, some coast. 68 locations (53 train, 15 val); ~15 % are `split: val`
  and only ever used for the synthetic val set. Attribution per image in
  `backgrounds.json`. `--prefer <regex on title>` fetches matching locations
  first (used for harbours/coast), but OpenAerialMap has few real marinas, so
  the pier false positive may need a dedicated source.
- **Compositor** (`compose.py`): per image, a source-resolution canvas for a
  random level (L0 30 %, L1 50 %, L2 20 %), background scaled to 0.12-0.26 m per
  source px, 0-8 objects (classes balanced) with any rotation (box re-fitted to
  the rotated mask), +-15 % size, colour jitter, soft drop shadows, some across
  the view edge; then INTER_AREA down to 960x540 like the evaluator. 12 % of
  images have no objects (hard negatives). Seeded per image: the same dataset
  whatever the number of workers, and `--resume` reproduces it exactly.

## Colour and lighting are not allowed to be cues

- compositor: whole-image white balance, gamma, hue and saturation shifts on
  every image, and 25 % of images greyscale (`--gray-share`, `--color-strength`);
- `train_detector.py`: `--color-aug strong` (default now: hsv_h 0.05, hsv_s 0.9,
  hsv_v 0.6, red/blue swap on 10 % of images; `default` = the old settings) and
  `--gray-share 0.2` (share of the generated helsinki TRAIN crops written
  greyscale; val untouched; 0 rebuilds exactly the old dataset).

## What is in git, what is not

- in git: the cut-out bank (`datasets/synth_assets/cutouts/`, ~2 MB) and
  `synth/backgrounds_pinned.json` (ids, splits, sizes and hashes of the 68
  backgrounds; refresh it with `python -m synth.fetch_backgrounds --pin` after
  fetching new ones);
- not in git: the background images (~130 MB) and synth_v2 itself (5.5 GB).
  Never ship synth_v2: compose it where it is used. From the same inputs and
  library versions it is byte-identical (checked on a copy of a fresh clone).
- backgrounds on a new machine, either
  `python -m synth.fetch_backgrounds --pinned` (re-downloads exactly the pinned
  set, checks sizes and hashes, reports anything that vanished upstream), or a
  copy of the laptop's folder, e.g.
  `scp -r datasets/synth_assets/backgrounds <user>@transfer.gbar.dtu.dk:<clone>/drone-flyby/datasets/synth_assets/`.

## Train (A100)

One job does it all: `jobs/train.lsf`, after the one-time `jobs/hpc_setup.sh`
on the login node (venv, cached checkpoints and font, pinned backgrounds; the
compute nodes are offline). The job checks its inputs, builds the cut-outs if
missing, composes synth_v2 if not composed yet, trains `yolo11s` on helsinki
crops + `synth_v2/train`, and scores the result on synth_v2 val. It uses
`--patience 0 --no-install` and hands over `last.pt` as the candidate, because
the helsinki-val fitness ultralytics tracks does not measure transfer and must
neither stop training nor pick the model. Details in the header of `train.lsf`.

## Evaluate and pick

```cmd
python -m synth.evaluate weights/detector.pt synth_best.pt --recording <merged recording folder> --gallery eval_gallery
```

- synthetic val (held-out locations, real labels): mAP50, per class;
- recorded validation run (evaluation only): detections per view by level and
  the class mix. One class dominating is a systematic false positive; look at
  the per-class galleries;
- helsinki: `python debug_replay.py --detector yolo --weights synth_best.pt`.

Only install a checkpoint as `weights/detector.pt` after all three look right.
