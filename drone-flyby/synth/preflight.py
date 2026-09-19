"""Input checks for the training scripts (jobs/train.lsf, Train/train_local.sh).

    python3 -m synth.preflight synth_v3                 # all checks
    python3 -m synth.preflight synth_v3 --check recipe  # only: is a composed set stale?
    python3 -m synth.preflight synth_v3 --check inputs  # only: pinned backgrounds / cut-outs present?

Prints one line per problem (nothing when ready) and exits 1 if there is any.
Standard library only, so it runs without the venv:

- a composed datasets/<name> must match the current recipe (compose.RECIPE, or
  'v2' for synth_v2); an older one would be trained on silently otherwise;
- if it still has to be composed: every pinned background must be on disk.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'datasets' / 'synth_assets'


def recipe_problems(name: str):
    expected = 'v2' if name == 'synth_v2' else \
        re.search(r"^RECIPE = '([^']+)'", (ROOT / 'synth' / 'compose.py').read_text(encoding='utf-8'), re.M).group(1)
    manifest = ROOT / 'datasets' / name / 'manifest.json'
    if manifest.is_file():
        recipe = json.loads(manifest.read_text(encoding='utf-8')).get('recipe', 'v2' if name == 'synth_v2' else 'older')
        if recipe != expected:
            yield (f'datasets/{name} was composed with recipe {recipe!r}, the code now makes {expected!r}: '
                   f'delete datasets/{name} so it is composed again')


def input_problems(name: str):
    if (ROOT / 'datasets' / name / 'manifest.json').is_file():
        return      # composed already; recipe_problems says whether it is current
    pinned = ROOT / 'synth' / 'backgrounds_pinned.json'
    if not pinned.is_file():
        yield 'synth/backgrounds_pinned.json missing: git pull'
        return
    want = json.loads(pinned.read_text(encoding='utf-8'))['images']
    if name == 'synth_v2':
        want = [e for e in want if not e.get('tag')]
    missing = [e['file'] for e in want if not (ASSETS / 'backgrounds' / e['file']).is_file()]
    if missing or not (ASSETS / 'backgrounds' / 'backgrounds.json').is_file():
        yield (f'{len(missing)} of {len(want)} pinned backgrounds missing in datasets/synth_assets/backgrounds: '
               'python -m synth.fetch_backgrounds --pinned (needs internet), or copy them from the laptop')
    if not (ASSETS / 'cutouts' / 'cutouts.json').is_file() and not (ASSETS / 'mobile_sam.pt').is_file():
        yield 'no cut-out bank (datasets/synth_assets/cutouts, committed: git pull) and no SAM checkpoint to build one'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--check', choices=('all', 'recipe', 'inputs'), default='all')
    args = parser.parse_args()
    found = []
    if args.check in ('all', 'recipe'):
        found += list(recipe_problems(args.name))
    if args.check in ('all', 'inputs'):
        found += list(input_problems(args.name))
    for line in found:
        print(line)
    sys.exit(1 if found else 0)
