"""Pool the per-fold held-out records written by jobs/rl_grpo_train.py into a
leave-conversation-out CV score, next to E9's (results_E9_cv.json, 0.726) and
the init policy's own HF-greedy numbers (the apples-to-apples baseline: same
decoding path, same reward table).

Usage (from medical-appointment/):
    python3 tools/rl_pool.py rl_results/r1            # pooled init vs final, per fold
    python3 tools/rl_pool.py rl_results/r1 --write results_RL_r1_cv.json   # pipeline_eval_cv-style file for pickrule.py
"""
import glob
import json
import os
import sys

run_dir = sys.argv[1]
write = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else None


def score(records):
    pos = [x for x in records if x['gold'] is not None]
    acc = sum(x['said'] == x['want'] for x in records) / len(records)
    mt = sum(x['iou'] for x in pos) / len(pos) if pos else 0.0
    yes = [x for x in pos if x['said']]
    return {
        'n': len(records), 'acc': round(acc, 3), 'tiou': round(mt, 3), 'score': round(0.4 * acc + 0.6 * mt, 4),
        'FN': sum(1 for x in pos if not x['said']), 'FP': sum(1 for x in records if x['gold'] is None and x['said']),
        'oracle_pick': f"{sum(1 for x in yes if x.get('cited', x.get('chosen_idx')) == x.get('oracle_idx'))}/{len(yes)}",
    }


def load(kind):
    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, f'fold*_{kind}.json'))):
        recs += json.load(open(f))
    return recs


init, final = load('init'), load('final')
e9_path = os.path.join(os.path.dirname(__file__), 'results_E9_cv.json')
e9 = json.load(open(e9_path)) if os.path.isfile(e9_path) else []

folds = sorted(set(x['fold'] for x in init + final))
print(f'{run_dir}: folds done init={sorted(set(x["fold"] for x in init))} final={sorted(set(x["fold"] for x in final))}')
print(f'{"fold":>6} | {"E9 (Mac/ollama Q4)":>22} | {"init (HF bf16 greedy)":>22} | {"RL final (HF bf16 greedy)":>26}')
for f in folds:
    row = [f]
    for recs in (e9, init, final):
        fr = [x for x in recs if x['fold'] == f]
        row.append(f"{score(fr)['score']:.4f}" if fr else '-')
    print(f'{row[0]:>6} | {row[1]:>22} | {row[2]:>22} | {row[3]:>26}')
for name, recs in (('E9 pooled (Mac)', e9), ('init pooled (HF)', init), ('RL final pooled (HF)', final)):
    if recs:
        print(f'{name:>22}: {score(recs)}')

if write and final:
    out = os.path.join(os.path.dirname(__file__), write)
    json.dump(final, open(out, 'w'), indent=1)
    print(f'wrote {out} (compare: python3 tools/pickrule.py tools/results_E9_cv.json {out})')
