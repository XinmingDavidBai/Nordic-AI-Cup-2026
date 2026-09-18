"""E8 step 2: fine-tune the cross-encoder span-refinement picker on the 195
gold spans, 5-fold CV by conversation (same fold split as
tools/build_finetune_dataset.py: sorted conversation_id, index % 5).

Each candidate (whole segment / clause / clause-pair of the oracle segment,
same set as tools/refine_eval.py) is labelled with its IoU against gold and
trained as a regression target (MSE). A fresh model is trained per fold on
the other 4 folds and evaluated -- by picking the candidate with the highest
predicted score -- on the held-out fold, so the CV-pooled `best_sim` is a
genuine out-of-conversation number, comparable to refine_eval.py's zero-shot
0.7082 and tools/refine_eval_crossenc.py's zero-shot cross-encoder number.

No LLM calls; CPU is fine (22M-parameter model, ~1000 training pairs/fold).

Usage: python3 tools/finetune_crossenc.py
"""
import csv, json, os
os.environ.setdefault('USE_TF', '0')
import numpy as np
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

MODEL_NAME = os.getenv('CROSSENC_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = [r for r in csv.DictReader(open('data/question_train.csv'))
        if r['evidence_start'] and r['evidence_end'] and r['transcript_id'] in cache]

sids = sorted(cache)
fold_of = {sid: i % 5 for i, sid in enumerate(sids)}


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def clauses(words, pause=0.6):
    units, cur = [], []
    for w in words:
        if cur and w['start'] - cur[-1]['end'] > pause:
            units.append(cur); cur = []
        cur.append(w)
        if w['word'].strip().endswith(('.', '?', '!', ',', ';')):
            units.append(cur); cur = []
    if cur:
        units.append(cur)
    return [(u[0]['start'], u[-1]['end'], ''.join(w['word'] for w in u).strip()) for u in units]


def candidates(seg):
    cl = clauses(seg['words'])
    cands = [(seg['start'], seg['end'], seg['text'])]
    if len(cl) > 1:
        cands += [(s, e, t) for s, e, t in cl]
        cands += [(cl[i][0], cl[i + 1][1], cl[i][2] + ' ' + cl[i + 1][2]) for i in range(len(cl) - 1)]
    return cands


# Precompute, per question: fold, gold-segment candidates, and their IoU labels.
per_question = []
for r in rows:
    fold = fold_of[r['transcript_id']]
    g = (float(r['evidence_start']), float(r['evidence_end']))
    segs = cache[r['transcript_id']]
    si = max(range(len(segs)), key=lambda i: iou(g, (segs[i]['start'], segs[i]['end'])))
    seg = segs[si]
    cands = candidates(seg)
    labels = [iou(g, (c[0], c[1])) for c in cands]
    per_question.append({'fold': fold, 'question': r['question'], 'cands': cands, 'labels': labels})

cv_ious = []
for held_out in range(5):
    train_q = [q for q in per_question if q['fold'] != held_out]
    eval_q = [q for q in per_question if q['fold'] == held_out]

    examples = []
    for q in train_q:
        if len(q['cands']) == 1:
            continue  # nothing to rank within this question
        for c, lbl in zip(q['cands'], q['labels']):
            examples.append(InputExample(texts=[q['question'], c[2]], label=float(lbl)))

    model = CrossEncoder(MODEL_NAME, num_labels=1)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(train_dataloader=train_dataloader, epochs=3, warmup_steps=10, show_progress_bar=False)

    for q in eval_q:
        if len(q['cands']) == 1:
            cv_ious.append(q['labels'][0])
            continue
        scores = model.predict([(q['question'], c[2]) for c in q['cands']], show_progress_bar=False)
        best = int(np.argmax(scores))
        cv_ious.append(q['labels'][best])
    print(f'fold {held_out}: {len(eval_q)} held-out questions, running CV mean so far = {np.mean(cv_ious):.4f}', flush=True)

print(f'\nCV-pooled best_sim (fine-tuned cross-encoder): {np.mean(cv_ious):.4f}  (n={len(cv_ious)})')
print('compare against: refine_eval.py baseline 0.7082, refine_eval_crossenc.py zero-shot, E4 bar 0.72')
