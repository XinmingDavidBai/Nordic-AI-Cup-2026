"""E8 step 1: zero-shot cross-encoder as the span-refinement picker.

Same candidate set and same held-out (gold-segment-given) evaluation as
tools/refine_eval.py, but scores each (question, candidate) pair with a
pretrained cross-encoder instead of comparing nomic embeddings by cosine.
No LLM calls. Uses tools/words_cache.json, which must already reflect any
transcribe() changes (E7) — regenerate with tools/cache_words.py first if not.
"""
import csv, json, os
os.environ.setdefault('USE_TF', '0')
import numpy as np
from sentence_transformers import CrossEncoder

CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = [r for r in csv.DictReader(open('data/question_train.csv'))
        if r['evidence_start'] and r['evidence_end'] and r['transcript_id'] in cache]

MODEL_NAME = os.getenv('CROSSENC_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
model = CrossEncoder(MODEL_NAME)


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


ious = []
for r in rows:
    g = (float(r['evidence_start']), float(r['evidence_end']))
    segs = cache[r['transcript_id']]
    si = max(range(len(segs)), key=lambda i: iou(g, (segs[i]['start'], segs[i]['end'])))
    seg = segs[si]
    cands = candidates(seg)
    if len(cands) == 1:
        ious.append(iou(g, (seg['start'], seg['end'])))
        continue
    pairs = [(r['question'], c[2]) for c in cands]
    scores = model.predict(pairs)
    best = cands[int(np.argmax(scores))]
    ious.append(iou(g, (best[0], best[1])))

print(f'n={len(rows)} model={MODEL_NAME}')
print(f'  zero_shot_crossenc  best_sim={np.mean(ious):.4f}')
