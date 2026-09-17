"""Given the oracle segment per gold span, test span-refinement strategies.

Isolates localization quality from LLM yes/no + segment choice.
"""
import csv, json, os, sys
import numpy as np, requests

CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = [r for r in csv.DictReader(open('data/question_train.csv'))
        if r['evidence_start'] and r['evidence_end'] and r['transcript_id'] in cache]


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def embed(texts):
    r = requests.post('http://localhost:11434/api/embed',
                      json={'model': 'nomic-embed-text', 'input': texts}, timeout=60)
    return np.array(r.json()['embeddings'])


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
    cands = [(seg['start'], seg['end'], seg['text'], 'seg')]
    if len(cl) > 1:
        cands += [(s, e, t, 'clause') for s, e, t in cl]
        cands += [(cl[i][0], cl[i + 1][1], cl[i][2] + ' ' + cl[i + 1][2], 'pair') for i in range(len(cl) - 1)]
    return cands


res = {}
def add(k, v): res.setdefault(k, []).append(v)

for r in rows:
    g = (float(r['evidence_start']), float(r['evidence_end']))
    segs = cache[r['transcript_id']]
    si = max(range(len(segs)), key=lambda i: iou(g, (segs[i]['start'], segs[i]['end'])))
    seg = segs[si]
    cands = candidates(seg)
    # neighbour extensions for gold spans straddling a segment boundary
    nb = []
    if si + 1 < len(segs):
        nxt = segs[si + 1]; ncl = clauses(nxt['words'])
        if ncl:
            nb.append((seg['start'], ncl[0][1], seg['text'] + ' ' + ncl[0][2], 'nb'))
        nb.append((seg['start'], nxt['end'], seg['text'] + ' ' + nxt['text'], 'nb'))
    if si > 0:
        prv = segs[si - 1]; pcl = clauses(prv['words'])
        if pcl:
            nb.append((pcl[-1][0], seg['end'], pcl[-1][2] + ' ' + seg['text'], 'nb'))
    cands_nb = cands + nb
    add('whole_seg', iou(g, (seg['start'], seg['end'])))
    add('oracle_cand', max(iou(g, (c[0], c[1])) for c in cands))
    add('oracle_cand_nb', max(iou(g, (c[0], c[1])) for c in cands_nb))
    q = embed([f'search_query: {r["question"]}'])[0]
    cn = embed([f'search_document: {x[2]}' for x in cands_nb])
    sn = cn @ q / (np.linalg.norm(cn, axis=1) * np.linalg.norm(q))
    bn = cands_nb[int(np.argmax(sn))]
    add('best_sim_nb', iou(g, (bn[0], bn[1])))
    if len(cands) == 1:
        for k in ('best_sim', 'best_sim_lenprior', 'clause_only', 'refine_if_long4', 'refine_if_long5'):
            add(k, iou(g, (seg['start'], seg['end'])))
        continue
    q = embed([f'search_query: {r["question"]}'])[0]
    c = embed([f'search_document: {x[2]}' for x in cands])
    sims = c @ q / (np.linalg.norm(c, axis=1) * np.linalg.norm(q))
    best = cands[int(np.argmax(sims))]
    add('best_sim', iou(g, (best[0], best[1])))
    # mild preference for longer units (short clauses are noisy)
    lens = np.array([x[1] - x[0] for x in cands])
    adj = sims + 0.01 * np.minimum(lens, 4.0)
    b2 = cands[int(np.argmax(adj))]
    add('best_sim_lenprior', iou(g, (b2[0], b2[1])))
    cl_idx = [i for i, x in enumerate(cands) if x[3] == 'clause']
    bc = cands[cl_idx[int(np.argmax(sims[cl_idx]))]]
    add('clause_only', iou(g, (bc[0], bc[1])))
    seglen = seg['end'] - seg['start']
    for T in (4, 5):
        add(f'refine_if_long{T}', iou(g, (best[0], best[1])) if seglen > T else iou(g, (seg['start'], seg['end'])))

print(f'n={len(rows)} gold spans over {len(cache)} samples')
for k, v in res.items():
    print(f'  {k:<20} {np.mean(v):.3f}')
