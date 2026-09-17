"""Evaluate segment-pick rules combining the LLM citation with retrieval sims.

Uses pipeline_results_refine.json (LLM picks) + fresh nomic sims (no LLM calls).
"""
import json, os, sys
import numpy as np, requests
D = os.path.dirname(__file__)
cache = json.load(open(os.path.join(D, 'words_cache.json')))
res = json.load(open(os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else 'pipeline_results_refine.json')))


def embed(texts):
    r = requests.post('http://localhost:11434/api/embed', json={'model': 'nomic-embed-text', 'input': texts}, timeout=120)
    return np.array(r.json()['embeddings'])


seg_emb = {}
for sid, segs in cache.items():
    e = embed([f'search_document: {s["text"]}' for s in segs])
    seg_emb[sid] = e / np.linalg.norm(e, axis=1, keepdims=True)

pos = [r for r in res if r['gold'] is not None and r['said']]
for r in pos:
    segs = cache[r['sid']]
    if r.get('chosen_idx') is None and r['span']:
        r['chosen_idx'] = next((i for i, s in enumerate(segs) if s['start'] - 1e-3 <= r['span'][0] and r['span'][1] <= s['end'] + 1e-3), None)
    q = embed([f'search_query: {r["question"]}'])[0]
    q /= np.linalg.norm(q)
    r['sims'] = (seg_emb[r['sid']] @ q).tolist()
pos = [r for r in pos if r['chosen_idx'] is not None]
print('n', len(pos))


def evaluate(name, rule):
    right = sum(rule(r) == r['oracle_idx'] for r in pos)
    print(f'  {name:<40} {right}/{len(pos)}')


def llm_rank(r):
    order = sorted(r['top_idx'], key=lambda i: -r['sims'][i])
    return order.index(r['chosen_idx']) if r['chosen_idx'] in order else 99


evaluate('llm pick', lambda r: r['chosen_idx'])
evaluate('retrieval best', lambda r: r['best_idx'])
evaluate('earlier of both', lambda r: min(r['chosen_idx'], r['best_idx']))
for k in (1, 2, 3):
    evaluate(f'llm if retrieval rank<={k} else best', lambda r, k=k: r['chosen_idx'] if llm_rank(r) <= k else r['best_idx'])
for m in (0.02, 0.04, 0.06, 0.1):
    evaluate(f'llm unless best sim > llm sim + {m}', lambda r, m=m: r['chosen_idx'] if r['sims'][r['chosen_idx']] + m >= r['sims'][r['best_idx']] else r['best_idx'])
for m in (0.02, 0.04):
    evaluate(f'earlier-of-both unless sim gap > {m}', lambda r, m=m: min(r['chosen_idx'], r['best_idx']) if abs(r['sims'][r['chosen_idx']] - r['sims'][r['best_idx']]) <= m else max((r['chosen_idx'], r['best_idx']), key=lambda i: r['sims'][i]))
# sim-weighted earliest: among top-k segments whose sim >= max_sim - m, take the earliest
for m in (0.02, 0.04, 0.06):
    evaluate(f'earliest top-k within {m} of max sim', lambda r, m=m: min(i for i in r['top_idx'] if r['sims'][i] >= max(r['sims'][j] for j in r['top_idx']) - m))
# llm pick, but if an earlier segment within m sim of the pick exists among top-k, take it
for m in (0.02, 0.04, 0.06):
    evaluate(f'llm, or earlier top-k within {m} of llm sim', lambda r, m=m: min(i for i in r['top_idx'] if r['sims'][i] >= r['sims'][r['chosen_idx']] - m and i <= r['chosen_idx']))
