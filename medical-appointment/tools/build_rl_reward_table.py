"""RL-tIoU data prep: precompute, for every training question, the exact span
and tIoU the live pipeline (example.ask) would produce for EVERY possible
segment index the LLM could cite -- so that on the HPC the RL reward is a
pure table lookup of the real evaluation metric, with no ollama/embedding
model needed on the compute node.

Runs locally (needs ollama for nomic-embed-text, same as
build_finetune_dataset.py). Deterministic given tools/words_cache.json and
example.py's retrieve/refine_span/E1-union logic, which this replicates
exactly (see `span_for_citation`).

Output: tools/rl_reward_table.json, one entry per question_id:
    want          bool   gold answer
    gold          [s,e]  or null
    n_segments    int
    top_idx       list   the top-8 shown in the prompt (same as the prompt's context)
    best_idx      int    retrieval's top-1
    span_by_idx   {idx: [s,e]}  pipeline span if the LLM cites idx (only idx in top_idx)
    iou_by_idx    {idx: iou}    tIoU of that span vs gold (positives only, else 0)
    fallback_span [s,e]  span when the citation is absent/out-of-range/not in top-8
    fallback_iou  float
    oracle_idx    int    segment with max whole-segment IoU vs gold (positives)
    max_iou       float  best achievable tIoU over all citable indices

Usage (from medical-appointment/):  python3 tools/build_rl_reward_table.py
"""
import csv, json, os, sys, time
sys.path.insert(0, os.getcwd())
import example  # noqa: E402  (loads whisper once; unused here)

OUT = os.path.join(os.path.dirname(__file__), 'rl_reward_table.json')
CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = list(csv.DictReader(open('data/question_train.csv')))


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def span_for_citation(segments, question, top_indices, best_idx, llm_seg_idx, refined):
    """Replicates the tail of example.ask() for answer=True. `refined` is a
    memo of refine_span(segments[i], question) keyed by i."""
    if isinstance(llm_seg_idx, int) and 0 <= llm_seg_idx < len(segments) and llm_seg_idx in top_indices:
        if abs(llm_seg_idx - best_idx) == 1:
            lo, hi = min(llm_seg_idx, best_idx), max(llm_seg_idx, best_idx)
            return (refined[lo][0], refined[hi][1])
        center_idx = min(llm_seg_idx, best_idx)
    else:
        center_idx = best_idx
    return refined[center_idx]


by_sample = {}
for r in rows:
    by_sample.setdefault(r['transcript_id'], []).append(r)

table = {}
t0 = time.time()
for sid, qs in sorted(by_sample.items()):
    segments = [dict(s) for s in cache[sid]]
    emb = example.embed_segments(segments)
    for r in qs:
        question = example._normalize_question(r['question'])  # ask() normalises before retrieve+refine
        top, best_idx = example.retrieve(segments, question, seg_embeddings=emb)
        top_indices = [i for i, _ in top]
        gold = None
        if r['evidence_start'] and r['evidence_end']:
            gold = (float(r['evidence_start']), float(r['evidence_end']))

        # refine_span is deterministic per (question, segment); every span the
        # pipeline can emit for this question is built from these 8 (+best_idx,
        # which is always in the top-8 by construction).
        refined = {i: example.refine_span(segments[i], question) for i in set(top_indices) | {best_idx}}

        span_by_idx, iou_by_idx = {}, {}
        for i in top_indices:
            sp = span_for_citation(segments, question, set(top_indices), best_idx, i, refined)
            span_by_idx[str(i)] = [round(sp[0], 3), round(sp[1], 3)]
            iou_by_idx[str(i)] = round(iou(gold, sp), 4) if gold else 0.0
        fb = span_for_citation(segments, question, set(top_indices), best_idx, None, refined)
        entry = {
            'sid': sid,
            'type': r['question_type'],
            'want': r['answer'] == 'yes',
            'gold': list(gold) if gold else None,
            'n_segments': len(segments),
            'top_idx': top_indices,
            'best_idx': best_idx,
            'span_by_idx': span_by_idx,
            'iou_by_idx': iou_by_idx,
            'fallback_span': [round(fb[0], 3), round(fb[1], 3)],
            'fallback_iou': round(iou(gold, fb), 4) if gold else 0.0,
        }
        if gold:
            oracle = max(range(len(segments)), key=lambda i: iou(gold, (segments[i]['start'], segments[i]['end'])))
            entry['oracle_idx'] = oracle
            entry['oracle_in_topk'] = oracle in top_indices
            entry['max_iou'] = max(list(iou_by_idx.values()) + [entry['fallback_iou']])
        table[r['question_id']] = entry
    print(f'{sid}: {len(qs)} questions done ({time.time()-t0:.0f}s)', flush=True)

json.dump(table, open(OUT, 'w'), indent=1)
pos = [e for e in table.values() if e['gold']]
print(f'\nwrote {len(table)} entries to {OUT}')
print(f'positives {len(pos)}: mean max-achievable tIoU (best citable index) '
      f'{sum(e["max_iou"] for e in pos)/len(pos):.4f}; '
      f'mean tIoU if citing oracle-whole-segment idx (when in top-8) '
      f'{sum(e["iou_by_idx"].get(str(e["oracle_idx"]), 0.0) for e in pos)/len(pos):.4f}; '
      f'mean fallback (cite nothing -> best_idx) {sum(e["fallback_iou"] for e in pos)/len(pos):.4f}; '
      f'oracle in top-8 {sum(e["oracle_in_topk"] for e in pos)}/{len(pos)}')
