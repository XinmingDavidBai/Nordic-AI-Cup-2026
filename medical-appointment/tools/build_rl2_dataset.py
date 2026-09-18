"""Stage 2 data prep (see RL_TIOU_NOTES.md section 4): the clause-level action
space. Same retrieval, same top-8, same rules text as the live prompt, but each
segment's text is shown with numbered clause markers and the model answers with
a clause range instead of leaving span refinement to the embedding argmax:

    [4] [11.9-16.4s]: (1) I need my prescriptions renewed. (2) The asthma medicine and the one for stomach acid.
    -> {"answer": true, "segment": 4, "from": 2, "to": 2}

Clauses are exactly example._clauses(words) (split at .?!,; or >0.6s pauses),
so the span for (segment, from, to) is (clause[from].start, clause[to].end),
computable from tools/words_cache.json alone -- the RL reward on the HPC is
then plain tIoU with no embedding model and no ollama.

Labels: the (segment in top-8, from <= to) candidate with max tIoU vs gold
(the oracle over the clause action space, ceiling 0.808 mean tIoU). A positive
whose oracle whole-segment is not in the top-8 is taught as a clean "no", same
convention as tools/build_finetune_dataset.py (E9).

Runs locally (needs ollama for retrieval embeddings). Output:
tools/rl2_dataset.jsonl, one record per question with system/prompt/target_json
(same field names as finetune_dataset.jsonl) plus `clauses`: {seg_idx: [[s,e],...]}.

Usage (from medical-appointment/):  python3 tools/build_rl2_dataset.py
"""
import csv, json, os, sys, time
sys.path.insert(0, os.getcwd())
import example  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'rl2_dataset.jsonl')
CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = list(csv.DictReader(open('data/question_train.csv')))

# example._PROMPT with only the citation/output lines changed. The paraphrase
# rules and the exactness rules are byte-identical to the live prompt.
_PROMPT2 = example._PROMPT.replace(
    '- If YES, give the index of the ONE segment that most directly proves it.\n'
    '- If NO, "segment" must be null.\n\n'
    'Output JSON: {{"answer": true or false, "segment": integer or null}}',
    '- If YES, give the index of the ONE segment that most directly proves it, and the range of numbered clauses (from, to) inside that segment that states the evidence -- as narrow as possible while still containing it.\n'
    '- If NO, "segment", "from" and "to" must be null.\n\n'
    'Output JSON: {{"answer": true or false, "segment": integer or null, "from": integer or null, "to": integer or null}}',
)
assert _PROMPT2 != example._PROMPT, 'prompt rewrite did not apply; example._PROMPT changed?'
# `.replace` operates on the raw template with {{ }} escaping, so keep format() semantics
_PROMPT2_TEMPLATE = _PROMPT2


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def seg_clauses(seg):
    cl = example._clauses(seg.get('words') or [])
    if not cl:
        return [(seg['start'], seg['end'], seg['text'])]
    return cl


def context_line(i, seg, cl):
    body = ' '.join(f'({k + 1}) {c[2]}' for k, c in enumerate(cl))
    return f'[{i}] [{seg["start"]:.1f}-{seg["end"]:.1f}s]: {body}'


sids = sorted(set(r['transcript_id'] for r in rows))
fold_of = {sid: i % 5 for i, sid in enumerate(sids)}
by_sample = {}
for r in rows:
    by_sample.setdefault(r['transcript_id'], []).append(r)

records = []
t0 = time.time()
for sid, qs in sorted(by_sample.items()):
    segments = [dict(s) for s in cache[sid]]
    emb = example.embed_segments(segments)
    for r in qs:
        question = example._normalize_question(r['question'])
        top, best_idx = example.retrieve(segments, question, seg_embeddings=emb)
        cl_of = {i: seg_clauses(s) for i, s in top}
        context = '\n'.join(context_line(i, s, cl_of[i]) for i, s in top)
        prompt = _PROMPT2_TEMPLATE.format(context=context, question=question)

        is_yes = r['answer'] == 'yes'
        gold = None
        target = {'answer': False, 'segment': None, 'from': None, 'to': None}
        best_iou = 0.0
        if is_yes and r['evidence_start'] and r['evidence_end']:
            gold = (float(r['evidence_start']), float(r['evidence_end']))
            oracle = max(range(len(segments)), key=lambda i: iou(gold, (segments[i]['start'], segments[i]['end'])))
            if oracle in cl_of:
                best = None
                for i, cl in cl_of.items():
                    for a in range(len(cl)):
                        for b in range(a, len(cl)):
                            v = iou(gold, (cl[a][0], cl[b][1]))
                            if best is None or v > best[0]:
                                best = (v, i, a + 1, b + 1)
                best_iou = best[0]
                target = {'answer': True, 'segment': best[1], 'from': best[2], 'to': best[3]}
            # else: oracle not shown -> unlearnable from this context, teach as a clean no (E9 convention)

        records.append({
            'question_id': r['question_id'],
            'sid': sid,
            'fold': fold_of[sid],
            'question_type': r['question_type'],
            'want': is_yes,
            'gold': list(gold) if gold else None,
            'top_idx': [i for i, _ in top],
            'best_idx': best_idx,
            'clauses': {str(i): [[round(c[0], 3), round(c[1], 3)] for c in cl] for i, cl in cl_of.items()},
            'target_iou': round(best_iou, 4),
            'system': example._SYSTEM,
            'prompt': prompt,
            'target': target,
            'target_json': json.dumps(target),
        })
    print(f'{sid}: {len(qs)} questions done ({time.time()-t0:.0f}s)', flush=True)

with open(OUT, 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')

pos = [x for x in records if x['gold']]
n_yes = sum(1 for x in records if x['target']['answer'])
print(f'\nwrote {len(records)} records to {OUT}')
print(f'  positives {len(pos)}, taught as yes {n_yes}; mean oracle tIoU over the clause action space '
      f'{sum(x["target_iou"] for x in pos)/len(pos):.4f}')
import statistics
print(f'  prompt length (chars): median {statistics.median(len(x["prompt"]) for x in records):.0f}, '
      f'max {max(len(x["prompt"]) for x in records)}; '
      f'E9 prompt median for comparison: see finetune_dataset.jsonl')
print(records[0]['prompt'][:900])
