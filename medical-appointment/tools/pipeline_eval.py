"""Offline replay of retrieve+ask on cached whisper segments (no audio/whisper).

Writes per-question records so selection failures can be analysed.
"""
import csv, json, os, sys, time
sys.path.insert(0, os.getcwd())
import example  # noqa: E402  (loads whisper once; unused here)

if os.getenv('PROMPT_FILE'):
    example._PROMPT = open(os.getenv('PROMPT_FILE')).read()
    print('using prompt from', os.getenv('PROMPT_FILE'))

CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
OUT = os.path.join(os.path.dirname(__file__), sys.argv[1] if len(sys.argv) > 1 else 'pipeline_results.json')
cache = json.load(open(CACHE))
rows = list(csv.DictReader(open('data/question_train.csv')))


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


by_sample = {}
for r in rows:
    by_sample.setdefault(r['transcript_id'], []).append(r)

records = []
t0 = time.time()
for sid, qs in sorted(by_sample.items()):
    segments = [dict(s) for s in cache[sid]]
    emb = example.embed_segments(segments)
    for r in qs:
        gold = None
        if r['evidence_start'] and r['evidence_end']:
            gold = (float(r['evidence_start']), float(r['evidence_end']))
        top, best_idx = example.retrieve(segments, r['question'], seg_embeddings=emb)
        top_idx = [i for i, _ in top]
        answer, span = example.ask(segments, r['question'], seg_embeddings=emb)
        rec = {
            'qid': r['question_id'], 'sid': sid, 'type': r['question_type'],
            'question': r['question'], 'want': r['answer'] == 'yes', 'said': answer,
            'span': span, 'gold': gold, 'top_idx': top_idx, 'best_idx': best_idx,
        }
        if gold is not None:
            oracle = max(range(len(segments)), key=lambda i: iou(gold, (segments[i]['start'], segments[i]['end'])))
            rec['oracle_idx'] = oracle
            rec['oracle_iou'] = iou(gold, (segments[oracle]['start'], segments[oracle]['end']))
            rec['oracle_in_topk'] = oracle in top_idx
            rec['chosen_idx'] = next((i for i, s in enumerate(segments) if span and s['start'] - 1e-3 <= span[0] and span[1] <= s['end'] + 1e-3), None)
            rec['iou'] = iou(gold, span) if span else 0.0
        records.append(rec)
    print(f'{sid}: done ({time.time()-t0:.0f}s)', flush=True)

json.dump(records, open(OUT, 'w'), indent=1)
pos = [x for x in records if x['gold'] is not None]
acc = sum(x['said'] == x['want'] for x in records) / len(records)
mt = sum(x['iou'] for x in pos) / len(pos)
yes = [x for x in pos if x['said']]
print(f'\naccuracy {acc:.3f}  mean tIoU {mt:.3f}  score {0.4*acc+0.6*mt:.3f}')
print(f'positives answered yes {len(yes)}/{len(pos)}; tIoU when yes {sum(x["iou"] for x in yes)/len(yes):.3f}')
print(f'oracle seg in top-k: {sum(x["oracle_in_topk"] for x in pos)}/{len(pos)}')
print(f'best_idx == oracle: {sum(x["best_idx"]==x["oracle_idx"] for x in pos)}/{len(pos)}')
print(f'chosen == oracle (of yes): {sum(x["chosen_idx"]==x["oracle_idx"] for x in yes)}/{len(yes)}')
print(f'mean oracle-seg IoU: {sum(x["oracle_iou"] for x in pos)/len(pos):.3f}')
