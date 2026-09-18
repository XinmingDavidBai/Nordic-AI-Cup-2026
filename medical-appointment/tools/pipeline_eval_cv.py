"""E9 evaluation: leave-conversation-out CV-pooled score for the fine-tuned
models. For each fold, points OLLAMA_MODEL at that fold's model and replays
ONLY that fold's held-out conversations (never the ones it was trained on),
then pools all 195 conversations' worth of results into one score comparable
to results_E3.json / results_E7.json.

Requires: the fold models already created locally via tools/export_gguf.py
(llama3.2-medqa-ft-fold0 .. fold4), and tools/finetune_dataset.jsonl (for the
fold assignment -- must be the same file used to train them).

Usage: python3 tools/pipeline_eval_cv.py results_E9_cv.json
"""
import csv, json, os, sys, time
sys.path.insert(0, os.getcwd())

MODEL_PREFIX = os.getenv('CV_MODEL_PREFIX', 'llama3.2-medqa-ft-fold')
OUT = os.path.join(os.path.dirname(__file__), sys.argv[1] if len(sys.argv) > 1 else 'pipeline_results_cv.json')

CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = list(csv.DictReader(open('data/question_train.csv')))
by_sample = {}
for r in rows:
    by_sample.setdefault(r['transcript_id'], []).append(r)

# fold assignment must match tools/build_finetune_dataset.py exactly
sids = sorted(by_sample)
fold_of = {sid: i % 5 for i, sid in enumerate(sids)}


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


records = []
t0 = time.time()
for fold in range(5):
    os.environ['OLLAMA_MODEL'] = f'{MODEL_PREFIX}{fold}'
    import example
    import importlib
    importlib.reload(example)  # pick up the new OLLAMA_MODEL env var
    fold_sids = [sid for sid in sids if fold_of[sid] == fold]
    print(f'--- fold {fold}: model {os.environ["OLLAMA_MODEL"]}, {len(fold_sids)} held-out conversations ---')
    for sid in fold_sids:
        segments = [dict(s) for s in cache[sid]]
        emb = example.embed_segments(segments)
        for r in by_sample[sid]:
            gold = None
            if r['evidence_start'] and r['evidence_end']:
                gold = (float(r['evidence_start']), float(r['evidence_end']))
            top, best_idx = example.retrieve(segments, r['question'], seg_embeddings=emb)
            top_idx = [i for i, _ in top]
            answer, span = example.ask(segments, r['question'], seg_embeddings=emb)
            rec = {
                'qid': r['question_id'], 'sid': sid, 'type': r['question_type'], 'fold': fold,
                'want': r['answer'] == 'yes', 'said': answer, 'span': span, 'gold': gold,
                'top_idx': top_idx, 'best_idx': best_idx,
            }
            if gold is not None:
                oracle = max(range(len(segments)), key=lambda i: iou(gold, (segments[i]['start'], segments[i]['end'])))
                rec['oracle_idx'] = oracle
                rec['oracle_in_topk'] = oracle in top_idx
                rec['iou'] = iou(gold, span) if span else 0.0
            records.append(rec)
        print(f'  {sid}: done ({time.time()-t0:.0f}s)', flush=True)

json.dump(records, open(OUT, 'w'), indent=1)
pos = [x for x in records if x['gold'] is not None]
acc = sum(x['said'] == x['want'] for x in records) / len(records)
mt = sum(x['iou'] for x in pos) / len(pos)
print(f'\nCV-pooled (leave-conversation-out, {len(sids)} conversations across 5 folds):')
print(f'accuracy {acc:.3f}  mean tIoU {mt:.3f}  score {0.4*acc+0.6*mt:.3f}')
print(f'compare against results_E7.json (or the current best offline baseline) with tools/pickrule.py')
