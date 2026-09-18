"""E9 data prep: build the exact (system, prompt) -> target JSON pairs the
live pipeline sends to the LLM, labelled with the oracle segment instead of
whatever llama3.2:3b happens to cite. Runs locally (needs ollama for
embeddings, not a GPU). Output is consumed by jobs/finetune_train.py on HPC.

5-fold split is by conversation_id (sorted, index % 5) so a fold never sees
a conversation's other questions during training -- 39 conversations is small
enough that leaking a question from an otherwise held-out conversation would
let the model memorise it instead of generalising.
"""
import csv, json, os, sys
sys.path.insert(0, os.getcwd())
import example  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'finetune_dataset.jsonl')
CACHE = os.path.join(os.path.dirname(__file__), 'words_cache.json')
cache = json.load(open(CACHE))
rows = list(csv.DictReader(open('data/question_train.csv')))

sids = sorted(set(r['transcript_id'] for r in rows))
fold_of = {sid: i % 5 for i, sid in enumerate(sids)}
print(f'{len(sids)} conversations, 5-fold split by conversation_id')

by_sample = {}
for r in rows:
    by_sample.setdefault(r['transcript_id'], []).append(r)

records = []
for sid, qs in sorted(by_sample.items()):
    segments = [dict(s) for s in cache[sid]]
    emb = example.embed_segments(segments)
    for r in qs:
        question = example._normalize_question(r['question'])
        top, best_idx = example.retrieve(segments, question, seg_embeddings=emb)
        context = '\n'.join(f'[{i}] [{s["start"]:.1f}-{s["end"]:.1f}s]: {s["text"]}' for i, s in top)
        prompt = example._PROMPT.format(context=context, question=question)

        is_yes = r['answer'] == 'yes'
        target_segment = None
        if is_yes and r['evidence_start'] and r['evidence_end']:
            gold = (float(r['evidence_start']), float(r['evidence_end']))

            def iou(a, b):
                inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
                union = max(a[1], b[1]) - min(a[0], b[0])
                return inter / union if union > 0 else 0.0

            oracle = max(range(len(segments)), key=lambda i: iou(gold, (segments[i]['start'], segments[i]['end'])))
            # Only usable as a training target if the oracle segment is
            # actually shown to the model (it's in this question's top-8).
            top_indices = {i for i, _ in top}
            target_segment = oracle if oracle in top_indices else None
            if target_segment is None:
                is_yes = False  # unlearnable from this context; teach it as a clean no rather than a wrong index

        target = {'answer': bool(is_yes), 'segment': target_segment}
        records.append({
            'question_id': r['question_id'],
            'sid': sid,
            'fold': fold_of[sid],
            'question_type': r['question_type'],
            'system': example._SYSTEM,
            'prompt': prompt,
            'target': target,
            'target_json': json.dumps(target),
        })
    print(f'{sid}: {len(qs)} questions done', flush=True)

with open(OUT, 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')

n_yes = sum(1 for r in records if r['target']['answer'])
n_seg = sum(1 for r in records if r['target']['segment'] is not None)
print(f'\nwrote {len(records)} records to {OUT}')
print(f'  yes={n_yes} no={len(records)-n_yes}, has usable oracle segment={n_seg}')
for f in range(5):
    print(f'  fold {f}: {sum(1 for r in records if r["fold"]==f)} records, '
          f'{sum(1 for sid in sids if fold_of[sid]==f)} conversations')
