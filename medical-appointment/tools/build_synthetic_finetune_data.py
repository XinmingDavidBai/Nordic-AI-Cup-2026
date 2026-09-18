"""E10 -> E9 bridge: convert jobs/synthetic_data.jsonl (generated on the HPC,
see jobs/synthetic_generate.py) into the same record format as
tools/build_finetune_dataset.py, and append to tools/finetune_dataset.jsonl.

Synthetic records get fold=-1, a sentinel that never matches any real fold
(0-4) in jobs/finetune_train.py's `r['fold'] != fold` filter -- so they are
included in every fold's TRAINING set but never selected as held-out
evaluation data. This matches E10's risk note in NEXT_STEPS.md section 5b:
synthetic transcripts have no ASR noise, so any model trained on them must be
judged by its held-out REAL folds only (tools/pipeline_eval_cv.py already
only evaluates real conversations).

Turns have no real timestamps (no audio was generated); each turn is given a
nominal, strictly increasing span sized ~2.7 words/second (a rough spoken-
rate estimate) purely so example._PROMPT's "[i] [start-end s]: text" context
formatting has something to show -- these numbers are never used as
evidence_start/evidence_end targets.

Usage: python3 tools/build_synthetic_finetune_data.py
"""
import difflib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import example  # noqa: E402

IN = os.path.join(os.path.dirname(__file__), '..', 'jobs', 'synthetic_data.jsonl')
OUT = os.path.join(os.path.dirname(__file__), 'finetune_dataset.jsonl')
WORDS_PER_SECOND = 2.7
MATCH_RATIO_THRESHOLD = 0.6

synthetic = [json.loads(l) for l in open(IN)]
print(f'{len(synthetic)} synthetic consultations from {IN}')

new_records = []
skipped_unmatched_evidence = 0
for ci, conv in enumerate(synthetic):
    turns = conv['transcript']
    segments = []
    t = 0.0
    for turn in turns:
        text = turn['text'].strip()
        if not text:
            continue
        dur = max(1.0, len(text.split()) / WORDS_PER_SECOND)
        segments.append({'start': t, 'end': t + dur, 'text': text, 'words': []})
        t += dur + 0.3  # small gap between turns

    if not segments:
        continue
    emb = example.embed_segments(segments)

    for qi, q in enumerate(conv['questions']):
        question = example._normalize_question(q['question'])
        top, best_idx = example.retrieve(segments, question, seg_embeddings=emb)
        context = '\n'.join(f'[{i}] [{s["start"]:.1f}-{s["end"]:.1f}s]: {s["text"]}' for i, s in top)
        prompt = example._PROMPT.format(context=context, question=question)

        is_yes = q['answer'] == 'yes'
        target_segment = None
        if is_yes and q.get('evidence_sentence'):
            top_indices = {i for i, _ in top}
            ev = q['evidence_sentence'].strip().lower()
            best_i, best_ratio = None, 0.0
            for i in top_indices:
                ratio = difflib.SequenceMatcher(None, ev, segments[i]['text'].strip().lower()).ratio()
                if ev and ev in segments[i]['text'].strip().lower():
                    ratio = max(ratio, 0.99)  # exact substring wins outright
                if ratio > best_ratio:
                    best_ratio, best_i = ratio, i
            if best_ratio >= MATCH_RATIO_THRESHOLD:
                target_segment = best_i
            else:
                skipped_unmatched_evidence += 1
                is_yes = False  # can't locate the quoted sentence in the shown context; teach a clean no instead

        target = {'answer': bool(is_yes), 'segment': target_segment}
        new_records.append({
            'question_id': f'synthetic_{ci}_q{qi}',
            'sid': f'synthetic_{ci}',
            'fold': -1,
            'question_type': q.get('type', 'positive' if is_yes else 'hard_negative'),
            'system': example._SYSTEM,
            'prompt': prompt,
            'target': target,
            'target_json': json.dumps(target),
        })
    if (ci + 1) % 100 == 0:
        print(f'{ci+1}/{len(synthetic)} consultations processed', flush=True)

with open(OUT, 'a') as f:
    for rec in new_records:
        f.write(json.dumps(rec) + '\n')

n_yes = sum(1 for r in new_records if r['target']['answer'])
print(f'\nappended {len(new_records)} synthetic records to {OUT}')
print(f'  yes={n_yes} no={len(new_records)-n_yes}')
print(f'  {skipped_unmatched_evidence} positive questions had an unmatched evidence_sentence '
      f'(kept as no rather than a wrong segment index)')
