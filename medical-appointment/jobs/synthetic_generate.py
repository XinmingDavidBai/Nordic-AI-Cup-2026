"""E10: generate synthetic doctor-patient consultations + questions in the
style of data/question_train.csv, using a large instruct model served by
vLLM on the HPC A100. Few-shot prompted with 3 real transcripts (from
tools/words_cache.json) and their questions, matching the training
distribution: ~50% positive (question quotes the evidence sentence
verbatim), ~36% hard_negative (one number/drug/duration changed from the
transcript), ~14% off_topic.

Output: jobs/synthetic_data.jsonl, one record per consultation:
  {"transcript": [{"speaker": "doctor"|"patient", "text": "..."}],
   "questions": [{"question": "...", "answer": "yes"|"no", "type": "...",
                  "evidence_sentence": "..." or null}]}

evidence_sentence is the verbatim quoted sentence for a positive question --
tools/build_synthetic_finetune_data.py locates it in the synthetic
transcript (segment = sentence) to convert this into E9's training format.
No audio, no ASR noise: E9 must judge any model trained on this by its
held-out REAL folds only (see NEXT_STEPS.md section 5b, E10 risk note).
"""
import argparse
import json
import os
import random
import re

from openai import OpenAI  # vLLM's OpenAI-compatible server

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'synthetic_data.jsonl'))
ap.add_argument('--base-url', default='http://localhost:8000/v1')
ap.add_argument('--model', default=os.getenv('VLLM_MODEL', 'Qwen/Qwen2.5-32B-Instruct-AWQ'))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--few-shot-samples', default=os.path.join(os.path.dirname(__file__), '..', 'tools', 'words_cache.json'))
ap.add_argument('--questions-csv', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'question_train.csv'))
args = ap.parse_args()
random.seed(args.seed)

client = OpenAI(base_url=args.base_url, api_key='not-needed')

# --- build few-shot examples from 3 real transcripts ------------------------
import csv
cache = json.load(open(args.few_shot_samples))
rows = list(csv.DictReader(open(args.questions_csv)))
by_sid = {}
for r in rows:
    by_sid.setdefault(r['transcript_id'], []).append(r)

few_shot_sids = random.sample(sorted(cache), 3)
few_shot_blocks = []
for sid in few_shot_sids:
    segs = cache[sid]
    transcript_text = '\n'.join(f'- {s["text"]}' for s in segs)
    qs = by_sid.get(sid, [])
    q_lines = []
    for r in qs:
        q_lines.append(f'  - question: "{r["question"]}"\n    answer: {r["answer"]}\n    type: {r["question_type"]}')
    few_shot_blocks.append(f'TRANSCRIPT ({sid}):\n{transcript_text}\n\nQUESTIONS:\n' + '\n'.join(q_lines))

FEW_SHOT = '\n\n---\n\n'.join(few_shot_blocks)

SYSTEM = (
    'You generate synthetic doctor-patient consultation transcripts and yes/no '
    'comprehension questions for training data, in the exact style of the examples given.'
)

INSTRUCTIONS = """\
Below are real examples of short doctor-patient consultation transcripts, each
followed by 10 yes/no questions about it.

{few_shot}

---

Generate ONE new, different synthetic consultation in the same style (a short,
natural spoken dialogue between doctor and patient -- a routine check-up,
follow-up, or minor complaint; vary the topic from the examples above). Then
write exactly 10 questions about it, matching this distribution:
- 5 "positive" questions: each one's answer is "yes", and it must be
  answerable by a single sentence you can quote VERBATIM from your transcript
  (put that exact sentence in "evidence_sentence").
- ~3-4 "hard_negative" questions: phrased like a positive question but with
  ONE number, drug name, or duration changed from what the transcript
  actually says (so the correct answer is "no" -- a near miss, not an absurd
  claim).
- ~1-2 "off_topic" questions: about something never discussed in this
  transcript at all (answer "no").

Output ONLY this JSON (no markdown fences, no commentary):
{{
  "transcript": [{{"speaker": "doctor" or "patient", "text": "..."}}, ...],
  "questions": [
    {{"question": "...", "answer": "yes" or "no", "type": "positive"|"hard_negative"|"off_topic",
      "evidence_sentence": "verbatim sentence from transcript, or null for hard_negative/off_topic"}}
  ]
}}"""

records = []
n_fail = 0
while len(records) < args.n:
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': INSTRUCTIONS.format(few_shot=FEW_SHOT)},
        ],
        temperature=0.9,
        max_tokens=1500,
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        rec = json.loads(text)
        assert 'transcript' in rec and 'questions' in rec and len(rec['questions']) > 0
    except Exception as exc:
        n_fail += 1
        if n_fail % 20 == 0:
            print(f'  {n_fail} parse failures so far (last: {exc})')
        continue
    records.append(rec)
    if len(records) % 50 == 0:
        print(f'{len(records)}/{args.n} generated ({n_fail} failures)', flush=True)

with open(args.out, 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')
print(f'wrote {len(records)} synthetic consultations to {args.out} ({n_fail} generation failures discarded)')
