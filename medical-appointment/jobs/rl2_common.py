"""Stage 2 (clause-level action space) counterpart of jobs/rl_common.py, with
the same interface so jobs/rl_grpo_train.py can run either task:

    load_records / load_table / render_prompt / parse_completion /
    REWARDS / make_reward_fn / eval_records / summarize

Data: tools/rl2_dataset.jsonl (tools/build_rl2_dataset.py). Each record carries
its own `clauses` table {seg_idx: [[start,end],...]}, so the "table" here is
just the records keyed by question_id, and the reward is closed-form tIoU of
(clause[from].start, clause[to].end) against gold -- no ollama, no embeddings.

Pipeline semantics for the eval (what a stage-2 example.ask() would do):
  answer false                      -> (False, None)
  segment not in top-8 / missing    -> whole segment `best_idx` (retrieval top-1), like ask()'s fallback
  from/to missing or out of range   -> whole cited segment
  from > to                         -> swapped
"""
import json
import os
import re

from rl_common import render_prompt, FORMAT_PENALTY  # noqa: F401  (identical template)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'tools', 'rl2_dataset.jsonl')


def load_records():
    return [json.loads(l) for l in open(DATA_PATH)]


def load_table():
    return {r['question_id']: r for r in load_records()}


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


_ANS_RE = re.compile(r'"answer"\s*:\s*(true|false)', re.IGNORECASE)
_INT_RE = {k: re.compile(rf'"{k}"\s*:\s*(-?\d+|null)', re.IGNORECASE) for k in ('segment', 'from', 'to')}


def _as_int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def parse_completion(text):
    """-> (answer: bool|None, (segment, from, to)) ; answer None == unparseable."""
    if isinstance(text, list):
        text = text[-1].get('content', '') if text else ''
    text = (text or '').strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
        if isinstance(data, dict) and 'answer' in data:
            return bool(data.get('answer', False)), (_as_int(data.get('segment')), _as_int(data.get('from')), _as_int(data.get('to')))
    except Exception:
        pass
    ma = _ANS_RE.search(text)
    if not ma:
        return None, (None, None, None)
    vals = []
    for k in ('segment', 'from', 'to'):
        mk = _INT_RE[k].search(text)
        vals.append(int(mk.group(1)) if mk and mk.group(1) != 'null' else None)
    return ma.group(1).lower() == 'true', tuple(vals)


def span_of(entry, seg, frm, to):
    """Span a stage-2 pipeline would emit for this citation (see module docstring)."""
    cl = entry['clauses']
    key = str(seg) if seg is not None else None
    if key not in cl:
        key = str(entry['best_idx'])
        return (cl[key][0][0], cl[key][-1][1])
    c = cl[key]
    if frm is None or to is None:
        return (c[0][0], c[-1][1])
    a, b = sorted((frm, to))
    if a < 1 or b > len(c):
        return (c[0][0], c[-1][1])
    return (c[a - 1][0], c[b - 1][1])


def pipeline_outcome(entry, answer, cite):
    if not answer:
        return False, None, 0.0
    span = span_of(entry, *cite)
    return True, list(span), (iou(entry['gold'], span) if entry['gold'] else 0.0)


def reward_metric(entry, answer, cite):
    """Same shape as rl_common.reward_metric: 0.4*[correct] + 1.2*IoU*[positive]."""
    if answer is None:
        return FORMAT_PENALTY
    said, span, v = pipeline_outcome(entry, answer, cite)
    r = 0.4 * float(said == entry['want'])
    if entry['want'] and said:
        r += 1.2 * v
    return r


def reward_tiou(entry, answer, cite):
    if answer is None:
        return FORMAT_PENALTY
    said, span, v = pipeline_outcome(entry, answer, cite)
    if entry['want']:
        return v if said else 0.0
    return 1.0 if not said else 0.0


REWARDS = {'metric': reward_metric, 'tiou': reward_tiou}


def make_reward_fn(table, kind='metric'):
    fn = REWARDS[kind]

    def reward_fn(completions, qid, **kwargs):
        out = []
        for comp, q in zip(completions, qid):
            answer, cite = parse_completion(comp)
            out.append(fn(table[q], answer, cite))
        return out

    reward_fn.__name__ = f'reward2_{kind}'
    return reward_fn


def eval_records(model, tok, records, table, fold, batch_size=16, max_new_tokens=64, desc=''):
    import torch
    model.eval()
    tok.padding_side = 'left'
    out = []
    prompts = [render_prompt(r) for r in records]
    for i in range(0, len(records), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for r, text in zip(records[i:i + batch_size], texts):
            e = table[r['question_id']]
            answer, cite = parse_completion(text)
            said, span, v = pipeline_outcome(e, bool(answer), cite)
            rec = {
                'qid': r['question_id'], 'sid': e['sid'], 'type': e['question_type'], 'fold': fold,
                'want': e['want'], 'said': said, 'span': span, 'gold': e['gold'],
                'top_idx': e['top_idx'], 'best_idx': e['best_idx'],
                'cited': cite[0], 'cite': list(cite), 'parsed': answer is not None, 'raw': text[:200],
                'reward': reward_metric(e, answer, cite),
            }
            if e['gold']:
                rec['iou'] = v
                rec['oracle_idx'] = e['target']['segment']  # oracle in the clause action space
                rec['oracle_in_topk'] = e['target']['answer']
                rec['target_iou'] = e['target_iou']
            out.append(rec)
        print(f'  {desc} eval {min(i + batch_size, len(records))}/{len(records)}', flush=True)
    model.train()
    return out


from rl_common import summarize  # noqa: E402,F401  (same record keys)
