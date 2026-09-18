"""Shared pieces for the RL-with-tIoU-reward experiment (see RL_TIOU_NOTES.md).

- prompt rendering: byte-identical to jobs/finetune_train.py (E9), so an E9
  adapter can be continued and a trained adapter can be exported/served by the
  unchanged live pipeline.
- completion parsing: mirrors what example.ask() does with ollama's JSON.
- reward: the *actual* evaluation metric, per question, looked up from
  tools/rl_reward_table.json (built locally by tools/build_rl_reward_table.py,
  which replays example.ask()'s deterministic span logic for every citable
  index). No embedding model or ollama is needed on the HPC.
- held-out evaluation with plain HF greedy decoding, emitting records in the
  same schema as tools/pipeline_eval_cv.py so tools/rl_pool.py and
  tools/pickrule.py can compare against results_E9_cv.json.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'tools', 'finetune_dataset.jsonl')
TABLE_PATH = os.path.join(HERE, '..', 'tools', 'rl_reward_table.json')

FORMAT_PENALTY = -0.1  # unparseable completion; ollama's `format: json` makes this near-impossible live


def load_records():
    return [json.loads(l) for l in open(DATA_PATH)]


def load_table():
    return json.load(open(TABLE_PATH))


def render_prompt(r, eos_token='<|eot_id|>'):
    """Same string jobs/finetune_train.py trained E9 on (BOS added by the tokenizer)."""
    return (
        '<|start_header_id|>system<|end_header_id|>\n\n'
        'Cutting Knowledge Date: December 2023\n\n'
        f"{r['system']}<|eot_id|>"
        '<|start_header_id|>user<|end_header_id|>\n\n'
        f"{r['prompt']}<|eot_id|>"
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )


_ANS_RE = re.compile(r'"answer"\s*:\s*(true|false)', re.IGNORECASE)
_SEG_RE = re.compile(r'"segment"\s*:\s*(-?\d+|null)', re.IGNORECASE)


def parse_completion(text):
    """-> (answer: bool | None, segment: int | None). answer None == unparseable."""
    if isinstance(text, list):  # conversational format: [{"role":..., "content":...}]
        text = text[-1].get('content', '') if text else ''
    text = (text or '').strip()
    # take the first {...} block if there is one
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
        if isinstance(data, dict) and 'answer' in data:
            answer = bool(data.get('answer', False))
            seg = data.get('segment')
            seg = seg if isinstance(seg, int) and not isinstance(seg, bool) else None
            return answer, seg
    except Exception:
        pass
    ma = _ANS_RE.search(text)
    if not ma:
        return None, None
    answer = ma.group(1).lower() == 'true'
    ms = _SEG_RE.search(text)
    seg = int(ms.group(1)) if ms and ms.group(1) != 'null' else None
    return answer, seg


def pipeline_outcome(entry, answer, seg):
    """What the live pipeline would return for this citation: (said, span, iou)."""
    if not answer:
        return False, None, 0.0
    key = str(seg) if seg is not None else None
    if key is not None and key in entry['span_by_idx']:
        span, iou = entry['span_by_idx'][key], entry['iou_by_idx'][key]
    else:
        span, iou = entry['fallback_span'], entry['fallback_iou']
    return True, span, (iou if entry['gold'] else 0.0)


def reward_metric(entry, answer, seg):
    """Per-question marginal contribution to score = 0.4*acc + 0.6*mean_tIoU.
    acc is averaged over all 390 questions, tIoU over the 195 positives, so a
    positive's IoU carries 2x weight relative to a correctness unit:
        r = 0.4 * [answer correct] + 1.2 * IoU * [positive]
    GRPO normalises within a prompt group, so only the within-question
    ordering matters: negatives  false(0.4) > true(0);
    positives  true+good idx (up to 1.6) > true+bad idx (0.4) > false (0).
    """
    if answer is None:
        return FORMAT_PENALTY
    said, span, iou = pipeline_outcome(entry, answer, seg)
    r = 0.4 * float(said == entry['want'])
    if entry['want'] and said:
        r += 1.2 * iou
    return r


def reward_tiou(entry, answer, seg):
    """Ablation: pure tIoU on positives, correctness on negatives (equal scale)."""
    if answer is None:
        return FORMAT_PENALTY
    said, span, iou = pipeline_outcome(entry, answer, seg)
    if entry['want']:
        return iou if said else 0.0
    return 1.0 if not said else 0.0


_SEG_BOUNDS = None


def _seg_bounds(sid):
    """Whole-segment (start, end) per index from tools/words_cache.json (lazy)."""
    global _SEG_BOUNDS
    if _SEG_BOUNDS is None:
        cache = json.load(open(os.path.join(HERE, '..', 'tools', 'words_cache.json')))
        _SEG_BOUNDS = {k: [(s['start'], s['end']) for s in v] for k, v in cache.items()}
    return _SEG_BOUNDS[sid]


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def reward_segment(entry, answer, seg):
    """Like reward_metric but the localisation term is the IoU of the CITED
    segment itself (whole-segment bounds), without the pipeline's earlier-pick /
    E1-union post-processing. Run m1 (2026-09-18) showed that post-processing
    creates reward ties (any later index falls back to retrieval's pick and
    scores the same), which the policy exploited by drifting one index later --
    a habit that transferred to held-out prompts and cost tIoU. This reward has
    a unique best index per prompt."""
    if answer is None:
        return FORMAT_PENALTY
    said = bool(answer)
    r = 0.4 * float(said == entry['want'])
    if entry['want'] and said:
        if seg is not None and seg in entry['top_idx']:
            r += 1.2 * _iou(entry['gold'], _seg_bounds(entry['sid'])[seg])
    return r


REWARDS = {'metric': reward_metric, 'tiou': reward_tiou, 'segment': reward_segment}


def make_reward_fn(table, kind='metric'):
    fn = REWARDS[kind]

    def reward_fn(completions, qid, **kwargs):
        out = []
        for comp, q in zip(completions, qid):
            answer, seg = parse_completion(comp)
            out.append(fn(table[q], answer, seg))
        return out

    reward_fn.__name__ = f'reward_{kind}'
    return reward_fn


def eval_records(model, tok, records, table, fold, batch_size=16, max_new_tokens=48, desc=''):
    """Greedy-decode every record; return pipeline_eval_cv-style records."""
    import torch
    model.eval()
    tok.padding_side = 'left'
    out = []
    prompts = [render_prompt(r) for r in records]
    for i in range(0, len(records), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for r, text in zip(records[i:i + batch_size], texts):
            e = table[r['question_id']]
            answer, seg = parse_completion(text)
            said, span, iou = pipeline_outcome(e, bool(answer), seg)
            rec = {
                'qid': r['question_id'], 'sid': e['sid'], 'type': e['type'], 'fold': fold,
                'want': e['want'], 'said': said, 'span': span, 'gold': e['gold'],
                'top_idx': e['top_idx'], 'best_idx': e['best_idx'],
                'cited': seg, 'parsed': answer is not None, 'raw': text[:200],
                'reward': reward_metric(e, answer, seg),
            }
            if e['gold']:
                rec['iou'] = iou
                rec['oracle_idx'] = e['oracle_idx']
                rec['oracle_in_topk'] = e['oracle_in_topk']
            out.append(rec)
        print(f'  {desc} eval {min(i + batch_size, len(records))}/{len(records)}', flush=True)
    model.train()
    return out


def summarize(records):
    pos = [x for x in records if x['gold'] is not None]
    acc = sum(x['said'] == x['want'] for x in records) / len(records)
    mt = sum(x['iou'] for x in pos) / len(pos) if pos else 0.0
    yes = [x for x in pos if x['said']]
    fn = sum(1 for x in pos if not x['said'])
    fp = sum(1 for x in records if x['gold'] is None and x['said'])
    right = sum(1 for x in yes if x.get('cited') == x['oracle_idx'])
    unparsed = sum(1 for x in records if not x.get('parsed', True))
    return {
        'n': len(records), 'accuracy': round(acc, 4), 'mean_tiou': round(mt, 4),
        'score': round(0.4 * acc + 0.6 * mt, 4), 'FN': fn, 'FP': fp,
        'cited_oracle_of_yes': f'{right}/{len(yes)}', 'unparsed': unparsed,
    }
