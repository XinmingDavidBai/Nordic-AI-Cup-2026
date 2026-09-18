"""Exact expected-reward policy optimization over the ENUMERABLE action space
(RL_TIOU_NOTES.md section 6) -- the sampling-free alternative to GRPO.

Both tasks here have a small, fully enumerable set of valid completions:
  seg    : {"answer": false} + {"answer": true, "segment": i} for the 8 shown i     -> 9 candidates
  clause : {"answer": false} + {true, segment i, from a, to b} for a<=b in each i  -> ~37 candidates
and tools/rl_reward_table.json / tools/rl2_dataset.jsonl give the exact reward of
every one of them. So instead of sampling G completions and normalising rewards
within the group (GRPO; degenerates to zero advantage whenever the samples
agree, which is most of the time for a peaked SFT-initialised policy), score
every candidate by teacher forcing and optimise the exact expectation

    J = sum_a q(a|x) r(a),   q = softmax_a( l(a|x) ),   l = (mean) token log-prob of candidate a

plus an entropy bonus, following FGPO ("Why Sample What You Can Enumerate?",
arXiv 2609.10221). Gradient computed exactly and memory-light via the policy-
gradient identity  grad J = sum_a q(a) (r(a) - J) grad l(a): a no-grad pass gets
all l(a) and the per-candidate weights, then candidates are back-propagated in
chunks with loss = -sum_a w_a l(a). Same prompt bytes, same init adapters, same
held-out evaluation as jobs/rl_grpo_train.py, so results are directly comparable.

Usage (from medical-appointment/):
    python jobs/rl_exact_train.py --task seg    --fold 0 --run x1
    python jobs/rl_exact_train.py --task clause --fold 0 --run xc1 --init-dir checkpoints_rl2_sft
"""
import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault('USE_TF', '0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

BASE_MODEL = os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct')

ap = argparse.ArgumentParser()
ap.add_argument('--task', choices=['seg', 'clause'], default='seg')
ap.add_argument('--fold', required=True)
ap.add_argument('--run', default='x1')
ap.add_argument('--init', choices=['e9', 'base'], default='e9')
ap.add_argument('--init-dir', default=None)
ap.add_argument('--reward', choices=['metric', 'tiou'], default='metric')
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--entropy', type=float, default=0.03, help='entropy bonus weight (FGPO uses 0.03)')
ap.add_argument('--no-length-norm', action='store_true', help='use summed instead of mean token log-prob')
ap.add_argument('--prompts-per-step', type=int, default=4)
ap.add_argument('--chunk', type=int, default=8, help='candidates per backward chunk')
ap.add_argument('--skip-eval', action='store_true')
ap.add_argument('--eval-only', action='store_true')
ap.add_argument('--max-steps', type=int, default=-1)
ap.add_argument('--model', default=None)
ap.add_argument('--eval-limit', type=int, default=0)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

if args.task == 'seg':
    import rl_common as rc
else:
    import rl2_common as rc
INIT_DIR = args.init_dir or ('checkpoints_e9' if args.task == 'seg' else 'checkpoints_rl2_sft')
reward_fn = rc.REWARDS[args.reward]

base_model = args.model or BASE_MODEL
res_dir = os.path.join('rl_results', args.run)
tag = 'final' if args.fold == 'all' else f'fold{args.fold}'
ckpt_dir = os.path.join('checkpoints_rl', args.run, tag)
os.makedirs(res_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(res_dir, f'{tag}_args.json'), 'w'), indent=1)
random.seed(args.seed)
torch.manual_seed(args.seed)

records = rc.load_records()
table = rc.load_table()
if args.fold == 'all':
    train_records, held_out, fold = records, [], None
    init_adapter = os.path.join(INIT_DIR, 'final')
else:
    fold = int(args.fold)
    train_records = [r for r in records if r['fold'] != fold]
    held_out = [r for r in records if r['fold'] == fold]
    if args.eval_limit:
        held_out = held_out[:args.eval_limit]
    init_adapter = os.path.join(INIT_DIR, f'fold{fold}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(base_model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16 if device == 'cuda' else torch.float32).to(device)
if args.init == 'e9':
    if not os.path.isfile(os.path.join(init_adapter, 'adapter_config.json')):
        sys.exit(f'missing init adapter {init_adapter}')
    model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    print(f'continuing SFT adapter from {init_adapter}', flush=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False


# ---------------------------------------------------------------- candidates
def candidates_for(r):
    """-> list of (completion_text, reward). Completion text matches target_json formatting exactly."""
    e = table[r['question_id']]
    out = []
    if args.task == 'seg':
        out.append((json.dumps({'answer': False, 'segment': None}), reward_fn(e, False, None)))
        for i in e['top_idx']:
            out.append((json.dumps({'answer': True, 'segment': i}), reward_fn(e, True, i)))
    else:
        out.append((json.dumps({'answer': False, 'segment': None, 'from': None, 'to': None}), reward_fn(e, False, (None, None, None))))
        for i in e['top_idx']:
            n = len(e['clauses'][str(i)])
            for a in range(1, n + 1):
                for b in range(a, n + 1):
                    out.append((json.dumps({'answer': True, 'segment': i, 'from': a, 'to': b}), reward_fn(e, True, (i, a, b))))
    return out


def encode(prompt, completions):
    """Tokenize prompt + each completion (+eos); return input_ids, attention, and a completion mask."""
    p_ids = tok(prompt, add_special_tokens=True)['input_ids']
    seqs, masks = [], []
    for c in completions:
        c_ids = tok(c + tok.eos_token, add_special_tokens=False)['input_ids']
        seqs.append(p_ids + c_ids)
        masks.append([0] * len(p_ids) + [1] * len(c_ids))
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    cm = torch.zeros((len(seqs), L), dtype=torch.float)
    for k, (s, m) in enumerate(zip(seqs, masks)):
        ids[k, :len(s)] = torch.tensor(s)
        att[k, :len(s)] = 1
        cm[k, :len(s)] = torch.tensor(m, dtype=torch.float)
    return ids.to(device), att.to(device), cm.to(device)


def seq_logprobs(ids, att, cm):
    """Per-sequence (mean or summed) log-prob of the completion tokens."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = torch.gather(F.log_softmax(logits, dim=-1), 2, tgt.unsqueeze(-1)).squeeze(-1)
    m = cm[:, 1:]
    s = (lp * m).sum(1)
    return s if args.no_length_norm else s / m.sum(1).clamp(min=1)


# ---------------------------------------------------------------- eval
def do_eval(name):
    if args.skip_eval or not held_out:
        return None
    t = time.time()
    model.config.use_cache = True
    recs = rc.eval_records(model, tok, held_out, table, fold, desc=name)
    model.config.use_cache = False
    path = os.path.join(res_dir, f'{tag}_{name}.json')
    json.dump(recs, open(path, 'w'), indent=1)
    s = rc.summarize(recs)
    print(f'[{tag} {name}] {s}  ({time.time()-t:.0f}s) -> {path}', flush=True)
    return s


init_summary = do_eval('init')
if args.eval_only:
    sys.exit(0)

# ---------------------------------------------------------------- train
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train_records) / args.prompts_per_step)
total_steps = int(steps_per_epoch * args.epochs) if args.max_steps < 0 else args.max_steps
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
print(f'{tag} [{args.task}] exact-EV training: {len(train_records)} prompts, {total_steps} optimizer steps '
      f'({args.prompts_per_step} prompts/step), lr {args.lr}, entropy {args.entropy}', flush=True)

log = []
step = 0
t0 = time.time()
model.train()
done = False
while not done:
    order = list(range(len(train_records)))
    random.shuffle(order)
    for s0 in range(0, len(order), args.prompts_per_step):
        batch_idx = order[s0:s0 + args.prompts_per_step]
        opt.zero_grad(set_to_none=True)
        ev_sum = ent_sum = pbest_sum = 0.0
        for bi in batch_idx:
            r = train_records[bi]
            cands = candidates_for(r)
            texts = [c for c, _ in cands]
            rew = torch.tensor([v for _, v in cands], device=device)
            prompt = rc.render_prompt(r)
            ids, att, cm = encode(prompt, texts)
            # pass 1: all candidate log-probs, no grad -> q, weights
            with torch.no_grad():
                lps = torch.cat([seq_logprobs(ids[i:i + args.chunk], att[i:i + args.chunk], cm[i:i + args.chunk])
                                 for i in range(0, len(texts), args.chunk)])
                q = torch.softmax(lps, 0)
                J = (q * rew).sum()
                H = -(q * torch.log(q + 1e-12)).sum()
                # d/dl [J + beta*H] = q*(r - J) + beta * (-q*(log q + H))
                w = q * (rew - J) + args.entropy * (-q * (torch.log(q + 1e-12) + H))
                w = w / len(batch_idx)
            # pass 2: exact gradient via sum_a w_a * l_a, in chunks
            for i in range(0, len(texts), args.chunk):
                lp_chunk = seq_logprobs(ids[i:i + args.chunk], att[i:i + args.chunk], cm[i:i + args.chunk])
                loss = -(w[i:i + args.chunk] * lp_chunk).sum()
                loss.backward()
            ev_sum += J.item(); ent_sum += H.item(); pbest_sum += q[rew.argmax()].item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        n = len(batch_idx)
        rec = {'step': step, 'expected_reward': ev_sum / n, 'entropy': ent_sum / n,
               'p_best_candidate': pbest_sum / n, 'lr': sched.get_last_lr()[0], 'elapsed_s': round(time.time() - t0)}
        log.append(rec)
        if step % 5 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if step >= total_steps:
            done = True
            break

print(f'training done in {(time.time()-t0)/60:.1f} min', flush=True)
json.dump(log, open(os.path.join(res_dir, f'{tag}_train_log.json'), 'w'), indent=1)
model.save_pretrained(ckpt_dir)
tok.save_pretrained(ckpt_dir)
print(f'saved adapter to {ckpt_dir}', flush=True)

final_summary = do_eval('final')
if init_summary and final_summary:
    print(f'[{tag}] init score {init_summary["score"]} -> final score {final_summary["score"]} '
          f'(tIoU {init_summary["mean_tiou"]} -> {final_summary["mean_tiou"]}, '
          f'acc {init_summary["accuracy"]} -> {final_summary["accuracy"]})', flush=True)
