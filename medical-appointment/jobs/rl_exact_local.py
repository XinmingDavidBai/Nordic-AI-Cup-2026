"""Local (Apple MPS) version of jobs/rl_exact_train.py for the stage-1 action
space, written 2026-09-18 evening when the DTU HPC went into a service window
that outlasts the competition deadline (Sunday 16:00).

Same objective as rl_exact_train.py -- exact expected reward over the 9
enumerable completions, J = sum_a q(a) r(a) with q = softmax of the mean token
log-prob, plus an entropy bonus -- but engineered for a 17GB M1 Pro that also
serves the live endpoint:

- fp16 base weights (bf16 is not native on MPS), fp32 LoRA (peft's default
  autocast), no gradient checkpointing (it is incompatible with the KV cache
  and did not reduce peak memory here anyway).
- The ~560-token prompt is forwarded ONCE with a KV cache (with grad); the 9
  candidate tails (~13 tokens each) are scored as one batch against the
  repeated cache. Measured cost is ~1 full-sequence forward+backward per
  prompt instead of 9-18.
- Adapter checkpoint every --save-every prompts and at every epoch end; --resume
  continues from the last checkpoint (prompt order is seeded per epoch, so
  resuming mid-epoch replays the same order and skips what is done).
- An OOM on one prompt (the live server's ollama shares the GPU) empties the
  cache and skips that prompt rather than dying.

Usage (from medical-appointment/, in tools/.export-venv of the main checkout):
    python jobs/rl_exact_local.py --fold 0 --run m1 --epochs 2
    python jobs/rl_exact_local.py --fold all --run m1 --epochs 2 --skip-eval   # final model
Results: rl_results/<run>/fold{k}_{init,final}.json (same schema as the HPC runs),
adapter: checkpoints_rl/<run>/fold{k}/ (exportable by tools/export_gguf.py).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, get_cosine_schedule_with_warmup  # noqa: E402

import rl_common as rc  # noqa: E402

BASE_MODEL = os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct')

ap = argparse.ArgumentParser()
ap.add_argument('--fold', required=True)
ap.add_argument('--run', default='m1')
ap.add_argument('--init-dir', default='checkpoints_e9')
ap.add_argument('--reward', choices=['metric', 'tiou'], default='metric')
ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--entropy', type=float, default=0.03)
ap.add_argument('--prompts-per-step', type=int, default=4)
ap.add_argument('--save-every', type=int, default=50)
ap.add_argument('--skip-eval', action='store_true')
ap.add_argument('--eval-only', action='store_true')
ap.add_argument('--eval-batch', type=int, default=4)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--limit', type=int, default=0, help='debug: only this many training prompts per epoch')
ap.add_argument('--eval-limit', type=int, default=0)
ap.add_argument('--check', action='store_true', help='verify cached scoring == full-sequence scoring on one prompt, then exit')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--prompt-grad', action='store_true',
                help='also backpropagate through the prompt forward (exact gradient; measured 212s/19GB per prompt on the M1 Pro, so off by default: the default trains only through the candidate tails reading a fixed prompt encoding)')
args = ap.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = torch.float16 if device == 'mps' else torch.float32
reward_fn = rc.REWARDS[args.reward]
res_dir = os.path.join('rl_results', args.run)
tag = 'final' if args.fold == 'all' else f'fold{args.fold}'
ckpt_dir = os.path.join('checkpoints_rl', args.run, tag)
state_path = os.path.join(ckpt_dir, 'train_state.json')
os.makedirs(res_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(res_dir, f'{tag}_args.json'), 'w'), indent=1)

records = rc.load_records()
table = rc.load_table()
if args.fold == 'all':
    train_records, held_out, fold = records, [], None
    init_adapter = os.path.join(args.init_dir, 'final')
else:
    fold = int(args.fold)
    train_records = [r for r in records if r['fold'] != fold]
    held_out = [r for r in records if r['fold'] == fold]
    if args.eval_limit:
        held_out = held_out[:args.eval_limit]
    init_adapter = os.path.join(args.init_dir, f'fold{fold}')
if args.limit:
    train_records = train_records[:args.limit]

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
resume_from = ckpt_dir if args.resume and os.path.isfile(os.path.join(ckpt_dir, 'adapter_config.json')) else init_adapter
model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
model.config.use_cache = True
print(f'{tag}: {len(train_records)} train prompts, {len(held_out)} held out; adapter from {resume_from}; device {device} {dtype}', flush=True)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f'trainable params {sum(p.numel() for p in trainable)/1e6:.1f}M in {trainable[0].dtype}', flush=True)


def candidates_for(r):
    e = table[r['question_id']]
    out = [(json.dumps({'answer': False, 'segment': None}), reward_fn(e, False, None))]
    for i in e['top_idx']:
        out.append((json.dumps({'answer': True, 'segment': i}), reward_fn(e, True, i)))
    return out


def cand_logprobs(prompt, texts):
    """Mean token log-prob of each candidate completion (+eos) given the prompt,
    with the prompt forwarded once. Returns a tensor of shape (n,) with grad."""
    p_ids = tok(prompt, add_special_tokens=True)['input_ids']
    c_ids = [tok(c + tok.eos_token, add_special_tokens=False)['input_ids'] for c in texts]
    n, P, C = len(texts), len(p_ids), max(len(c) for c in c_ids)
    p_in = torch.tensor([p_ids], device=device)
    cache = DynamicCache()
    if args.prompt_grad:
        out = model(input_ids=p_in, past_key_values=cache, use_cache=True, logits_to_keep=1)
    else:
        with torch.no_grad():
            out = model(input_ids=p_in, past_key_values=cache, use_cache=True, logits_to_keep=1)
    last = out.logits[0, -1].float()  # predicts the first candidate token
    cache = out.past_key_values
    cache.batch_repeat_interleave(n)
    cand = torch.full((n, C), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n, C), dtype=torch.long)
    for k, c in enumerate(c_ids):
        cand[k, :len(c)] = torch.tensor(c)
        mask[k, :len(c)] = 1
    cand, mask = cand.to(device), mask.to(device)
    att = torch.cat([torch.ones((n, P), dtype=torch.long, device=device), mask], 1)
    pos = torch.arange(P, P + C, device=device).unsqueeze(0).expand(n, C)
    out2 = model(input_ids=cand, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True)
    logits = out2.logits.float()  # (n, C, V); logits[:, t] predicts cand[:, t+1]
    lp_first = F.log_softmax(last, -1)[cand[:, 0]]  # (n,)
    lp_rest = torch.gather(F.log_softmax(logits[:, :-1], -1), 2, cand[:, 1:].unsqueeze(-1)).squeeze(-1)  # (n, C-1)
    lp_rest = (lp_rest * mask[:, 1:]).sum(1)
    total = lp_first + lp_rest
    return total / mask.sum(1)


def full_logprob(prompt, text):
    """Reference: same quantity from one full-sequence forward (for --check)."""
    p_ids = tok(prompt, add_special_tokens=True)['input_ids']
    c = tok(text + tok.eos_token, add_special_tokens=False)['input_ids']
    ids = torch.tensor([p_ids + c], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, len(p_ids) - 1:-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 1, torch.tensor(c, device=device).unsqueeze(-1)).squeeze(-1)
    return lp.mean().item()


if args.check:
    r = train_records[0]
    cands = candidates_for(r)
    prompt = rc.render_prompt(r)
    t0 = time.time()
    with torch.no_grad():
        lps = cand_logprobs(prompt, [c for c, _ in cands])
    t_cached = time.time() - t0
    t0 = time.time()
    ref = [full_logprob(prompt, c) for c, _ in cands[:3]]
    t_full = (time.time() - t0) / 3
    print('cached :', [round(x, 4) for x in lps[:3].tolist()], f'({t_cached:.1f}s for all {len(cands)})')
    print('full   :', [round(x, 4) for x in ref], f'({t_full:.1f}s per candidate)')
    print('max abs diff', max(abs(a - b) for a, b in zip(lps[:3].tolist(), ref)))
    rew = torch.tensor([v for _, v in cands], device=device)
    for it in range(3):
        t0 = time.time()
        lps = cand_logprobs(prompt, [c for c, _ in cands])
        q = torch.softmax(lps, 0)
        loss = -((q * rew).sum())
        loss.backward()
        if device == 'mps':
            torch.mps.synchronize()
        g = sum(p.grad.float().norm() ** 2 for p in trainable if p.grad is not None) ** 0.5
        print(f'iter {it}: forward+backward (prompt_grad={args.prompt_grad}): {time.time()-t0:.1f}s; q(best)={q[rew.argmax()].item():.3f}; '
              f'grad norm {g:.4f}; mps driver mem {torch.mps.driver_allocated_memory()/1e9:.1f} GB', flush=True)
        model.zero_grad(set_to_none=True)
        if device == 'mps':
            torch.mps.empty_cache()
    sys.exit(0)


def do_eval(name):
    if args.skip_eval or not held_out:
        return None
    t = time.time()
    recs = rc.eval_records(model, tok, held_out, table, fold, batch_size=args.eval_batch, desc=name)
    path = os.path.join(res_dir, f'{tag}_{name}.json')
    json.dump(recs, open(path, 'w'), indent=1)
    s = rc.summarize(recs)
    print(f'[{tag} {name}] {s}  ({time.time()-t:.0f}s) -> {path}', flush=True)
    return s


state = {'epoch': 0, 'idx': 0, 'step': 0, 'log': []}
if args.resume and os.path.isfile(state_path):
    state = json.load(open(state_path))
    print(f'resuming at epoch {state["epoch"]} prompt {state["idx"]} step {state["step"]}', flush=True)

init_summary = None
if not (args.resume and state['step'] > 0):
    init_summary = do_eval('init')
elif os.path.isfile(os.path.join(res_dir, f'{tag}_init.json')):
    init_summary = rc.summarize(json.load(open(os.path.join(res_dir, f'{tag}_init.json'))))
if args.eval_only:
    sys.exit(0)

opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train_records) / args.prompts_per_step)
total_steps = steps_per_epoch * args.epochs
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
for _ in range(state['step']):
    sched.step()
model.train()
t0 = time.time()
n_oom = 0
print(f'training: {total_steps} optimizer steps of {args.prompts_per_step} prompts, lr {args.lr}, entropy {args.entropy}', flush=True)


def save(state):
    model.save_pretrained(ckpt_dir)
    json.dump(state, open(state_path, 'w'))


for epoch in range(state['epoch'], args.epochs):
    order = list(range(len(train_records)))
    random.Random(args.seed + epoch).shuffle(order)
    start = state['idx'] if epoch == state['epoch'] else 0
    acc_n = 0
    opt.zero_grad(set_to_none=True)
    ev_sum = ent_sum = pbest_sum = 0.0
    for i in range(start, len(order)):
        r = train_records[order[i]]
        cands = candidates_for(r)
        rew = torch.tensor([v for _, v in cands], device=device)
        try:
            lps = cand_logprobs(rc.render_prompt(r), [c for c, _ in cands])
            q = torch.softmax(lps, 0)
            J = (q * rew).sum()
            H = -(q * torch.log(q + 1e-12)).sum()
            loss = -(J + args.entropy * H) / args.prompts_per_step
            loss.backward()
            ev_sum += J.item(); ent_sum += H.item(); pbest_sum += q[rew.argmax()].item()
            acc_n += 1
        except RuntimeError as ex:
            if 'out of memory' not in str(ex).lower():
                raise
            n_oom += 1
            print(f'OOM on prompt {i} ({n_oom} total), skipping', flush=True)
            opt.zero_grad(set_to_none=True)
            acc_n = 0
            if device == 'mps':
                torch.mps.empty_cache()
            time.sleep(5)
            continue
        if acc_n == args.prompts_per_step or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            state['step'] += 1
            rec = {'step': state['step'], 'epoch': epoch, 'prompt': i + 1, 'expected_reward': ev_sum / acc_n,
                   'entropy': ent_sum / acc_n, 'p_best_candidate': pbest_sum / acc_n,
                   'lr': sched.get_last_lr()[0], 'elapsed_min': round((time.time() - t0) / 60, 1)}
            state['log'].append(rec)
            if state['step'] % 5 == 0 or state['step'] == 1:
                print(json.dumps(rec), flush=True)
            acc_n = 0
            ev_sum = ent_sum = pbest_sum = 0.0
        if device == 'mps':
            torch.mps.empty_cache()
        if (i + 1) % args.save_every == 0:
            state['epoch'], state['idx'] = epoch, i + 1
            save(state)
            print(f'checkpoint saved (epoch {epoch}, prompt {i+1}, {(time.time()-t0)/60:.0f} min)', flush=True)
    state['epoch'], state['idx'] = epoch + 1, 0
    save(state)
    print(f'=== epoch {epoch} done ({(time.time()-t0)/60:.0f} min) ===', flush=True)

print(f'training done in {(time.time()-t0)/60:.1f} min, {n_oom} OOM skips', flush=True)
json.dump(state['log'], open(os.path.join(res_dir, f'{tag}_train_log.json'), 'w'), indent=1)
tok.save_pretrained(ckpt_dir)
model.eval()
final_summary = do_eval('final')
if init_summary and final_summary:
    print(f'[{tag}] init score {init_summary["score"]} -> final score {final_summary["score"]} '
          f'(tIoU {init_summary["mean_tiou"]} -> {final_summary["mean_tiou"]}, '
          f'acc {init_summary["accuracy"]} -> {final_summary["accuracy"]})', flush=True)
