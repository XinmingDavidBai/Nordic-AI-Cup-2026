"""Minimal standalone SFT (no trl) for the M1 Pro, written 2026-09-20 ~04:00
after jobs/finetune_train.py hit two environment issues in tools/.export-venv
(device_map='auto' offloading to meta/disk instead of MPS; trl 1.13's
SFTConfig rejecting warmup_ratio -- likely a trl/transformers version skew
specific to this venv). Reuses the exact same explicit-device-placement
pattern that jobs/rl_exact_local.py used reliably all night, just with plain
next-token cross-entropy on the completion (prompt masked out) instead of
the multi-candidate exact-expectation objective -- ordinary SFT, matching
E9's own recipe (LoRA r=16, same target modules), pointed at any --data file
in the {system, prompt, target_json} record schema (tools/finetune_dataset.jsonl
for stage 1 [not needed, E9 already exists], tools/rl2_dataset.jsonl for stage 2).

Usage (from medical-appointment/, in tools/.export-venv):
    python jobs/finetune_local.py --fold 0 --data tools/rl2_dataset.jsonl --out checkpoints_rl2_sft/fold0
    python jobs/finetune_local.py --fold 0 --data tools/rl2_dataset.jsonl --out ... --resume
"""
import argparse
import json
import math
import os
import random
import sys
import gc
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

import rl_common as rc  # noqa: E402  (render_prompt is identical for both tasks)


def render_prompt_phi3(r):
    """ollama's exact phi3.5:3.8b template (checked via `ollama show phi3.5:3.8b
    --template`): <|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n
    The stop token is <|end|>, not the tokenizer's base eos (<|endoftext|>)."""
    return f"<|system|>\n{r['system']}<|end|>\n<|user|>\n{r['prompt']}<|end|>\n<|assistant|>\n"


TEMPLATES = {'llama': rc.render_prompt, 'phi3': render_prompt_phi3}
TURN_END = {'llama': '<|eot_id|>', 'phi3': '<|end|>'}
LORA_TARGETS = {
    'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'phi3': ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],  # Phi-3.5 fuses q/k/v and gate/up
}

BASE_MODEL = os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct')

ap = argparse.ArgumentParser()
ap.add_argument('--fold', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--batch-size', type=int, default=4)
ap.add_argument('--save-every', type=int, default=100)
ap.add_argument('--eval-batch', type=int, default=16, help='generation batch size for held-out eval (training always uses batch 1). Lower for bigger models / tight memory.')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--eval-limit', type=int, default=0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max-steps', type=int, default=-1, help='hard cap on successful optimizer steps, for bounded smoke tests')
ap.add_argument('--init-adapter', default=None, help='continue SFT from this existing adapter dir instead of a fresh LoRA (e.g. checkpoints_e9/fold0)')
ap.add_argument('--focal-gamma', type=float, default=0.0,
                help='focal-loss exponent applied per training EXAMPLE (not per token): loss *= (1-p)^gamma, '
                     'p = exp(-plain CE) = model\'s current probability on the whole target sequence. '
                     '0 = plain CE (default); >0 down-weights examples the model already gets confidently right. '
                     'DIAGNOSED INEFFECTIVE on this data 2026-09-20: E9 already fits its own training set near-'
                     'perfectly (max training loss observed 0.054 over 29 sampled steps), so there is no genuine '
                     'hard-vs-easy signal in TRAINING loss to exploit -- use --hard-boost instead, which is driven '
                     'by domain knowledge (the reward table), not the model\'s own (already-overfit) confidence.')
ap.add_argument('--base-template', choices=list(TEMPLATES), default='llama',
                help='prompt/chat template + turn-end token to train with (must match how the base model is served)')
ap.add_argument('--hard-boost', type=float, default=1.0,
                help='fixed loss multiplier (not confidence-based) for stage-1 training positives where '
                     'tools/rl_reward_table.json has oracle_idx != best_idx -- i.e. plain retrieval alone would '
                     'cite the wrong segment, so getting these right requires genuine LLM disambiguation, not just '
                     'the pipeline. 52/150 fold-0 training positives qualify. 1.0 = no boost (default).')
args = ap.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = torch.float16 if device == 'mps' else torch.float32
random.seed(args.seed)
torch.manual_seed(args.seed)

records = [json.loads(l) for l in open(args.data)]

HARD_IDS = set()
if args.hard_boost != 1.0:
    _table = json.load(open('tools/rl_reward_table.json'))
    HARD_IDS = {q for q, e in _table.items() if e.get('gold') and e.get('oracle_idx') != e.get('best_idx')}
    print(f'--hard-boost {args.hard_boost}: {len(HARD_IDS)} question_ids flagged hard (oracle_idx != best_idx)', flush=True)
if args.fold == 'all':
    train_records, held_out = records, []
else:
    fold = int(args.fold)
    train_records = [r for r in records if r['fold'] != fold]
    held_out = [r for r in records if r['fold'] == fold]
    if args.eval_limit:
        held_out = held_out[:args.eval_limit]

os.makedirs(args.out, exist_ok=True)
state_path = os.path.join(args.out, 'train_state.json')

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if args.base_template == 'phi3':
    # eval_records() (shared with the llama path) stops generation at tok.eos_token_id;
    # phi3.5's actual turn-end token per ollama's own template is <|end|>, not the base
    # tokenizer eos (<|endoftext|>) -- point eos_token_id at <|end|> so greedy eval matches
    # how ollama actually serves it. pad stays a real, rarely-generated token (<|endoftext|>).
    tok.pad_token_id = tok.convert_tokens_to_ids('<|endoftext|>')
    tok.eos_token_id = tok.convert_tokens_to_ids('<|end|>')
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
from peft import PeftModel
if args.resume and os.path.isfile(os.path.join(args.out, 'adapter_config.json')):
    model = PeftModel.from_pretrained(model, args.out, is_trainable=True)
    print(f'resumed adapter from {args.out}', flush=True)
elif args.init_adapter:
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    print(f'continuing SFT from {args.init_adapter}', flush=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=LORA_TARGETS[args.base_template]))
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
trainable = [p for p in model.parameters() if p.requires_grad]
print(f'{args.fold}: {len(train_records)} train, {len(held_out)} held out; '
      f'trainable {sum(p.numel() for p in trainable)/1e6:.1f}M; device {device} {dtype}', flush=True)


_render = TEMPLATES[args.base_template]
_turn_end = TURN_END[args.base_template]


def encode_batch(recs):
    p_ids = [tok(_render(r), add_special_tokens=True)['input_ids'] for r in recs]
    c_ids = [tok(r['target_json'] + _turn_end, add_special_tokens=False)['input_ids'] for r in recs]
    seqs = [p + c for p, c in zip(p_ids, c_ids)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(recs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(recs), L), dtype=torch.long)
    mask = torch.zeros((len(recs), L), dtype=torch.float)  # 1 on completion tokens (loss target)
    for k, (p, c) in enumerate(zip(p_ids, c_ids)):
        s = p + c
        ids[k, :len(s)] = torch.tensor(s)
        att[k, :len(s)] = 1
        mask[k, len(p):len(s)] = 1
    return ids.to(device), att.to(device), mask.to(device)


def ce_loss(ids, att, mask, gamma=0.0):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = ids[:, 1:]
    m = mask[:, 1:]
    lp = torch.gather(F.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)
    tok_loss = -(lp * m)
    plain = tok_loss.sum() / m.sum().clamp(min=1)  # mean per-token CE for this example (batch size 1 assumed for focal)
    if gamma <= 0:
        return plain, plain.item()
    p = torch.exp(-plain.detach())  # model's current per-token-geometric-mean probability on the target
    weight = (1 - p).clamp(min=1e-4) ** gamma
    return weight * plain, plain.item()


def do_eval(name, rc2):
    if not held_out:
        return None
    t = time.time()
    model.eval()
    recs = rc2.eval_records(model, tok, held_out, rc2.load_table(), None if args.fold == 'all' else int(args.fold), batch_size=args.eval_batch, desc=name)
    model.train()
    path = os.path.join(args.out, f'{name}.json')
    json.dump(recs, open(path, 'w'), indent=1)
    s = rc2.summarize(recs)
    print(f'[{name}] {s} ({time.time()-t:.0f}s) -> {path}', flush=True)
    return s


# module for eval: same field schema as rl_common, but this file's records
# already carry 'want'/'gold'/etc. (stage-1 schema) or need rl2_common's
# richer pipeline_outcome (stage-2 schema with clauses). Pick by filename.
if 'rl2' in args.data:
    import rl2_common as rc2
else:
    rc2 = rc
if args.base_template != 'llama':
    rc2.render_prompt = _render  # eval_records() calls rc2.render_prompt; keep it in sync with training

state = {'epoch': 0, 'idx': 0, 'step': 0}
if args.resume and os.path.isfile(state_path):
    state = json.load(open(state_path))
    print(f'resuming at epoch {state["epoch"]} idx {state["idx"]} step {state["step"]}', flush=True)

if not (args.resume and state['step'] > 0):
    do_eval('init', rc2)

opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train_records) / args.batch_size)
total_steps = int(steps_per_epoch * args.epochs)
if args.max_steps > 0:
    total_steps = min(total_steps, args.max_steps)
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)
for _ in range(state['step']):
    sched.step()

t0 = time.time()
model.train()
n_oom = 0
print(f'training: {total_steps} steps, batch {args.batch_size}, lr {args.lr}', flush=True)

for epoch in range(state['epoch'], math.ceil(args.epochs)):
    order = list(range(len(train_records)))
    random.Random(args.seed + epoch).shuffle(order)
    start = state['idx'] if epoch == state['epoch'] else 0
    step_in_epoch = start // args.batch_size
    for i in range(start, len(order), args.batch_size):
        if state['step'] >= total_steps:
            break
        if n_oom > 20:
            print(f'ABORTING: {n_oom} consecutive/total OOMs -- batch size {args.batch_size} does not fit; lower it', flush=True)
            sys.exit(1)
        batch = [train_records[j] for j in order[i:i + args.batch_size]]
        try:
            ids, att, mask = encode_batch(batch)
            opt.zero_grad(set_to_none=True)
            loss, plain_loss = ce_loss(ids, att, mask, gamma=args.focal_gamma)
            is_hard = args.hard_boost != 1.0 and any(r['question_id'] in HARD_IDS for r in batch)
            if is_hard:
                loss = loss * args.hard_boost
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
        except RuntimeError as ex:
            if 'out of memory' not in str(ex).lower():
                raise
            n_oom += 1
            print(f'OOM at idx {i}, skipping batch ({n_oom} total)', flush=True)
            opt.zero_grad(set_to_none=True)
            if device == 'mps':
                torch.mps.empty_cache()
            continue
        loss_item = loss.item()
        del ids, att, mask, loss
        gc.collect()
        state['step'] += 1
        if state['step'] % 10 == 0:
            print(json.dumps({'step': state['step'], 'epoch': epoch, 'idx': i + args.batch_size,
                               'loss': plain_loss, 'weighted_loss': loss_item, 'hard': is_hard if args.hard_boost != 1.0 else None,
                               'lr': sched.get_last_lr()[0], 'elapsed_min': round((time.time() - t0) / 60, 1)}), flush=True)
        if device == 'mps':
            torch.mps.empty_cache()
        if (i + args.batch_size) % args.save_every < args.batch_size:
            state['epoch'], state['idx'] = epoch, i + args.batch_size
            model.save_pretrained(args.out)
            json.dump(state, open(state_path, 'w'))
            print(f'checkpoint saved (epoch {epoch}, idx {state["idx"]}, {(time.time()-t0)/60:.0f} min)', flush=True)
    state['epoch'], state['idx'] = epoch + 1, 0
    model.save_pretrained(args.out)
    json.dump(state, open(state_path, 'w'))
    print(f'=== epoch {epoch} done ({(time.time()-t0)/60:.0f} min) ===', flush=True)
    if state['step'] >= total_steps:
        break

print(f'training done in {(time.time()-t0)/60:.1f} min, {n_oom} OOM skips', flush=True)
tok.save_pretrained(args.out)
model.eval()
do_eval('final', rc2)
