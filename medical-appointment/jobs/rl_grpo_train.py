"""GRPO training of the segment-citing policy with the real evaluation metric
(0.4*acc + 0.6*tIoU, per question) as the reward -- see RL_TIOU_NOTES.md.

One fold per invocation, leave-conversation-out, same 5-fold split as E9
(tools/finetune_dataset.jsonl `fold`). Per fold this script:
  1. loads the init policy (--init e9: continue E9's fold adapter, the
     validated 0.726 CV model; --init base: fresh LoRA on the base instruct model)
  2. evaluates it greedily on the held-out fold  -> <out>/fold{k}_init.json
  3. runs GRPO on the training folds (8 samples/prompt, reward = table lookup)
  4. evaluates the trained policy on the held-out fold -> <out>/fold{k}_final.json
  5. saves the adapter (single LoRA, exportable by tools/export_gguf.py as-is)

Usage (inside jobs/rl_grpo.lsf, from medical-appointment/):
    python jobs/rl_grpo_train.py --fold 0 --run r1
    python jobs/rl_grpo_train.py --fold all --run r1 --skip-eval   # final model, no held-out
Pool + compare: python3 tools/rl_pool.py rl_results/r1
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('USE_TF', '0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402


BASE_MODEL = os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct')

ap = argparse.ArgumentParser()
ap.add_argument('--fold', required=True, help='0-4 (leave that fold out) or "all"')
ap.add_argument('--run', default='r1', help='run name: results -> rl_results/<run>/, adapters -> checkpoints_rl/<run>/')
ap.add_argument('--init', choices=['e9', 'base'], default='e9', help='e9: continue an SFT adapter from --init-dir; base: fresh LoRA')
ap.add_argument('--init-dir', default=None, help='adapter dir with fold0..4,final (default: checkpoints_e9 for --task seg, checkpoints_rl2_sft for --task clause)')
ap.add_argument('--task', choices=['seg', 'clause'], default='seg', help='seg: stage 1 (segment index, jobs/rl_common); clause: stage 2 (clause range, jobs/rl2_common)')
ap.add_argument('--reward', choices=['metric', 'tiou'], default='metric')
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--beta', type=float, default=0.0, help='KL coef vs the init policy (0 = off)')
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--num-generations', type=int, default=8)
ap.add_argument('--prompts-per-step', type=int, default=4)
ap.add_argument('--max-completion-length', type=int, default=48)
ap.add_argument('--skip-eval', action='store_true')
ap.add_argument('--eval-only', action='store_true', help='just evaluate the init policy on the held-out fold')
ap.add_argument('--max-steps', type=int, default=-1, help='debug: cap optimizer steps')
ap.add_argument('--model', default=None, help='debug: override base model (e.g. a tiny one for a CPU smoke test)')
ap.add_argument('--eval-limit', type=int, default=0, help='debug: evaluate only the first N held-out records')
args = ap.parse_args()

if args.task == 'seg':
    import rl_common as rc
else:
    import rl2_common as rc
E9_DIR = args.init_dir or ('checkpoints_e9' if args.task == 'seg' else 'checkpoints_rl2_sft')

base_model = args.model or BASE_MODEL
res_dir = os.path.join('rl_results', args.run)
ckpt_dir = os.path.join('checkpoints_rl', args.run, 'final' if args.fold == 'all' else f'fold{args.fold}')
os.makedirs(res_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
tag = 'final' if args.fold == 'all' else f'fold{args.fold}'
json.dump(vars(args), open(os.path.join(res_dir, f'{tag}_args.json'), 'w'), indent=1)

records = rc.load_records()
table = rc.load_table()
if args.fold == 'all':
    train_records, held_out = records, []
    init_adapter = os.path.join(E9_DIR, 'final')
else:
    fold = int(args.fold)
    train_records = [r for r in records if r['fold'] != fold]
    held_out = [r for r in records if r['fold'] == fold]
    if args.eval_limit:
        held_out = held_out[:args.eval_limit]
    init_adapter = os.path.join(E9_DIR, f'fold{fold}')
print(f'{tag} [{args.task}]: {len(train_records)} train prompts, {len(held_out)} held out; init={args.init} '
      f'reward={args.reward} lr={args.lr} beta={args.beta} T={args.temperature} G={args.num_generations}', flush=True)

tok = AutoTokenizer.from_pretrained(base_model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

peft_config = None
if args.init == 'e9':
    if not os.path.isdir(init_adapter):
        sys.exit(f'missing init adapter {init_adapter}')
    model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    print(f'continuing SFT adapter from {init_adapter}', flush=True)
else:
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )


def do_eval(name):
    if args.skip_eval or not held_out:
        return None
    t = time.time()
    recs = rc.eval_records(model, tok, held_out, table, fold, desc=name)
    path = os.path.join(res_dir, f'{tag}_{name}.json')
    json.dump(recs, open(path, 'w'), indent=1)
    s = rc.summarize(recs)
    print(f'[{tag} {name}] {s}  ({time.time()-t:.0f}s) -> {path}', flush=True)
    return s


init_summary = do_eval('init')  # for --init base this is the zero-shot base model; still worth a number
if args.eval_only:
    sys.exit(0)

ds = Dataset.from_list([{'prompt': rc.render_prompt(r), 'qid': r['question_id']} for r in train_records])

G = args.num_generations
cfg = GRPOConfig(
    output_dir=os.path.join(ckpt_dir, 'trainer_state'),
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=G,                      # one prompt group per micro-batch
    gradient_accumulation_steps=args.prompts_per_step,  # prompts per optimizer step
    steps_per_generation=args.prompts_per_step,         # generate all of a step's groups in one go
    num_generations=G,
    max_completion_length=args.max_completion_length,   # (trl 1.13 has no max_prompt_length: prompts are never truncated)
    mask_truncated_completions=True,
    temperature=args.temperature,
    top_p=1.0,
    top_k=0,
    beta=args.beta,
    epsilon=0.2,
    scale_rewards='group',
    learning_rate=args.lr,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy='no',
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to=[],
    log_completions=False,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=rc.make_reward_fn(table, args.reward),
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=peft_config,
)
t0 = time.time()
trainer.train()
print(f'training done in {(time.time()-t0)/60:.1f} min', flush=True)

# GRPO's own log has reward/std/frac_reward_zero_std per logging step; keep it
json.dump(trainer.state.log_history, open(os.path.join(res_dir, f'{tag}_train_log.json'), 'w'), indent=1)
trainer.save_model(ckpt_dir)
tok.save_pretrained(ckpt_dir)
print(f'saved adapter to {ckpt_dir}', flush=True)

final_summary = do_eval('final')
if init_summary and final_summary:
    print(f'[{tag}] init score {init_summary["score"]} -> final score {final_summary["score"]} '
          f'(tIoU {init_summary["mean_tiou"]} -> {final_summary["mean_tiou"]}, '
          f'acc {init_summary["accuracy"]} -> {final_summary["accuracy"]})', flush=True)
