"""E9: LoRA SFT of unsloth/Llama-3.2-3B-Instruct on the exact (system, prompt)
-> target JSON pairs the live pipeline sends to llama3.2:3b, built by
tools/build_finetune_dataset.py.

Usage (run from medical-appointment/, inside jobs/finetune.lsf):
    python jobs/finetune_train.py --fold 0        # leave fold 0 out, train on 1-4
    python jobs/finetune_train.py --fold all       # train on all 39 conversations

5-fold CV by conversation_id (see build_finetune_dataset.py) so a held-out
fold's conversations never appear in that fold's training data -- 39
conversations is small enough to memorise otherwise.

Uses trl's native prompt-completion dataset format (trl>=1.x): a Dataset with
"prompt"/"completion" string columns, which SFTTrainer auto-detects and masks
the prompt out of the loss (completion_only_loss) -- no separate data
collator needed.
"""
import argparse
import json
import os

os.environ.setdefault('USE_TF', '0')  # avoid the TF/Keras-3 import path in transformers

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.getenv('BASE_MODEL', 'unsloth/Llama-3.2-3B-Instruct')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'tools', 'finetune_dataset.jsonl')

ap = argparse.ArgumentParser()
ap.add_argument('--fold', required=True, help='0-4 to leave that fold out, or "all" to train on every conversation')
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--out', default=None)
ap.add_argument('--batch-size', type=int, default=4)
ap.add_argument('--data', default=DATA_PATH, help='prompt/target jsonl (default: E9 data; stage-2 uses tools/rl2_dataset.jsonl)')
args = ap.parse_args()

records = [json.loads(l) for l in open(args.data)]
if args.fold == 'all':
    train_records = records
    out_dir = args.out or 'checkpoints/final'
else:
    fold = int(args.fold)
    train_records = [r for r in records if r['fold'] != fold]
    held_out = [r for r in records if r['fold'] == fold]
    out_dir = args.out or f'checkpoints/fold{fold}'
    print(f'fold {fold}: {len(train_records)} train, {len(held_out)} held out '
          f'({len(set(r["sid"] for r in held_out))} conversations)')

os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# IMPORTANT: example.py's ask() calls ollama's /api/generate with `system` +
# `prompt` and no `raw: true` flag, so ollama DOES apply the model's Modelfile
# chat template before it ever reaches the weights -- this is NOT a raw
# completion. Training has to replicate that exact template or every serve
# call sees a different token layout than training did. This is ollama's
# literal `llama3.2:3b --template` output (checked 2026-09-18) specialised to
# our always-one-system-plus-one-user-turn, no-tools case; BOS is added by
# the tokenizer automatically, same as ollama's tokenizer layer adds it, so
# it is not written into this string. The completion is the raw JSON the
# model must generate, ending in its actual end-of-turn token (`<|eot_id|>`,
# == tok.eos_token here and == the stop string ollama serves the model with).
def to_prompt_completion(r):
    prompt = (
        '<|start_header_id|>system<|end_header_id|>\n\n'
        'Cutting Knowledge Date: December 2023\n\n'
        f"{r['system']}<|eot_id|>"
        '<|start_header_id|>user<|end_header_id|>\n\n'
        f"{r['prompt']}<|eot_id|>"
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )
    return {'prompt': prompt, 'completion': f"{r['target_json']}{tok.eos_token}"}


ds = Dataset.from_list([to_prompt_completion(r) for r in train_records])

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map='auto',
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
)

training_args = SFTConfig(
    output_dir=os.path.join(out_dir, 'trainer_state'),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy='no',
    bf16=torch.cuda.is_available(),  # True on the GPU nodes (as E9 ran); lets a CPU smoke test run
    max_length=2048,
    packing=False,
    completion_only_loss=True,  # requires the prompt/completion dataset format above
    loss_type='nll',  # plain (unchunked) loss; the default 'chunked_nll' path hits a
                       # torch.distributed.tensor.DTensor check that needs a newer torch
                       # than this repo's pinned 2.2.1 -- not needed at this scale anyway
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tok,
    peft_config=lora_config,
)
trainer.train()

trainer.save_model(out_dir)
tok.save_pretrained(out_dir)
print(f'saved LoRA adapter to {out_dir}')
