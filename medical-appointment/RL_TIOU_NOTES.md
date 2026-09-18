# RL with tIoU reward -- working notes (branch `medical-appointment-rl-tiou`)

Started 2026-09-18 from `TIOU_RL_SCOPE.md` (the scoping note) after reading
`NEXT_STEPS.md`. Both are gitignored on the leaderboard branch; copies live in
this branch's working tree only. This file is committed and is the running
log for this direction. Baseline to beat: **E9, CV-pooled 0.7255**
(leave-conversation-out 5-fold; per fold 0.753 / 0.736 / 0.718 / 0.706 / 0.708).

## 1. What is built (stage 1: RL over the existing action space)

The policy is the same model, same prompt, same JSON schema as the live
pipeline and E9 (`{"answer": bool, "segment": int|null}` over the top-8
retrieved segments). Only the *training signal* changes: GRPO with the real
per-question evaluation metric as the reward, instead of E9's next-token
cross-entropy on an oracle label.

| piece | file | notes |
|---|---|---|
| reward table (Mac, once) | `tools/build_rl_reward_table.py` -> `tools/rl_reward_table.json` | For every question and every citable index, replays `example.ask()`'s deterministic tail (retrieval `best_idx`, the `min(llm, best)` earlier-pick rule, E1 union, embedding `refine_span`) and stores the resulting span + tIoU. Validated: reproduces all 189 spans E9 actually emitted in `results_E9_cv.json`, max IoU diff 5e-5 (rounding). Needs ollama locally (embeddings); nothing on the HPC needs ollama. |
| reward / parse / eval | `jobs/rl_common.py` | `reward_metric = 0.4*[correct] + 1.2*IoU*[positive]` = the question's exact marginal contribution to `0.4*acc + 0.6*mean_tIoU` (tIoU is averaged over 195 positives vs acc over 390, hence the 2x). Unparseable completion: -0.1. `reward_tiou` ablation = IoU on positives / correctness on negatives. Held-out eval = HF greedy decode -> parse -> table lookup, records in the `pipeline_eval_cv.py` schema. |
| trainer | `jobs/rl_grpo_train.py` | trl 1.13 `GRPOTrainer`. `--init e9` continues E9's fold adapter (`checkpoints_e9/fold{k}`, a snapshot of E9's `checkpoints/`), so the result is a single LoRA that `tools/export_gguf.py` can export unchanged. `--init base` = fresh LoRA on the base instruct model (RL without SFT, ablation). Evaluates the init policy and the trained policy on the held-out fold. |
| job | `jobs/rl_grpo.lsf` | queue `gpua10` (8 pending vs ~2000 on gpua100 on 2026-09-18). Env `RUN`, `FOLDS`, `EXTRA` select the run. Resumable per fold. |
| HPC setup | `jobs/hpc_setup_rl.sh` | own workspace `/work3/s234812/nordic_cup_rl/medical-appointment`; reuses the leaderboard workspace's `.venv` and `.hf_cache` read-only. |
| pooling | `tools/rl_pool.py` | per-fold and pooled: E9 (Mac/ollama Q4) vs init (HF bf16 greedy) vs RL final (HF bf16 greedy). `--write` emits a `pickrule.py`-compatible file. |

**Why "init (HF greedy)" matters:** the HPC-side evaluation decodes bf16
weights greedily without ollama's `format: json` grammar or Q4_K_M
quantisation. The init policy *is* E9, so `init` vs the 0.7255 Mac number
calibrates how faithful the HPC proxy is; RL is judged first against `init`
(same decoding path), and a winner is confirmed the E9 way (export one fold at
a time with `tools/export_gguf.py`, `pipeline_eval_cv.py --fold k`,
`ollama rm`, next fold -- disk discipline from NEXT_STEPS.md section 7).

**Headroom in this action space** (from the reward table): if accuracy were
1.0 and every positive cited its best index, the score would be 0.794 (mean
achievable tIoU 0.657, vs E9's 0.560). E9 already answers 97.4% correctly, so
almost all remaining upside is in *which* index gets cited -- exactly what the
reward differentiates and the SFT label (one oracle index) does not.

## 2. Design decisions

- **Reward = the metric, precomputed.** GRPO only needs the within-prompt
  ordering of the 8 sampled completions; the table gives the real
  pipeline tIoU for each, including the E1-union and earlier-pick rules that
  make "best index" differ from "oracle whole-segment index" (0.657 vs 0.629
  mean tIoU). A model trained against next-token loss on the oracle index
  cannot see that difference.
- **Warm start from E9, not from scratch.** GRPO needs reward variance within
  a group. E9 puts nearly all mass on valid JSON, so variance comes from
  genuine disagreement about the index/answer, which is the signal we want.
  `frac_reward_zero_std` in `rl_results/<run>/fold*_train_log.json` is the
  health metric: near 1.0 means the policy is too peaked for the temperature
  (raise `--temperature`) and nothing is learned.
- **No KL by default (`--beta 0`)**, matching trl's default and the DAPO/Dr.GRPO
  practice; clipping (eps 0.2) plus a LoRA-scale LR (2e-5) bound the drift over
  ~230 optimizer steps. trl 1.13 does support KL-to-init for a continued
  adapter (it clones a "ref" adapter), exposed as `--beta`.
- **Prompt rendering is byte-identical to E9's SFT** (`rl_common.render_prompt`
  == `finetune_train.py.to_prompt_completion`), and trl 1.13 never truncates
  prompts (there is no `max_prompt_length` any more), so the ~700-token
  prompts are seen whole.
- **Not repeated from the rejected list:** no prompt text changes, no wider
  top-k, no verbatim quoting, no new picker heuristics. The action space is
  unchanged; only the optimiser is.

## 3. Runs

| run | config | status / result |
|---|---|---|
| smoke | SmolLM2-135M on the login node CPU, 2 steps | wiring OK (init eval -> train -> save -> final eval), continue-adapter path OK with beta 0 and 0.02 |
| r1 | `--init e9` lr 2e-5 T 1.0 beta 0 G 8, 3 epochs, 5 folds | submitted 2026-09-18 evening, gpua10 |
| r2 | `--init e9` lr 1e-5 T 1.3 beta 0.02 G 8, 3 epochs, 5 folds | submitted alongside r1 (diversified in case T=1.0 gives too little group variance) |
| smoke (stage 2) | SmolLM2-135M on the login node CPU: SFT on `rl2_dataset.jsonl` (batch 1; the login node caps process memory), GRPO `--task clause` fresh LoRA, GRPO continuing that adapter | wiring OK |
| c1 | stage 2 (`jobs/rl2.lsf`): per fold SFT warm start (E9 recipe on the clause format) -> GRPO `--task clause` lr 2e-5 T 1.0 beta 0 G 8, 3 epochs | submitted 2026-09-18 evening, gpua10. Its `init` eval (the SFT policy) is itself a result: a learned segment+clause picker vs E9's SFT + embedding argmax |

Adoption rule (same spirit as NEXT_STEPS.md): RL final pooled must beat the
*init* pooled HF number by >= 0.01 AND its Mac-side CV replay must beat
0.7255; then the `final` (all-39) adapter gets exported and the live worst-case
round trip checked <= 50s (unchanged model size, so latency should be E9's
35.5s).

## 4. Stage 2 (built 2026-09-18, job c1): clause-level action space

The refinement picker is the other known gap (0.708 achieved vs 0.823 oracle
given the gold segment, embedding argmax, six rejected heuristic variants in
E4/E8). With RL the model can pick the *clause range* itself: mark clause
boundaries inside each segment in the prompt (`[4a] ... | [4b] ...`) and emit
`{"answer", "segment", "clauses": [i, j]}`; reward = tIoU computed directly from
word timestamps (`example._clauses`, pure Python, no embeddings). Needs an SFT
warm start on the oracle clause range first (new format), then GRPO. Costs
+14% prompt characters (median 1905 -> 2179) and ~+8 output tokens per
question, i.e. well under the latency budget, but the 3B model's known
fragility to format changes (E2, E6) is the risk -- mitigated by the SFT
warm start on the exact format, which is what made E9 work where prompt
edits failed.

Built as: `tools/build_rl2_dataset.py` -> `tools/rl2_dataset.jsonl` (prompt
format `[4] [11.9-16.4s]: (1) clause. (2) clause.`, output
`{"answer", "segment", "from", "to"}`, labels = oracle clause range, 189/195
positives learnable, oracle 0.806), `jobs/rl2_common.py` (closed-form reward,
fallbacks: unknown segment -> whole `best_idx`, bad range -> whole segment),
`jobs/rl_grpo_train.py --task clause`, `jobs/finetune_train.py --data`,
`jobs/rl2.lsf`. Serving it live needs a small stage-2 `ask()` variant in
`example.py` (parse from/to -> clause bounds from the word timestamps) -- not
written yet, only worth doing if c1 beats E9 on CV.

Headroom, computed in closed form from `tools/words_cache.json` + the reward
table (2026-09-18, positives only, score column assumes accuracy 1.0):

| action space | ceiling mean tIoU | ceiling score |
|---|---|---|
| stage 1: best citable segment index, pipeline refine (E9 actual: 0.560) | 0.657 | 0.794 |
| stage 2: whole / single clause / adjacent clause pair, over all top-8 segments | 0.808 | 0.885 |
| any contiguous clause range over all top-8 (36 candidates/question on average) | 0.808 | 0.885 |
| any word range (upper bound, not a practical action space) | 0.850 | 0.910 |

So the pipeline's existing candidate set (whole/clause/adjacent pair) already
carries all the clause-level headroom; longer clause ranges add nothing. The
stage-2 action is therefore "segment index + one of {whole, clause k, clauses
k..k+1}", ~5 candidates per segment.

## 5. Operational notes (HPC)

- Workspace `/work3/s234812/nordic_cup_rl/medical-appointment`, synced from
  this worktree with (from the worktree root):
  `rsync -az --delete --exclude data/audio --exclude '__pycache__' --exclude '.venv*' --exclude 'checkpoints*' --exclude rl_results --exclude logs --exclude 'tools/.export*' --exclude tools/.llama.cpp --exclude 'tools/results_*.json' --exclude 'tools/pipeline_results*.json' medical-appointment/ hpc:/work3/s234812/nordic_cup_rl/medical-appointment/`
  then `ssh hpc 'bash -lc "cd /work3/s234812/nordic_cup_rl/medical-appointment && bash jobs/hpc_setup_rl.sh"'`.
- `bsub`/`bjobs` need a login shell over ssh: `ssh hpc 'bash -lc "bjobs -w"'`.
- Same gotchas as NEXT_STEPS.md: `unset PYTHONPATH` after `module load`,
  `HF_HOME` on /work3 (plus `HF_HUB_OFFLINE=1`: compute nodes have no
  internet), `XDG_CACHE_HOME` on /work3. No CUDA module is needed (no vLLM /
  JIT-compiled deps in this loop).
- Pull results: `rsync -avz hpc-transfer:/work3/s234812/nordic_cup_rl/medical-appointment/rl_results/ medical-appointment/rl_results/`
  then `python3 tools/rl_pool.py rl_results/r1`.
