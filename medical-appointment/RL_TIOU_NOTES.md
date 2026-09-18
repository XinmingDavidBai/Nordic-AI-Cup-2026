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
| x1 | stage 1, exact expected-reward (`jobs/rl_exact.lsf`, TASK=seg), init E9, lr 2e-5, entropy 0.03, 3 epochs | submitted 2026-09-18 ~20:00: per-fold `rl_x1_f0..4` on gpua100 (-W 0:45) + whole-run `medqa_rl_x1` on gpua10 |
| xc1 | stage 2, exact expected-reward (TASK=clause), init = rl2 SFT adapters (trained by whichever of c1/xc1 gets there first) | submitted alongside: `rl_xc1_f0..4` on gpua100 (-W 1:30) + `medqa_rl_xc1` on gpua10 |
| m1 (Mac) | stage 1 exact-EV on the M1 Pro (`jobs/rl_exact_local.py`, tail-only gradient, fp16 base, init E9 fold0 / final), 2 epochs, lr 2e-5, entropy 0.03 | started 2026-09-18 21:20 local, `rl_results/m1/overnight.log`; fold 0 with init/final held-out eval, then the all-39 `final` adapter chained. Decision Sat morning: export + Mac CV replay of fold 0 only if it beats E9's fold-0 HF number by a clear margin |
| c1 | stage 2 (`jobs/rl2.lsf`): per fold SFT warm start (E9 recipe on the clause format) -> GRPO `--task clause` lr 2e-5 T 1.0 beta 0 G 8, 3 epochs | submitted 2026-09-18 evening, gpua10. Its `init` eval (the SFT policy) is itself a result: a learned segment+clause picker vs E9's SFT + embedding argmax |

Queue layout (2026-09-18 ~19:00): gpua10 turned out to be a single physical
GPU shared by everyone (its short pending list serialises), so each run is
queued twice with per-fold `mkdir` locks: a whole-run job on gpua10 and five
per-fold jobs (`-W 1:30` / `2:00`, which backfill far better than 6h requests)
on gpua100. Whichever dispatches first takes the fold; the other skips it.
Jobs: `medqa_rl_grpo` (r1), `medqa_rl_grpo_r2`, `medqa_rl2_clause` (c1) on
gpua10; `rl_r1_f0..4`, `rl2_c1_f0..4` on gpua100. Status: `ssh hpc 'bash -lc "bjobs -w"'`.

**2026-09-18 20:00: DTU HPC service window until Monday 21/09 08:00, no
logins.** All 25 jobs were still PEND when it started. On Monday: check
`bjobs -w` first; if the queue was flushed by the maintenance, resubmit with
the same commands (section 3 / section 5), the fold locks and result files
make that idempotent.

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

## 6. Literature review (2026-09-18, while the first jobs queued)

Searched for work on the same problem shape: a yes/no claim grounded in a
spoken transcript, scored by temporal IoU, small model, ~400 labelled items.

**Directly relevant, and it changes the plan:**

- *Why Sample What You Can Enumerate? Exact Policy Optimization*
  (FGPO, arXiv 2609.10221, Sept 2026). For enumerable action spaces, score
  every candidate by teacher forcing, softmax over the (length-normalised)
  candidate log-probs, and optimise the exact expectation sum_a q(a) r(a) plus
  an entropy bonus (0.03). Beat GRPO in all 15 of their cells (+6.75 avg) and
  diagnosed why: as the policy sharpens, GRPO's sampled rewards collide and the
  group-normalised advantage vanishes ("dead groups", 20-80% of prompts for a
  converged policy). That is exactly our situation: 9 valid completions in
  stage 1, ~37 in stage 2, a peaked SFT init, and a precomputed reward for
  every candidate. Built as `jobs/rl_exact_train.py` + `jobs/rl_exact.lsf`
  (runs x1 = stage 1, xc1 = stage 2). No generation, so a fold takes minutes.
- *Time-R1* (NeurIPS 2025, arXiv 2503.13377): r = IoU * (1-|ds|/T) * (1-|de|/T)
  + binary format reward, GRPO, 2.5K samples chosen by difficulty (Gaussian
  around IoU 0.3, drop IoU>0.7 each epoch), 150-sample LoRA cold start. RL
  60.8 vs SFT-LoRA 51.7 R1@0.5 on Charades (7B). Lesson for us: filter/weight
  prompts where the init already gets IoU>0.7 (they add nothing) and keep the
  ones in the middle.
- *TempSamp-R1* (NeurIPS 2025): mixes the ground-truth completion into each
  GRPO group with an anchored advantage, because on-policy samples rarely hit
  high-tIoU spans. Our exact-expectation trainer makes this moot (the oracle
  candidate is always scored), but it is the fix if GRPO r1/r2 show high
  `frac_reward_zero_std`.
- *Temporal-R1* (github appletea233): variance-aware data selection (repeat
  inference, keep prompts whose samples disagree) and RL 53.9 vs SFT 46.0 mIoU;
  the SFT model lost the ability to emit valid options on other tasks, RL did
  not.
- *Topic-to-Timestamp Alignment by Constrained Evidence Selection*
  (arXiv 2606.20890): transcript domain. Select a chunk ID instead of
  generating a timestamp (fewer invalid outputs, smaller error tail), hybrid
  dense+BM25 retrieval was the biggest win, and their error analysis names the
  same "later restatement vs earliest mention" failure we see. Mistral-7B,
  no fine-tuning. Confirms the index-selection framing; nothing new on the
  duplicate-mention problem.
- *ECPO* (arXiv 2605.21993): a "feasible sampler" that masks invalid
  candidate indices during RL rollouts. Same effect as our reward-table
  fallback plus format penalty; the exact trainer only ever scores valid
  candidates, so this is covered.

**Spoken QA with time-span answers** -- NMSQA / DUAL (Interspeech 2022),
SpeechDPR, GSQA: the metric family is ours (Audio Overlap Score = IoU on
time), but the models are textless speech encoders trained on tens of
thousands of TTS questions; not transferable at 39 conversations.

**Timestamp precision -- checked, not a lever.** Whisper word boundaries in
`tools/words_cache.json` sit within 0.1s of the gold boundaries for 93% (start)
/ 97% (end) of spans; the oracle word-range IoU over the conversation is
0.980. WhisperX / "Whisper has an internal word aligner" (arXiv 2509.09987) /
MFA would not help. The gold spans were evidently produced from the same kind
of word alignment.

**What the analysis found instead: gold spans straddle whisper segments.**
45/195 positives (23%) cover more than one whisper segment. Ceilings (mean
tIoU, positives):

| action space | ceiling |
|---|---|
| clause range inside one top-8 segment (stage 2 as built) | 0.808 |
| clause range allowed to end in the next transcript segment, both in top-8 | 0.868 |
| same, any segments | 0.906 |
| any word range across segments | 0.980 |

This is the same phenomenon E1's union rule patched (+0.004 live) and explains
why "best citable index" (0.657) beats "oracle whole segment" (0.629). Stage-2
v2 should let a range end in the following segment (e.g. `"to_segment"`), or
merge adjacent top-8 segments into one candidate block in the prompt.

**Medical dialogue datasets** (ACI-Bench, the 2026 robot/doctor-patient
dialogue benchmark, PriMock57-style corpora): note generation or response
selection, no evidence-span labels; not usable as extra supervision for this
metric. The E10 synthetic route remains the only data-scaling option.

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
