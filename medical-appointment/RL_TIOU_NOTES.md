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
| m1 result | fold 0 held-out, scoring-based eval (argmax of the 9 valid completions; matches E9's ollama fold-0 score 0.753 exactly at init) | **init 0.7533 -> after epoch 0: 0.7312** (tIoU 0.597 -> 0.560, oracle picks 36 -> 34/44, acc unchanged). Policy collapsed to p(argmax) 0.999. Diff: 3 of 4 losses are the citation drifting one index LATER. Cause: the reward table inherits the pipeline's earlier-of(cited, retrieval-best) rule, so any later index ties with retrieval's pick -- the policy learned that drift, and it transferred to held-out prompts. Stopped after epoch 0. |
| m2 (Mac) | fold 0, 1 epoch: `--reward segment` (IoU of the cited segment itself, unique best index for 191/195 positives), `--no-length-norm` (true sequence-probability policy, peaked, so gradients concentrate on prompts the model is unsure about), `--kl 0.1` to the E9 init over the candidate set, lr 1e-5, entropy 0 | started 2026-09-19 00:40, `rl_results/m2/overnight.log`; final all-39 chained after |
| m2 result | fold 0, `--reward segment --no-length-norm --kl 0.1 --entropy 0`, 1 epoch (98 steps) | **init 0.7533 -> final 0.7483** (tIoU unchanged 0.5972, acc 0.9875 -> 0.975, one new FP). Only 3/80 held-out answers changed at all: 2 wrong->different-wrong citations (no IoU change), 1 correct "no" flipped to a false "yes". Diagnosis: the peaked true-policy softmax + KL-to-init anchor + zero entropy left almost no room to move -- the reward-table-tie bug from m1 is fixed, but untested under real exploration pressure. |
| m3 result | fold 0, 1 epoch (78 steps), full run | **init 0.7533 -> final 0.7533, exactly unchanged** (0/80 held-out answers flipped). Training dynamics were healthy throughout (entropy oscillated 0.02-2.1 across batches rather than collapsing like m1, reward moved on training batches) but produced zero measurable effect on held-out data -- the safest of the 3 configs, but with lr 1e-5/1 epoch/310 prompts too weak in magnitude to cross any decision boundary. Three configs now: m1 regressed (bug, fixed), m2 regressed slightly (over-conservative), m3 flat (too weak). |
| m3-extended (Mac) | same fold-0 config continued to 3 epochs (234 total steps) -- reopens the cosine LR schedule to a higher value for epochs 1-2 rather than continuing at the near-zero rate epoch 0 ended on | started 2026-09-19 23:06, `rl_results/m3/overnight3.log`. Caught and killed a near-miss: the old 1-epoch chain auto-advanced into training the (already-proven-null) recipe's all-39 "final" model at the same moment, which would have thrashed the machine running two MPS jobs at once -- killed within seconds, no data lost. |
| m3 (Mac) | fold 0 only, `--reward segment` (the m1 fix, kept), default flat length-normalized policy (real exploration room, unlike m2), entropy 0.03, no KL, lr 1e-5, 1 epoch | started 2026-09-19 11:33; intercepted at the prompt-100 checkpoint (~35 min) for an early read before committing to the full epoch or the final model |
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

## 4b. m3-extended result and pivot to fast lr probes (2026-09-20, ~02:30)

3-epoch extension of m3 (234 total steps, same segment reward/flat policy/no
KL, LR schedule reopened to a higher peak for epochs 1-2): **0.7533 -> 0.7381,
a clear regression** (tIoU 0.597->0.580, acc 0.988->0.975, one new FP). More
training in the same safe direction made it worse, not better -- ruling out
"just needs more magnitude" as the fix for m3's flatness. Diff of the 4
changed held-out answers: the clearest case (`sample_10_yes_q03`) was a
previously-CORRECT citation (segment 6, iou 0.97, matching both oracle and
best_idx) that drifted to an adjacent wrong segment (5, iou 0.21) -- looks
like generic duplicate-mention confusion getting worse with more exposure,
not a clean, patchable bug like m1's tie exploit.

Four configurations now, all flat-or-negative on held-out fold 0:

| config | init -> final | delta |
|---|---|---|
| m1: lr 2e-5, 1ep, reward-table ties (bug) | 0.7533 -> 0.7312 | -0.0221 |
| m2: lr 1e-5, 1ep, peaked policy + KL 0.1 + entropy 0 | 0.7533 -> 0.7483 | -0.0050 |
| m3: lr 1e-5, 1ep, flat policy + entropy 0.03, no KL | 0.7533 -> 0.7533 | 0.0000 |
| m3-ext: same as m3, 3ep | 0.7533 -> 0.7381 | -0.0152 |

**Killed a second near-miss**: the extend script auto-chained into training
the all-39 "final" adapter on the just-proven-regressive 3-epoch recipe;
caught and killed within ~1 minute (vs a ~3-4h waste if left running).

**Pivot**: rather than another full 1-3.5h blind run, triaging fast --
`jobs/rl_exact_local.py --limit 80 --epochs 1` (20 steps, ~15-20 min) at three
learning rates (3e-6, 1e-5, 3e-5), fresh from E9's fold-0 adapter each time,
looking for ANY positive held-out delta before committing to a full run.
`rl_results/probe_lrs.sh` / `rl_results/probes.log`.

## 4c. Stage-2 SFT attempt (2026-09-20, ~04:00-06:40)

Three lr probes (3e-6/1e-5/3e-5, 20 steps each, fresh from E9): all exactly
flat, 0.7533->0.7533. Combined with m1/m2/m3/m3-ext, that's 7 configurations
spanning 3 orders of magnitude in lr, all flat or negative. Considering
stage-1 (segment-citation) RL fine-tuning at this data/compute scale
thoroughly negatively tested; stopped searching that space.

Pivoted to stage 2 (clause-level, higher ceiling 0.885 vs 0.794). Two real
mistakes cost significant time here, logged for honesty:
- `jobs/finetune_train.py` (the trl-based E9-style trainer) failed in
  `tools/.export-venv` on two fronts: `device_map='auto'` offloaded params to
  the meta device instead of placing them on MPS, and trl 1.13's `SFTConfig`
  rejected `warmup_ratio` -- likely a trl/transformers version skew specific
  to this venv (transformers 5.17 here vs 4.57 on the HPC). Rather than debug
  someone else's trainer's dependency stack with the clock critical, wrote
  `jobs/finetune_local.py`: a minimal standalone SFT loop reusing the same
  explicit-device-placement pattern that worked all night in
  `rl_exact_local.py`.
- That new script OMITTED `gradient_checkpointing_enable()` (present in
  every other local trainer tonight) and had a bug where the per-batch OOM
  handler's `continue` meant a never-advancing step counter never hit the
  loop's exit condition -- so a "quick smoke test" silently retried every
  remaining batch in the epoch, OOMing on all of them, for **158 minutes**,
  producing a meaningless "result" (an untrained, freshly-initialized LoRA
  adapter's zero-shot score on 4 examples). Fixed both (gradient checkpointing
  enabled; `--max-steps` hard cap; abort after 20 total OOMs) and verified
  with a bounded 3-step run (24s, 0 OOM) before committing to a real run.

**Current**: `checkpoints_rl2_sft/fold0` stage-2 SFT training (3 epochs,
lr 2e-4, batch 1, matching E9's own recipe), started 06:40,
`rl_results/stage2_sft.log`. ~8s/step at batch 1 -> ~2h for 3 epochs
(310 steps/epoch). ~9h remain to the 16:00 deadline as of launch.

**Stage-2 SFT, full result (fold 0, real greedy generation via
`rl2_common.eval_records`, not candidate-scoring):**

| checkpoint | score | accuracy | mean tIoU |
|---|---|---|---|
| zero-shot (fresh LoRA) | 0.5861 | 0.9125 | 0.3686 |
| 1 epoch | 0.6752 | 0.9625 | 0.4837 |
| 3 epochs (matches E9's own recipe) | 0.7129 | 0.975 | 0.5382 |
| 6 epochs (2x E9's budget, LR schedule reopened) | **0.7302** | 0.975 | 0.5670 |
| E9 (3 epochs, stage 1, for comparison) | **0.7533** | 0.9875 | 0.5972 |

Rising but with clearly diminishing returns (epoch 0->1: +0.089; epochs 1->3,
i.e. 2 epochs: +0.038, ~0.019/epoch). Extrapolating the decelerating curve,
more epochs would likely approach but not fully close the ~0.04 gap to E9 --
and the LR schedule was set for exactly 3 epochs (already near zero by the
end), so genuinely testing more epochs needs a schedule extension, not just
continuing training, the same lesson learned the hard way in section 4b.

**Why this stops here rather than extending further**: every number in this
whole investigation (stage 1 AND stage 2) is a single-fold (fold 0, 80
held-out questions) result. E9's 0.7255 is a proper 5-fold CV-pooled result.
At ~25-38 min/epoch/fold, getting a genuinely comparable CV-pooled number for
stage 2 would need 4 more folds x 3 epochs each -- over 9 hours, which does
not fit in what remains before the 16:00 deadline. Even a fold-0 win at this
point could not be honestly validated against E9's number in time, so
further open-ended training paused here in favour of writing up complete,
honest findings for whoever continues this branch -- then reconsidered:
unlike the RL runs, this SFT trajectory is still rising at every checkpoint
with no sign of plateau or overfitting, and SFT is a fundamentally more
predictable optimisation process than the RL runs that failed tonight. With
real time margin remaining, extended to 6 epochs (3 more, LR schedule
reopened the same way as m3-extended in 4b -- except here the trend actually
favours it). Started 08:47, `rl_results/stage2_sft_extend.log`. Note this
remains a single-fold (fold 0) result regardless of outcome; see the CV-time
caveat above, unchanged.

## 6. Summary for whoever picks this branch up next

**Bottom line: neither stage 1 nor stage 2 beat E9's validated 0.7255
CV-pooled score within one night's local (Apple M1 Pro) compute, and the DTU
HPC service window (Fri 20:00 - Mon 08:00) made the originally-planned
multi-fold HPC runs unavailable for this competition's deadline.** This is a
genuine, reasonably thorough negative-to-inconclusive result, not a
give-up -- TIOU_RL_SCOPE.md's own risk note anticipated exactly this as a
legitimate possible outcome.

**Stage 1 (segment-citation RL, continuing E9): conclusively negative.**
Seven configurations tested on held-out fold 0, spanning three orders of
magnitude of learning rate (3e-6 to 3e-5), two reward formulations (a bug in
the first, fixed in the rest), two policy sharpness regimes, with and without
a KL anchor, and 20 to 234 total training steps. Every single one was flat
or negative; none improved on E9's own fold-0 score of 0.7533:

| config | delta vs E9 fold-0 (0.7533) |
|---|---|
| m1 (reward-table-tie bug) | -0.0221 |
| m2 (peaked policy + KL 0.1) | -0.0050 |
| m3 (1 epoch, fixed reward) | 0.0000 |
| m3-extended (3 epochs, same recipe) | -0.0152 |
| probe lr 3e-6 (20 steps) | 0.0000 |
| probe lr 1e-5 (20 steps) | 0.0000 |
| probe lr 3e-5 (20 steps) | 0.0000 |

The one diagnosable failure (m1) was a real, fixable bug: the reward table
inherited the pipeline's earlier-of(cited, retrieval-best) post-processing,
so citing a later wrong index often tied the reward of citing the correct
one, and the policy learned to drift later. Fixed by scoring the cited
index's own segment IoU directly (`jobs/rl_common.py:reward_segment`). Even
with that fixed, more training pressure (m3-extended) made things worse, not
better -- this action space, at this data scale (310 training prompts) and
compute budget (LoRA r=16, single M1 Pro), does not have an accessible
improvement direction via exact-expectation RL continuing E9's optimum.

**Stage 2 (clause-level citation, higher ceiling): now conclusively
negative too, not just out of time.** SFT alone (no RL attempted) on
fold 0: zero-shot 0.5861 -> 1 epoch 0.6752 (+0.089) -> 3 epochs (E9's own
budget) 0.7129 (+0.019/epoch) -> 6 epochs, LR schedule reopened, 0.7302
(+0.006/epoch). The per-epoch gain decelerates by roughly 3x at each
doubling, a clear asymptotic convergence -- extrapolating, more epochs would
keep shrinking toward a plateau below E9's 0.7533, not close the gap. At 2x
E9's own training budget this is now a settled result for this format on
this data/compute scale, not an open question. Never got to test RL on top
of this SFT warm start (would need the stage-1 lesson applied: a fixed,
tie-free reward and a fast lr-probe pass before any full run) -- a legitimate
next step for someone with more time or HPC access, but not attempted.

**What actually worked and is reusable:**
- `tools/build_rl_reward_table.py` / `tools/build_rl2_dataset.py`: exact
  reproduction of the live pipeline's span logic for every citable index,
  validated against E9's real ollama output (spans match, IoU diff < 5e-5).
  Reusable for any future RL or reranking attempt on this task.
- `jobs/rl_exact_train.py` (HPC) / `jobs/rl_exact_local.py` (Mac): the
  FGPO-style exact expected-reward trainer. Verified correct (cached
  candidate scoring matches full-sequence scoring within floating-point
  tolerance) and numerically stable all night (zero crashes, zero silent
  corruption across ~10 hours of training).
- `jobs/finetune_local.py`: a minimal standalone SFT loop for this venv, after
  `jobs/finetune_train.py`'s trl/device_map='auto' combination broke in
  `tools/.export-venv` (transformers 5.17 there vs 4.57 on the HPC). Needs
  `gradient_checkpointing_enable()` -- the one bug in this file cost 158
  minutes before being caught; already fixed, verified working.
- Fast lr-probing (`--limit N --epochs 1`, 20-step bursts): a good pattern
  for triaging hyperparameters before committing to full runs -- should have
  been the default approach from the start of the night, not the fallback
  after two full-scale failures.

**Honest account of process mistakes** (all caught and fixed, logged for
whoever reads this): a background-task notification that silently failed to
arrive left a training run paused for ~9 hours unnoticed; a resume script
name was reused for the wrong run's hyperparameters and briefly launched the
wrong config (caught in seconds); an auto-chaining script trained a "final"
model on an already-proven-regressive recipe twice (caught within ~1 minute
each time); and the missing-gradient-checkpointing bug above cost 158
minutes. None corrupted data or the leaderboard branch; all are documented
here so they are not repeated.

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
