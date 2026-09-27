# Pre-Registration: Experiment Program M1-M5 (TMLR paper redesign)

Registered: 2026-07-27, before any full experimental run.
All predictions, estimators, and success gates below are fixed before data collection.
Deviations must be documented in `DEVIATIONS.md` with a rationale.

## 0. Verified platform facts (Phase 0, completed 2026-07-27)

### 0.1 Model cutoffs (source-tiered)

Verified via OpenRouter models API (`knowledge_cutoff` field) and the June-2026
cross-vendor fact-check (metehan.ai, itself sourced to vendor docs). Tier 1 =
vendor docs / API metadata; Tier 2 = consistent secondary reporting.

| Model (OpenRouter slug)              | Cutoff        | Tier | in/out $ per M | Role |
|--------------------------------------|---------------|------|----------------|------|
| openai/gpt-5                         | 2024-09-30    | 1 (OR API + OpenAI docs) | 1.25 / 10.00 | M1 clean row; M4-C target |
| openai/gpt-5-mini                    | 2024-05-31    | 1 (OR API)   | 0.25 / 2.00   | M4-C target |
| openai/gpt-5.4                       | 2025-08-31    | 1 (OpenAI docs) | 2.50 / 15.00 | M4-F target (replaces DeepSeek-V3.2, whose cutoff is unverifiable) |
| openai/gpt-5.5                       | 2025-12-01    | 1 (OpenAI docs + OR API) | 5.00 / 30.00 | M4-F target; M1 clean row |
| google/gemini-3.1-pro-preview        | 2025-01 (Jan) | 1 (Google docs) | 2.00 / 12.00 | M1 clean row; M5 control candidate |
| moonshotai/kimi-k2.6                 | 2025-04 (Apr) | 2 (vendor specs via reviews; some directories disagree) | 0.646 / 2.72 | M4-F target (boundary sweep mandatory); M1 row (footnoted); M5 secondary target |
| moonshotai/kimi-k2-0905              | 2024-12-31    | 2 (OR API)   | 0.60 / 2.50   | M4-C target |
| deepseek/deepseek-chat-v3.1          | 2025-03-31    | 1 (OR API)   | 0.25 / 0.95   | M4-F target |
| deepseek/deepseek-chat-v3-0324       | 2024-07-31    | 1 (OR API)   | 0.27 / 1.12   | M4-C target |
| minimax/minimax-m3                   | 2026-01       | 1 (HF chat template) | 0.30 / 1.20 | M4-F/M4-C control; M1 clean row; M5 secondary |
| anthropic/claude-opus-4.7            | 2026-01       | 1 (Anthropic docs) | 5.00 / 25.00 | secondary control (subsample only, cost) |
| qwen/qwen3.5-35b-a3b                 | "2026" year-level | 1 (DashScope prompt) | 0.15 / 1.00 | M5 primary target; M3 base (twin differencing handles vague cutoff) |
| z-ai/glm-4.7                         | unpublished   | -    | 0.40 / 1.75   | M5 secondary target (target cleanliness not required) |
| openai/gpt-3.5-turbo                 | 2021-09-30    | 1    | 0.50 / 1.50   | M5 deliberate weak-control arm |
| openai/gpt-4o (-2024-08-06)          | 2023-10-31    | 1 (OR API) | 2.50 / 10.00 | M4-C continuity target (current paper's E3) |

Pre-registered decision rule: models whose cutoff cannot be documented at
Tier 1 or Tier 2 are NOT used in any role that requires cleanliness
(M1 clean rows, M4 controls, M5 control). DeepSeek-V3.2 was excluded from
M4-F targets for exactly this reason and replaced by GPT-5.4.

### 0.2 Tinker catalog (verified against tinker-docs pricing page)

| Tinker ID | Type | Train $/M | Sample $/M | Role |
|-----------|------|-----------|------------|------|
| Qwen/Qwen3.5-35B-A3B-Base | MoE base | 1.177 | 1.335 | (contingency QA twins) |
| Qwen/Qwen3.5-35B-A3B      | MoE instruct | 1.177 | 1.335 | M3 twin base |
| Qwen/Qwen3.5-4B           | dense | 0.737 | - | G2 pilot |
| Qwen/Qwen3.5-9B(-Base)    | dense | 1.463 | - | G2 escalation |
| Kimi-K2.6                 | MoE | 4.84 | 5.49 | deferred replication |

LoRA rank 32 default; `create_lora_training_client(base_model=...)`;
per-token logprob scoring via `forward_backward` without `optim_step`.
LR for Qwen3.5 via cookbook `get_lr(model_name)`.

### 0.3 ForecastBench archive audit (raw files, 2026-07-27)

32 question sets + 32 resolution sets (2024-07-21 .. 2026-07-19), CC BY-SA.

- 1,646 unique resolved binary market questions (Polymarket 40%+, Metaculus,
  Manifold, INFER), 100% with `freeze_datetime_value` crowd anchors.
  By quarter: 24Q3 29, 24Q4 63, 25Q1 57, 25Q2 208, 25Q3 86, 25Q4 349,
  26Q1 311, 26Q2 461, 26Q3 81.
- 2,452 dataset-source questions (ACLED/FRED/yfinance/Wikipedia/DBnomics),
  31,405 resolved (question, date) pairs, dense from 2025Q2.
- Consequence: forecasting RD is powered at 2025 cutoffs only; 2024-cutoff
  models are covered in the code domain (LiveCodeBench v6).

## 1. Claim contract

| Claim | Primary experiment | Pre-registered headline statistic |
|-------|--------------------|------------------------------------|
| C2a: recency alone fails the naive check | M1 | naive pre/post gap > 0 with 95% CI excluding 0 for >= 3 provably-clean flagships |
| C1 + C2b: mechanism and law B = E[b0 w(2-w)] | M2a (Hubble anchor) + M3 law arm | monotone dose-response; law-predicted tercile bins inside observed CIs; T3/T1 >= 4 |
| C3 (RD/DiD recover known temporal leakage) | M3 recovery arm | twin-DiD slope in [0.9, 1.1] vs injected truth; naive check starred at dose 0, ours not |
| C3-PRC ground truth | M3 PRC arm | PRC CI excludes 0 only in (treatment twin x injected questions) cell |
| C3-RD in the wild | M4-F + M4-C | >= 2 families with starred own-cutoff diagonal; permutation p < 0.01 |
| Discipline null | M5 | all adjusted DiD within +-0.01 of 0, upper bound <= 0.03; weak-control arm spuriously positive; power >= 90% at 0.05 |

## 2. Per-experiment protocols

### M1 - Provably-clean flagships fail the naive check

- Panel: ForecastBench market questions; for each model keep only questions
  resolving strictly after its documented cutoff (+30-day guard band).
- Rows: GPT-5 (post 2024-10-31), Gemini-3.1-Pro (post 2025-02-28),
  GPT-5.5 (post 2025-12-31), MiniMax-M3 (post 2026-01-31),
  Kimi-K2.6 (post 2025-05-31; Tier-2 cutoff footnote).
- Pre/post boundary: median resolution date of each model's clean window.
- Elicitation: probability of YES, reasoning disabled, temperature 0,
  single call, strict numeric parse with one retry.
- Statistic: naive gap = mean Brier(pre) - mean Brier(post); block bootstrap
  (cluster by question source x month), 10,000 draws, BCa 95% CI.
- Prediction: all rows positive; >= 3 of 5 starred.
- Gate to proceed to M4/M5 spend: G1 smoke test passes (parse failure < 5%).

### M3 - Forecasting twins with injected temporal leakage (centerpiece)

- Base: Qwen/Qwen3.5-35B-A3B. Screen: base P(YES) within +-0.15 of crowd
  anchor, or |logit difference| < 1; target pool 1,000-1,300 questions.
- Doses {0,1,4,16,64} randomized across questions, stratified by control-twin
  b0 tercile and resolution month. One treatment corpus (outcome documents),
  one control corpus (same documents, outcomes scrubbed; matched source/
  topic/length within 10%).
- Two LoRA runs (treatment, control), rank 32, identical schedule and seed.
- Scoring: P(YES) elicited by sampling (n=8, temperature 0.7, mean) and
  cross-checked by option-token logprobs.
- Recovery estimators: naive pre/post at injection boundary; RD at boundary
  (+2 placebo dates, +-60d); twin-DiD; realistic DiD (OpenRouter
  qwen/qwen3.5-35b-a3b as non-twin control).
- Law analyses: per-question dose-response; b0-tercile x dose concentration
  vs b0*w(2-w) with w fitted once at dose r=16 (not per bin); convex-pull
  ratio curve vs w(2-w); PRC 2x2 with paraphrases (DeepSeek-V3.1-generated,
  validated by back-checking meaning preservation).
- Success gates: listed in claim contract. Fallback: report empirical
  L(b0, r) shape and scope Assumption 2 (a finding, not a failure).

### M4-F - Wild RD on the forecasting panel

- Targets: DeepSeek-V3.1 (2025-03-31), Kimi-K2.6 (2025-04, sweep mandatory),
  GPT-5.4 (2025-08-31), GPT-5.5 (2025-12-01).
  Exploratory row: Gemini-3.1-Pro (2025-01, thin boundary; flagged).
- Controls: MiniMax-M3 primary; Claude-Opus-4.7 on a 500-question subsample.
- Estimator: crowd-anchored excess Brier, DiD vs control pool, local
  bandwidth +-90d (sensitivity: 60/120d), placebo boundaries at +-6 months.
- Specificity matrix: each target evaluated at every other target's cutoff;
  permutation test over cutoff assignments (10,000 draws).
- Clean-anchor arm: where ForecastBench holds real-time forecasts from a
  target (or same-family) model, paired comparison of archived real-time
  vs our retrospective forecast, split by pre/post-cutoff resolution.
- Prediction: >= 2 starred diagonal cells; off-diagonal and controls null.
  A null target stays in the matrix (heterogeneous w).

### M4-C - LiveCodeBench v6 generality + 2024-cutoff arm

- Targets: GPT-4o (Oct'23, continuity), GPT-5-Mini (May'24),
  DeepSeek-V3-0324 (Jul'24), GPT-5 (Sep'24), Kimi-K2-0905 (Dec'24).
- Controls: MiniMax-M3, DeepSeek-V3.2 (both released after the v6 window
  ends Apr'25 - control use only requires release date after window, which
  is Tier-1 documented, not cutoff precision).
- Protocol: public per-problem submissions first; self-generate missing
  models on OpenRouter (single greedy completion, pass@1); grade locally
  against released tests. Harness gate: reproduce a published pass@1 within
  +-3 points on one overlapping model before trusting self-generated scores.
- Same specificity matrix + permutation test as M4-F.

### M5 - Disciplined null (forecasting)

- Target: qwen/qwen3.5-35b-a3b. Boundary: 2025-07-01 (within its plausible
  training window). Panel: all market questions resolving 2025-01..2026-06.
- Control selection (pre-registered protocol): among {Gemini-3.1-Pro,
  GPT-5}, pick the model minimizing pre-boundary profile distance
  (calibration curve L2 + topic-stratified Brier L2) on post-cutoff-clean
  questions; report the selection table.
- Secondary targets: Kimi-K2.6, DeepSeek-V3.2, GLM-4.7, MiniMax-M3.
- Weak-control arm: GPT-3.5-turbo (predicted spurious positive).
- Robustness: boundary sweep, recall probe, power injection (semi-synthetic
  leakage at 0.05 / 0.09), one-sided upper bound.
- Prediction: adjusted DiD null for all secondary targets; naive gap
  positive for at least the primary target.

## 3. Pilot gates (no run > $100 without a passed gate)

- G1 (API smoke, ~$5): 50 questions x each OpenRouter model; parse failure
  and refusal < 5%; provider stability; M4-C harness reproduction check.
- G2 (M3 pilot, ~$45): Qwen3.5-4B twins on ~10% corpus. Gates:
  (a) memorization: treatment logprob on inserted docs >> control;
  (b) signal: movement toward injected outcomes at r=64, CI excludes 0;
  (c) null: r=0 CI contains 0;
  (d) direction: movement toward realized outcome, not generic confidence;
  (e) recency: both twins beat un-tuned base on post-cutoff-topic questions.
  Fail (a): fix format/LR, re-pilot. Fail (b) only: escalate 9B (~$90).
  Fail at 9B: trigger contingency QA-insertion arm; document.
- G3 (budget checkpoint): reconcile actuals vs budget before any deferred
  replication.

## 4. Budget (list prices)

M1 ~$20; M4-F ~$35-45; M4-C ~$15-20; M5 ~$20-35;
M3 training 2 x 250-350M x $1.177/M = $590-825 + scoring ~$20;
pilots ~$50-140. Program total ~$750-1,050 (~$1,400 with 2x retry buffer).
