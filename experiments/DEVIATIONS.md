# Deviations and refinements relative to PREREGISTRATION.md

1. **M3 screening rule (refinement, 2026-07-27).** The pre-registered screen
   ("base P(YES) within +-0.15 of crowd anchor, or |logit diff| < 1") was a
   proxy for the design requirement that injected documents be the marginal
   information source. Measured on the actual base model
   (Qwen3.5-35B-A3B-Base), that proxy keeps only 342/1,103 pre-T* questions
   because the base deviates from the crowd on most questions while being
   genuinely uncertain (base Brier 0.167 vs crowd 0.067). The direct
   operationalization - exclude questions the base is confident AND correct
   on (p>0.85 & Y=1, or p<0.15 & Y=0; 115 questions) - implements the stated
   goal and keeps 988 questions. Decided BEFORE any twin training run.

2. **M3 base model (forced by catalog, 2026-07-27).** The plan named
   Qwen3.5-35B-A3B (instruct); the Tinker catalog only offers
   Qwen3.5-35B-A3B-Base at this scale. The Base variant is used. This
   strengthens the pretraining-contamination analogy (raw-document
   continued training) and P(YES) is read exactly from ' Yes'/' No'
   continuation logprobs with a fixed few-shot prefix instead of sampling.

3. **G2 pilot model (improvement, 2026-07-27).** Pre-registration listed a
   Qwen3.5-4B pilot with 9B escalation. Since the 35B MoE trains at $1.177/M
   (cheaper than the 9B dense at $1.463/M), the pilot runs on the target
   35B model itself with the 10% corpus (~$25-30/pair) - strictly more
   informative at comparable cost. The escalation ladder is unnecessary;
   a failed pilot proceeds directly to format/LR fixes or the contingency.

4. **M4-F target swap (pre-registered rule applied).** DeepSeek-V3.2's
   cutoff could not be verified at Tier 1/2; per the pre-registered decision
   rule it was replaced by GPT-5.4 (2025-08-31, Tier 1).

5. **M1 statistic (implementation).** CIs are cluster bootstrap percentile
   intervals (10,000 draws, clusters = source x resolution month) rather
   than BCa; BCa with clustered resampling is not well-defined in standard
   tooling.

6. **M4-F clean-anchor horizon banding (added robustness, 2026-07-28).**
   Archived real-time anchors for late-resolving questions can come from
   more recent submission rounds (shorter horizons), which could mimic a
   cutoff jump. Added a primary variant restricting anchor horizons to
   10-75 days (balances mean pre/post horizons); the unbanded variant is
   reported alongside. Decided after observing the horizon imbalance,
   before interpreting the GPT-5.4 arm (which the banding nullifies) as a
   detection.

7. **M4-F protocol placebo-jump (added robustness, 2026-07-28).**
   DeepSeek-V3.1's own retro-minus-real-time score difference is tested
   for jumps at candidate boundaries. No jump at the December 2025
   boundary (+0.026 [-0.011,+0.064], n=42/419) where the GPT-5.5
   signature appears; worst drift anywhere is -0.036 at March 2026
   (opposite sign, one third of the banded GPT-5.5 signal).

8. **M3 concentration ex-ante binning (added robustness, 2026-07-28).**
   The pre-registered tercile variable (control-twin realized b0) risks
   selection-on-noise: binning on the minuend of the contrast can inflate
   top-bin estimates mechanically. Added binning by ex-ante crowd
   uncertainty c0(1-c0), which is outcome-free. Conclusion unchanged in
   kind: lower two terciles match the law; the hardest tercile falls
   short under BOTH binnings (d64: +0.053 obs vs +0.127 pred ex-ante),
   so the shortfall is real w-heterogeneity, not a binning artifact.
   Per-tercile fitted w at dose 64 (identity form): 0.55 / 0.44 / 0.18.

9. **M3 spillover mechanism check (added analysis, 2026-07-28).** The
   treatment twin's P(YES) shift relative to control is uniform across
   pools (-0.088 injected / -0.096 dose-0 / -0.097 post-T*), matching the
   83% NO rate of injected outcomes (base-rate learning). Cost asymmetry
   is explained by pool base rates: pre-T* YES rate 0.17-0.18 (lean
   cost-free, dose-0 null), post-T* YES rate 0.355 (lean costs 0.028
   Brier). Dose-0 differencing removes the shift exactly.
