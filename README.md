# Temporal Leakage in LLM Backtesting — Reproducibility Package

Code, data, and result files for the TMLR submission *Temporal Leakage in LLM
Backtesting: From Statistical Modeling to Leakage-Adjusted Evaluation*.
Appendix F (Table 9) of the paper maps every reported result to its generating
script and output file; the layout here mirrors that table.

## Layout

| Result | Script | Primary output |
|---|---|---|
| Panel build | `redesign/build_panel.py` | `redesign/data/panel_market.jsonl` |
| M1 clean flagships | `redesign/analysis/m1_analyze.py` | `redesign/m1/results.json` |
| M2 Hubble | `M2_hubble/run_m2.py` | `M2_hubble/results.json` |
| M3 corpus and twins | `redesign/m3/build_corpus.py`, `redesign/m3/train_twin.py` | (corpus rebuilt from panel) |
| M3 analysis | `redesign/m3/analyze_m3.py` | `redesign/m3/analysis_full.json` |
| M3 PRC | `redesign/m3/prc.py` | `redesign/m3/prc_differenced.json` |
| M4-F matrix and anchor | `redesign/analysis/m4f_analyze.py`, `redesign/analysis/m4f_anchor.py` | `redesign/m4f/*.json` |
| M4-C | `redesign/m4c/analyze.py` | `redesign/m4c/m4c_results.json` |
| M5 | `redesign/analysis/m5_analyze.py` | `redesign/m5/results.json` |
| E1 synthetic (T1–T6) | `M1_synthetic/run_m1.py` | `M1_synthetic/results.json` |
| Matched PRC | `M6_prc_matched/run_prc_matched.py` | `M6_prc_matched/results.json` |
| Figures | `redesign/analysis/paper_figures.py`, `redesign/analysis/appendix_figures.py` | `figures/*.png` |

## Pre-registration

`redesign/PREREGISTRATION.md` is the frozen pre-analysis plan;
`redesign/DEVIATIONS.md` logs every deviation from it with dates and reasons
(including the M4-F anchor-horizon banding discussed in Section 6.4).

## Data

`redesign/data/panel_market.jsonl` is the assembled forecasting panel: 1,646
resolved binary market questions (Polymarket, Metaculus, Manifold, INFER) with
resolution dates, realized outcomes, and frozen contemporaneous crowd
forecasts, built from the public ForecastBench archive by
`redesign/build_panel.py`.

## Rerunning

Analysis scripts consume the panel and the per-experiment result JSONs
included here and require only `numpy`, `scipy`, `pandas`, and `matplotlib`.
Scripts that query models (panel scoring, M4-F anchor arm) require an
OpenRouter API key; twin training (`redesign/m3/train_twin.py`) requires
Tinker API access. All randomness is seeded; bootstrap and permutation
settings are those documented in Appendix D.

Twin checkpoints (LoRA adapters for the treatment and control twins) exceed
the size limit of anonymous hosting and will be released with the
de-anonymized version.
