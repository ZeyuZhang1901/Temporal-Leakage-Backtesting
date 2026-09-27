# Temporal Leakage in LLM Backtesting — Reproducibility Package

Code, data, and result files for the TMLR submission *Temporal Leakage in LLM
Backtesting: Measurement, Validation, and Adjusted Scores*. Appendix F
(Table 9) of the paper maps every reported result to its generating script and
output file; the layout here mirrors that table. Paths are relative to this
directory.

## Layout

| Result | Script | Primary output |
|---|---|---|
| Panel build | `experiments/build_panel.py` | `experiments/data/panel_market.jsonl` |
| M1 clean flagships | `experiments/analysis/m1_analyze.py` | `experiments/m1/results.json` |
| M2 Hubble | `hubble/run_m2.py` | `hubble/results.json` |
| M3 corpus and twins | `experiments/m3/build_corpus.py`, `experiments/m3/train_twin.py` | (corpus rebuilt from panel) |
| M3 analysis | `experiments/m3/analyze_m3.py` | `experiments/m3/analysis_full.json` |
| M3 PRC | `experiments/m3/prc.py` | `experiments/m3/prc_differenced.json` |
| M4-F matrix and anchor | `experiments/analysis/m4f_analyze.py`, `experiments/analysis/m4f_anchor.py` | `experiments/m4f/*.json` |
| M4-C | `experiments/m4c/analyze.py` | `experiments/m4c/m4c_results.json` |
| M5 | `experiments/analysis/m5_analyze.py` | `experiments/m5/results.json` |
| E1 synthetic (T1–T6) | `synthetic/run_synthetic.py` | `synthetic/results.json` |
| Matched PRC | `prc_matched/run_prc_matched.py` | `prc_matched/results.json` |
| Trend test (E.9) | `experiments/analysis/e9_trends.py` | `experiments/e9/trends.json` |
| GPT-5.5 sensitivity (E.9) | `experiments/analysis/e9_gpt55_sensitivity.py`, `experiments/analysis/e9_gpt55_prediction.py` | `experiments/e9/gpt55_*.json` |
| Power floors (E.7, E.9) | `experiments/analysis/e9_power_controls.py` | `experiments/e9/power_controls.json` |
| Anchor-arm windows (D.5) | `experiments/analysis/e9_anchor_windows.py` | `experiments/e9/anchor_windows.json` |
| Main-text figures | `experiments/analysis/paper_figures.py` | `experiments/figures/*.png` |
| Appendix figures | `experiments/analysis/appendix_figures.py` | `experiments/figures/*.png` |

`livecodebench/` holds the code-domain robustness results read by the
appendix figures.

## Pre-registration

`experiments/PREREGISTRATION.md` is the frozen pre-analysis plan;
`experiments/DEVIATIONS.md` logs every deviation from it with dates and reasons
(including the M4-F anchor-horizon band discussed in Section 7.3 and
Appendix D.5).

## Data

`experiments/data/panel_market.jsonl` is the assembled forecasting panel: 1,646
resolved binary market questions (Polymarket, Metaculus, Manifold, INFER) with
resolution dates, realized outcomes, and frozen contemporaneous crowd
forecasts, built from the public ForecastBench archive by
`experiments/build_panel.py`. `hubble/hubble_testset.csv` holds the public
Hubble model evaluations used by M2.

The M4-F anchor arm and the E.9 analyses also read the ForecastBench processed
forecast sets (public). Set `FORECASTBENCH_FSETS` to their location, or place
them at `experiments/data/forecastbench-processed-forecast-sets/`.

## Rerunning

Analysis scripts consume the panel and the per-experiment result JSONs
included here and require only `numpy`, `scipy`, `pandas`, and `matplotlib`.
Scripts that query models (panel scoring, M4-F anchor arm, matched PRC) require
`OPENROUTER_API_KEY` in the environment and write their responses to
`experiments/cache/`; the matched-PRC question file is not redistributed (set
`PRC_QUESTIONS` to its location). Twin training
(`experiments/m3/train_twin.py`) requires Tinker API access. All randomness is
seeded; bootstrap and permutation settings are those documented in Appendix D.

Twin checkpoints (LoRA adapters for the treatment and control twins) exceed
the size limit of anonymous hosting and will be released with the
de-anonymized version.
