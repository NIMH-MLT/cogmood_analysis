# Regenerated analysis artifacts

The analysis data and result artifacts live under `data/exploratory/` and are
**gitignored** (participant data must not be committed). This manifest pins each
regenerated artifact to the source commit, inputs, and configuration that produced
it, so the git-ignored files can be reproduced and audited. Each artifact also
embeds the same provenance record (under a `provenance` key for pickles, or a
`<name>.provenance.json` sidecar) via `cogmood_analysis.provenance`.

Common inputs (this remediation round):
- `training_data.csv` SHA-256: `e86542ef6b3141ddea67beb5fe952016e476b69f726f3f33b9acfd4fd7184f80`
  (matches the externally audited value).
- Analysis cohort: training half, model-fit convergence exclusion `max_rhat > 1.1`
  on any task → **N = 1298**, subject-set SHA-256
  `7c7f63c4167475bba82a12dc4a8dd86c3ce4bdcc98104ac2d830a8da4a8583e9`.

| artifact | analysis | N | source commit | key config | created (UTC) |
|---|---|---|---|---|---|
| `sharp_ladder_results_rhat1p1.pkl` | CCA ladder (7 arms) + SHARP | 1298 | `509551e2` | J=30, K=5; inference=score_test; signed correlations; exclude_rhat_above=1.1; n_perm raw/kernel=1000, deep/fm/tabfm=100, fm_deep/tabfm_deep=50; seed=0 | 2026-07-22 |
| `ak_results.parquet` (+ `.provenance.json`) | Anna Karenina deviation→symptom | 1298 | `6c4b62fe` | unsupervised (max\|z\|/top-k/count/dist) + supervised elastic-net (nested ElasticNetCV, true OOF R²); n_perm_uni=2000, n_perm_sup=200; seed=0 | 2026-07-22 |
| `normative_deviations.pkl` | training-half HV normative deviations | 1298 (285 HV) | (pre-provenance build) | per-parameter QC (rhat≤1.1 & ess≥400) + subject max_rhat>1.1 exclusion; log/Yeo-Johnson transforms | 2026-07-22 |
| `xgb_nonlinear_results.parquet` (+ `.provenance.json`) | XGBoost full-vs-null (age+sex+30 params) + SHARP | 1298 | `d9a0578a` | J=30, K=5; one-sided score test; raw d_a/d_b retained; depth-3 early-stopped XGB; seed=0 | 2026-07-22 |
| `xgb_subscores_sharp_results.parquet` (+ `.provenance.json`) | XGBoost full-vs-null (age+sex+4 task scores) + SHARP | 1298 | `d9a0578a` | J=30, K=5; one-sided score test; raw d_a/d_b retained; seed=0 | 2026-07-22 |
| `xgb_kfold_results.parquet` | full-sample K-fold R² cross-check (no SHARP) | 1298 | (descriptive; pre-provenance) | repeated 5-fold, n_reps=20; descriptive R² gains only (unaffected by the inference fix) | 2026-07-22 |

Notes:
- The `sharp_ladder_results_rhat1p1.pkl` `provenance.dirty` flag is `true` because the
  CCA notebook source patch was uncommitted while the (long) rerun was writing its
  checkpoint; the ladder-affecting code (score test, signed correlations) was already
  committed (`5445c85`, `7d3767b`), and the recorded commit `509551e2` adds only an
  unrelated test fix.
- Rows for the Anna Karenina and XGBoost reruns are appended as those analyses are
  regenerated on their `-fix` branches.
