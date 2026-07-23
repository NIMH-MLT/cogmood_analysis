# Regenerated analysis artifacts

The analysis data and result artifacts live under `data/exploratory/` and are
**gitignored** (participant data must not be committed). This manifest pins each
regenerated artifact to the source commit, inputs, and configuration that produced it.
Each artifact also has a `<name>.provenance.json` **sidecar** (the authoritative
record) carrying: schema version, source commit, `dirty` flag, `training_data.csv`
SHA-256, subject-set SHA-256, run config, `uv.lock` (environment) SHA-256, the
artifact's own SHA-256, and a timestamp. Sidecars were restamped from a clean
committed tree (`scripts/restamp_provenance.py`), so all read `dirty=false`.

Common inputs (round-2):
- `training_data.csv` SHA-256: `e86542ef6b3141ddea67beb5fe952016e476b69f726f3f33b9acfd4fd7184f80`
  (matches the externally audited value).
- Analysis cohort: training half, `max_rhat > 1.1` excluded → **N = 1298**, subject-set
  SHA-256 `7c7f63c4167475bba82a12dc4a8dd86c3ce4bdcc98104ac2d830a8da4a8583e9` — identical
  across the CCA, AK, and XGBoost artifacts.
- Environment pinned by `uv.lock` (hash recorded in each sidecar).

| artifact | analysis | N | artifact SHA-256 (head) | key config |
|---|---|---|---|---|
| `sharp_ladder_results_rhat1p1.pkl` | CCA ladder (5 arms) + SHARP | 1298 | `1524ac27fe92` | **J=60, K=5**; null-constrained **score test (global optimizer)** + test-inversion CIs; **signed** correlations; arms raw/kernel/deep/fm/tabfm (fine-tuned arms dropped); **cross-fitted** FM embeddings; n_perm raw/kernel=1000, deep=200, fm/tabfm=100; seed=0 |
| `ak_results.parquet` | Anna Karenina deviation→symptom | 1298 | `6856ce50ba77` | unsupervised (max\|z\|/top-k/count/dist) + elastic-net (nested ElasticNetCV, true OOF R², **in-fold imputation**, **identical folds for obs+perm**); n_perm_uni=2000, n_perm_sup=200; seed=0 |
| `normative_deviations.pkl` | training-half HV normative deviations | 1298 (285 HV) | `b61e85bc7432` | per-parameter QC (rhat≤1.1 & ess≥400) + subject max_rhat>1.1 exclusion; **HV out-of-fold (cross-fit)** deviations; transform clipping negligible (mean 0.05%, max 0.62%) |
| `xgb_nonlinear_results.parquet` | XGBoost full-vs-null (age+sex+30 params) + SHARP | 1298 | `8e65a9c3ebe7` | **J=60**, K=5; one-sided score test (global optimizer); raw d_a/d_b retained; depth-3 early-stopped XGB; seed=0 |
| `xgb_subscores_sharp_results.parquet` | XGBoost full-vs-null (age+sex+4 task scores) + SHARP | 1298 | `87662c94c553` | **J=60**, K=5; one-sided score test; raw d_a/d_b retained; seed=0 |
| `xgb_kfold_results.parquet` | full-sample K-fold R² cross-check (no SHARP) | 1298 | `a616a65258d3` | repeated 5-fold, n_reps=20; descriptive R² gains only (unaffected by the inference fix) |

Preserved round-1 copies (before the round-2 reruns) are kept alongside as `*.r1.*`
(gitignored) for comparison; their sidecars are left verbatim.

Reproduce any artifact from the recorded commit + `uv.lock`, e.g.:
```
SV_MAX_RHAT=1.1 uv run python scripts/run_sharp_ladder.py   # ladder (N=1298, J=60)
uv run python scripts/build_normative_deviations.py && AK_NJOBS=6 uv run python scripts/run_ak.py
uv run python scripts/run_xgb_nonlinear.py && uv run python scripts/run_xgb_subscores.py
```
