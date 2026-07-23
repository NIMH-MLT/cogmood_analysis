# Regenerated analysis artifacts

The analysis data and result artifacts live under `data/exploratory/` and are
**gitignored** (participant data must not be committed). This manifest pins each
regenerated artifact to its inputs and configuration. Each artifact has a
`<name>.provenance.json` **sidecar** (the authoritative record) carrying: schema version
(v3), **`analysis_source_commit`** (code that produced the artifact) and
**`provenance_stamp_commit`** (commit that last wrote/refreshed the sidecar), a `dirty`
flag, the `training_data.csv` + subject-set + `uv.lock` SHA-256s, the run config, the
artifact's own SHA-256, and a timestamp. Sidecars are restamped from a clean committed tree
(`scripts/restamp_provenance.py`), so all read `dirty=false`.

Common inputs:
- `training_data.csv` SHA-256: `e86542ef6b3141ddea67beb5fe952016e476b69f726f3f33b9acfd4fd7184f80`
  (matches the externally audited value).
- Analysis cohort: training half, `max_rhat > 1.1` excluded → **N = 1298**, subject-set
  SHA-256 `7c7f63c4167475bba82a12dc4a8dd86c3ce4bdcc98104ac2d830a8da4a8583e9` — identical
  across the CCA, AK, and XGBoost artifacts.
- Environment pinned by `uv.lock` (hash recorded in each sidecar).

Round-3 note: the SHARP inference (CIs, comparisons, one-sided p) for the ladder and the two
XGBoost artifacts was **recomputed from the saved fold-difference arrays** with the
boundary-aware global optimizer (`scripts/recompute_sharp_inference.py`) — no predictive
models were refit — so their `analysis_source_commit` reflects the recompute, while the
underlying `D_A`/`D_B` come from the J=60 round-2 fits.

| artifact | analysis | N | artifact SHA-256 (head) | key config |
|---|---|---|---|---|
| `sharp_ladder_results_rhat1p1.pkl` | CCA ladder (5 arms) + SHARP | 1298 | `2d7aaf0e3142` | J=60, K=5; null-constrained **score test (global optimizer)** + test-inversion CIs; **signed** correlations; arms raw/kernel/deep/fm/tabfm (fine-tuned dropped); **cross-fitted** FM embeddings; n_perm raw/kernel=1000, deep=200, fm/tabfm=100; seed=0 |
| `ak_results.parquet` | Anna Karenina deviation→symptom | 1298 | `6856ce50ba77` | unsupervised (max\|z\|/top-k/count/dist) + elastic-net (nested ElasticNetCV, true OOF R², in-fold imputation, identical folds obs+perm); n_perm_uni=2000, n_perm_sup=200; seed=0 |
| `ak_maxstat.parquet` | AK family-wise maxT (joint) | 1298 | `8be5cd03235c` | Westfall–Young step-down maxT over 5 unsupervised approaches × 21 resid targets; 10,000 perms; seed=0. **0 joint survivors** (top count_gt2→hitop_hypsom adj_p≈0.077) |
| `ak_within_maxT.parquet` | AK maxT (within-approach) | 1298 | `7706c24d3db5` | per-approach step-down maxT; 10,000 perms; seed=0. 1 within-approach survivor (count_gt2→hitop_hypsom within_maxT≈0.038) |
| `normative_deviations.pkl` | training-half HV normative deviations | 1298 (285 HV) | `b61e85bc7432` | per-parameter QC (rhat≤1.1 & ess≥400) + subject max_rhat>1.1 exclusion; **HV out-of-fold (cross-fit)** deviations; transform clipping negligible (~0.08% mean, ~0.74% max) |
| `xgb_nonlinear_results.parquet` | XGBoost full-vs-null (age+sex+30 params) + SHARP | 1298 | `50538d6ba464` | J=60, K=5; one-sided score test (global optimizer); raw d_a/d_b retained; depth-3 early-stopped XGB; seed=0 |
| `xgb_subscores_sharp_results.parquet` | XGBoost full-vs-null (age+sex+4 task scores) + SHARP | 1298 | `81796442d134` | J=60, K=5; one-sided score test; raw d_a/d_b retained; seed=0 |
| `xgb_kfold_results.parquet` | full-sample K-fold R² cross-check (no SHARP) | 1298 | `a616a65258d3` | repeated 5-fold, n_reps=20; descriptive R² gains only |

Preserved backups (gitignored, each with a sidecar): `*.r1.*` = pre-round-2 copies
(`*-fix-r1` tags), `*.r2.*` = pre-round-3-recompute copies (`*-fix-r2` tags), for comparison.

Reproduce from the recorded commit + `uv.lock`:
```
SV_MAX_RHAT=1.1 uv run python scripts/run_sharp_ladder.py     # ladder (N=1298, J=60)
uv run python scripts/build_normative_deviations.py && AK_NJOBS=6 uv run python scripts/run_ak.py
uv run python scripts/run_xgb_nonlinear.py && uv run python scripts/run_xgb_subscores.py
uv run python scripts/recompute_sharp_inference.py            # refresh SHARP inference from saved D
uv run python scripts/regen_ak_maxt.py                        # canonical AK maxT
```
