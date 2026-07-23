# Exploratory analyses: cognitive task parameters vs. psychiatric symptoms

**Scope & hard constraint.** All analyses below are **exploratory** and use **only the
training half** of a preregistered 50/50 split (`data/exploratory/training_data.csv`).
The held-out half is never read; a single confirmatory test on it is a separate,
explicitly-approved future step. Participant-level data and result artifacts are
gitignored; each regenerated artifact carries a provenance sidecar (see `ARTIFACTS.md`).

**Question.** Do fitted cognitive task-model parameters (BART, RDM, CAB, Flanker) carry
information about self-report psychiatric symptom scores (BAARS, PHQ-8, GAD-7, HiTOP),
tested as a *shared axis* (CCA), a *nonlinear predictive* signal (XGBoost), or
*idiosyncratic deviation* from healthy norms (Anna Karenina)?

---

## 0. Shared data, cohort, and views

- **Cognitive view (View A): 30 fitted parameters** — BART (5), RDM (7), CAB (10),
  Flanker (8). Fit diagnostics (`sub_score`, `map_dif`, `abs_map_dif`, `max_rhat`) are
  excluded from the feature set.
- **Symptom view (View B): 20 baseline scale/subscale scores** — `baars`, `phq8`,
  `gad7`, `hitop` summaries + HiTOP/BAARS subscales (`baars_inattentive`,
  `hitop_anhdep`, `hitop_anxwor`, `hitop_hypsom`, `hitop_welbe`, …); `today*` and
  `attnbin` excluded. Derived programmatically from `survey_helpers.SCALES`.
- **Analysis cohort: N = 1298.** Complete-case on both views, then **model-fit
  convergence exclusion**: drop any subject with `{task}__max_rhat > 1.1` on any task
  (1399 → 1298). This identical cohort (subject-set SHA-256 `7c7f63c4…`) is used by the
  CCA, Anna Karenina, and XGBoost analyses.
- **Covariates:** `age`, `sex` (and `age²`, interactions where a linear model needs them).
- **Stratification variable:** `backfilled_prolific_screen_group` (recruitment group).
- `training_data.csv` SHA-256 `e86542ef…7184f80` (matches the audited input).

---

## 1. Participant exclusion funnel (CONSORT)  — `scripts/exclusion_table.py` (branch `exclusion-table`)

- **Design:** reconstruct recruitment → analysis-eligible cohort from `data_quality.csv`,
  survey responses, and per-task fit files. Behavioral criterion `good10` (>10%
  non-response/RT-outlier tolerated).
- **Funnel:** interacted 4987 → provided survey response 4068 → completed all 4 tasks
  3614 → passed behavioral QC (good10) 3023 → successful model fits (all 4 tasks) 2809 →
  50/50 split: **training 1399 / held-out 1410** → training with converged fits
  (`max_rhat≤1.1`) **1298**.
- **Per-reason (behavioral, overlapping):** below-chance accuracy and >10%
  non-response/RT-outliers per task; union failing ≥1 task = 591. Model-fit missing ≥1
  task = 214.
- **Finding:** the analysis-eligible training cohort is 1399; the rhat-converged subset
  used by all downstream analyses is **1298** (101 dropped).

---

## 2. CCA ladder + SHARP inference  — `sharp.py`, `shared_variance.py`, `notebooks/cca_shared_variance.ipynb`

**Goal.** Estimate the *generalizable* leading shared variance between the two views and
test whether added model flexibility raises it.

**Ladder (5 arms), each scored on the same held-out folds:**
- `raw` — regularized linear CCA (rCCA) on standardized features, each view **PCA-reduced
  to 10 components** first.
- `kernel` — kernel CCA (RBF) directly.
- `deep` — Deep CCA: trained MLP encoders (layers 64→32, ≤100 epochs, lr 1e-3, batch 128)
  then linear CCA on the learned representations.
- `fm` — frozen **TabPFN v2** embeddings + linear CCA.
- `tabfm` — frozen **Google TabFM** embeddings + linear CCA.
- **Excluded:** the jointly fine-tuned arms `fm_deep`/`tabfm_deep` are dropped from the
  comparison — a leakage-free version requires nested cross-fit of the fine-tuning itself
  (infeasible), and they did not beat linear in earlier runs.

**Key hyperparameters.** rCCA/KCCA: latent dims `k=5`, ridge `c=0.9`; PCA before linear
CCA `n_pca=10` for `raw`, `n_pca=5` for the FM arms (`fm`/`tabfm`). FM pseudo-target =
within-view PC1 (each view embedded in isolation → no cross-view leakage). Naming: `fm` =
TabPFN v2 (the original foundation-model arm), `tabfm` = Google TabFM.

**Leakage-free FM embeddings (cross-fitting).** For the frozen FM arms, TRAIN rows are
embedded by **5-fold cross-fitting** — each training fold is queried against a
TabPFN/TabFM context built from the *other* folds only, so no queried row appears in its
own labeled support; TEST rows are embedded against the full-train context (never
in-context). A disjointness test asserts this.

**Sampling design (SHARP; Zeng et al. 2026, bioRxiv 2026.05.17.724301).**
- **J = 60 repetitions, K = 5-fold** (paper-scale). Each repetition splits subjects into
  two **disjoint** stratified halves; K-fold CV is run within each half; fold statistics
  (leading held-out **signed** canonical correlation) are averaged to one value per half
  → a pair `(D_Aj, D_Bj)` independent within the repetition.
- **Inference: null-constrained score test** with a **global** (multimodal-safe, grid +
  refine) profile optimizer for `(σ², ρ)`, and **test-inversion** 95% CIs. Correlations
  are **signed** (a negative held-out r means the fitted direction did not generalize; it
  is *not* folded to |r|). The earlier method-of-moments + Wald test (which inflated the
  false-positive rate) is retained only as a calibration reference.
- **Above-chance permutation null (corroborating):** within-strata label permutation, run
  at a coarser `J_perm=5, K_perm=3`; **n_perm** = raw/kernel 1000, deep 200, fm/tabfm 100.
  Because J_perm/K_perm differ from the main estimate, the permutation "observed"
  statistic is reported separately. Arm-level permutation p-values are BH-corrected.

**Findings (N=1298, J=60):**
- Leading held-out canonical r (signed) with score-test 95% CI, and permutation p:
  - `raw`   r = +0.085, CI [+0.018, +0.152], r² = 0.007, p_perm = 0.009
  - `kernel` r = +0.088, CI [+0.029, +0.147], r² = 0.008, p_perm = 0.010
  - `deep`  r = +0.046, CI [+0.016, +0.076], r² = 0.002, p_perm = 0.025
  - `fm`    r = **−0.006**, CI [−0.045, +0.033], r² ≈ 0, p_perm = 0.347  (null even leakage-free)
  - `tabfm` r = +0.059, CI [+0.001, +0.118], r² = 0.004, p_perm = 0.020
- **No flexible arm beats linear `raw`.** Pairwise SHARP vs raw: kernel p=0.93, deep
  p=0.041, fm p=0.016, tabfm p=0.35 — and the significant contrasts are in the direction
  of **raw being better** (the flexible arm is worse). **Nothing survives BH-FDR** across
  the 10 pairwise comparisons (min q ≈ 0.08).
- **Bottom line:** the generalizable shared axis is small (~0.7–0.8% leading-dimension
  shared variance, essentially linear); kernel ≈ raw; foundation-model embeddings do not
  help.

---

## 3. Normative models of the cognitive parameters  — `normative.py`, `scripts/build_normative_deviations.py`

**Goal.** A healthy-volunteer (HV) reference for each cognitive parameter, to score every
subject's deviation (prerequisite for the Anna Karenina test).

- **Reference:** training-half HV only (`backfilled_prolific_screen_group == "hv"`),
  **285 HV** after the `max_rhat>1.1` exclusion.
- **Per-parameter convergence QC:** a parameter is used for a subject only if that
  parameter's MCMC diagnostics pass **rhat ≤ 1.1 and ESS ≥ 400** (finer than the
  subject-level max_rhat filter). ~97.9% of subject×param cells usable.
- **Transforms:** CAB parameters (10, LogNormal priors) → log; the other 20 → Yeo-Johnson
  (fit on HV). Transformed values are clipped to the fitted reference range to guard
  extrapolation; **clipping quantified as negligible** (mean ~0.08%, max ~0.74% of subjects
  on any parameter).
- **Model:** covariate-adjusted mean (design `[1, a, a², sex]`) and residual SD; deviation
  `z`, centile, and extremeness indicator (`|z| > 2`).
- **HV cross-fitting:** HV subjects receive **out-of-fold** deviations (5-fold within HV;
  each HV fold scored by a norm fit on the other folds); non-HV subjects are scored from
  the full HV reference. Avoids in-sample optimism for HV.
- **Calibration:** HV k-fold out-of-fold z ≈ N(0,1) (mean ~0.00, SD ~1.02) across all 30
  parameters.
- **Artifact:** `normative_deviations.pkl` — 1298 subjects × 30 params (`z`, `centile`,
  `indicator`, `qc_ok`, `is_hv`).

---

## 4. Anna Karenina test  — `anna_karenina.py`, `scripts/run_ak.py`, `notebooks/anna_karenina.ipynb`

**Hypothesis.** Symptomatic individuals are each abnormal *in their own way*, so no shared
axis exists — instead the **extremeness** (|z|) of a person's deviation from cognitive
norms should track symptoms.

**Sample.** All 1298 training subjects (deviations vs the HV norm; symptom scores treated
dimensionally). Deviations encoded as **extremeness** (|z|), since signed deviations carry
a random sign under the hypothesis.

**Symptom targets (21).** 20 View-B scores + a **PC1 general factor** (first PC of the
standardized symptom scores), each in **raw** and **age/sex-residualized** variants.
`hitop_welbe` (positive valence) is sign-flipped so higher = more pathology.
**Pre-specified primary:** `max|z|` × **PC1-residualized** (one test).

**Approaches.**
- Unsupervised (from |z|): `max_abs`, `top3_mean`, `top5_mean`, `count_gt2` (# |z|>2),
  `dist` (RMS). Inference: Spearman + **2000-permutation** null (permute the symptom).
- Supervised: **elastic-net** on |z| → true out-of-fold **R² = 1 − SSE/SST**.
  - **Nested CV:** `ElasticNetCV` (l1_ratio ∈ {0.5, 0.9, 1.0}, 30 alphas, inner 3-fold)
    selected **inside each outer training fold** (no outcome leakage).
  - Repeated K-fold: `n_splits=5`, `n_repeats=2`; **identical outer folds reused for the
    observed outcome and every permutation**; median imputation of missing |z| fit
    **inside each outer fold**. Permutation null: **200** permutations rerunning the full
    pipeline.
- Multiple comparisons: BH-FDR within the secondary set per approach; plus a family-wise
  **Westfall–Young step-down maxT** (10,000 shared permutations) over the joint
  **5 unsupervised approaches × 21 targets** family (elastic-net is *not* in the maxT
  family) and a per-approach `within_maxT`.

**Findings (N=1298):**
- **Pre-specified primary** (max|z| → PC1-resid): r = −0.002, **p = 0.54 (null)**.
- **Elastic-net:** null across every target (0 BH survivors; OOF R² ≤ ~0.006; **sparse**
  selection — median 1, mean ~2.3, up to 10 of 30 features across targets). Earlier floor-p
  "hits" were an artifact of selecting hyperparameters on the
  full outcome and an OOF-R² null below zero — they disappear under the nested pipeline.
- **Family-wise maxT:** **0 of 105 survive** (strongest `count_gt2 → hitop_hypsom`,
  adj_p ≈ 0.08).
- **BH-FDR:** a single secondary survivor — the unsupervised **`count_gt2 → hitop_hypsom`
  (raw), r = 0.10, q = 0.02** (cognitive-abnormality burden ↔ hyposomnia / reduced sleep
  need). Its residualized form is not significant and it does not survive maxT. This is
  disclosed, not overclaimed.
- **Bottom line:** the AK hypothesis is **not supported** — prespecified and joint-maxT
  tests are null; one weak, family-dependent per-approach lead remains.

---

## 5. XGBoost full-vs-null nonlinear check  — `xgb_nonlinear.py`, `scripts/run_xgb_nonlinear.py` / `run_xgb_subscores.py` / `run_xgb_kfold.py`, `notebooks/xgb_nonlinear.ipynb`

**Goal.** A direct predictive test for any *nonlinear* cognition→symptom signal the CCA
and AK framings might miss. For each survey score, does a gradient-boosted tree with the
cognitive features out-predict an equally-expressive demographics-only model on held-out
data?

- **Models (per survey score, 20 targets):**
  - null = `age`, `sex`; full = `age`, `sex` + **30 cognitive parameters** (and a second
    feature view: full = age, sex + **4 per-task summary scores** `{task}__sub_score`).
  - Squared/interaction terms omitted (a tree recovers them by splitting), so the arms
    differ only in the cognitive features.
- **XGBoost hyperparameters (fixed, identical across arms):** `max_depth=3`,
  `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`,
  `tree_method="hist"`; number of trees tuned by **early stopping** (cap 2000,
  `early_stopping_rounds=30`) on an inner validation split carved from each training fold.
- **Metric & inference:** held-out **R²** per fold; per-half statistic = fold-averaged
  `R²_full − R²_null`; **SHARP one-sided score test** (full > null) with the global
  optimizer + test-inversion CI. **J = 60, K = 5.** Raw `D_A`/`D_B` half-statistics are
  retained in the artifact. No permutation null needed (analytic score test). BH-FDR
  across the 20 survey scores.
- **Cross-check (descriptive, no SHARP):** full-sample repeated **5-fold** (`n_reps=20`)
  R² comparison — each model trains on ~830 rows/fold vs ~415 in the split-half — to check
  whether more training data surfaces any positive gain.

**Findings (N=1298, J=60):**
- **30-parameter features:** every survey score has a **negative** held-out R² gain;
  **0/20 survive BH-FDR**; min one-sided p = 0.83 (e.g. `hitop_hypsom` p = 0.85).
- **4 task-summary-score features:** same null (19/20 negative; 0/20 survive FDR).
- **Full-sample K-fold cross-check:** more data only shrinks the gaps toward zero — 3/20
  targets tip marginally positive (max +0.006 R²), none distinguishable from noise.
- **Bottom line:** no incremental predictive gain for these features with this XGBoost
  pipeline — robust across feature representations (30 params vs 4 scores) and evaluation
  schemes (SHARP split-half vs full-sample K-fold). Scoped claim, not a model-agnostic
  proof of absence.

---

## 6. Overall synthesis

- Across **three complementary framings** — *shared axis* (CCA ladder), *idiosyncratic
  deviation* (Anna Karenina), and *nonlinear prediction* (XGBoost) — the fitted cognitive
  task parameters carry **very little recoverable information about symptom scores** in the
  training half.
- The only shared structure is a **small, essentially linear** leading canonical
  correlation (~0.7–0.8% shared variance) that no kernel/deep/foundation-model arm
  improves on.
- The lone flagged signal is a weak, family-dependent **`count_gt2 → hitop_hypsom`** (AK,
  q≈0.02, but not surviving family-wise maxT) — disclosed as exploratory, not a confirmed
  effect.
- **Recommendation:** the training-half exploratory results are null / borderline-at-best,
  so spending the reserved held-out half on a confirmatory test is not warranted on this
  evidence.

---

## 7. Inference validity, provenance, and limitations

- **SHARP inference** uses the paper's null-constrained **score test** (not method-of-
  moments + Wald, which inflates the FPR) with a **global** profile optimizer (the profiled
  likelihood is multimodal; a single bounded minimize could pick a non-global mode) and
  **test-inversion** CIs. Validated by null simulations: valid FPR control (conservative
  at low/moderate ρ), a power check, and ~95% CI coverage.
- **Conservatism / power:** at J=60 the score test is still somewhat conservative at
  low/moderate ρ, so the null results reflect limited power, not proof of absence.
- **Excluded arms:** the jointly fine-tuned FM arms are not made leakage-free (infeasible
  nested cross-fit of fine-tuning); conclusions are limited to the frozen-FM and non-FM
  arms.
- **Provenance:** every regenerated artifact has a `*.provenance.json` sidecar
  (schema v3: `analysis_source_commit` + `provenance_stamp_commit`, `training_data.csv` +
  subject-set + artifact + `uv.lock` SHA-256, config, `dirty=false`); see `ARTIFACTS.md`.
  Prior artifacts preserved as `*.r1.*` (pre-round-2) and `*.r2.*` (pre-round-3-recompute).
- **Reproducibility:** all analyses are training-half only; artifacts are reproducible from
  the recorded commit + `uv.lock` via the `scripts/run_*.py` runners.
