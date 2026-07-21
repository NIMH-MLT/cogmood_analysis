"""Build notebooks/xgb_nonlinear.ipynb from the saved results parquet."""

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "xgb_nonlinear.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# XGBoost check: does cognition add *nonlinear* predictive value for symptoms?\n\n"
    "The CCA ladder found only ~1% shared variance and no nonlinear/foundation-model "
    "arm beat the linear baseline; the Anna Karenina deviation test was similarly null. "
    "Those framings look for a *shared axis* or a *deviation-magnitude* signal. This is a "
    "direct predictive check for any nonlinearity they might miss.\n\n"
    "For each survey score we compare two gradient-boosted tree models on held-out data:\n\n"
    "* **null** — features = `age`, `sex`\n"
    "* **full** — features = `age`, `sex` + the 30 cognitive task parameters\n\n"
    "A tree recovers `age²` and `age×sex` from `age` and `sex` by splitting, so the arms "
    "differ **only** in whether the cognitive parameters are available. XGBoost captures "
    "arbitrary nonlinear interactions among the 30 parameters that CCA/rCCA cannot, so a "
    "null result here is strong evidence there is no recoverable nonlinear cognition→symptom "
    "relationship.\n\n"
    "Inference uses the **SHARP** estimator (Zeng et al. 2026) on the paired per-half "
    "difference in held-out R² (full − null): a valid analytic, fold-dependence-aware "
    "p-value, one-sided (full > null). Both arms use the identical algorithm (fixed shallow "
    "trees; number of rounds tuned by early stopping inside each training fold).\n\n"
    "**Training half only** (N=1298 after the `max_rhat>1.1` exclusion, matching the CCA run); "
    "the held-out half is never read."
))

cells.append(nbf.v4.new_code_cell(
    "import polars as pl\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n\n"
    "sns.set_context(\"notebook\")\n"
    "res = pl.read_parquet(\"../data/exploratory/xgb_nonlinear_results.parquet\")\n"
    "print(f\"targets: {res.height} | N: {res['n'][0]} | J={res['J'][0]}, K={res['K'][0]}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Per-target held-out R² gain (full − null), SHARP inference\n\n"
    "`mean_r2_gain` is the SHARP D̄ (mean over halves/reps); `p_one_sided` tests full > null; "
    "`q_fdr` is BH-FDR across the 20 survey scores."
))

cells.append(nbf.v4.new_code_cell(
    "tbl = (res.select(['target', 'mean_r2_gain', 'ci_lo', 'ci_hi',\n"
    "                   'p_one_sided', 'q_fdr', 'rho'])\n"
    "          .with_columns([pl.col(c).round(4) for c in\n"
    "                         ['mean_r2_gain', 'ci_lo', 'ci_hi', 'p_one_sided', 'q_fdr', 'rho']]))\n"
    "print(tbl)\n"
    "n_sig = int((res['q_fdr'] < 0.05).sum())\n"
    "print(f\"\\nTargets surviving BH-FDR (q<0.05): {n_sig} / {res.height}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Held-out R² gain per survey score with SHARP 95% CI\n\n"
    "Points right of 0 mean the cognitive parameters improved held-out prediction. "
    "Red = survives BH-FDR (q<0.05)."
))

cells.append(nbf.v4.new_code_cell(
    "d = res.sort('mean_r2_gain')\n"
    "targets = d['target'].to_list()\n"
    "gain = d['mean_r2_gain'].to_numpy()\n"
    "lo, hi = d['ci_lo'].to_numpy(), d['ci_hi'].to_numpy()\n"
    "q = d['q_fdr'].to_numpy()\n"
    "y = np.arange(len(targets))\n"
    "colors = np.where(q < 0.05, 'crimson', '0.4')\n\n"
    "fig, ax = plt.subplots(figsize=(7, 7))\n"
    "ax.axvline(0, color='k', lw=0.8, zorder=0)\n"
    "for i in range(len(targets)):\n"
    "    ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=1.5, alpha=0.7)\n"
    "ax.scatter(gain, y, c=colors, s=40, zorder=5)\n"
    "ax.set_yticks(y); ax.set_yticklabels(targets)\n"
    "ax.set_xlabel('held-out R² gain (full − null), SHARP 95% CI')\n"
    "ax.set_title('Does adding cognitive parameters improve prediction?')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Cross-check: full-sample K-fold (no SHARP split-half)\n\n"
    "SHARP splits each repetition into two disjoint halves, so each model trains on only "
    "~415 rows here — a concern given the modest N. As a robustness check we repeat the same "
    "full-vs-null XGBoost contrast with **repeated stratified K-fold on the whole training-half "
    "sample** (each model trains on ~830 rows, roughly double), and simply compare the held-out "
    "R² directly. No split-half means the SHARP variance estimator does not apply, so this is "
    "purely descriptive — the question is whether more training data surfaces any *positive* "
    "R² gain. (Still training half only; the held-out half is never read.)"
))

cells.append(nbf.v4.new_code_cell(
    "kf = pl.read_parquet(\"../data/exploratory/xgb_kfold_results.parquet\")\n"
    "kftbl = (kf.select(['target', 'mean_r2_gain', 'ci_lo', 'ci_hi',\n"
    "                    'frac_folds_positive', 'max_fold_gain', 'mean_r2_full', 'mean_r2_null'])\n"
    "           .with_columns([pl.col(c).round(4) for c in\n"
    "                          ['mean_r2_gain', 'ci_lo', 'ci_hi', 'frac_folds_positive',\n"
    "                           'max_fold_gain', 'mean_r2_full', 'mean_r2_null']]))\n"
    "print(kftbl)\n"
    "n_pos = int((kf['mean_r2_gain'] > 0).sum())\n"
    "print(f\"\\nTargets with a positive mean R2 gain: {n_pos} / {kf.height}\")\n"
    "print(f\"Largest mean R2 gain: {kf['mean_r2_gain'].max():+.4f} \"\n"
    "      f\"({kf.sort('mean_r2_gain', descending=True)['target'][0]})\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "The gaps shrink toward zero relative to the split-half test (the extra features needed more "
    "data to stop hurting), and a few targets tip marginally positive — but the fold-level 95% "
    "bands all straddle zero, the fractions of positive folds sit near a coin-flip, and the "
    "full-model absolute R² for the 'positive' targets is ~0. More data does not surface a real "
    "nonlinear cognition→symptom signal."
))

cells.append(nbf.v4.new_code_cell(
    "d = kf.sort('mean_r2_gain')\n"
    "targets = d['target'].to_list()\n"
    "gain = d['mean_r2_gain'].to_numpy()\n"
    "lo, hi = d['ci_lo'].to_numpy(), d['ci_hi'].to_numpy()\n"
    "y = np.arange(len(targets))\n"
    "colors = np.where(gain > 0, 'seagreen', '0.4')\n\n"
    "fig, ax = plt.subplots(figsize=(7, 7))\n"
    "ax.axvline(0, color='k', lw=0.8, zorder=0)\n"
    "for i in range(len(targets)):\n"
    "    ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=1.5, alpha=0.6)\n"
    "ax.scatter(gain, y, c=colors, s=40, zorder=5)\n"
    "ax.set_yticks(y); ax.set_yticklabels(targets)\n"
    "ax.set_xlabel('held-out R² gain (full − null), full-sample K-fold; bars = 2.5–97.5% of folds')\n"
    "ax.set_title('Full-sample K-fold cross-check (no split-half)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Feature view: the 4 per-task summary scores (SHARP)\n\n"
    "The 30 fitted parameters are a high-dimensional, noisy view of task behavior. As a "
    "lower-dimensional alternative we repeat the SHARP full-vs-null test with the full model's "
    "cognitive features set to the **4 per-task summary scores** (`{task}__sub_score`) instead "
    "of the 30 parameters — everything else identical (same N=1298, same age+sex null, same "
    "one-sided SHARP). If a coarse behavioral summary carried symptom signal that the parameters "
    "diluted, it would show here."
))

cells.append(nbf.v4.new_code_cell(
    "ss = pl.read_parquet(\"../data/exploratory/xgb_subscores_sharp_results.parquet\")\n"
    "sstbl = (ss.select(['target', 'mean_r2_gain', 'ci_lo', 'ci_hi', 'p_one_sided', 'q_fdr'])\n"
    "           .with_columns([pl.col(c).round(4) for c in\n"
    "                          ['mean_r2_gain', 'ci_lo', 'ci_hi', 'p_one_sided', 'q_fdr']]))\n"
    "print(sstbl)\n"
    "n_sig = int((ss['q_fdr'] < 0.05).sum())\n"
    "n_pos = int((ss['mean_r2_gain'] > 0).sum())\n"
    "print(f\"\\nPositive mean R2 gain: {n_pos}/{ss.height} | surviving BH-FDR (q<0.05): {n_sig}/{ss.height}\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "Same null. 19/20 targets negative, only `hitop_welbe` a hair positive (+0.0002, p≈0.48); "
    "no target survives FDR. Fewer, cleaner features make the gaps slightly less negative and the "
    "SHARP CIs tighter, but nothing crosses into a real positive gain. The result is robust across "
    "feature representations (30 parameters vs 4 summary scores) as well as evaluation schemes "
    "(SHARP split-half vs full-sample K-fold)."
))

cells.append(nbf.v4.new_markdown_cell(
    "## Conclusion\n\n"
    "**Decisive null.** For every one of the 20 survey scores the held-out R² gain from "
    "adding the 30 cognitive parameters is **negative** — the full model predicts *worse* "
    "than the age+sex baseline, i.e. the parameters act as noise that a gradient-boosted "
    "tree cannot turn into signal. All one-sided SHARP p-values are ≥ 0.52 (most ≈ 1.0), and "
    "**no target survives BH-FDR** (all q = 1.0). The closest-to-zero targets (`hitop_hypsom`, "
    "`hitop_welbe`) are still negative and non-significant.\n\n"
    "This is a strong, model-agnostic corroboration of the earlier results: the CCA ladder "
    "found no shared axis (~1% linear variance, no nonlinear/FM arm beating linear) and the "
    "Anna Karenina deviation test was null. XGBoost — which can capture arbitrary nonlinear "
    "interactions among the parameters — finds no nonlinear cognition→symptom relationship "
    "either. Within this training half, cognitive task parameters carry no recoverable "
    "predictive information about symptom scores beyond demographics.\n\n"
    "The full-sample K-fold cross-check (§3) confirms this is not an artifact of SHARP's "
    "split-half: doubling the per-fold training data only shrinks the negative gaps toward zero "
    "and nudges three targets a hair positive, none distinguishable from noise. Swapping the 30 "
    "parameters for the 4 per-task summary scores (§4) gives the same null. The result therefore "
    "holds across feature representations and evaluation schemes.\n\n"
    "_Training half only; the held-out half remains untouched for any future confirmatory step._"
))

nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT}")
