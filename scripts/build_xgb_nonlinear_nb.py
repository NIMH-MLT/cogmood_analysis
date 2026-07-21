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
    "_Training half only; the held-out half remains untouched for any future confirmatory step._"
))

nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Wrote {OUT}")
