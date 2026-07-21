"""XGBoost full-vs-null check for nonlinear cognition->symptom prediction.

The CCA ladder found only ~1% shared variance and no nonlinear/foundation-model
arm beat the linear baseline; the Anna Karenina deviation test was similarly null.
Both look for a *shared axis* or a *deviation-magnitude* signal. This module is a
direct predictive check for any nonlinearity those framings might miss: for each
survey score, does a gradient-boosted tree model that has access to the cognitive
parameters predict held-out symptom scores better than an equally-expressive model
built on demographics alone?

For each survey score we compare two XGBoost regressors:

* ``null``  - features = ``age``, ``sex``.
* ``full``  - features = ``age``, ``sex`` + the 30 cognitive task parameters.

The squared/interaction terms a linear model would need (``age^2``, ``age*sex``)
are omitted: a tree recovers them from ``age`` and ``sex`` by splitting. The two
arms therefore differ only in whether the cognitive parameters are available, so a
performance gain isolates their (possibly nonlinear, interactive) contribution.

Inference uses the **SHARP** estimator (:func:`cogmood_analysis.sharp._sharp_moments`,
Zeng et al. 2026) on the paired per-half performance *difference* (full - null),
giving a valid analytic p-value under fold dependence - no permutation null needed.
The test is one-sided (full > null): we ask whether cognition *adds* predictive
value. Both arms use the identical algorithm - fixed shallow trees with the number
of boosting rounds tuned by early stopping on an inner validation split carved from
each outer training fold (never touching that fold's held-out test rows).

Exclusion matches the final CCA run: subjects with ``{task}__max_rhat > 1.1`` on
any task are dropped (N=1298). **Training half only**; the held-out half is never
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import os

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBRegressor

from cogmood_analysis import shared_variance as sv
from cogmood_analysis.anna_karenina import _bh_fdr
from cogmood_analysis.sharp import _sharp_moments

#: Demographic covariates in the null (and full) design. Squared/interaction
#: terms are omitted - trees recover them from ``age`` and ``sex`` by splitting.
COVARIATE_COLUMNS = ("age", "sex")

#: Fixed XGBoost hyperparameters shared by both arms. Only the number of trees
#: is tuned (early stopping), so null and full differ purely in feature set.
XGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=2000,
    early_stopping_rounds=30,
    tree_method="hist",
    n_jobs=1,
)


@dataclass
class XGBData:
    """Row-aligned covariates, cognitive parameters, and symptom targets."""

    covars: NDArray[np.float64]  # (n, len(COVARIATE_COLUMNS))
    params: NDArray[np.float64]  # (n, 30) cognitive parameters
    symptoms: NDArray[np.float64]  # (n, n_targets)
    strata: NDArray[np.str_]  # (n,) stratification labels
    sub_ids: NDArray[np.str_]
    covar_cols: list[str]
    param_cols: list[str]
    symptom_cols: list[str]
    n_total: int  # rows before complete-case filtering


def load_xgb_data(
    csv_path: str | os.PathLike,
    covar_columns: Sequence[str] = COVARIATE_COLUMNS,
    param_columns: Sequence[str] | None = None,
    symptom_columns: Sequence[str] | None = None,
    strata_column: str = sv.STRATA_COLUMN,
    exclude_rhat_above: float | None = 1.1,
    rhat_tasks: Sequence[str] = sv.RHAT_TASKS,
) -> XGBData:
    """Load covariates, cognitive parameters, and symptom scores from the CSV.

    Uses the same rhat-convergence exclusion and complete-case discipline as
    :func:`cogmood_analysis.shared_variance.load_views`; the default
    ``exclude_rhat_above=1.1`` reproduces the N=1298 CCA analysis sample.
    Complete-case is enforced jointly across covariates, parameters, and every
    symptom target so all XGBoost fits run on an identical row set.
    """
    csv_path = Path(csv_path)
    df = pl.read_csv(csv_path, infer_schema_length=10000)
    available = set(df.columns)

    covar_cols = [c for c in covar_columns if c in available]
    param_cols = [c for c in (param_columns or sv.VIEW_A_COLUMNS) if c in available]
    symptom_cols = [c for c in (symptom_columns or sv.VIEW_B_COLUMNS) if c in available]
    if not covar_cols:
        raise ValueError("No covariate columns found in the CSV.")
    if not param_cols:
        raise ValueError("No cognitive-parameter columns found in the CSV.")
    if not symptom_cols:
        raise ValueError("No symptom columns found in the CSV.")
    if strata_column not in available:
        raise ValueError(f"Strata column {strata_column!r} not found in the CSV.")

    n_total = df.height

    # model-fit convergence exclusion (any task max_rhat over threshold), mirroring
    # shared_variance.load_views so the analysis sample matches the CCA run.
    if exclude_rhat_above is not None:
        rhat_cols = [f"{t}__max_rhat" for t in rhat_tasks if f"{t}__max_rhat" in available]
        if not rhat_cols:
            raise ValueError("exclude_rhat_above set but no {task}__max_rhat columns found.")
        df = df.filter(
            pl.all_horizontal([pl.col(c) <= exclude_rhat_above for c in rhat_cols])
        )

    analysis_cols = [*covar_cols, *param_cols, *symptom_cols]
    sub = df.select(["sub_id", strata_column, *analysis_cols]).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in analysis_cols]
    )
    sub = sub.drop_nulls(subset=analysis_cols)

    return XGBData(
        covars=sub.select(covar_cols).to_numpy().astype(np.float64),
        params=sub.select(param_cols).to_numpy().astype(np.float64),
        symptoms=sub.select(symptom_cols).to_numpy().astype(np.float64),
        strata=sub.select(pl.col(strata_column).fill_null("missing"))
        .to_numpy()
        .ravel()
        .astype(str),
        sub_ids=sub.select("sub_id").to_numpy().ravel().astype(str),
        covar_cols=covar_cols,
        param_cols=param_cols,
        symptom_cols=symptom_cols,
        n_total=n_total,
    )


def _fit_r2(
    X_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    X_te: NDArray[np.float64],
    y_te: NDArray[np.float64],
    seed: int,
) -> float:
    """Fit one XGBoost regressor with early stopping; return held-out R^2.

    The number of boosting rounds is tuned on an inner validation split carved
    from the *training* rows only (``X_tr``/``y_tr``), so the outer held-out set
    ``X_te`` is never used for model selection.
    """
    Xf, Xv, yf, yv = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    model = XGBRegressor(random_state=seed, **XGB_PARAMS)
    model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    return float(r2_score(y_te, model.predict(X_te)))


def _rep_half_diff(
    X_null: NDArray[np.float64],
    X_full: NDArray[np.float64],
    y: NDArray[np.float64],
    strata: NDArray[np.str_],
    K: int,
    rep_seed: int,
) -> tuple[float, float]:
    """One SHARP repetition: fold-averaged ``R2_full - R2_null`` per disjoint half.

    Parallels :func:`cogmood_analysis.sharp._eval_one_rep`: split into two disjoint
    strata-balanced halves, then run K-fold CV within each half. For every fold both
    models are fit on the same train rows (each tuned by early stopping) and scored
    on the same held-out rows; the per-fold R^2 differences are averaged. Returns the
    two per-half means ``(D_A, D_B)``.
    """
    hs = StratifiedKFold(n_splits=2, shuffle=True, random_state=rep_seed)
    halves = [idx for _, idx in hs.split(X_null, strata)]
    out: list[float] = []
    for half_idx in halves:
        Xn_h, Xf_h, y_h, strat_h = (
            X_null[half_idx], X_full[half_idx], y[half_idx], strata[half_idx],
        )
        inner = StratifiedKFold(n_splits=K, shuffle=True, random_state=rep_seed)
        fold_diffs: list[float] = []
        for tr, te in inner.split(Xn_h, strat_h):
            r2_null = _fit_r2(Xn_h[tr], y_h[tr], Xn_h[te], y_h[te], seed=rep_seed)
            r2_full = _fit_r2(Xf_h[tr], y_h[tr], Xf_h[te], y_h[te], seed=rep_seed)
            fold_diffs.append(r2_full - r2_null)
        out.append(float(np.mean(fold_diffs)))
    return out[0], out[1]


def sharp_xgb(
    covars: NDArray[np.float64],
    params: NDArray[np.float64],
    y: NDArray[np.float64],
    strata: NDArray[np.str_],
    J: int = 30,
    K: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict[str, float]:
    """SHARP test that the full (covars+params) model beats the null (covars) model.

    Runs ``J`` disjoint-half repetitions (:func:`_rep_half_diff`), each yielding a
    paired ``(D_A, D_B)`` of fold-averaged held-out ``R2_full - R2_null``. The SHARP
    moment estimator (:func:`cogmood_analysis.sharp._sharp_moments`) gives the mean
    gain and its fold-dependence-aware variance; the p-value is one-sided (full > null).

    Returns a dict with ``mean_r2_gain`` (D_bar), ``se``, ``ci_lo``/``ci_hi`` (95%),
    ``z``, ``p_one_sided``, ``rho``, ``sigma2``, ``J``, ``K``.
    """
    X_null = covars
    X_full = np.column_stack([covars, params])

    def _one(j: int) -> tuple[float, float]:
        return _rep_half_diff(X_null, X_full, y, strata, K, rep_seed=seed + j)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        pairs = Parallel(n_jobs=n_jobs)(delayed(_one)(j) for j in range(J))
    else:
        pairs = [_one(j) for j in range(J)]

    D_A = np.array([p[0] for p in pairs], dtype=float)
    D_B = np.array([p[1] for p in pairs], dtype=float)
    D_bar, var_Dbar, sigma2, rho = _sharp_moments(D_A, D_B)
    se = float(np.sqrt(var_Dbar)) if np.isfinite(var_Dbar) else float("nan")
    z = D_bar / se if se and se > 1e-12 else 0.0
    p_one_sided = float(1.0 - stats.norm.cdf(z))
    return {
        "mean_r2_gain": D_bar,
        "se": se,
        "ci_lo": D_bar - 1.96 * se,
        "ci_hi": D_bar + 1.96 * se,
        "z": float(z),
        "p_one_sided": p_one_sided,
        "rho": rho,
        "sigma2": sigma2,
        "J": J,
        "K": K,
    }


def run_xgb_nonlinear(
    data: XGBData,
    targets: Sequence[str] | None = None,
    J: int = 30,
    K: int = 5,
    seed: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run the full-vs-null SHARP test for every survey score.

    Returns a tidy table (one row per target) with the mean held-out R^2 gain, its
    SHARP 95% CI, the one-sided p-value, and a BH-FDR q-value across the target
    family. Targets default to every symptom column loaded in ``data``.
    """
    target_cols = list(targets) if targets is not None else list(data.symptom_cols)
    rows: list[dict[str, float | str]] = []
    for name in target_cols:
        y = data.symptoms[:, data.symptom_cols.index(name)]
        res = sharp_xgb(
            data.covars, data.params, y, data.strata, J=J, K=K, seed=seed, n_jobs=n_jobs,
        )
        rows.append({"target": name, "n": int(data.covars.shape[0]), **res})
        if verbose:
            print(
                f"{name:22s}  R2 gain={res['mean_r2_gain']:+.4f} "
                f"[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  "
                f"p={res['p_one_sided']:.4f}"
            )

    tbl = pl.DataFrame(rows)
    q = _bh_fdr(tbl["p_one_sided"].to_numpy())
    return tbl.with_columns(pl.Series("q_fdr", q)).sort("p_one_sided")


# --- full-sample K-fold cross-check (no SHARP split-half) --------------------


def kfold_r2_gain(
    covars: NDArray[np.float64],
    params: NDArray[np.float64],
    y: NDArray[np.float64],
    strata: NDArray[np.str_],
    K: int = 5,
    n_reps: int = 20,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Full-sample repeated stratified K-fold: held-out ``R2_full - R2_null``.

    A robustness check for :func:`sharp_xgb`, which trains each model on only a
    disjoint half (~415 rows here). This uses the **whole training-half sample**
    for K-fold CV, so each model trains on ~(K-1)/K of it (~830 rows) - roughly
    double the data - to see whether more data surfaces any positive gain. Still
    training half only; the held-out half is never touched. No split-half means the
    SHARP variance estimator does not apply, so this reports only descriptive
    statistics of the fold-level R^2 differences, not a p-value.

    Returns the mean held-out R^2 gain and its spread across the ``n_reps * K``
    folds, the fraction of folds with a positive gain, the single best fold gain,
    and the mean held-out R^2 of each arm.
    """
    X_null = covars
    X_full = np.column_stack([covars, params])

    def _one_rep(rep: int) -> tuple[list[float], list[float], list[float]]:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed + rep)
        gains, r2_full, r2_null = [], [], []
        for tr, te in skf.split(X_null, strata):
            a = _fit_r2(X_null[tr], y[tr], X_null[te], y[te], seed=seed + rep)
            b = _fit_r2(X_full[tr], y[tr], X_full[te], y[te], seed=seed + rep)
            r2_null.append(a)
            r2_full.append(b)
            gains.append(b - a)
        return gains, r2_full, r2_null

    if n_jobs != 1:
        from joblib import Parallel, delayed

        reps = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(r) for r in range(n_reps))
    else:
        reps = [_one_rep(r) for r in range(n_reps)]

    gains = np.array([g for rep in reps for g in rep[0]], dtype=float)
    r2_full = np.array([v for rep in reps for v in rep[1]], dtype=float)
    r2_null = np.array([v for rep in reps for v in rep[2]], dtype=float)
    return {
        "mean_r2_gain": float(gains.mean()),
        "sd_r2_gain": float(gains.std(ddof=1)),
        "ci_lo": float(np.percentile(gains, 2.5)),
        "ci_hi": float(np.percentile(gains, 97.5)),
        "frac_folds_positive": float((gains > 0).mean()),
        "max_fold_gain": float(gains.max()),
        "mean_r2_full": float(r2_full.mean()),
        "mean_r2_null": float(r2_null.mean()),
        "n_folds": int(gains.size),
    }


def run_xgb_kfold(
    data: XGBData,
    targets: Sequence[str] | None = None,
    K: int = 5,
    n_reps: int = 20,
    seed: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pl.DataFrame:
    """Full-sample repeated K-fold R^2 comparison for every survey score.

    Companion to :func:`run_xgb_nonlinear`: same full-vs-null XGBoost contrast but
    trained on the whole training-half sample per fold (no SHARP split-half), to
    check whether the extra training data yields any positive R^2 gain. Returns a
    tidy table sorted by ``mean_r2_gain`` (descending).
    """
    target_cols = list(targets) if targets is not None else list(data.symptom_cols)
    rows: list[dict[str, float | str]] = []
    for name in target_cols:
        y = data.symptoms[:, data.symptom_cols.index(name)]
        res = kfold_r2_gain(
            data.covars, data.params, y, data.strata,
            K=K, n_reps=n_reps, seed=seed, n_jobs=n_jobs,
        )
        rows.append({"target": name, "n": int(data.covars.shape[0]), **res})
        if verbose:
            print(
                f"{name:22s}  R2 gain={res['mean_r2_gain']:+.4f} "
                f"[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  "
                f"folds+={res['frac_folds_positive']:.2f}  "
                f"max={res['max_fold_gain']:+.4f}"
            )

    return pl.DataFrame(rows).sort("mean_r2_gain", descending=True)
