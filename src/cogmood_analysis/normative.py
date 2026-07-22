"""Normative models for the cognitive parameters, built on training-half HV.

Purpose: produce per-subject *deviation* features (how far each cognitive
parameter is from the healthy-volunteer norm, adjusted for age and sex) to test
the Anna Karenina hypothesis - that symptomatic individuals are each abnormal in
their own way, so symptom variance is carried by the *magnitude/extremeness* of
deviations rather than a shared linear axis (which the CCA ladder already ruled
out).

Design (established empirically, see the exclusion/normative feasibility work):

* **Reference** = training-half healthy volunteers only (``backfilled_prolific_
  screen_group == "hv"``). No pooling across the held-out half.
* **Per-parameter QC** - each parameter's norm is fit only on HV subjects whose
  *per-parameter* MCMC diagnostics are good (``rhat <= 1.1`` and ``ess >= 400``),
  read from ``data/task/<task>_results.csv``. This catches non-convergence that
  the task-level ``max_rhat`` filter misses.
* **Scale** - CAB parameters have ``LogNormal`` priors, so they are log-
  transformed (this alone makes them near-Gaussian); all other parameters get a
  Yeo-Johnson transform fit on HV. Parametric transforms (not quantile) so the
  tail *extrapolates* - AK needs graded magnitude beyond the healthy range, which
  a saturating quantile transform would destroy.
* **Covariate adjustment** - transformed value is modelled with mean and
  (log-)variance as functions of ``[1, age, age^2, sex]`` (variance uses
  ``[1, age, sex]``), so the deviation z is calibrated across the covariate range.

Deviation outputs per subject x parameter: ``z`` (standardized residual),
``centile`` (Phi(z)), and a 3-level extremeness ``indicator`` (+1 above, -1
below, 0 within +/-2 SD). Non-converged or missing estimates yield NaN and are
flagged in the QC mask.

For the AK tests downstream, use *absolute* deviations / extremeness (``|z|``,
``|indicator|``) - signed deviations carry a random sign under AK and are not
predictive (verified by simulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import os

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from sklearn.preprocessing import PowerTransformer

from .shared_variance import COGNITIVE_PARAMS, VIEW_A_COLUMNS

# --- configuration ----------------------------------------------------------

HV_COLUMN = "backfilled_prolific_screen_group"
HV_VALUE = "hv"
COVARIATE_AGE = "age"
COVARIATE_SEX = "sex"

#: CAB parameters have LogNormal priors -> log-transform; others Yeo-Johnson.
LOG_PARAMS: tuple[str, ...] = tuple(f"cab__{p}" for p in COGNITIVE_PARAMS["cab"])

#: Per-parameter convergence QC thresholds.
RHAT_MAX = 1.1
ESS_MIN = 400.0

#: Extremeness indicator threshold (|z| > this -> abnormal).
INDICATOR_Z = 2.0

#: Bias of E[log(chi^2_1)] = -(gamma_euler + log 2); log(sigma^2) = E[log r^2] + this.
_LOGCHI2_BIAS = 1.2703628454614782


def _diag_columns(param: str) -> tuple[str, str, str, str]:
    """Map a training-data param column to its results file + diagnostic columns.

    Returns ``(task, results_filename, rhat_col, ess_col)``. flkr uses the
    ``<param>_rhat`` / ``<param>_ess`` naming; the other tasks use
    ``rhat__<param>`` / ``ess__<param>``.
    """
    task, pname = param.split("__")
    if task == "flkr":
        return task, "flkr_results.csv", f"{pname}_rhat", f"{pname}_ess"
    return task, f"{task}_results.csv", f"rhat__{pname}", f"ess__{pname}"


def load_diagnostics(
    data_dir: str | os.PathLike, params: Sequence[str] = VIEW_A_COLUMNS
) -> pl.DataFrame:
    """Per-subject per-parameter rhat/ess, keyed by ``sub_id``.

    Columns: ``sub_id`` plus ``rhat::<param>`` and ``ess::<param>`` for each
    requested parameter, joined from the per-task results CSVs.
    """
    data_dir = Path(data_dir)
    out: pl.DataFrame | None = None
    for task in ("bart", "rdm", "cab", "flkr"):
        tparams = [p for p in params if p.split("__")[0] == task]
        if not tparams:
            continue
        _, fname, _, _ = _diag_columns(tparams[0])
        r = pl.read_csv(data_dir / "task" / fname, infer_schema_length=20000)
        idc = "sub_id" if "sub_id" in r.columns else "subject"
        exprs = [pl.col(idc).alias("sub_id")]
        for p in tparams:
            _, _, rcol, ecol = _diag_columns(p)
            exprs.append(pl.col(rcol).alias(f"rhat::{p}"))
            exprs.append(pl.col(ecol).alias(f"ess::{p}"))
        sel = r.select(exprs)
        out = sel if out is None else out.join(sel, on="sub_id", how="full", coalesce=True)
    assert out is not None
    return out


# --- fitted model -----------------------------------------------------------


@dataclass
class ParamNorm:
    """Fitted normative model for one parameter."""

    param: str
    kind: str  # "log" or "yeojohnson"
    pt: PowerTransformer | None  # fitted transformer for yeo-johnson
    age_mean: float
    age_sd: float
    beta_mean: NDArray[np.float64]  # mean coefs for [1, a, a^2, sex]
    gamma_var: NDArray[np.float64] | None  # log-variance coefs for [1, a, sex]
    const_sd: float  # fallback / floor sd (robust)
    n_fit: int
    t_lo: float  # fitted transformed-value clip bounds (guards extrapolation)
    t_hi: float

    def _transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kind == "log":
            t = np.log(np.clip(x, 1e-12, None))
        else:
            t = self.pt.transform(x.reshape(-1, 1)).ravel()
        # clip to the fitted range (+pad): power transforms explode when
        # extrapolating past the reference range on new subjects.
        return np.clip(t, self.t_lo, self.t_hi)

    def _design(self, age: NDArray, sex: NDArray) -> tuple[NDArray, NDArray]:
        a = (age - self.age_mean) / self.age_sd
        Xm = np.column_stack([np.ones_like(a), a, a**2, sex])
        Xv = np.column_stack([np.ones_like(a), a, sex])
        return Xm, Xv

    def z(self, x: NDArray, age: NDArray, sex: NDArray) -> NDArray[np.float64]:
        """Deviation z-scores for arbitrary subjects."""
        t = self._transform(np.asarray(x, float))
        Xm, Xv = self._design(np.asarray(age, float), np.asarray(sex, float))
        mu = Xm @ self.beta_mean
        if self.gamma_var is not None:
            # bias-corrected: E[log(r^2)] = log(sigma^2) - 1.2704 for Gaussian r
            sd = np.sqrt(np.exp(Xv @ self.gamma_var + _LOGCHI2_BIAS))
            sd = np.clip(sd, 0.25 * self.const_sd, 4.0 * self.const_sd)
        else:
            sd = np.full(t.shape, self.const_sd)
        return (t - mu) / sd


def _fit_param(
    x: NDArray, age: NDArray, sex: NDArray, param: str,
    heteroscedastic: bool = False, seed: int = 0,
) -> ParamNorm:
    """Fit transform + covariate mean/variance model on a (QC'd HV) sample.

    ``heteroscedastic=False`` (default) uses an unbiased constant residual SD,
    which calibrates cleanly. Setting it True adds a bias-corrected log-variance
    model in ``[1, age, sex]`` on top (use only if variance clearly varies with
    covariates and calibration holds).
    """
    x = np.asarray(x, float)
    age = np.asarray(age, float)
    sex = np.asarray(sex, float)
    kind = "log" if param in LOG_PARAMS else "yeojohnson"
    pt = None
    if kind == "log":
        t = np.log(np.clip(x, 1e-12, None))
    else:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        t = pt.fit_transform(x.reshape(-1, 1)).ravel()
    # clip bounds: fitted transformed range padded by 25% of its span
    span = float(t.max() - t.min()) or 1.0
    t_lo, t_hi = float(t.min()) - 0.25 * span, float(t.max()) + 0.25 * span
    t = np.clip(t, t_lo, t_hi)

    age_mean, age_sd = float(age.mean()), float(age.std() or 1.0)
    a = (age - age_mean) / age_sd
    Xm = np.column_stack([np.ones_like(a), a, a**2, sex])
    beta_mean, *_ = np.linalg.lstsq(Xm, t, rcond=None)
    resid = t - Xm @ beta_mean
    const_sd = float(resid.std(ddof=Xm.shape[1]))  # unbiased for the fitted mean
    if not np.isfinite(const_sd) or const_sd <= 0:
        const_sd = float(1.4826 * stats.median_abs_deviation(resid)) or 1.0

    gamma_var: NDArray[np.float64] | None = None
    if heteroscedastic:
        try:
            Xv = np.column_stack([np.ones_like(a), a, sex])
            g, *_ = np.linalg.lstsq(Xv, np.log(resid**2 + 1e-8), rcond=None)
            pred = np.exp(Xv @ g + _LOGCHI2_BIAS)
            if np.all(np.isfinite(pred)) and pred.min() > 0:
                gamma_var = g
        except np.linalg.LinAlgError:
            gamma_var = None

    return ParamNorm(param=param, kind=kind, pt=pt, age_mean=age_mean, age_sd=age_sd,
                     beta_mean=beta_mean, gamma_var=gamma_var, const_sd=const_sd,
                     n_fit=len(x), t_lo=t_lo, t_hi=t_hi)


# --- fit all + compute deviations -------------------------------------------


def _hv_qc_mask(df: pl.DataFrame, diag: pl.DataFrame, param: str) -> NDArray[np.bool_]:
    """Per-subject QC pass for a parameter (converged & present)."""
    d = df.select("sub_id").join(diag, on="sub_id", how="left")
    rhat = d[f"rhat::{param}"].to_numpy()
    ess = d[f"ess::{param}"].to_numpy()
    x = df[param].to_numpy()
    ok = (rhat <= RHAT_MAX) & (ess >= ESS_MIN) & np.isfinite(x.astype(float))
    return np.asarray(ok, bool)


def fit_normative(
    df: pl.DataFrame,
    diag: pl.DataFrame,
    params: Sequence[str] = VIEW_A_COLUMNS,
    hv_column: str = HV_COLUMN,
    hv_value: str = HV_VALUE,
) -> dict[str, ParamNorm]:
    """Fit per-parameter normative models on QC'd training-half HV subjects."""
    is_hv = (df[hv_column] == hv_value).to_numpy()
    age = df[COVARIATE_AGE].to_numpy().astype(float)
    sex = df[COVARIATE_SEX].cast(pl.Float64).to_numpy()
    models: dict[str, ParamNorm] = {}
    for p in params:
        qc = _hv_qc_mask(df, diag, p)
        fit_mask = is_hv & qc
        if fit_mask.sum() < 30:
            continue  # too few converged HV to fit a norm
        models[p] = _fit_param(df[p].to_numpy()[fit_mask], age[fit_mask],
                               sex[fit_mask], p)
    return models


@dataclass
class Deviations:
    """Per-subject deviation features (n_subjects x n_params)."""

    sub_ids: NDArray[np.str_]
    params: list[str]
    z: NDArray[np.float64]
    centile: NDArray[np.float64]
    indicator: NDArray[np.float64]  # +1/-1/0 (0 also where QC fails -> see mask)
    qc_ok: NDArray[np.bool_]        # False where estimate non-converged/missing


def compute_deviations(
    df: pl.DataFrame, diag: pl.DataFrame, models: dict[str, ParamNorm],
    cross_fit_hv: bool = False, k: int = 5, seed: int = 0,
    hv_column: str = HV_COLUMN, hv_value: str = HV_VALUE,
) -> Deviations:
    """Deviation z / centile / extremeness-indicator for every subject.

    QC-failing (non-converged / missing) estimates are set to NaN in ``z`` and
    ``centile``, 0 in ``indicator``, and False in ``qc_ok``.

    ``cross_fit_hv=True`` gives the healthy-volunteer subjects **out-of-fold**
    deviations (k-fold within HV: each HV fold scored by a norm fit on the other HV
    folds), so HV are not scored in-sample by a norm fit on themselves; non-HV
    subjects are scored from the full HV reference ``models`` as usual.
    """
    from sklearn.model_selection import KFold

    params = list(models.keys())
    n = df.height
    age = df[COVARIATE_AGE].to_numpy().astype(float)
    sex = df[COVARIATE_SEX].cast(pl.Float64).to_numpy()
    Z = np.full((n, len(params)), np.nan)
    qc = np.zeros((n, len(params)), bool)
    for j, p in enumerate(params):
        ok = _hv_qc_mask(df, diag, p)
        zj = models[p].z(df[p].to_numpy(), age, sex)
        Z[:, j] = np.where(ok, zj, np.nan)
        qc[:, j] = ok

    if cross_fit_hv:
        is_hv = (df[hv_column] == hv_value).to_numpy()
        for j, p in enumerate(params):
            x = df[p].to_numpy()
            idx = np.where(is_hv & qc[:, j])[0]      # QC-good HV for this param
            if len(idx) < max(2 * k, 30):
                continue
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            for tr, te in kf.split(idx):
                m = _fit_param(x[idx[tr]], age[idx[tr]], sex[idx[tr]], p)
                Z[idx[te], j] = m.z(x[idx[te]], age[idx[te]], sex[idx[te]])

    centile = stats.norm.cdf(Z)
    indicator = np.where(np.isnan(Z), 0.0, np.where(Z > INDICATOR_Z, 1.0,
                         np.where(Z < -INDICATOR_Z, -1.0, 0.0)))
    return Deviations(sub_ids=df["sub_id"].to_numpy().astype(str), params=params,
                      z=Z, centile=centile, indicator=indicator, qc_ok=qc)


def clipping_fraction(df: pl.DataFrame, models: dict[str, ParamNorm]) -> dict[str, float]:
    """Per-parameter fraction of subjects whose transformed value hits the fitted
    clip bounds ``[t_lo, t_hi]`` (quantifies how often the transform extrapolation
    guard attenuates extreme deviations)."""
    out: dict[str, float] = {}
    for p, m in models.items():
        x = df[p].to_numpy().astype(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out[p] = 0.0
            continue
        if m.kind == "log":
            t = np.log(np.clip(x, 1e-12, None))
        else:
            t = m.pt.transform(x.reshape(-1, 1)).ravel()
        out[p] = float(((t < m.t_lo) | (t > m.t_hi)).mean())
    return out


# --- calibration check (k-fold within HV) -----------------------------------


def hv_calibration(
    df: pl.DataFrame, diag: pl.DataFrame, params: Sequence[str] = VIEW_A_COLUMNS,
    k: int = 5, seed: int = 0,
    hv_column: str = HV_COLUMN, hv_value: str = HV_VALUE,
) -> pl.DataFrame:
    """K-fold within-HV calibration: honest deviation z for HV should be ~N(0,1).

    For each parameter, fit on k-1 HV folds and score the held-out HV fold;
    aggregate those out-of-fold z and report mean/sd (target 0/1) and normality
    (skew, excess kurtosis). Uses the QC'd HV subjects.
    """
    from sklearn.model_selection import KFold

    is_hv = (df[hv_column] == hv_value).to_numpy()
    age = df[COVARIATE_AGE].to_numpy().astype(float)
    sex = df[COVARIATE_SEX].cast(pl.Float64).to_numpy()
    rows = []
    for p in params:
        qc = _hv_qc_mask(df, diag, p)
        idx = np.where(is_hv & qc)[0]
        if len(idx) < 50:
            continue
        x = df[p].to_numpy()
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        zoof = []
        for tr, te in kf.split(idx):
            m = _fit_param(x[idx[tr]], age[idx[tr]], sex[idx[tr]], p)
            zoof.append(m.z(x[idx[te]], age[idx[te]], sex[idx[te]]))
        z = np.concatenate(zoof)
        z = z[np.isfinite(z)]
        rows.append({"param": p, "n_hv": len(idx), "cv_z_mean": float(z.mean()),
                     "cv_z_sd": float(z.std()), "cv_z_skew": float(stats.skew(z)),
                     "cv_z_exkurt": float(stats.kurtosis(z))})
    return pl.DataFrame(rows)
