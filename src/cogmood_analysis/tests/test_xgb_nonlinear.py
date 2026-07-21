"""Tests for the XGBoost full-vs-null nonlinear check.

The SHARP moment reuse is covered in ``test_sharp.py``; here we verify that the
XGBoost harness detects a genuine nonlinear cognition->symptom signal, is
calibrated under the null, and that the loader reproduces the CCA analysis sample.
"""

from pathlib import Path

import numpy as np
import pytest

from cogmood_analysis import xgb_nonlinear as xg

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CSV = REPO_ROOT / "data" / "exploratory" / "training_data.csv"


def _synthetic(n=800, p=30, nonlinear=True, seed=0):
    """Covariates (age, sex), cognitive params, and a symptom target.

    The target always has a demographic component (which the null model captures).
    When ``nonlinear`` is True it additionally depends on a genuine interaction /
    quadratic of two parameters that only the full model can see.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    sex = rng.integers(0, 2, n).astype(float)
    covars = np.column_stack([age, sex])
    params = rng.normal(size=(n, p))
    strata = np.array(["g0", "g1", "g2", "g3"])[rng.integers(0, 4, size=n)]
    demo = 0.03 * (age - 50) + 1.5 * sex
    signal = (2.0 * params[:, 0] * params[:, 1] + 1.5 * params[:, 2] ** 2) if nonlinear else 0.0
    y = demo + signal + rng.normal(scale=1.0, size=n)
    return covars, params, y, strata


def test_sharp_xgb_detects_nonlinear_signal():
    covars, params, y, strata = _synthetic(nonlinear=True, seed=1)
    res = xg.sharp_xgb(covars, params, y, strata, J=8, K=3, seed=0)
    assert res["mean_r2_gain"] > 0  # cognition adds predictive value
    assert res["p_one_sided"] < 0.05
    assert 0.0 <= res["rho"] < 0.5


def test_sharp_xgb_null_is_calibrated():
    # target depends only on demographics -> full model should not beat null
    covars, params, y, strata = _synthetic(nonlinear=False, seed=2)
    res = xg.sharp_xgb(covars, params, y, strata, J=8, K=3, seed=0)
    assert res["mean_r2_gain"] < 0.02
    assert res["p_one_sided"] > 0.05


def test_rep_half_diff_returns_two_finite_values():
    covars, params, y, strata = _synthetic(seed=3)
    X_null = covars
    X_full = np.column_stack([covars, params])
    d_a, d_b = xg._rep_half_diff(X_null, X_full, y, strata, K=3, rep_seed=0)
    assert np.isfinite(d_a) and np.isfinite(d_b)


@pytest.mark.skipif(not TRAINING_CSV.exists(), reason="training_data.csv not present")
def test_loader_matches_cca_sample():
    data = xg.load_xgb_data(TRAINING_CSV)
    # rows aligned across every returned array
    n = data.covars.shape[0]
    assert (
        data.params.shape[0]
        == data.symptoms.shape[0]
        == data.strata.shape[0]
        == data.sub_ids.shape[0]
        == n
    )
    # rhat exclusion reproduces the CCA analysis sample (N=1298)
    from cogmood_analysis import shared_variance as sv

    views = sv.load_views(TRAINING_CSV, exclude_rhat_above=1.1)
    assert n == views.A.shape[0]
    assert set(data.sub_ids.tolist()) == set(views.sub_ids.tolist())
    # shapes: 2 covariates, 30 params, complete-case (no missing)
    assert data.covars.shape[1] == 2 and data.params.shape[1] == 30
    assert not np.isnan(data.covars).any()
    assert not np.isnan(data.params).any()
    assert not np.isnan(data.symptoms).any()
