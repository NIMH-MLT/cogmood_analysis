"""Tests for the SHARP ladder module.

The estimator and the cheap arms (raw/kernel/deep) are tested on synthetic data.
The FM arm downloads TabPFN weights, so its smoke test is opt-in via
``TABPFN_SMOKE=1``.
"""

import os

import numpy as np
import pytest

from cogmood_analysis import shared_variance as sv
from cogmood_analysis import sharp


def _synthetic_views(n=600, shared=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, shared))
    A = z @ rng.normal(size=(shared, 12)) + rng.normal(scale=0.5, size=(n, 12))
    B = z @ rng.normal(size=(shared, 9)) + rng.normal(scale=0.5, size=(n, 9))
    strata = np.array(["g0", "g1", "g2", "g3"])[rng.integers(0, 4, size=n)]
    return sv.Views(
        A=A, B=B, sub_ids=np.array([f"s{i}" for i in range(n)]), strata=strata,
        a_columns=[f"a{i}" for i in range(12)], b_columns=[f"b{i}" for i in range(9)],
        n_total=n,
    )


# --- SHARP moment estimator -------------------------------------------------


def test_sharp_moments_recovers_sigma2_when_independent():
    # Independent entries (rho ~ 0): sigma^2 should be recovered and the variance
    # of the mean should collapse toward sigma^2/(2J).
    rng = np.random.default_rng(0)
    J, sigma2, mu = 4000, 0.04, 0.3
    sigma = np.sqrt(sigma2)
    D_A = mu + rng.normal(scale=sigma, size=J)
    D_B = mu + rng.normal(scale=sigma, size=J)
    D_bar, var_Dbar, s2, rho = sharp._sharp_moments(D_A, D_B)
    assert abs(s2 - sigma2) < 0.01
    assert rho < 0.1
    assert abs(var_Dbar - sigma2 / (2 * J)) < sigma2 / (2 * J)


def test_sharp_moments_variance_formula():
    # direct check that var_Dbar matches sigma^2 (1/(2J) + (J-1)/J rho)
    D_A = np.array([0.5, 0.4, 0.6, 0.55, 0.45])
    D_B = np.array([0.48, 0.42, 0.58, 0.53, 0.47])
    D_bar, var_Dbar, s2, rho = sharp._sharp_moments(D_A, D_B)
    J = 5
    expected = s2 * (1 / (2 * J) + (J - 1) / J * rho)
    assert abs(var_Dbar - expected) < 1e-9
    assert abs(D_bar - 0.5 * (D_A.mean() + D_B.mean())) < 1e-12


# --- arm scorers (cheap arms) -----------------------------------------------


@pytest.mark.parametrize("arm", ["raw", "kernel", "deep"])
def test_score_arm_detects_shared_structure(arm):
    views = _synthetic_views()
    cfg = sharp.DEFAULT_CONFIGS[arm]
    if arm == "deep":
        cfg = sharp.ArmConfig(max_epochs=60, device="cpu")  # keep test fast/CPU
    r = sharp.score_arm(views.A[:450], views.B[:450], views.A[450:], views.B[450:],
                        arm, cfg, seed=0)
    assert r.ndim == 1 and (r >= 0).all() and (r <= 1.0001).all()
    assert r[0] > 0.4  # real shared structure recovered on held-out data


# --- SHARP evaluation + CI + comparison -------------------------------------


def test_sharp_eval_ci_and_compare():
    views = _synthetic_views()
    res = sharp.sharp_eval(
        views, arms=["raw", "kernel"], J=6, K=3,
        configs={"raw": sharp.ArmConfig(n_pca=8), "kernel": sharp.ArmConfig()},
        seed=0, verbose=False,
    )
    assert res.D_A["raw"].shape == (6,) and res.D_B["kernel"].shape == (6,)
    ci = sharp.sharp_ci(res, "raw")
    assert 0 < ci["mean"] <= 1 and ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["se"] >= 0 and 0 <= ci["rho"] < 0.5
    cmp = sharp.sharp_compare(res, "raw", "kernel")
    assert set(["diff", "z", "p"]).issubset(cmp) and 0 <= cmp["p"] <= 1


def test_sharp_permutation_within_strata_below_observed():
    views = _synthetic_views()
    cfg = sharp.ArmConfig(n_pca=8)
    out = sharp.sharp_permutation_null(
        views, "raw", J=4, K=3, n_perm=12, config=cfg, seed=0, verbose=False,
    )
    assert out["block_within_strata"] is True
    assert np.median(out["null"]) < out["observed"]
    assert 0 < out["p"] <= 1


@pytest.mark.skipif(os.environ.get("TABPFN_SMOKE") != "1", reason="set TABPFN_SMOKE=1")
def test_score_arm_fm_smoke():
    views = _synthetic_views(n=200)
    r = sharp.score_arm(views.A[:150], views.B[:150], views.A[150:], views.B[150:],
                        "fm", sharp.ArmConfig(n_pca=5), seed=0)
    assert r.ndim == 1 and np.isfinite(r).all()


@pytest.mark.skipif(os.environ.get("TABPFN_SMOKE") != "1", reason="set TABPFN_SMOKE=1")
def test_score_arm_fm_deep_smoke():
    views = _synthetic_views(n=200)
    cfg = sharp.ArmConfig(n_pca=5, ft_epochs=5)  # tiny fine-tune for the smoke test
    r = sharp.score_arm(views.A[:150], views.B[:150], views.A[150:], views.B[150:],
                        "fm_deep", cfg, seed=0)
    assert r.ndim == 1 and np.isfinite(r).all()


@pytest.mark.skipif(os.environ.get("TABFM_SMOKE") != "1", reason="set TABFM_SMOKE=1")
def test_score_arm_tabfm_smoke():
    views = _synthetic_views(n=200)
    r = sharp.score_arm(views.A[:150], views.B[:150], views.A[150:], views.B[150:],
                        "tabfm", sharp.ArmConfig(n_pca=5), seed=0)
    assert r.ndim == 1 and np.isfinite(r).all()


@pytest.mark.skipif(os.environ.get("TABFM_SMOKE") != "1", reason="set TABFM_SMOKE=1")
def test_score_arm_tabfm_deep_smoke():
    views = _synthetic_views(n=200)
    cfg = sharp.ArmConfig(n_pca=5, ft_epochs=4)  # tiny fine-tune for the smoke test
    r = sharp.score_arm(views.A[:150], views.B[:150], views.A[150:], views.B[150:],
                        "tabfm_deep", cfg, seed=0)
    assert r.ndim == 1 and np.isfinite(r).all()
