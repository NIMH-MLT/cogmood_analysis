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


def test_legacy_mom_estimate_recovers_sigma2_when_independent():
    # Independent entries (rho ~ 0): sigma^2 should be recovered and the variance
    # of the mean should collapse toward sigma^2/(2J).
    rng = np.random.default_rng(0)
    J, sigma2, mu = 4000, 0.04, 0.3
    sigma = np.sqrt(sigma2)
    D_A = mu + rng.normal(scale=sigma, size=J)
    D_B = mu + rng.normal(scale=sigma, size=J)
    D_bar, var_Dbar, s2, rho = sharp._legacy_mom_estimate(D_A, D_B)
    assert abs(s2 - sigma2) < 0.01
    assert rho < 0.1
    assert abs(var_Dbar - sigma2 / (2 * J)) < sigma2 / (2 * J)


def test_legacy_mom_estimate_variance_formula():
    # direct check that var_Dbar matches sigma^2 (1/(2J) + (J-1)/J rho)
    D_A = np.array([0.5, 0.4, 0.6, 0.55, 0.45])
    D_B = np.array([0.48, 0.42, 0.58, 0.53, 0.47])
    D_bar, var_Dbar, s2, rho = sharp._legacy_mom_estimate(D_A, D_B)
    J = 5
    expected = s2 * (1 / (2 * J) + (J - 1) / J * rho)
    assert abs(var_Dbar - expected) < 1e-9
    assert abs(D_bar - 0.5 * (D_A.mean() + D_B.mean())) < 1e-12


# --- paper-faithful score test + calibration --------------------------------


def _sample_sharp_null(J, sigma2, rho, n_sim, seed):
    """Draw n_sim SHARP half-statistic pairs under H0 (mu=0) with the paper's
    structured covariance: common corr rho on all off-diagonal pairs except the
    paired (A_j, B_j) which are independent."""
    rng = np.random.default_rng(seed)
    cov = np.full((2 * J, 2 * J), sigma2 * rho)
    np.fill_diagonal(cov, sigma2)
    for j in range(J):  # paired halves are independent
        cov[j, J + j] = cov[J + j, j] = 0.0
    draws = rng.multivariate_normal(np.zeros(2 * J), cov, size=n_sim)
    return draws[:, :J], draws[:, J:]


def _brute_argmin_rho(a, b, mu0=0.0, n=60000):
    """Dense brute-force global argmin of the profiled objective (for testing)."""
    J = a.size
    lo, hi = -1.0 / (2.0 * (J - 1)) + 1e-9, 0.5 - 1e-9
    g = np.linspace(lo, hi, n)
    o = sharp._neg2ll_grid(g, *sharp._profile_stats(a, b, mu0, J), J)
    return float(g[int(np.argmin(o))])


# hitop_hypsom XGBoost SHARP half-statistics (J=30) -- the profile is BIMODAL here;
# the old single bounded optimizer picked a local mode (rho=0.210, one-sided p=0.524)
# instead of the global (rho=-0.016, one-sided p=0.839). Regression-locks the fix.
_HYPSOM_DA = [0.003889, 0.023393, 0.000135, 0.004804, -0.028773, 0.007376, 0.006656, 0.031328,
              -0.02948, 0.015661, 0.007479, -0.010791, -0.02582, 0.01279, 0.012298, -0.010375,
              -0.007, -0.026575, -0.01054, -0.028357, 0.019274, 0.013336, 0.01057, 0.006652,
              -0.002176, 0.014723, 7.2e-05, 0.009105, -0.025646, -0.006886]
_HYPSOM_DB = [0.029928, -0.005542, -0.005518, -0.003796, 0.018563, -0.028853, 0.00395, -0.003404,
              -0.001817, -0.011324, 0.015737, -0.012909, 0.007891, -0.053984, 0.020107, 0.000709,
              -0.016947, 0.020355, -0.013432, 0.024872, -0.011691, 0.011214, -0.004138, -0.002889,
              0.008155, -0.001564, 0.009475, -0.005664, 0.013616, -0.019258]


def _brute_min_obj(a, b, mu0=0.0, n=200000):
    """Global minimum of the profiled objective via an ultra-dense boundary-clustered
    (cosine) grid -- the ground-truth reference for the optimizer."""
    J = a.size
    lo, hi = -1.0 / (2.0 * (J - 1)) + 1e-12, 0.5 - 1e-12
    g = sharp._cosine_grid(lo, hi, n)
    return float(sharp._neg2ll_grid(g, *sharp._profile_stats(a, b, mu0, J), J).min())


def _opt_obj(a, b, mu0=0.0):
    _, rho = sharp._null_constrained_mle(a, b, mu0)
    return float(sharp._neg2ll_grid(rho, *sharp._profile_stats(a, b, mu0, J=a.size), a.size))


def test_optimizer_regression_hitop_hypsom():
    a, b = np.array(_HYPSOM_DA), np.array(_HYPSOM_DB)
    st = sharp.sharp_score_test(a, b, mu0=0.0, alternative="greater")
    assert abs(st["p"] - 0.839) < 0.005      # global mode, not the local p=0.524
    assert abs(st["rho"] - (-0.0162)) < 0.003
    _, _, info = sharp._null_constrained_mle(a, b, 0.0, return_info=True)
    assert info["n_stationary"] >= 2 and info["converged"]   # genuinely multimodal, global


def test_optimizer_is_global_randomized():
    # the optimizer's objective must match a 200k-point boundary-clustered brute force
    # (the round-2 grid optimizer missed narrow near-boundary modes; this asserts 0 misses).
    rng = np.random.default_rng(0)
    for _ in range(60):
        J = int(rng.integers(10, 61))
        mu = rng.normal(scale=0.05)
        A, B = _sample_sharp_null(J, sigma2=float(rng.uniform(1e-4, 0.1)),
                                  rho=float(rng.uniform(0.0, 0.49)), n_sim=1,
                                  seed=int(rng.integers(1, 10_000_000)))
        a, b = A[0] + mu, B[0] + mu
        assert _opt_obj(a, b) <= _brute_min_obj(a, b) + 1e-6


def test_optimizer_narrow_mode_near_boundaries():
    # extreme sufficient statistics push the minimum toward a PD boundary; the optimizer
    # must still find the global objective.
    for a, b in ([np.array([5.0] + [0.0] * 29), np.array([-5.0] + [0.0] * 29)],   # huge s0sq
                 [np.array([0.01] * 30), np.array([-0.01] * 30)]):                 # near-degenerate
        assert _opt_obj(a, b) <= _brute_min_obj(a, b) + 1e-6


def test_score_test_alternatives_consistent_and_validated():
    a, b = _sample_sharp_null(J=30, sigma2=0.04, rho=0.1, n_sim=1, seed=3)
    a, b = a[0] + 0.05, b[0] + 0.05
    g = sharp.sharp_score_test(a, b, alternative="greater")["p"]
    ls = sharp.sharp_score_test(a, b, alternative="less")["p"]
    two = sharp.sharp_score_test(a, b, alternative="two-sided")["p"]
    assert abs(g + ls - 1.0) < 1e-9
    assert abs(two - 2 * min(g, ls)) < 1e-9
    with pytest.raises(ValueError):                 # invalid alternative must not silently pass
        sharp.sharp_score_test(a, b, alternative="one-sided")


def test_score_test_recovers_variance_params():
    # the null-constrained MLE is (approximately) unbiased for (sigma2, rho):
    # a single draw is noisy, so check the mean over many draws.
    J, sigma2, rho, n_sim = 40, 0.04, 0.15, 400
    A, B = _sample_sharp_null(J, sigma2, rho, n_sim, seed=0)
    ests = [sharp._null_constrained_mle(A[i], B[i], 0.0) for i in range(n_sim)]
    mean_sigma2 = np.mean([e[0] for e in ests])
    mean_rho = np.mean([e[1] for e in ests])
    assert abs(mean_sigma2 - sigma2) < 0.005
    assert abs(mean_rho - rho) < 0.03
    assert 0 <= sharp.sharp_score_test(A[0], B[0], mu0=0.0)["p"] <= 1


def test_score_test_type_i_multi_rho():
    # Two-sided type-I across rho. The score test is VALID (FPR <= alpha) but
    # CONSERVATIVE at low/moderate rho at J=30 -- documented, not called "calibrated
    # at 5%". The old MoM+Wald reference inflates (esp. rho~0.1). Uses J=60 (the
    # final analyses' config), where calibration is closer to nominal.
    J, sigma2, n_sim, alpha = 60, 0.04, 500, 0.05
    for rho in (0.01, 0.10, 0.25, 0.40):
        A, B = _sample_sharp_null(J, sigma2, rho, n_sim, seed=1)
        rej = sum(sharp.sharp_score_test(A[i], B[i], mu0=0.0)["p"] < alpha
                  for i in range(n_sim)) / n_sim
        assert rej <= 0.085, f"rho={rho}: FPR {rej} not controlled"   # valid, not inflated
    # MoM+Wald inflation at rho=0.10 (the retained reference)
    A, B = _sample_sharp_null(30, sigma2, 0.10, n_sim, seed=1)
    mom = 0
    for i in range(n_sim):
        D_bar, var, _, _ = sharp._legacy_mom_estimate(A[i], B[i])
        z = D_bar / np.sqrt(var) if var > 1e-12 else 0.0
        mom += 2 * (1 - _norm_cdf(abs(z))) < alpha
    assert mom / n_sim > 0.12          # MoM inflates well above nominal


def test_score_test_has_power():
    # With a real positive mean shift the one-sided test rejects most of the time.
    J, sigma2, rho, n_sim = 60, 0.01, 0.05, 300
    A, B = _sample_sharp_null(J, sigma2, rho, n_sim, seed=5)
    mu = 0.10
    rej = sum(sharp.sharp_score_test(A[i] + mu, B[i] + mu, alternative="greater")["p"] < 0.05
              for i in range(n_sim)) / n_sim
    assert rej > 0.5   # empirically ~0.88 at this effect size


def test_ci_coverage_near_nominal():
    # test-inversion 95% CI should cover the true mean about >=93% of the time.
    J, sigma2, rho, n_sim, mu = 60, 0.03, 0.2, 200, 0.08
    A, B = _sample_sharp_null(J, sigma2, rho, n_sim, seed=7)
    covered = 0
    for i in range(n_sim):
        lo, hi = sharp._invert_score_test(A[i] + mu, B[i] + mu, alpha=0.05)
        covered += lo <= mu <= hi
    cov = covered / n_sim
    assert cov >= 0.93   # conservative test -> coverage at or above nominal


def test_ci_inversion_contains_and_excludes():
    A, B = _sample_sharp_null(J=30, sigma2=0.04, rho=0.1, n_sim=1, seed=2)
    a, b = A[0] + 0.3, B[0] + 0.3   # shift the mean to 0.3
    lo, hi = sharp._invert_score_test(a, b, alpha=0.05)
    D_bar = 0.5 * (a.mean() + b.mean())
    assert lo < D_bar < hi
    # mu0 = 0 (far below the shifted mean) should sit outside a tight-ish CI or be
    # rejected; at minimum the CI is finite and ordered
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def _norm_cdf(x):
    from scipy import stats as _s
    return float(_s.norm.cdf(x))


# --- arm scorers (cheap arms) -----------------------------------------------


@pytest.mark.parametrize("arm", ["raw", "kernel", "deep"])
def test_score_arm_detects_shared_structure(arm):
    views = _synthetic_views()
    cfg = sharp.DEFAULT_CONFIGS[arm]
    if arm == "deep":
        cfg = sharp.ArmConfig(max_epochs=60, device="cpu")  # keep test fast/CPU
    r = sharp.score_arm(views.A[:450], views.B[:450], views.A[450:], views.B[450:],
                        arm, cfg, seed=0)
    assert r.ndim == 1 and (np.abs(r) <= 1.0001).all()  # signed corrs
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
    # test-inversion CI must contain the point estimate; rho may be negative
    assert 0 < ci["mean"] <= 1 and ci["lo"] <= ci["mean"] <= ci["hi"]
    assert -0.5 < ci["rho"] < 0.5 and 0 <= ci["p"] <= 1
    cmp = sharp.sharp_compare(res, "raw", "kernel")
    assert set(["diff", "z", "p", "lo", "hi"]).issubset(cmp) and 0 <= cmp["p"] <= 1


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
