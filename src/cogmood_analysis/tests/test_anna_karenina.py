"""Tests for the Anna Karenina deviation-based test module."""

import numpy as np
import pytest

from cogmood_analysis import anna_karenina as ak


def _ak_absz_and_symptom(n=800, P=30, k=3, effect=1.5, seed=0):
    """AK generative model: each subject idiosyncratically extreme on ONE of k
    relevant params; symptom tracks that extreme's magnitude."""
    r = np.random.default_rng(seed)
    Z = r.normal(size=(n, P))
    which = r.integers(0, k, size=n)
    mag = np.abs(r.normal(size=n)) * effect
    Z[np.arange(n), which] += mag * r.choice([-1, 1], size=n)
    y = mag + r.normal(scale=2.0, size=n)
    return np.abs(Z), y


# --- helpers ----------------------------------------------------------------


def test_bh_fdr_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    q = ak._bh_fdr(p)
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(q >= p)  # q-values never below raw p for BH


def test_deviation_features_shapes_and_nan():
    absz = np.array([[0.5, 3.1, np.nan], [1.0, 1.0, 1.0], [np.nan, np.nan, 2.5]])
    f = ak.deviation_features(absz, topk=(2,))
    assert f["max_abs"][0] == pytest.approx(3.1)
    assert f["count_gt2"][0] == 1 and f["count_gt2"][1] == 0
    assert np.isfinite(f["dist"]).all()
    assert f["top2_mean"].shape == (3,)


# --- univariate test calibration + power ------------------------------------


def test_ak_univariate_detects_and_is_calibrated():
    absz, y = _ak_absz_and_symptom(seed=1)
    feats = ak.deviation_features(absz)
    hit = ak.ak_univariate(feats["max_abs"], y, n_perm=500, seed=0)
    assert hit["effect"] > 0 and hit["p"] < 0.05           # detects real signal
    # under no association, permutation p is not systematically small
    rng = np.random.default_rng(3)
    nullp = [ak.ak_univariate(feats["max_abs"], rng.permutation(y), n_perm=200, seed=s)["p"]
             for s in range(20)]
    assert np.mean(np.array(nullp) < 0.05) < 0.25          # roughly calibrated


def test_max_beats_distance_under_sparsity():
    # only 2 of 40 params relevant -> global distance dilutes, max holds up
    absz, y = _ak_absz_and_symptom(n=800, P=40, k=2, effect=2.0, seed=2)
    feats = ak.deviation_features(absz)
    r_max = ak.ak_univariate(feats["max_abs"], y, n_perm=1, seed=0)["effect"]
    r_dist = ak.ak_univariate(feats["dist"], y, n_perm=1, seed=0)["effect"]
    assert r_max > r_dist


# --- supervised test --------------------------------------------------------


def test_ak_supervised_recovers_signal_and_null_calibrated():
    absz, y = _ak_absz_and_symptom(n=700, P=30, k=3, effect=2.0, seed=4)
    Xi = ak._impute_absz(absz)
    res = ak.ak_supervised(Xi, y, n_repeats=2, n_perm=60, seed=0)
    assert res["effect"] > 0 and res["p"] < 0.1            # detects
    assert 1 <= res["n_selected"] <= 30
    # permuted target -> no predictive signal
    rng = np.random.default_rng(5)
    res0 = ak.ak_supervised(Xi, rng.permutation(y), n_repeats=2, n_perm=60, seed=0)
    assert res0["effect"] < 0.02
