"""Tests for the Anna Karenina deviation-based test module."""

import numpy as np
import polars as pl
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
    # inject some missing |z| to exercise the in-fold median imputation
    rng0 = np.random.default_rng(11)
    absz = absz.copy()
    mask = rng0.random(absz.shape) < 0.05
    absz[mask] = np.nan
    res = ak.ak_supervised(absz, y, n_repeats=2, n_perm=60, seed=0)     # raw |z| (imputed in-fold)
    assert res["effect"] > 0 and res["p"] < 0.1            # detects
    assert 1 <= res["n_selected"] <= 30
    # permuted target -> no predictive signal; identical folds used for obs + perms
    rng = np.random.default_rng(5)
    res0 = ak.ak_supervised(absz, rng.permutation(y), n_repeats=2, n_perm=60, seed=0)
    assert res0["effect"] < 0.02


# --- reverse-scored (positive-valence) targets ------------------------------


def test_reverse_scored_welbe_is_flipped():
    rng = np.random.default_rng(0)
    n = 200
    cols = ["phq8", "hitop_welbe"]
    S = rng.normal(size=(n, 2))
    data = ak.AKData(
        sub_ids=np.array([str(i) for i in range(n)]), params=["a"],
        absz=np.abs(rng.normal(size=(n, 1))), symptom_cols=cols, symptoms=S,
        age=rng.uniform(18, 80, n), sex=rng.integers(0, 2, n).astype(float),
        is_hv=np.zeros(n, bool))
    tg = {(t.name, t.variant): t for t in ak.symptom_targets(data)}
    # positive-valence scale flipped (higher = more pathology); others unchanged
    assert np.allclose(tg[("hitop_welbe", "raw")].y, -S[:, 1])
    assert np.allclose(tg[("phq8", "raw")].y, S[:, 0])


# --- maxT (Westfall-Young step-down) correction -----------------------------


def _targets(Y, names):
    return [ak.Target(name=n, variant="resid", y=Y[:, i], primary=(n == "PC1"))
            for i, n in enumerate(names)]


def test_maxstat_stepdown_recovers_monotone_and_one_sided():
    rng = np.random.default_rng(0)
    n, P = 500, 10
    absz = np.abs(rng.normal(size=(n, P)))
    burden = absz.max(1)
    Y = np.column_stack([
        burden + rng.normal(scale=1.0, size=n),   # sig: positively assoc with max|z|
        -burden + rng.normal(scale=1.0, size=n),  # neg: one-sided test should miss it
        rng.normal(size=n), rng.normal(size=n),    # noise
    ])
    tgts = _targets(Y, ["sig", "neg", "noise1", "noise2"])
    tbl, maxnull = ak.maxstat_correction(absz, tgts, n_perm=1000, seed=0)
    assert tbl["adj_p_maxT"].min() >= 0 and tbl["adj_p_maxT"].max() <= 1
    # the planted (max_abs, sig) cell is the strongest and significant
    sig = tbl.filter((pl.col("approach") == "max_abs") & (pl.col("target") == "sig"))
    assert sig["adj_p_maxT"][0] < 0.05
    # one-sided: a strong NEGATIVE association is not significant
    neg = tbl.filter((pl.col("approach") == "max_abs") & (pl.col("target") == "neg"))
    assert neg["adj_p_maxT"][0] > 0.5
    # step-down monotonicity: adj_p non-decreasing as observed effect decreases
    s = tbl.sort("effect", descending=True)["adj_p_maxT"].to_numpy()
    assert np.all(np.diff(s) >= -1e-9)


def test_within_approach_le_joint():
    rng = np.random.default_rng(1)
    absz = np.abs(rng.normal(size=(400, 8)))
    burden = absz.max(1)
    Y = np.column_stack([burden + rng.normal(size=400)] + [rng.normal(size=400) for _ in range(5)])
    tgts = _targets(Y, ["sig"] + [f"n{i}" for i in range(5)])
    joint, _ = ak.maxstat_correction(absz, tgts, n_perm=1000, seed=0)
    within = ak.maxstat_within_approach(absz, tgts, n_perm=1000, seed=0)
    j = joint.rename({"adj_p_maxT": "joint"}).join(
        within.select(["approach", "target", "within_maxT"]), on=["approach", "target"])
    # smaller family -> within-approach adjusted p <= joint adjusted p
    assert (j["within_maxT"] <= j["joint"] + 1e-9).all()
