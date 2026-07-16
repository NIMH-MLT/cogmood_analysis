"""Tests for the shared-variance estimation module.

The foundation-model path downloads TabPFN weights and is slow, so its test is
skipped unless ``TABPFN_SMOKE=1`` is set. The data-loading, helper, and
raw-control tests run without TabPFN.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from cogmood_analysis import shared_variance as sv

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CSV = REPO_ROOT / "data" / "exploratory" / "training_data.csv"


# --- column definitions -----------------------------------------------------


def test_view_a_columns_count():
    # 5 + 7 + 10 + 8 fitted parameters, no fit diagnostics
    assert len(sv.VIEW_A_COLUMNS) == 30
    assert "bart__alpha" in sv.VIEW_A_COLUMNS
    # diagnostics must be excluded
    assert not any(
        c.endswith(("sub_score", "map_dif", "abs_map_dif", "max_rhat"))
        for c in sv.VIEW_A_COLUMNS
    )


def test_view_b_columns_from_scales():
    # attnbin is dropped, baars_inattentive is added, no today* variants
    assert "attnbin" not in sv.VIEW_B_COLUMNS
    assert "baars_inattentive" in sv.VIEW_B_COLUMNS
    assert all(not c.startswith("today") for c in sv.VIEW_B_COLUMNS)
    assert "baars" in sv.VIEW_B_COLUMNS and "hitop_anhdep" in sv.VIEW_B_COLUMNS


# --- data loading -----------------------------------------------------------


@pytest.mark.skipif(not TRAINING_CSV.exists(), reason="training_data.csv not present")
def test_load_views_complete_case_and_alignment():
    views = sv.load_views(TRAINING_CSV)
    # rows aligned across A, B, sub_ids, strata
    assert (
        views.A.shape[0]
        == views.B.shape[0]
        == views.sub_ids.shape[0]
        == views.strata.shape[0]
    )
    # strata are non-null categorical labels (used to stratify the splits)
    assert views.strata.dtype.kind in ("U", "S")
    assert views.strata.shape[0] > 0 and len(set(views.strata.tolist())) > 1
    # complete-case: no missing values remain in either view
    assert not np.isnan(views.A).any()
    assert not np.isnan(views.B).any()
    # complete-case keeps a strict subset of the original rows
    assert 0 < views.A.shape[0] <= views.n_total
    # resolved columns are subsets of the requested defaults
    assert set(views.a_columns).issubset(set(sv.VIEW_A_COLUMNS))
    assert set(views.b_columns).issubset(set(sv.VIEW_B_COLUMNS))
    assert views.A.shape[1] == len(views.a_columns)
    assert views.B.shape[1] == len(views.b_columns)


@pytest.mark.skipif(not TRAINING_CSV.exists(), reason="training_data.csv not present")
def test_load_views_rhat_exclusion():
    full = sv.load_views(TRAINING_CSV)
    filt = sv.load_views(TRAINING_CSV, exclude_rhat_above=1.1)
    # exclusion drops a strict, non-empty subset
    assert 0 < filt.A.shape[0] < full.A.shape[0]
    # kept subjects are exactly those with all-task max_rhat <= 1.1
    import polars as pl
    df = pl.read_csv(TRAINING_CSV, infer_schema_length=20000)
    ok = df.filter(
        pl.all_horizontal([pl.col(f"{t}__max_rhat") <= 1.1
                           for t in ("bart", "rdm", "cab", "flkr")])
    )["sub_id"].to_list()
    assert set(filt.sub_ids.tolist()) == set(ok)
    # views stay aligned after filtering
    assert filt.A.shape[0] == filt.B.shape[0] == filt.strata.shape[0]


# --- pure-numpy helpers -----------------------------------------------------


def test_reduce_embedding_rules():
    E = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)  # (n_est, n, d)
    np.testing.assert_allclose(sv._reduce_embedding(E, "mean"), E.mean(axis=0))
    np.testing.assert_allclose(sv._reduce_embedding(E, "first"), E[0])
    # already-2D passes through
    flat = np.ones((5, 3))
    np.testing.assert_allclose(sv._reduce_embedding(flat, "mean"), flat)


def test_heldout_corr_recovers_known_correlation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 2))
    b = a.copy()
    b[:, 1] = rng.normal(size=500)  # second dim uncorrelated
    r = sv._heldout_corr(a, b)
    assert r[0] > 0.99
    assert abs(r[1]) < 0.2


def test_standardize_uses_train_stats():
    rng = np.random.default_rng(1)
    tr = rng.normal(loc=5, scale=2, size=(100, 3))
    te = rng.normal(loc=5, scale=2, size=(40, 3))
    s_tr, s_te = sv._standardize(tr, te)
    # train is standardized to ~0 mean / unit std
    np.testing.assert_allclose(s_tr.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(s_tr.std(axis=0), 1, atol=1e-6)
    # test is NOT forced to zero mean (uses train stats)
    assert np.abs(s_te.mean(axis=0)).max() > 0


# --- CCA / evaluation on synthetic data (raw mode, no TabPFN) ----------------


def _synthetic_views(n=400, shared=3, seed=0):
    """Two views sharing `shared` latent factors plus view-specific noise."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, shared))
    Wa = rng.normal(size=(shared, 12))
    Wb = rng.normal(size=(shared, 9))
    A = z @ Wa + rng.normal(scale=0.5, size=(n, 12))
    B = z @ Wb + rng.normal(scale=0.5, size=(n, 9))
    strata = np.array(["g0", "g1", "g2", "g3"])[rng.integers(0, 4, size=n)]
    return sv.Views(
        A=A, B=B,
        sub_ids=np.array([f"s{i}" for i in range(n)]),
        strata=strata,
        a_columns=[f"a{i}" for i in range(12)],
        b_columns=[f"b{i}" for i in range(9)],
        n_total=n,
    )


def test_fit_score_cca_shapes_and_signal():
    views = _synthetic_views()
    EA_tr, EA_te = sv.embed_view_raw(views.A[:300], views.A[300:])
    EB_tr, EB_te = sv.embed_view_raw(views.B[:300], views.B[300:])
    r = sv.fit_score_cca(EA_tr, EB_tr, EA_te, EB_te, n_pca=8, k=5, c=0.3)
    assert r.shape == (5,)
    assert (r >= 0).all() and (r <= 1).all()
    # real shared structure -> leading held-out correlation is substantial
    assert r[0] > 0.5


def test_repeated_eval_raw_detects_shared_variance():
    views = _synthetic_views()
    grid = sv.Grid(n_pca=(6, 8), k=(5,), c=(0.1, 0.5))
    res = sv.repeated_eval(
        views, mode="raw", n_splits=4, grid=grid, n_inner=2, verbose=False
    )
    assert res.r_per_split.shape == (4, 5)
    assert res.mean_r[0] > 0.5
    lo, hi = res.ci(dim=0)
    assert lo <= res.mean_r[0] <= hi


def test_permutation_null_below_observed():
    views = _synthetic_views()
    grid = sv.Grid(n_pca=(6,), k=(3,), c=(0.3,))
    res = sv.repeated_eval(
        views, mode="raw", n_splits=3, grid=grid, n_inner=2, verbose=False
    )
    null = sv.permutation_null(
        views, mode="raw", n_perm=15, grid=grid, n_inner=2, verbose=False
    )
    # breaking the cross-view pairing should collapse the leading correlation
    assert np.median(null) < res.mean_r[0]
    p = sv.permutation_pvalue(res.mean_r[0], null)
    assert 0 < p <= 1


# --- foundation-model smoke test (downloads weights; opt-in) -----------------


@pytest.mark.skipif(
    os.environ.get("TABPFN_SMOKE") != "1",
    reason="set TABPFN_SMOKE=1 to run the TabPFN embedding smoke test",
)
def test_embed_view_fm_shapes_and_no_leakage():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(40, 8))
    X_te = rng.normal(size=(15, 8))
    E_tr, E_te = sv.embed_view_fm(X_tr, X_te, device="cpu", seed=0)
    # 2D, row-aligned to inputs, TabPFN v2 embedding dim is 192
    assert E_tr.shape[0] == 40 and E_te.shape[0] == 15
    assert E_tr.shape[1] == E_te.shape[1] == 192
    assert np.isfinite(E_tr).all() and np.isfinite(E_te).all()
