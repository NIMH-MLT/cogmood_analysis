"""Tests for the normative-modeling module."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cogmood_analysis import normative as nm

REPO = Path(__file__).resolve().parents[3]
TRAINING_CSV = REPO / "data" / "exploratory" / "training_data.csv"
DATA_DIR = REPO / "data"
HAVE_DATA = TRAINING_CSV.exists() and (DATA_DIR / "task" / "cab_results.csv").exists()


def test_diag_column_mapping():
    # non-flkr uses rhat__/ess__; flkr uses <param>_rhat/<param>_ess
    assert nm._diag_columns("cab__tau") == ("cab", "cab_results.csv", "rhat__tau", "ess__tau")
    assert nm._diag_columns("flkr__r") == ("flkr", "flkr_results.csv", "r_rhat", "r_ess")


def test_transform_clip_guards_extrapolation():
    # a fitted param should not explode on values far outside the fit range
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=400)
    age = rng.uniform(18, 80, 400)
    sex = rng.integers(0, 2, 400).astype(float)
    m = nm._fit_param(x, age, sex, "rdm__w_s")  # yeo-johnson param
    z = m.z(np.array([x.max() * 100]), np.array([50.0]), np.array([0.0]))
    assert np.isfinite(z).all() and abs(z[0]) < 20  # clipped, not exploded


@pytest.mark.skipif(not HAVE_DATA, reason="training data / results CSVs not present")
def test_calibration_scale_and_deviations():
    df = pl.read_csv(TRAINING_CSV, infer_schema_length=20000)
    diag = nm.load_diagnostics(DATA_DIR)
    # HV k-fold calibration: deviation z should be ~unit-SD, ~zero-mean
    cal = nm.hv_calibration(df, diag)
    assert cal.height >= 25
    assert cal["cv_z_sd"].median() == pytest.approx(1.0, abs=0.15)
    assert cal["cv_z_mean"].abs().max() < 0.25
    # full deviations: aligned, mostly finite, indicator in {-1,0,1}
    models = nm.fit_normative(df, diag)
    dev = nm.compute_deviations(df, diag, models)
    assert dev.z.shape == (df.height, len(models))
    assert dev.sub_ids.shape[0] == df.height
    assert np.isfinite(dev.z).mean() > 0.9
    assert set(np.unique(dev.indicator)).issubset({-1.0, 0.0, 1.0})
    # QC-failed cells are NaN in z and 0 in indicator
    bad = ~dev.qc_ok
    if bad.any():
        assert np.isnan(dev.z[bad]).all()
        assert (dev.indicator[bad] == 0).all()
