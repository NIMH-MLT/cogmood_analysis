from pathlib import Path
import numpy as np
from scipy.stats import boxcox
from cogmood_analysis.load import boxcoxmask, load_task
import polars as pl


def test_boxcoxmask():
    rng = np.random.default_rng()
    zbcxmax = 0
    xmin = -1
    x = np.load(Path(__file__).parent / "test_data/boxcox.npy")
    test_mask = boxcoxmask(x)
    xp = x[test_mask.squeeze()]
    bcx = boxcox(xp)[0]
    zbcx = np.abs((bcx - bcx.mean()) / bcx.std())
    zbcxmax = zbcx.max()
    assert zbcxmax < 3
    assert len(test_mask) == len(x)

    x = x * np.nan
    test_mask = boxcoxmask(x)
    assert test_mask.sum() == 0
    assert len(test_mask) == len(x)

    xmin = 1
    while xmin >= 0:
        x = np.hstack([rng.normal(1.5, 0.3, 200), rng.uniform(-1, 1, 22)])
        xmin = x.min()
    test_mask = boxcoxmask(x)
    assert test_mask.sum() == 0
    assert len(test_mask) == len(x)

    zbcxmax = 4
    xmin = -1
    while (zbcxmax > 3) or (xmin < 0):
        x = np.hstack([rng.normal(1.5, 0.3, 200), rng.uniform(0, 1, 22)])
        bcx = boxcox(x)[0]
        zbcx = np.abs((bcx - bcx.mean()) / bcx.std())
        zbcxmax = zbcx.max()
        xmin = x.min()
    test_mask = boxcoxmask(x)
    assert test_mask.mean() == 1
    assert len(test_mask) == len(x)


def test_load():
    # Compare against the golden parquets on the columns they contain, excluding
    # `date`. Two pre-existing, review-unrelated caveats are handled here:
    #  * `date` = pl.lit(zip mtime) (load.py) is a filesystem timestamp that changes
    #    on every checkout, so it can never match a stored golden.
    #  * the goldens predate the derived `coh_dif` column now emitted by load_task,
    #    so we compare on the golden's own column set (regenerating the goldens is a
    #    separate maintenance task).
    def _cmp(got, expected):
        cols = [c for c in expected.columns if c != "date"]
        assert got.select(cols).equals(expected.select(cols))

    zipped_path = Path(__file__).parent / "oneblock_test.zip"
    for task in ("flkr", "bart", "cab", "rdm"):
        expected = pl.read_parquet(Path(__file__).parent / f"test_data/{task}.parquet")
        _cmp(load_task(zipped_path, task, "load_task_test", 0), expected)

    expected_rdm = pl.read_parquet(Path(__file__).parent / "test_data/rdm.parquet")
    cols = [c for c in expected_rdm.columns if c != "date"]
    loddf = load_task(zipped_path, "rdm", "load_task_test", runnum=0, as_dateframe=True)
    assert loddf[cols].reset_index(drop=True).equals(
        expected_rdm.select(cols).to_pandas()
    )
