from pathlib import Path
import numpy as np
from scipy.stats import boxcox
from cogmood_analysis.load import boxcoxmask, load_task
import polars as pl


def test_boxcoxmask():
    rng = np.random.default_rng()
    zbcxmax = 0
    xmin = -1
    x = np.load(Path(__file__).parent / 'test_data/boxcox.npy')
    test_mask = boxcoxmask(x)
    xp = x[test_mask.squeeze()]
    bcx = boxcox(xp)[0]
    zbcx = np.abs((bcx - bcx.mean())/bcx.std())
    zbcxmax = zbcx.max()
    assert zbcxmax < 3
    assert len(test_mask) == len(x)

    x = x * np.nan
    test_mask = boxcoxmask(x)
    assert test_mask.sum() == 0
    assert len(test_mask) == len(x)

    xmin = 1
    while (xmin >= 0):
        x = np.hstack([rng.normal(1.5, 0.3, 200), rng.uniform(-1,1, 22)])
        xmin = x.min()
    test_mask = boxcoxmask(x)
    assert test_mask.sum() == 0
    assert len(test_mask) == len(x)

    zbcxmax = 4
    xmin = -1
    while (zbcxmax > 3) or (xmin < 0):
        x = np.hstack([rng.normal(1.5, 0.3, 200), rng.uniform(0,1, 22)])
        bcx = boxcox(x)[0]
        zbcx = np.abs((bcx - bcx.mean())/bcx.std())
        zbcxmax = zbcx.max()
        xmin = x.min()
    test_mask = boxcoxmask(x)
    assert test_mask.mean() == 1
    assert len(test_mask) == len(x)


def test_load(datafiles):
    zipped_path = Path(__file__).parent / 'oneblock_test.zip'
    expected_flkr = pl.read_parquet( Path(__file__).parent / 'test_data/flkr.parquet')
    expected_bart = pl.read_parquet( Path(__file__).parent / 'test_data/bart.parquet')
    expected_cab = pl.read_parquet( Path(__file__).parent / 'test_data/cab.parquet')
    expected_rdm = pl.read_parquet( Path(__file__).parent / 'test_data/rdm.parquet')
    loddf = load_task(zipped_path, 'flkr', 'load_task_test', 0)
    assert loddf.equals(expected_flkr)
    loddf = load_task(zipped_path, 'bart', 'load_task_test', 0)
    assert loddf.equals(expected_bart)
    loddf = load_task(zipped_path, 'cab', 'load_task_test', 0)
    assert loddf.equals(expected_cab)
    loddf = load_task(zipped_path, 'rdm', 'load_task_test', 0)
    assert loddf.equals(expected_rdm)
    loddf = load_task(zipped_path, 'rdm', 'load_task_test', runnum=0, as_dateframe=True)
    assert loddf.equals(expected_rdm.to_pandas())