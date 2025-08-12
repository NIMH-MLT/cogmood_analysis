import numpy as np
from scipy.stats import boxcox
from cogmood_analysis.load import boxcoxmask

def test_boxcoxmask():
    rng = np.random.default_rng()
    zbcxmax = 0
    xmin = -1
    while (zbcxmax < 3) or (xmin < 0):
        x = np.hstack([rng.normal(1.5, 0.3, 200), rng.uniform(0,1, 22)])
        bcx = boxcox(x)[0]
        zbcx = np.abs((bcx - bcx.mean())/bcx.std())
        zbcxmax = zbcx.max()
        xmin = x.min()
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
