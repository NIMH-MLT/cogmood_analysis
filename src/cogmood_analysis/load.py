from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import tempfile
import json
import os
import polars as pl
import numpy as np
from scipy.stats import boxcox
from numpy.typing import ArrayLike, NDArray
from . import log


def load_task(zipped_path: str | os.PathLike, task_name: str,  subject: str, runnum: int, as_dateframe: bool = False) -> pl.DataFrame:
    zipped_path= Path(zipped_path)
    if not zipped_path.exists():
        raise FileNotFoundError(zipped_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        slog_file = ZipFile(zipped_path).extract(f'log_{task_name}_0.slog', path=tmpdir)
        lod = log.log2dl(slog_file)
        loddf = pl.from_dicts(lod)
    try:
        loddf = loddf.filter(pl.col('run_num') == runnum)
    except pl.exceptions.ColumnNotFoundError:
        loddf = loddf.filter(pl.col('block') == runnum)
    file_date = datetime.fromtimestamp(zipped_path.stat().st_mtime)
    loddf = loddf.with_columns(
        sub_id=pl.lit(subject),
        zrn=pl.lit(runnum),
        date=pl.lit(file_date)
    )
    if task_name == 'cab':
        loddf = loddf.with_columns(
            rt=pl.col('resp_rt')
        )
    if as_dateframe:
        return loddf.to_pandas()
    else:
        return loddf


def nanboxcox(x : ArrayLike) -> NDArray[np.float64]:
    """ Run boxcox transformation with nan masking
    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    boxcox transformed values
    """
    try:
        x = x.to_numpy()
    except AttributeError:
        x = np.array(x)
    res = np.zeros_like(x) * np.nan

    try:
        xmask = ~np.isnan(x)
        goodx = x[xmask]
        goodxbc = boxcox(goodx)[0]
        res[xmask] = goodxbc
    except (IndexError, ValueError):
        pass
    return res


def boxcoxmask(x : ArrayLike, thresh : float = 3) -> NDArray[np.bool_]:
    """ Iteratively run boxcox transformations and drop any values that 
    are more than thresh standard deviations away from the mean.
    
    Parameters
    ----------
    x : ArrayLike

    Returns
    -------
    boolean mask for values to include
    """
    try:
        x = x.to_numpy()
    except AttributeError:
        x = np.array(x)
    tmp = np.zeros_like(x) * np.nan
    try:
        ogxmask = ~np.isnan(x)
        goodxbc: NDArray[np.float64] = boxcox(x[ogxmask])[0]
        tmp[ogxmask] = np.abs((goodxbc - goodxbc.mean()) / goodxbc.std())
        newmask = ogxmask.copy()
        while tmp[ogxmask].max() > thresh:
            newmask = newmask & (tmp < thresh)
            goodxbc: NDArray[np.float64] = boxcox(x[newmask])[0]
            tmp = np.zeros_like(x) * np.nan
            tmp[newmask] = np.abs((goodxbc - goodxbc.mean()) / goodxbc.std())
        return newmask
    except (IndexError, ValueError):
        return np.zeros_like(x, dtype=bool)
