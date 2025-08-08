from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import tempfile
import json
import os
import polars as pl
from . import log

def load_task(zipped_path: str | os.PathLike, task_name: str,  subject: str, runnum: int, as_dateframe=False):
    zipped_path= Path(zipped_path)
    if not zipped_path.exists():
        raise FileNotFoundError(zipped_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        slog_file = ZipFile(zipped_path).extract(f'log_{task_name}_0.slog', path=tmpdir)
        lod = log.log2dl(slog_file)
        loddf = pl.from_dicts(lod)
    try:
        loddf = loddf.filter(pl.col('run_num') == runnum)
    except AttributeError:
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
