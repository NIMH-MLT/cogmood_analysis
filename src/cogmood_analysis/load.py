from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import tempfile
import json
import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from numpy.typing import ArrayLike, NDArray
from typing import Any
from . import log
from . import survey_helpers as sh


def load_survey(
    json_path: str | os.PathLike,
) -> dict[str, Any]:
    """Load survey respones from a json to a dict.
    Adds the filename as a sub_id field and the file date as a survey_date field.
    Parameters
    ----------
    json_path : str

    Returns
    -------
    res : dict
    """

    json_path = Path(json_path)
    survey_date = datetime.fromtimestamp(json_path.stat().st_mtime)
    resp = dict(sub_id=json_path.parts[-1].split(".")[0], survey_date=survey_date)
    resp.update(
        sh.extract_responses(
            json.loads(json_path.read_text())[0]["response"], decoders=sh.SURVEY_DECODE
        )
    )
    return resp


def unpack_results(jdat, simple_keys, nested_keys):
    row = {}
    for sk in simple_keys:
        row[sk] = jdat[sk]
    for nk in nested_keys:
        nested_name = nk.split('_')[0]
        for nkk in jdat[nk].keys():
            row[f'{nested_name}__{nkk}'] = jdat[nk][nkk]
    return row


def load_flkr_results(filename):
    """
    Load in a simulation that was saved to a pickle.gz.
    """
    gf = gzip.open(filename, 'rb')
    res = pickle.loads(gf.read(), encoding='latin1')
    gf.close()
    return res


def proc_survey(responses: pl.DataFrame | list[dict[str, Any]]) -> pl.DataFrame:
    """Process list of survey responeses to score scales and subscales
    Parameters
    ----------
    """
    if not isinstance(responses, pl.DataFrame):
        responses = pl.DataFrame(responses)

    # clean up column names a little bit
    responses = responses.rename(sh.COL_LUT)

    # create attention check indicator columns
    responses = responses.with_columns(
        attnbin__2=pl.col("attn__2") == 4,
        attnbin__3=pl.col("attn__3") == 2,
        attnbin__4=pl.col("attn__4") == 3,
        attnbin__5=pl.col("attn__5") == 0,
    )

    # sum all of the scales and subscales
    exprs = []
    for scale, subscale in sh.SCALES:
        if subscale is None:
            scale_columns = pl.selectors.starts_with(scale)
            expr = responses.select(scale_columns).sum_horizontal().alias(scale)
            exprs.append(expr)
            if scale == "attnbin":
                scale = "todayattn"
            else:
                scale = "today" + scale
            scale_columns = pl.selectors.starts_with(scale)
            expr = responses.select(scale_columns).sum_horizontal().alias(scale)
            exprs.append(expr)
        else:
            scale_columns = pl.selectors.starts_with(scale) & pl.selectors.contains(
                f"_{subscale}_"
            )
            expr = (
                responses.select(scale_columns)
                .sum_horizontal()
                .alias(f"{scale}_{subscale}")
            )
            exprs.append(expr)
            scale = "today" + scale
            scale_columns = pl.selectors.starts_with(scale) & pl.selectors.contains(
                f"_{subscale}_"
            )
            expr = (
                responses.select(scale_columns)
                .sum_horizontal()
                .alias(f"{scale}_{subscale}")
            )
            exprs.append(expr)

    responses = responses.with_columns(*exprs)

    # create columns for screening groups
    responses = responses.with_columns(
        screen_group=pl.when(
            ~pl.col("ongoing_mentalhealth")
            & ~pl.col("experience_depression")
            & ~pl.col("experience_anxiety")
            & ~pl.col("have_adhd")
        )
        .then(pl.lit("hv"))
        .when(
            pl.col("ongoing_mentalhealth")
            & ~pl.col("experience_depression")
            & ~pl.col("experience_anxiety")
            & ~pl.col("have_adhd")
        )
        .then(pl.lit("othermh"))
        .when(
            pl.col("experience_depression")
            & ~pl.col("experience_anxiety")
            & ~pl.col("have_adhd")
        )
        .then(pl.lit("dep"))
        .when(
            ~pl.col("experience_depression")
            & pl.col("experience_anxiety")
            & ~pl.col("have_adhd")
        )
        .then(pl.lit("anx"))
        .when(
            ~pl.col("experience_depression")
            & ~pl.col("experience_anxiety")
            & pl.col("have_adhd")
        )
        .then(pl.lit("atn"))
        .when(
            pl.col("experience_depression")
            & pl.col("experience_anxiety")
            & ~pl.col("have_adhd")
        )
        .then(pl.lit("dep_anx"))
        .when(
            pl.col("experience_depression")
            & ~pl.col("experience_anxiety")
            & pl.col("have_adhd")
        )
        .then(pl.lit("dep_atn"))
        .when(
            ~pl.col("experience_depression")
            & pl.col("experience_anxiety")
            & pl.col("have_adhd")
        )
        .then(pl.lit("anx_atn"))
        .when(
            pl.col("experience_depression")
            & pl.col("experience_anxiety")
            & pl.col("have_adhd")
        )
        .then(pl.lit("dep_anx_atn"))
    )

    return responses


def load_task(
    zipped_path: str | os.PathLike,
    task_name: str,
    subject: str,
    runnum: int,
    as_dateframe: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Load task from a zipped file. The interior compressed slog corresponding to the task file is extracted without
    uncompressing the entire zipped file for security. Only rows from the slog
    where the run number matches the passed runnum are kept. A field for the subject
    (sub_id) is added, as well as the date of the file.

    Parameters
    ----------
    zipped_path : str
    task_name : str
    subject : str
    runnum : int
    as_dataframe : bool
        If true, return as pandas and not polars

    Returns
    -------
    result : polars or pandas dataframe
    """
    zipped_path = Path(zipped_path)
    if not zipped_path.exists():
        raise FileNotFoundError(zipped_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        slog_file = ZipFile(zipped_path).extract(f"log_{task_name}_0.slog", path=tmpdir)
        lod = log.log2dl(slog_file)
        loddf = pl.from_dicts(lod)
    try:
        loddf = loddf.filter(pl.col("run_num") == runnum)
    except pl.exceptions.ColumnNotFoundError:
        loddf = loddf.filter(pl.col("block") == runnum)
    file_date = datetime.fromtimestamp(zipped_path.stat().st_mtime)
    loddf = loddf.with_columns(
        sub_id=pl.lit(subject), zrn=pl.lit(runnum), date=pl.lit(file_date)
    )
    # alias columns from cab
    if task_name == "cab":
        loddf = loddf.with_columns(rt=pl.col("resp_rt"), correct=pl.col("resp_acc"))
    elif task_name == "rdm":
        # add coherence difference column to remd
        loddf = loddf.with_columns(
            coh_dif=(pl.col("left_coherence") - pl.col("right_coherence")).abs()
        )
    elif task_name == "bart":
        possible_keys = ["F", "J"]
        collect_keys = (
            loddf.group_by("balloon_id")
            .last()
            .filter(pl.col("pop_status") == "not_popped")
            .select("key_pressed")
            .to_series()
            .unique()
            .to_numpy()
        )
        if len(collect_keys) > 1:
            raise ValueError(
                f"There should only be one collect key per participant, but I found {collect_keys}."
            )
        pump_keys = (
            loddf.group_by("balloon_id")
            .last()
            .filter(pl.col("pop_status") == "popped")
            .select("key_pressed")
            .to_series()
            .unique()
            .to_numpy()
        )
        if len(pump_keys) > 1:
            raise ValueError(
                f"There should only be one pump key per participant, but I found {pump_keys}."
            )
        if len(collect_keys) == 0:
            if len(pump_keys) == 0:
                raise ValueError(
                    "Something has gone terribly wrong, this bart data has neither pumps nor collects."
                )
            else:
                pump_key = pump_keys[0]
                for key in possible_keys:
                    if key != pump_key:
                        collect_key = key
        else:
            collect_key = collect_keys[0]
            for key in possible_keys:
                if key != collect_key:
                    pump_key = key
        try:
            loddf = loddf.with_columns(
                pl.col("pump_button").fill_null(value=pump_key).alias("pump_button"),
                pl.col("collect_button")
                .fill_null(value=collect_key)
                .alias("collect_button"),
            )
        except pl.exceptions.ColumnNotFoundError:
            loddf = loddf.with_columns(
                pump_button = pl.lit(pump_key),
                collet_button=pl.lit(collect_key),
            )
    if as_dateframe:
        return loddf.to_pandas()
    else:
        return loddf


def nanboxcox(x: ArrayLike) -> NDArray[np.float64]:
    """Run boxcox transformation with nan masking
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


def boxcoxmask(x: ArrayLike, thresh: float = 3) -> NDArray[np.bool_]:
    """Iteratively run boxcox transformations and drop any values that
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
        while tmp[newmask].max() > thresh:
            newmask = newmask & (tmp < thresh)
            goodxbc: NDArray[np.float64] = boxcox(x[newmask])[0]
            tmp = np.zeros_like(x) * np.nan
            tmp[newmask] = np.abs((goodxbc - goodxbc.mean()) / goodxbc.std())
        return newmask
    except (IndexError, ValueError):
        return np.zeros_like(x, dtype=bool)


def load_prolific_data(prolific_data_path: str | os.PathLike) -> pl.DataFrame:
    pdf = pl.read_csv(prolific_data_path, separator='\t')
    # grab good records for folks where there are some good rows and some bad
    double_list = pdf.group_by('sub_id').agg(pl.col('age').n_unique()).sort('age').filter(pl.col('sub_id').is_not_null() & (pl.col('age') > 1))['sub_id'].to_list()
    pdf = pdf.filter((~pl.col('sub_id').is_in(double_list)) | (pl.col('status') != 'RETURNED'))
    pdf = (pdf
        .filter(pl.col('status') != 'Status')
        .filter(pl.col('sub_id').is_not_null())
        .group_by('sub_id').last()
        .with_columns(
            pl.when(pl.col('depression') == "Yes").then(True).when(pl.col('depression') == "No").then(False).alias('p_depression'),
            pl.when(pl.col('anxiety') == "Yes").then(True).when(pl.col('anxiety') == "No").then(False).alias('p_anxiety'),
            pl.when(pl.col('attention') == "Yes").then(True).when(pl.col('attention') == "No").then(False).alias('p_attention'),
            pl.when(pl.col('mental_health_ongoing') == "Yes").then(True).when(pl.col('mental_health_ongoing') == "No").then(False).alias('p_mental_health_ongoing'),
            pl.when(pl.col('mental_illness_impact') == "Yes").then(True).when(pl.col('mental_illness_impact') == "No").then(False).alias('p_mental_illness_impact'),
            pl.when(pl.col('age') == 'CONSENT_REVOKED').then(None).otherwise(pl.col('age')).str.to_integer().alias('p_age'),
            pl.when(pl.col('language') == 'CONSENT_REVOKED').then(None).otherwise(pl.col('language')).alias('p_language'),
            pl.when((pl.col('student_status') == 'CONSENT_REVOKED') | (pl.col('student_status') == 'DATA_EXPIRED')).then(None).otherwise(pl.col('student_status')).alias('p_student_status'),
            pl.when((pl.col('employment_status') == 'CONSENT_REVOKED') | (pl.col('employment_status') == 'DATA_EXPIRED')).then(None).otherwise(pl.col('employment_status')).alias('p_employment_status'),
            pl.when(pl.col('computer_os') == 'CONSENT_REVOKED').then(None).otherwise(pl.col('computer_os')).alias('p_computer_os'),
            pl.col('total_approvals').str.to_integer().alias('p_total_approvals'),
            pl.col('ses').str.to_integer().alias('p_ses'),
            pl.col('highest_education').alias('p_highest_education')
        )
    )
    # create columns for screening groups
    pdf = pdf.with_columns(
        prolific_screen_group=pl.when(
            ~pl.col("p_mental_health_ongoing")
            & ~pl.col("p_depression")
            & ~pl.col("p_anxiety")
            & ~pl.col("p_attention")
        )
        .then(pl.lit("hv"))
        .when(
            pl.col("p_mental_health_ongoing")
            & ~pl.col("p_depression")
            & ~pl.col("p_anxiety")
            & ~pl.col("p_attention")
        )
        .then(pl.lit("othermh"))
        .when(
            pl.col("p_depression")
            & ~pl.col("p_anxiety")
            & ~pl.col("p_attention")
        )
        .then(pl.lit("dep"))
        .when(
            ~pl.col("p_depression")
            & pl.col("p_anxiety")
            & ~pl.col("p_attention")
        )
        .then(pl.lit("anx"))
        .when(
            ~pl.col("p_depression")
            & ~pl.col("p_anxiety")
            & pl.col("p_attention")
        )
        .then(pl.lit("atn"))
        .when(
            pl.col("p_depression")
            & pl.col("p_anxiety")
            & ~pl.col("p_attention")
        )
        .then(pl.lit("dep_anx"))
        .when(
            pl.col("p_depression")
            & ~pl.col("p_anxiety")
            & pl.col("p_attention")
        )
        .then(pl.lit("dep_atn"))
        .when(
            ~pl.col("p_depression")
            & pl.col("p_anxiety")
            & pl.col("p_attention")
        )
        .then(pl.lit("anx_atn"))
        .when(
            pl.col("p_depression")
            & pl.col("p_anxiety")
            & pl.col("p_attention")
        )
        .then(pl.lit("dep_anx_atn"))
    )
    # make sure that there's only one row of prolific data per subject id
    assert pdf.group_by('sub_id').len()['len'].max() == 1
    return pdf
