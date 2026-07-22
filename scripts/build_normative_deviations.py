"""Build data/exploratory/normative_deviations.pkl (training half only).

Applies the SAME subject exclusion as the CCA / XGBoost analyses: subjects whose
``{task}__max_rhat > 1.1`` on any task are dropped (N=1298). This exclusion is
applied *before* fitting, so BOTH the normative model's HV reference and the
deviation sample use it - on top of the finer per-parameter convergence QC in
``normative.py`` (rhat<=1.1 AND ess>=400 per parameter). Previously the AK pipeline
kept all 1399 training subjects; this aligns it with the other analyses.

Writes a dict ``{sub_ids, params, z, centile, indicator, qc_ok, is_hv}`` matching
the format ``anna_karenina.load_ak_data`` expects. Held-out half is never read.
"""

from pathlib import Path
import pickle

import polars as pl

from cogmood_analysis import normative as nm
from cogmood_analysis import shared_variance as sv

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "exploratory" / "training_data.csv"
DATA_DIR = REPO / "data"
OUT = REPO / "data" / "exploratory" / "normative_deviations.pkl"

#: Subject-level exclusion threshold, matching sv.load_views(exclude_rhat_above=1.1).
RHAT_MAX_SUBJECT = 1.1


def main() -> None:
    df = pl.read_csv(CSV, infer_schema_length=20000)
    n_total = df.height
    keep = pl.all_horizontal(
        [pl.col(f"{t}__max_rhat") <= RHAT_MAX_SUBJECT for t in sv.RHAT_TASKS]
    )
    df = df.filter(keep)
    is_hv = (df[nm.HV_COLUMN] == nm.HV_VALUE).to_numpy()
    print(f"subject max_rhat<=1.1 exclusion: {n_total} -> {df.height} "
          f"(HV in sample: {int(is_hv.sum())})")

    diag = nm.load_diagnostics(DATA_DIR)
    models = nm.fit_normative(df, diag)
    print(f"fitted normative models for {len(models)}/{len(sv.VIEW_A_COLUMNS)} params")
    dev = nm.compute_deviations(df, diag, models)

    out = {
        "sub_ids": dev.sub_ids,
        "params": dev.params,
        "z": dev.z,
        "centile": dev.centile,
        "indicator": dev.indicator,
        "qc_ok": dev.qc_ok,
        "is_hv": is_hv,
    }
    OUT.write_bytes(pickle.dumps(out))
    usable = float(dev.qc_ok.mean())
    print(f"wrote {OUT}\n  {dev.z.shape[0]} subjects x {dev.z.shape[1]} params "
          f"| {usable:.1%} cells QC-usable")


if __name__ == "__main__":
    main()
