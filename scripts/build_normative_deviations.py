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

import numpy as np
import polars as pl

from cogmood_analysis import normative as nm
from cogmood_analysis import provenance as prov
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
    # cross-fit HV: healthy volunteers get out-of-fold deviations (not scored in-sample)
    dev = nm.compute_deviations(df, diag, models, cross_fit_hv=True, k=5, seed=0)

    clip = nm.clipping_fraction(df, models)
    cmax = max(clip.values()) if clip else 0.0
    cmean = float(np.mean(list(clip.values()))) if clip else 0.0
    worst = sorted(clip.items(), key=lambda kv: -kv[1])[:3]
    print(f"transform clipping: mean {cmean:.3%}, max {cmax:.3%} of subjects; "
          f"worst: {', '.join(f'{p}={f:.2%}' for p, f in worst)}")

    out = {
        "sub_ids": dev.sub_ids,
        "params": dev.params,
        "z": dev.z,
        "centile": dev.centile,
        "indicator": dev.indicator,
        "qc_ok": dev.qc_ok,
        "is_hv": is_hv,
        "cross_fit_hv": True,
        "clipping_fraction": clip,
    }
    OUT.write_bytes(pickle.dumps(out))
    prov.write_sidecar(OUT, prov.provenance(
        CSV, dev.sub_ids,
        config={"analysis": "normative_deviations", "exclude_rhat_above": RHAT_MAX_SUBJECT,
                "cross_fit_hv": True, "k": 5, "seed": 0,
                "n_hv": int(is_hv.sum()), "clip_mean": cmean, "clip_max": cmax}))
    usable = float(dev.qc_ok.mean())
    print(f"wrote {OUT} (+ provenance sidecar)\n  {dev.z.shape[0]} subjects x "
          f"{dev.z.shape[1]} params | {usable:.1%} cells QC-usable")


if __name__ == "__main__":
    main()
