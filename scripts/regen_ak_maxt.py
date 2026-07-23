"""Regenerate the canonical AK family-wise maxT artifacts from the CURRENT (cross-fitted)
normative deviations, so they stay in sync with ak_results.parquet and the notebook.

The notebook computes maxT inline (source of truth); these files exist for anyone loading
the canonical filenames. Before overwriting, the current files get a content-addressed,
never-clobbering backup (``provenance.backup_artifact``); the stale pre-cross-fit copies
(2026-07-17) remain preserved as ``*.r1.*``. Cheap (permutation on the deviation matrix; no
model refit). Training half only.
"""
from pathlib import Path

import polars as pl

from cogmood_analysis import anna_karenina as ak
from cogmood_analysis import provenance as prov

EXP = Path(__file__).resolve().parents[1] / "data" / "exploratory"
PKL = EXP / "normative_deviations.pkl"
CSV = EXP / "training_data.csv"
N_PERM = 10000


def main() -> None:
    data = ak.load_ak_data(PKL, CSV)
    resid = [t for t in ak.symptom_targets(data) if t.variant == "resid"]
    joint, _ = ak.maxstat_correction(data.absz, resid, n_perm=N_PERM, seed=0)
    within = ak.maxstat_within_approach(data.absz, resid, n_perm=N_PERM, seed=0)

    provenance = prov.provenance(
        CSV, data.sub_ids,
        config={"analysis": "ak_maxT", "family": "5 unsupervised approaches x 21 resid targets",
                "n_perm": N_PERM, "seed": 0, "procedure": "westfall_young_stepdown_maxT"})

    for name, tbl in (("ak_maxstat.parquet", joint), ("ak_within_maxT.parquet", within)):
        out = EXP / name
        prov.backup_artifact(out)     # content-addressed, never clobbers history (incl. stale .r1)
        tbl.write_parquet(out)
        prov.write_sidecar(out, provenance)

    top = joint.sort("adj_p_maxT").head(1).row(0, named=True)
    print(f"wrote ak_maxstat + ak_within_maxT ({N_PERM} perms) | "
          f"joint survivors<0.05: {joint.filter(pl.col('adj_p_maxT') < 0.05).height} | "
          f"top {top['approach']}->{top['target']} adj_p={top['adj_p_maxT']:.4f} "
          f"within={top['within_maxT']:.4f}")


if __name__ == "__main__":
    main()
