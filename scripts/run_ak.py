"""Run the Anna Karenina deviation-vs-symptom analysis (training half only).

Consumes the deviation matrix built by scripts/build_normative_deviations.py
(N=1298 after the max_rhat>1.1 exclusion, matching the CCA / XGBoost sample) and
writes the tidy results table to ``data/exploratory/ak_results.parquet`` (gitignored).
The held-out half is never read.
"""

import os
from pathlib import Path

import polars as pl

from cogmood_analysis import anna_karenina as ak
from cogmood_analysis import provenance as prov

REPO = Path(__file__).resolve().parents[1]
PKL = REPO / "data" / "exploratory" / "normative_deviations.pkl"
CSV = REPO / "data" / "exploratory" / "training_data.csv"
OUT = REPO / "data" / "exploratory" / "ak_results.parquet"


def main() -> None:
    n_jobs = int(os.environ.get("AK_NJOBS", "8"))
    data = ak.load_ak_data(PKL, CSV)
    print(f"AK sample: N={data.sub_ids.shape[0]} | {len(data.params)} params "
          f"| {len(data.symptom_cols)} symptom targets | HV in sample={int(data.is_hv.sum())} "
          f"| n_jobs_sup={n_jobs}")
    tbl = ak.run_ak(data, seed=0, verbose=True, n_jobs_sup=n_jobs)
    tbl.write_parquet(OUT)
    # provenance sidecar (parquet metadata is awkward for nested dicts)
    provenance = prov.provenance(
        CSV, data.sub_ids,
        config={"analysis": "anna_karenina", "n_perm_uni": 2000, "n_perm_sup": 200,
                "supervised": "nested_elasticnetcv_oof_r2", "seed": 0,
                "n_targets": int(tbl["target"].n_unique())},
    )
    prov.write_sidecar(OUT, provenance)
    print(f"\nWrote {OUT} ({tbl.height} rows) + provenance sidecar "
          f"(commit={str(provenance['analysis_source_commit'])[:9]} dirty={provenance['dirty']})")

    # canonical family-wise maxT artifacts (kept in sync with ak_results); the notebook
    # recomputes maxT inline as the source of truth. See scripts/regen_ak_maxt.py.
    resid = [t for t in ak.symptom_targets(data) if t.variant == "resid"]
    joint, _ = ak.maxstat_correction(data.absz, resid, n_perm=10000, seed=0)
    within = ak.maxstat_within_approach(data.absz, resid, n_perm=10000, seed=0)
    maxt_prov = prov.provenance(CSV, data.sub_ids, config={
        "analysis": "ak_maxT", "n_perm": 10000, "seed": 0,
        "procedure": "westfall_young_stepdown_maxT"})
    for name, mt in (("ak_maxstat.parquet", joint), ("ak_within_maxT.parquet", within)):
        mt.write_parquet(OUT.parent / name)
        prov.write_sidecar(OUT.parent / name, maxt_prov)
    print(f"Wrote ak_maxstat + ak_within_maxT (joint survivors<0.05: "
          f"{joint.filter(pl.col('adj_p_maxT') < 0.05).height})")

    primary = tbl.filter(
        (pl.col("target") == "PC1")
        & (pl.col("variant") == "resid")
        & (pl.col("approach") == "max_abs")
    )
    print("\nPre-specified primary (max|z| -> PC1 resid):")
    print(primary.select(["approach", "target", "variant", "effect", "p"]))
    sig = tbl.filter(pl.col("fdr_q") < 0.05)
    print(f"\nSecondary rows surviving BH-FDR (q<0.05): {sig.height}")
    if sig.height:
        print(sig.select(["approach", "target", "variant", "effect", "p", "fdr_q"]))


if __name__ == "__main__":
    main()
