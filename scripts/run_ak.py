"""Run the Anna Karenina deviation-vs-symptom analysis (training half only).

Consumes the deviation matrix built by scripts/build_normative_deviations.py
(N=1298 after the max_rhat>1.1 exclusion, matching the CCA / XGBoost sample) and
writes the tidy results table to ``data/exploratory/ak_results.parquet`` (gitignored).
The held-out half is never read.
"""

from pathlib import Path

import polars as pl

from cogmood_analysis import anna_karenina as ak

REPO = Path(__file__).resolve().parents[1]
PKL = REPO / "data" / "exploratory" / "normative_deviations.pkl"
CSV = REPO / "data" / "exploratory" / "training_data.csv"
OUT = REPO / "data" / "exploratory" / "ak_results.parquet"


def main() -> None:
    data = ak.load_ak_data(PKL, CSV)
    print(f"AK sample: N={data.sub_ids.shape[0]} | {len(data.params)} params "
          f"| {len(data.symptom_cols)} symptom targets | HV in sample={int(data.is_hv.sum())}")
    tbl = ak.run_ak(data, seed=0, verbose=True)
    tbl.write_parquet(OUT)
    print(f"\nWrote {OUT} ({tbl.height} rows)")

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
