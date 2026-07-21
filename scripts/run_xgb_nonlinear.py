"""Run the XGBoost full-vs-null nonlinear SHARP check on the training half.

Training half only (``data/exploratory/training_data.csv``), N=1298 after the
``max_rhat>1.1`` exclusion. Writes the tidy per-target results table to
``data/exploratory/xgb_nonlinear_results.parquet`` (gitignored).
"""

from pathlib import Path

from cogmood_analysis import xgb_nonlinear as xg

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "exploratory" / "training_data.csv"
OUT = REPO_ROOT / "data" / "exploratory" / "xgb_nonlinear_results.parquet"


def main() -> None:
    data = xg.load_xgb_data(CSV)
    print(f"Loaded N={data.covars.shape[0]} (of {data.n_total}), "
          f"{data.params.shape[1]} params, {len(data.symptom_cols)} targets")
    tbl = xg.run_xgb_nonlinear(data, J=30, K=5, seed=0, n_jobs=8, verbose=True)
    tbl.write_parquet(OUT)
    print(f"\nWrote {OUT}")
    print(tbl.select(["target", "mean_r2_gain", "ci_lo", "ci_hi",
                      "p_one_sided", "q_fdr"]))


if __name__ == "__main__":
    main()
