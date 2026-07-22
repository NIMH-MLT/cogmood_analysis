"""Full-sample repeated K-fold XGBoost full-vs-null R^2 cross-check.

Robustness check for the SHARP split-half test (scripts/run_xgb_nonlinear.py):
here each model trains on the whole training-half sample per fold (~830 rows,
roughly double the split-half), with no SHARP inference - just descriptive R^2
differences, to see whether more data surfaces any positive gain.

Training half only (``data/exploratory/training_data.csv``, N=1298 after the
``max_rhat>1.1`` exclusion). Writes ``data/exploratory/xgb_kfold_results.parquet``
(gitignored). The held-out half is never read.
"""

from pathlib import Path

from cogmood_analysis import provenance as prov

from cogmood_analysis import xgb_nonlinear as xg

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "exploratory" / "training_data.csv"
OUT = REPO_ROOT / "data" / "exploratory" / "xgb_kfold_results.parquet"


def main() -> None:
    data = xg.load_xgb_data(CSV)
    print(f"Loaded N={data.covars.shape[0]} (of {data.n_total}), "
          f"{data.params.shape[1]} params, {len(data.symptom_cols)} targets")
    tbl = xg.run_xgb_kfold(data, K=5, n_reps=20, seed=0, n_jobs=8, verbose=True)
    tbl.write_parquet(OUT)
    prov.write_sidecar(OUT, prov.provenance(
        CSV, data.sub_ids,
        config={"analysis": "xgb_kfold_descriptive", "features": "age+sex+30params",
                "K": 5, "n_reps": 20, "inference": "descriptive_only", "seed": 0}))
    print(f"\nWrote {OUT} (+ provenance sidecar)")
    print(tbl.select(["target", "mean_r2_gain", "ci_lo", "ci_hi",
                      "frac_folds_positive", "max_fold_gain",
                      "mean_r2_full", "mean_r2_null"]))
    n_pos = int((tbl["mean_r2_gain"] > 0).sum())
    print(f"\nTargets with positive mean R2 gain: {n_pos} / {tbl.height}")


if __name__ == "__main__":
    main()
