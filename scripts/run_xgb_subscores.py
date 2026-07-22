"""XGBoost full-vs-null SHARP test using the 4 per-task summary scores.

Companion to scripts/run_xgb_nonlinear.py: identical pipeline, but the full model's
cognitive features are the 4 per-task summary scores (``{task}__sub_score``) instead
of the 30 fitted model parameters. A lower-dimensional, arguably more robust feature
view of task behavior.

Same exclusion (``max_rhat>1.1``) and complete-case discipline as the parameter run
for a direct comparison. Training half only; the held-out half is never read. Writes
``data/exploratory/xgb_subscores_sharp_results.parquet`` (gitignored).
"""

from pathlib import Path

from cogmood_analysis import provenance as prov
from cogmood_analysis import xgb_nonlinear as xg

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "exploratory" / "training_data.csv"
OUT = REPO_ROOT / "data" / "exploratory" / "xgb_subscores_sharp_results.parquet"

SUBSCORES = ("bart__sub_score", "rdm__sub_score", "cab__sub_score", "flkr__sub_score")


def main() -> None:
    data = xg.load_xgb_data(CSV, param_columns=SUBSCORES)
    print(f"Loaded N={data.covars.shape[0]} (of {data.n_total}), "
          f"features={data.param_cols}, {len(data.symptom_cols)} targets")
    tbl = xg.run_xgb_nonlinear(data, J=30, K=5, seed=0, n_jobs=8, verbose=True)
    tbl.write_parquet(OUT)
    prov.write_sidecar(OUT, prov.provenance(
        CSV, data.sub_ids,
        config={"analysis": "xgb_subscores_sharp", "features": "age+sex+4subscores",
                "J": 30, "K": 5, "inference": "score_test_one_sided", "seed": 0,
                "xgb": xg.XGB_PARAMS}))
    print(f"\nWrote {OUT} (+ provenance sidecar)")
    print(tbl.select(["target", "mean_r2_gain", "ci_lo", "ci_hi",
                      "p_one_sided", "q_fdr"]))
    n_sig = int((tbl["q_fdr"] < 0.05).sum())
    print(f"\nTargets surviving BH-FDR (q<0.05): {n_sig} / {tbl.height}")


if __name__ == "__main__":
    main()
