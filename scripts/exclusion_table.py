"""Build participant exclusion tables (counts + reasons) for the cogmood analysis.

Reconstructs the CONSORT-style funnel from raw recruitment down to the
analysis-eligible cohort and its 50/50 train/test split, plus a per-reason
breakdown of the behavioral and model-fit exclusions. The funnel also shows the
model-fit convergence exclusion (any-task ``max_rhat > 1.1``) applied to the
training half by the CCA / XGBoost / AK analyses (1399 -> 1298).

Data sources (under ``data/``):
- survey/survey_responses.csv          -- one row per subject with a survey response
- task/to_model/data_quality.csv       -- per-subject behavioral QC flags
  (has_all, corr_ok_<task>, resp05/10_ok_<task>, good05/good10) from
  notebooks/exclusion_criteria.ipynb
- task/<task>_results.csv              -- per-subject model-fit diagnostics
- exploratory/training_data.csv        -- the training half of the analysis cohort

The number who *interacted with the survey in any way* is not in these files
(it comes from the recruitment platform) and is passed in via --n-interacted.

Behavioral criterion: ``good10`` (>10% non-response/RT-outlier tolerated), which
is the one the analysis cohort uses. Thresholds (from exclusion_criteria.ipynb):
below-chance accuracy = bart<24/36, cab<57/96, rdm<105/186, flkr<57/96 correct
(one-sided binomial p<0.05); non-response = >10% of trials non-response or
box-cox RT outliers (bart has no non-response criterion).

Usage:
    uv run python scripts/exclusion_table.py [--n-interacted 4987] [--threshold good10]
Writes data/exploratory/exclusion_funnel.csv and exclusion_by_reason.csv and
prints both as markdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TASKS = ["flkr", "cab", "rdm", "bart"]
TASK_LABEL = {"flkr": "Flanker", "cab": "CAB", "rdm": "RDM", "bart": "BART"}
ACC_THRESH = {"bart": "24/36", "cab": "57/96", "rdm": "105/186", "flkr": "57/96"}

#: Model-fit convergence exclusion applied by the CCA / XGBoost / AK analyses:
#: drop subjects whose ``{task}__max_rhat`` exceeds this on any task.
RHAT_MAX = 1.1


def _fit_subjects(task: str) -> set[str]:
    df = pl.read_csv(DATA / "task" / f"{task}_results.csv", infer_schema_length=20000)
    idc = "sub_id" if "sub_id" in df.columns else "subject"
    return set(df[idc].to_list())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-interacted", type=int, default=4987,
                    help="people who interacted with the survey in any way (recruitment platform)")
    ap.add_argument("--threshold", choices=["good10", "good05"], default="good10")
    args = ap.parse_args()
    g = args.threshold
    resp_ok = f"resp{'10' if g == 'good10' else '05'}_ok"
    nr_label = ">10%" if g == "good10" else ">5%"

    dq = pl.read_csv(DATA / "task" / "to_model" / "data_quality.csv", infer_schema_length=20000)
    srv = pl.read_csv(DATA / "survey" / "survey_responses.csv", infer_schema_length=20000)
    sidc = "sub_id" if "sub_id" in srv.columns else srv.columns[0]
    td = pl.read_csv(DATA / "exploratory" / "training_data.csv", infer_schema_length=20000)

    n_survey = srv[sidc].n_unique()
    n_complete = int(dq["has_all"].sum())          # all rows are complete in this file
    n_behavioral = int(dq.filter(pl.col(g))["sub_id"].len())
    all4 = set.intersection(*[_fit_subjects(t) for t in TASKS])
    good = set(dq.filter(pl.col(g))["sub_id"].to_list())
    eligible = good & all4
    n_eligible = len(eligible)
    n_train = td["sub_id"].n_unique()
    n_test = n_eligible - n_train

    # model-fit convergence exclusion on the training half (matches sv.load_views):
    # drop subjects with any-task max_rhat > 1.1 (the analysis sample, N=1298).
    rhat_cols = [f"{t}__max_rhat" for t in TASKS if f"{t}__max_rhat" in td.columns]
    n_train_conv = (
        td.filter(pl.all_horizontal([pl.col(c) <= RHAT_MAX for c in rhat_cols])).height
        if rhat_cols else n_train
    )

    # ---- CONSORT funnel ----
    funnel = [
        ("Interacted with survey", args.n_interacted, "", "recruitment platform"),
        ("Provided survey response", n_survey, args.n_interacted - n_survey,
         "started but no usable survey response"),
        ("Completed all 4 cognitive tasks", n_complete, n_survey - n_complete,
         "did not complete the full task battery"),
        (f"Passed behavioral QC ({g})", n_behavioral, n_complete - n_behavioral,
         f"below-chance accuracy and/or {nr_label} non-response/RT-outliers (any task)"),
        ("Successful model fits (all 4 tasks)", n_eligible, n_behavioral - n_eligible,
         "missing/failed model fit for >=1 task"),
        ("Analysis-eligible cohort", n_eligible, 0, "--"),
        ("  -> Training set (exploratory)", n_train, "", "50/50 split"),
        ("      -> Training, converged fits (max_rhat<=1.1)", n_train_conv,
         n_train - n_train_conv,
         "excl. any-task max_rhat>1.1 (analysis sample for CCA/XGBoost/AK)"),
        ("  -> Held-out test set", n_test, "", "50/50 split"),
    ]
    funnel_df = pl.DataFrame(
        {"stage": [r[0] for r in funnel],
         "n_remaining": [str(r[1]) for r in funnel],
         "n_excluded": [str(r[2]) for r in funnel],
         "reason": [r[3] for r in funnel]}
    )

    # ---- per-reason behavioral breakdown (among the n_complete completers; overlapping) ----
    rows = []
    for t in TASKS:
        rows.append({"stage": "behavioral", "task": TASK_LABEL[t],
                     "reason": f"below-chance accuracy (<{ACC_THRESH[t]} correct)",
                     "n_failing": int((~dq[f"corr_ok_{t}"]).sum())})
        if t != "bart":
            rows.append({"stage": "behavioral", "task": TASK_LABEL[t],
                         "reason": f"{nr_label} non-response / RT-outlier trials",
                         "n_failing": int((~dq[f"{resp_ok}_{t}"]).sum())})
    rows.append({"stage": "behavioral", "task": "any",
                 "reason": "failed behavioral QC on >=1 task (union)",
                 "n_failing": int(dq.filter(~pl.col(g))["sub_id"].len())})
    # model-fit: among behaviorally-good subjects, missing each task's fit
    for t in TASKS:
        rows.append({"stage": "model-fit", "task": TASK_LABEL[t],
                     "reason": "missing/failed model fit",
                     "n_failing": len(good - _fit_subjects(t))})
    rows.append({"stage": "model-fit", "task": "any",
                 "reason": "missing >=1 task fit (union)",
                 "n_failing": len(good - all4)})
    reason_df = pl.DataFrame(rows)

    out1 = DATA / "exploratory" / "exclusion_funnel.csv"
    out2 = DATA / "exploratory" / "exclusion_by_reason.csv"
    funnel_df.write_csv(out1)
    reason_df.write_csv(out2)

    def md(df: pl.DataFrame) -> str:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        for r in df.iter_rows():
            lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
        return "\n".join(lines)

    print(f"# Participant exclusions (threshold={g})\n")
    print("## CONSORT funnel\n")
    print(md(funnel_df))
    print("\n## Per-reason breakdown (counts overlap; among completers / behaviorally-good)\n")
    print(md(reason_df))
    print(f"\nSaved -> {out1}\n         {out2}")


if __name__ == "__main__":
    main()
