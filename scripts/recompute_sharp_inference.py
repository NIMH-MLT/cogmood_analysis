"""Recompute SHARP inference from the SAVED fold-difference arrays (no model refit).

The round-3 SHARP optimizer is boundary-aware and demonstrably global; this recomputes
the reported inference (CIs, comparisons, p-values) for the CCA ladder and the two
XGBoost artifacts directly from their stored ``D_A``/``D_B`` (ladder) and ``d_a``/``d_b``
(XGBoost) half-statistics — no predictive models are refit. Each artifact is preserved by a
content-addressed, never-clobbering backup (``provenance.backup_artifact`` →
``<name>.bak-<sha12>.<ext>`` + its sidecar) before being rewritten with self-consistent
provenance (embedded + sidecar, both commit fields = current tip); rerunning is therefore
safe and idempotent. Permutation nulls are optimizer-independent and carried over unchanged.
"""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from cogmood_analysis import provenance as prov
from cogmood_analysis import sharp

EXP = Path(__file__).resolve().parents[1] / "data" / "exploratory"


def _refresh(rec: dict, note: str) -> dict:
    """Refresh an existing provenance record to the current clean tip (both commits)."""
    rec = dict(rec)
    commit = prov.git_commit()
    rec["schema_version"] = prov.SCHEMA_VERSION
    rec["analysis_source_commit"] = commit          # recomputed at the tip from saved D
    rec["provenance_stamp_commit"] = commit
    rec["dirty"] = prov.git_dirty()
    rec["env_lockfile_sha256"] = prov._lockfile_sha256()
    rec.pop("source_commit", None)
    rec.setdefault("config", {})
    rec["config"]["inference_recomputed_from_saved_D"] = note
    rec["created_at"] = datetime.now(timezone.utc).isoformat()
    return rec


def _bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * n / (n - rank)); q[i] = prev
    return q


def recompute_ladder(path: Path) -> None:
    R = pickle.loads(path.read_bytes())
    import itertools
    res = sharp.SharpResult(D_A=R["D_A"], D_B=R["D_B"], arms=R["arms"], J=R["J"], K=R["K"])
    R["cis"] = {a: sharp.sharp_ci(res, a) for a in R["arms"]}
    R["comparisons"] = {f"{a1}_vs_{a2}": sharp.sharp_compare(res, a1, a2)
                        for a1, a2 in itertools.combinations(R["arms"], 2)}
    R["provenance"] = _refresh(R.get("provenance", {}), "ladder cis+comparisons")
    prov.backup_artifact(path)          # content-addressed, never clobbers history
    path.write_bytes(pickle.dumps(R))
    prov.write_sidecar(path, R["provenance"])
    print(f"ladder recomputed: raw r={R['cis']['raw']['mean']:+.3f}, "
          f"fm r={R['cis']['fm']['mean']:+.3f}")


def recompute_xgb(path: Path, alt: str = "greater") -> None:
    t = pl.read_parquet(path)
    rows = []
    for r in t.iter_rows(named=True):
        da, db = np.array(r["d_a"], float), np.array(r["d_b"], float)
        st = sharp.sharp_score_test(da, db, mu0=0.0, alternative=alt)
        lo, hi = sharp._invert_score_test(da, db, alpha=0.05)
        r = dict(r)
        r.update(mean_r2_gain=st["D_bar"], ci_lo=float(lo), ci_hi=float(hi),
                 z=st["z"], p_one_sided=st["p"], rho=st["rho"], sigma2=st["sigma2"])
        rows.append(r)
    out = pl.DataFrame(rows)
    out = out.with_columns(pl.Series("q_fdr", _bh(out["p_one_sided"].to_numpy()))).sort("p_one_sided")
    prov.backup_artifact(path)          # content-addressed, never clobbers history
    out.write_parquet(path)
    side = path.with_suffix(path.suffix + ".provenance.json")
    rec = json.loads(side.read_text()) if side.exists() else {}
    prov.write_sidecar(path, _refresh(rec, "xgb p/ci from saved d_a/d_b"))
    print(f"{path.name}: survivors q<0.05 = {int((out['q_fdr'] < 0.05).sum())}/{out.height}")


def main() -> None:
    if prov.git_dirty():
        raise SystemExit("refusing to recompute: working tree dirty (commit code first)")
    recompute_ladder(EXP / "sharp_ladder_results_rhat1p1.pkl")
    recompute_xgb(EXP / "xgb_nonlinear_results.parquet")
    recompute_xgb(EXP / "xgb_subscores_sharp_results.parquet")


if __name__ == "__main__":
    main()
