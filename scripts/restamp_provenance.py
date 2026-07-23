"""Restamp artifact provenance sidecars from the current clean committed state.

The round-2 analyses were regenerated while branches were being rebased/committed, so
some sidecars recorded ``dirty=true`` or an off-by-one commit. This refreshes every
``data/exploratory/*.provenance.json`` (skipping preserved ``*.r1.*`` round-1 copies)
with the current source commit, ``dirty`` flag (submodule-ignored), schema version,
lockfile hash, and the artifact's own SHA-256 -- preserving the run ``config``,
subject-set hash, and input hash. Run this only with a CLEAN committed tree; the
artifacts are byte-unchanged and reproducible from the recorded commit.
"""
import json
from pathlib import Path

from cogmood_analysis import provenance as prov

EXP = Path(__file__).resolve().parents[1] / "data" / "exploratory"


def main() -> None:
    commit, dirty = prov.git_commit(), prov.git_dirty()
    if dirty:
        raise SystemExit("refusing to restamp: working tree is dirty (commit code first)")
    lock = prov._lockfile_sha256()
    for side in sorted(EXP.glob("*.provenance.json")):
        artifact = EXP / side.name[: -len(".provenance.json")]
        if ".r1." in artifact.name or ".r2." in artifact.name:
            continue                            # preserve round-1/round-2 backups verbatim
        rec = json.loads(side.read_text())
        rec["schema_version"] = prov.SCHEMA_VERSION
        # preserve the analysis (run-time) commit; refresh only the stamp commit
        rec["analysis_source_commit"] = (
            rec.get("analysis_source_commit") or rec.pop("source_commit", None))
        rec["provenance_stamp_commit"] = commit
        rec.pop("source_commit", None)
        rec["dirty"] = dirty
        rec["env_lockfile_sha256"] = lock
        rec["artifact_sha256"] = prov.sha256_file(artifact) if artifact.exists() else None
        rec["restamped"] = True
        side.write_text(json.dumps(rec, indent=2, default=str))
        print(f"{artifact.name:44s} analysis={str(rec['analysis_source_commit'])[:9]} "
              f"stamp={str(commit)[:9]} dirty={dirty} sha={str(rec['artifact_sha256'])[:12]}")


if __name__ == "__main__":
    main()
