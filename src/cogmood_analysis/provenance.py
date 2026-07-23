"""Provenance stamps for regenerated analysis artifacts.

Every regenerated artifact (pkl/parquet) should embed a provenance record so the
git-ignored inputs and outputs are pinned to a source commit and a specific
subject set. Use :func:`provenance` and store the returned dict under a
``"provenance"`` key (pkl) or as a sidecar ``<artifact>.provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_commit() -> str | None:
    """Current HEAD SHA, or None outside a git tree."""
    return _run_git(["rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted modifications to **tracked
    analysis-code** files (None if unknown). Submodules are ignored (the repo's
    ``packages/Supreme`` pointer is unrelated) and untracked files are excluded
    (``--untracked-files=no``) so gitignored data/logs and transient NFS lock files
    (``.nfs*``) do not spuriously flag the tree dirty."""
    status = _run_git(["status", "--porcelain", "--ignore-submodules=all",
                       "--untracked-files=no"])
    return None if status is None else bool(status)


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_subjects(sub_ids: Sequence[str]) -> str:
    """SHA-256 of the sorted subject set (order-independent identity of the cohort)."""
    joined = "\n".join(sorted(str(s) for s in sub_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


#: Provenance record schema version (bump on breaking changes to the fields).
#: v3 splits the single source_commit into analysis_source_commit + provenance_stamp_commit.
SCHEMA_VERSION = "3"


def _lockfile_sha256() -> str | None:
    """SHA-256 of the repo's ``uv.lock`` (environment pin), if present."""
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    return sha256_file(lock) if lock.exists() else None


def provenance(
    training_csv: str | Path,
    sub_ids: Sequence[str],
    config: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a provenance record for an artifact.

    Records two commits: ``analysis_source_commit`` (the code that actually produced
    the artifact) and ``provenance_stamp_commit`` (the commit at which this record was
    written/refreshed). At generation time they are identical; a later
    :mod:`scripts.restamp_provenance` pass refreshes only the stamp commit and
    preserves the analysis commit. Also captures the ``dirty`` flag, the
    ``training_data.csv`` + subject-set + ``uv.lock`` SHA-256s, the run ``config``, a
    UTC timestamp, and the schema version. The artifact's own SHA-256 is added by
    :func:`write_sidecar`.
    """
    commit = git_commit()
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_source_commit": commit,
        "provenance_stamp_commit": commit,
        "dirty": git_dirty(),
        "training_csv_sha256": sha256_file(training_csv),
        "subject_set_sha256": sha256_subjects(sub_ids),
        "n_subjects": len(sub_ids),
        "config": config,
        "env_lockfile_sha256": _lockfile_sha256(),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def backup_artifact(path: str | Path) -> Path | None:
    """Content-addressed, **never-clobbering** backup of an artifact and its sidecar.

    Copies ``<name>.<ext>`` to ``<name>.bak-<sha12>.<ext>`` (and its
    ``.provenance.json`` sidecar) only if that exact backup does not already exist.
    Because the backup name embeds the artifact's SHA-256, distinct contents get
    distinct backups and an existing historical backup is never overwritten — safe to
    call on every rerun of a regeneration/recompute script. Returns the backup path
    (or ``None`` if the artifact is absent).
    """
    p = Path(path)
    if not p.exists():
        return None
    bak = p.with_suffix(f".bak-{sha256_file(p)[:12]}{p.suffix}")
    if not bak.exists():                       # immutable: do not clobber history
        shutil.copy2(p, bak)
        side = p.with_suffix(p.suffix + ".provenance.json")
        if side.exists():
            shutil.copy2(side, bak.with_suffix(bak.suffix + ".provenance.json"))
    return bak


def write_sidecar(artifact_path: str | Path, prov: dict[str, Any]) -> Path:
    """Write ``<artifact>.provenance.json`` (incl. the artifact's own SHA-256)."""
    p = Path(artifact_path)
    rec = dict(prov)
    rec["artifact_sha256"] = sha256_file(p) if p.exists() else None
    side = p.with_suffix(p.suffix + ".provenance.json")
    side.write_text(json.dumps(rec, indent=2, default=str))
    return side
