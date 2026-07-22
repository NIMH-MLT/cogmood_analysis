"""Provenance stamps for regenerated analysis artifacts.

Every regenerated artifact (pkl/parquet) should embed a provenance record so the
git-ignored inputs and outputs are pinned to a source commit and a specific
subject set. Use :func:`provenance` and store the returned dict under a
``"provenance"`` key (pkl) or as a sidecar ``<artifact>.provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
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
    """True if the working tree has uncommitted changes (None if unknown)."""
    status = _run_git(["status", "--porcelain"])
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


def provenance(
    training_csv: str | Path,
    sub_ids: Sequence[str],
    config: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a provenance record for an artifact.

    Captures the source commit + dirty flag, the ``training_data.csv`` SHA-256,
    the subject-set SHA-256 (hash of the sorted ``sub_ids`` actually analyzed), the
    run ``config``, and a UTC timestamp.
    """
    return {
        "source_commit": git_commit(),
        "dirty": git_dirty(),
        "training_csv_sha256": sha256_file(training_csv),
        "subject_set_sha256": sha256_subjects(sub_ids),
        "n_subjects": len(sub_ids),
        "config": config,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def write_sidecar(artifact_path: str | Path, prov: dict[str, Any]) -> Path:
    """Write ``<artifact>.provenance.json`` next to an artifact; return its path."""
    p = Path(artifact_path)
    side = p.with_suffix(p.suffix + ".provenance.json")
    side.write_text(json.dumps(prov, indent=2, default=str))
    return side
