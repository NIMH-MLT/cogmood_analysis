"""Tests for provenance helpers -- focus on the never-clobber artifact backup."""
import json

from cogmood_analysis import provenance as prov


def test_backup_artifact_content_addressed_and_never_clobbers(tmp_path):
    art = tmp_path / "result.parquet"
    art.write_bytes(b"round-2 content")
    side = tmp_path / "result.parquet.provenance.json"
    side.write_text(json.dumps({"analysis_source_commit": "abc123"}))

    bak = prov.backup_artifact(art)
    assert bak.exists() and bak.name.startswith("result.bak-")     # content-addressed
    assert bak.read_bytes() == b"round-2 content"
    assert (tmp_path / (bak.name + ".provenance.json")).exists()   # sidecar copied

    # simulate a re-run that CHANGES the live artifact, then re-backs-up
    import os
    mtime = os.path.getmtime(bak)
    art.write_bytes(b"round-3 content")
    bak3 = prov.backup_artifact(art)
    assert bak3 != bak                                             # new content -> new backup
    assert bak.read_bytes() == b"round-2 content"                  # HISTORICAL backup intact
    assert os.path.getmtime(bak) == mtime                          # untouched

    # backing up identical content again is idempotent (no clobber, same path)
    assert prov.backup_artifact(art) == bak3


def test_backup_artifact_missing_returns_none(tmp_path):
    assert prov.backup_artifact(tmp_path / "nope.pkl") is None
