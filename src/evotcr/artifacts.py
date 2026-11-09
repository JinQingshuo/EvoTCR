"""Atomic, hash-verified artifacts shared by datasets, pipelines and models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes root: {relative}")
    return path


def manifest(root: Path) -> dict:
    files = [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    ]
    return {"schema_version": 1, "files": files, "total_bytes": sum(x["bytes"] for x in files)}


def verify(root: Path, receipt: dict) -> int:
    seen = set()
    for row in receipt["files"]:
        if row["path"] in seen:
            raise ValueError("Duplicate manifest path")
        seen.add(row["path"])
        p = safe_path(root, row["path"])
        if not p.is_file() or p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError(f"Artifact integrity failure: {row['path']}")
    return len(seen)
