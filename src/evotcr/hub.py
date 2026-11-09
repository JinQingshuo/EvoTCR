"""Download an immutable Hub revision and verify every declared artifact."""

import json
import re
from pathlib import Path
from .artifacts import verify


def download(repo_id: str, revision: str, destination: Path, repo_type: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use a pinned 40-character commit, not a moving branch")
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(repo_id, repo_type=repo_type, revision=revision, local_dir=destination)
    )
    verify(root, json.loads((root / "MANIFEST.json").read_text()))
    return root
