"""Content-addressed DAG stages with locks, atomic receipts and output verification."""

from __future__ import annotations

import json
import time
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from filelock import FileLock

from .artifacts import atomic_json, fingerprint, manifest, verify


@dataclass(frozen=True)
class Stage:
    name: str
    version: str
    dependencies: tuple[str, ...]
    run: Callable[[Path, dict[str, Path]], None]


class Pipeline:
    def __init__(self, stages: list[Stage], root: Path):
        self.stages = {s.name: s for s in stages}
        if len(self.stages) != len(stages):
            raise ValueError("Duplicate pipeline stage")
        self.root = root

    def execute(self, target: str, inputs: dict, config: dict) -> Path:
        complete, visiting = {}, set()

        def visit(name):
            if name in visiting:
                raise ValueError("Pipeline dependency cycle")
            if name in complete:
                return complete[name]
            visiting.add(name)
            stage = self.stages[name]
            deps = {k: visit(k) for k in stage.dependencies}
            key = fingerprint(
                {
                    "stage": name,
                    "version": stage.version,
                    "inputs": inputs,
                    "config": config,
                    "dependencies": {k: str(v.name) for k, v in deps.items()},
                }
            )
            out = self.root / name / key
            out.mkdir(parents=True, exist_ok=True)
            receipt = out / "MANIFEST.json"
            with FileLock(str(out) + ".lock"):
                if receipt.exists():
                    verify(out, json.loads(receipt.read_text()))
                else:
                    start = time.time()
                    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=out.parent))
                    try:
                        stage.run(temporary, deps)
                        audit = manifest(temporary)
                        audit.update(
                            stage=name,
                            stage_version=stage.version,
                            cache_key=key,
                            seconds=time.time() - start,
                            inputs=inputs,
                            config=config,
                        )
                        atomic_json(temporary / "MANIFEST.json", audit)
                        out.rmdir()  # Only an empty, unpublished destination is replaceable.
                        temporary.replace(out)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
            visiting.remove(name)
            complete[name] = out
            return out

        return visit(target)
