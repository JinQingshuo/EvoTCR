"""Explicit dataset roles and group-complete observation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import atomic_json, digest, fingerprint, safe_path


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0)


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    role: Literal["train", "validation", "benchmark", "query"]
    label_semantics: str
    group_size: int = Field(default=10, gt=1)
    groups: int = Field(gt=0)
    provenance: dict
    assets: list[Asset]

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump())

    def verify(self, root: Path) -> None:
        if len({a.path for a in self.assets}) != len(self.assets):
            raise ValueError("Duplicate dataset assets")
        for asset in self.assets:
            p = safe_path(root, asset.path)
            if p.stat().st_size != asset.bytes or digest(p) != asset.sha256:
                raise ValueError(f"Dataset integrity failure: {asset.path}")


def register(spec: DatasetSpec, registry: Path) -> Path:
    target = registry / spec.name / f"{spec.version}.json"
    value = spec.model_dump()
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError("Dataset versions are immutable; choose a new version")
    atomic_json(target, value)
    return target


def validate_training(d: dict, features: np.ndarray) -> None:
    n = len(d["h"])
    if n != 1000 or features.shape != (10 * n, 6250):
        raise ValueError("Published recipe requires 1000 complete groups and 6250 descriptors")
    if d["base"].shape != (10 * n, 7) or d["D"].shape != (n, n):
        raise ValueError("Invalid confidence or distance dimensions")
    if not np.array_equal(np.unique(d["fold"]), np.arange(5)):
        raise ValueError("Five predefined group folds are required")
    if not np.isin(d["y"], [0, 1]).all() or not np.all(d["y"].reshape(n, 10).sum(1) == 1):
        raise ValueError("Each group must have one designed match and nine assumed negatives")
    if any(len(set(row)) != 10 for row in d["g"].reshape(n, 10)):
        raise ValueError("Training targets must be distinct within each group")
    if not np.all(d["HL"][d["g"].reshape(n, 10)] == d["h"][:, None]):
        raise ValueError("Cross-HLA rows in a ten-peptide group")
    for c in np.unique(d["cluster"]):
        if len(np.unique(d["fold"][d["cluster"] == c])) != 1:
            raise ValueError("Parent / identical-sequence component crosses folds")
    for k in ["base", "D"]:
        if not np.isfinite(d[k]).all():
            raise ValueError(f"Nonfinite {k}")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite descriptors")


def require_training(spec: DatasetSpec) -> None:
    if spec.role != "train":
        raise ValueError("Fitting and HPO accept registered training datasets only")
