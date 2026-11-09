"""Append-only promotion based on a fixed training validation protocol."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from filelock import FileLock
from .artifacts import atomic_json, fingerprint


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    invariance_passed: bool
    leakage_audit_passed: bool
    selection_split: str = "training_cv"


def promote(candidate: Candidate, registry: Path, minimum_gain: float = 0.0) -> bool:
    if minimum_gain < 0:
        raise ValueError("minimum_gain must be nonnegative")
    if (
        candidate.selection_split != "training_cv"
        or not candidate.invariance_passed
        or not candidate.leakage_audit_passed
    ):
        raise ValueError("Promotion requires training-only selection and passed audits")
    registry.mkdir(parents=True, exist_ok=True)
    with FileLock(str(registry / "promotion.lock")):
        current = registry / "current.json"
        if current.exists():
            old = Candidate.model_validate_json(current.read_text())
            if old.validation_protocol_sha256 != candidate.validation_protocol_sha256:
                raise ValueError("Changed validation protocol requires a separate model lineage")
            if candidate.validation_score <= old.validation_score + minimum_gain:
                return False
        value = candidate.model_dump()
        atomic_json(registry / "history" / f"{fingerprint(value)}.json", value)
        atomic_json(current, value)
    return True
