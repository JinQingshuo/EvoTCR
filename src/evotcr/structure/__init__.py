"""Portable, label-free structural feature banks from packed distance tensors.

The F1, F3 and F5 formulas and column ordering are preserved from the historical
builders. No files, labels, fitted weights, or global experiment state are read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from . import _f1, _f3, _f5

StructuralBank: TypeAlias = tuple[np.ndarray, tuple[str, ...]]
EXPECTED_FEATURES = {"F1": 2464, "F3": 2856, "F5": 930}
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_packed_distances(
    qa: np.ndarray,
    qb: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate N×seed×20×9×33 uint8 tensors and N×2 valid loop lengths.

    Each byte encodes 0.125 Å; 255 means absent. Padded positions are masked by
    ``length`` even if their bytes are nonmissing. Seed counts must agree across
    chains. Validation never copies a memory-mapped distance tensor.
    """
    qa, qb = np.asarray(qa), np.asarray(qb)
    if qa.dtype != np.uint8 or qb.dtype != np.uint8:
        raise ValueError("qa and qb must be uint8 packed distances (0.125 Å; 255 missing)")
    if (
        qa.ndim != 5
        or qa.shape[0] == 0
        or qa.shape[1] == 0
        or qa.shape[2:] != (20, 9, 33)
        or qb.shape != qa.shape
    ):
        raise ValueError("qa and qb must have identical nonempty shape (N, seeds, 20, 9, 33)")
    length = np.asarray(length)
    if length.shape != (len(qa), 2) or not np.issubdtype(length.dtype, np.integer):
        raise ValueError("length must be an integer array with shape (N, 2)")
    if np.any((length < 1) | (length > 20)):
        raise ValueError("CDR3 lengths must be between 1 and 20")
    return qa, qb, length


def _sequences(values: Sequence[str] | None, name: str, expected: np.ndarray) -> list[str]:
    if values is None:
        raise ValueError(f"F5 requires {name} sequences")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain one sequence per row")
    values = list(values)
    if len(values) != len(expected):
        raise ValueError(f"{name} must contain {len(expected)} sequences")
    for index, (sequence, size) in enumerate(zip(values, expected, strict=True)):
        if not isinstance(sequence, str) or len(sequence) != size:
            raise ValueError(f"{name} row {index} must have length {size}")
        if not set(sequence) <= AMINO_ACIDS:
            raise ValueError(f"{name} row {index} contains a noncanonical amino acid")
    return values


def build_structural_features(
    qa: np.ndarray,
    qb: np.ndarray,
    length: np.ndarray,
    cdr3a: Sequence[str] | None = None,
    cdr3b: Sequence[str] | None = None,
    peptides: Sequence[str] | None = None,
    *,
    families: Sequence[str] = ("F1", "F3", "F5"),
    batch_size: int = 500,
) -> dict[str, StructuralBank]:
    """Build selected structural banks, preserving exact historical names.

    Return ``{family: (float32 values[N, features], column_names)}``. F5 needs
    aligned canonical CDR3-alpha/beta and 9-mer peptide sequences. F1/F3 do not
    need sequence identity. Default three-bank output contains 6,250 columns.

    These columns are feature dimensions, not independent prediction seeds;
    do not pass a bank directly to a seed-vote operation. The computation is
    row-local and independent of labels, cohort size, and row order.
    """
    qa, qb, length = validate_packed_distances(qa, qb, length)
    if isinstance(families, str):
        raise ValueError("families must be a sequence such as ('F1', 'F3', 'F5')")
    families = tuple(families)
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be nonempty without duplicates")
    unknown = set(families) - set(EXPECTED_FEATURES)
    if unknown:
        raise ValueError(f"Unsupported structural families: {sorted(unknown)}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if "F5" in families:
        cdr3a = _sequences(cdr3a, "cdr3a", length[:, 0])
        cdr3b = _sequences(cdr3b, "cdr3b", length[:, 1])
        peptides = _sequences(peptides, "peptides", np.full(len(qa), 9))
    builders = {"F1": _f1.build, "F3": _f3.build, "F5": _f5.build}
    result: dict[str, StructuralBank] = {}
    for family in families:
        values, names = builders[family](qa, qb, length, cdr3a, cdr3b, peptides, batch_size)
        names = tuple(names)
        if values.shape != (len(qa), EXPECTED_FEATURES[family]):
            raise RuntimeError(f"Unexpected {family} feature dimensions: {values.shape}")
        if len(set(names)) != len(names) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid {family} feature names or values")
        result[family] = (values, names)
    return result


__all__ = [
    "EXPECTED_FEATURES",
    "StructuralBank",
    "build_structural_features",
    "validate_packed_distances",
]
