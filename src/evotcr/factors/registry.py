"""Versioned plugins: explicit dependencies and fit scopes, no import-time execution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Factor:
    name: str
    version: str
    family: str
    definition: str
    dependencies: tuple[str, ...]
    fitted: bool
    compute: Callable[[dict], np.ndarray]


class Registry:
    def __init__(self):
        self._factors: dict[str, Factor] = {}

    def register(self, factor: Factor) -> None:
        if factor.name in self._factors:
            raise ValueError(f"Duplicate factor: {factor.name}")
        self._factors[factor.name] = factor

    def load_plugins(self) -> None:
        for ep in entry_points(group="evotcr.factors"):
            ep.load()(self)

    def catalog(self) -> list[dict]:
        return [
            {k: v for k, v in vars(f).items() if k != "compute"} for f in self._factors.values()
        ]

    def build(self, names: list[str], context: dict) -> np.ndarray:
        if not names or len(names) != len(set(names)):
            raise ValueError("Choose unique nonempty factor names")
        values, visiting = {}, set()

        def visit(name):
            if name in values:
                return values[name]
            if name in visiting:
                raise ValueError("Factor dependency cycle")
            visiting.add(name)
            factor = self._factors[name]
            dep = {k: visit(k) for k in factor.dependencies}
            value = np.asarray(factor.compute(dict(context, dependencies=dep)))
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"Invalid output from {name}")
            values[name] = value
            visiting.remove(name)
            return value

        return np.column_stack([visit(name) for name in names]).astype(np.float32)
