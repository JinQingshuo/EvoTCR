"""Validated, frozen complete-panel inference; no fitting at serving time."""

from pathlib import Path
import json
import numpy as np
from .artifacts import verify
from .factors.builtin import NAMES, registry


class Predictor:
    def __init__(self, model: Path, seed: int = 1109, verify_files: bool = True):
        self.root, self.seed = Path(model), seed
        if verify_files:
            verify(self.root, json.loads((self.root / "MANIFEST.json").read_text()))
        selected = json.loads((self.root / "SELECTED_MODEL.json").read_text())
        from .parameters import factor_parameters

        factor_parameters(selected)
        if seed not in selected["final_seeds"]:
            raise ValueError("Seed is absent from the frozen model release")
        self.active = selected.get("active_channels", list(range(17)))
        if (
            not self.active
            or len(set(self.active)) != len(self.active)
            or not set(self.active) <= set(range(17))
        ):
            raise ValueError("Invalid frozen factor combination")
        self.registry = registry()

    def predict(self, data: dict) -> dict:
        from .models.frozen import score
        from .factors.runtime import FrozenFactors

        required = {"F", "base", "D", "g", "h"}
        if set(data) != required:
            raise ValueError(f"Query input must contain exactly {sorted(required)}; no labels")
        n = len(data["h"])
        expected = {
            "F": (n * 10, 6250),
            "base": (n * 10, 7),
            "D": (n, 1000),
            "g": (n * 10,),
            "h": (n,),
        }
        if not n:
            raise ValueError("Empty query")
        for key, shape in expected.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid {key}; expected finite array of shape {shape}")
        if not np.issubdtype(data["g"].dtype, np.integer) or not np.issubdtype(
            data["h"].dtype, np.integer
        ):
            raise ValueError("Target and HLA indices must be integers")
        if not np.isin(data["h"], [0, 1]).all() or not np.isin(data["g"], np.arange(20)).all():
            raise ValueError("This release supports the registered 20-target / 2-HLA panel")
        if np.any(data["D"] < 0) or not np.all(np.diff(data["g"].reshape(n, 10), axis=1) > 0):
            raise ValueError("Invalid distances or incomplete/unsorted panel")
        source = np.load(self.root / "train.npz", allow_pickle=False)
        if not np.all(source["HL"][data["g"].reshape(n, 10)] == data["h"][:, None]):
            raise ValueError("Panel HLA mismatch")
        raw = np.zeros((n * 10, 17), np.float32)
        raw[:, self.active] = self.registry.build(
            [NAMES[i] for i in self.active], {"runtime": FrozenFactors(self.root, data)}
        )
        return {"score": score(self.root, raw, self.seed), "raw17": raw, "g": data["g"]}

    def predict_all(self, data: dict) -> dict:
        from .models.frozen import score
        from .parameters import frozen_head

        result = self.predict(data)
        seeds = frozen_head()["final_seeds"]
        scores = [
            result["score"] if s == self.seed else score(self.root, result["raw17"], s)
            for s in seeds
        ]
        return {
            "scores": np.stack(scores),
            "seeds": np.array(seeds),
            "raw17": result["raw17"],
            "g": result["g"],
        }
