"""Exact, immutable numerical recipe of the published five-seed release."""

import json
from importlib.resources import files


def frozen_head() -> dict:
    return json.loads(files("evotcr").joinpath("resources/frozen_mlp.json").read_text())


def factor_parameters(selected: dict) -> dict:
    resource = json.loads(files("evotcr").joinpath("resources/frozen_factors.json").read_text())
    head = frozen_head()
    if selected["selected_factors"] != resource["selected_indices"]:
        raise ValueError("Factor configuration differs from the frozen Refill212 release")
    if selected["config"] != head["config"] or selected["final_seeds"] != head["final_seeds"]:
        raise ValueError("Model configuration differs from the frozen five-seed release")
    if (
        selected["selection"]["epoch"] != head["epoch"]
        or selected["selection"]["mode"] != head["mode"]
    ):
        raise ValueError("Normalization or training epoch differs from the frozen release")
    if selected.get("active_channels", list(range(17))) != list(range(17)):
        raise ValueError("All seventeen published factors are required")
    return resource["parameters"]
