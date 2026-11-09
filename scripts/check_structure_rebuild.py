"""Rebuild one registered complete group from its three-seed structure archives."""

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from evotcr.artifacts import atomic_json
from evotcr.features import build_query
from evotcr.inference import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/refill212"))
    p.add_argument("--model", type=Path, default=Path("weights/refill212"))
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/structure-check"))
    args = p.parse_args()
    if not 0 <= args.group < 1000:
        p.error("group must be 0..999")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.dataset / "metadata/tasks.jsonl").open() as stream:
        lines = list(itertools.islice(stream, args.group * 10, (args.group + 1) * 10))
    tasks = args.output / "tasks.jsonl"
    tasks.write_text("".join(lines))
    query = build_query(tasks, args.dataset, args.model, args.output / "pipeline")
    d = np.load(args.dataset / "training/train.npz", allow_pickle=False)
    rows = np.arange(args.group * 10, (args.group + 1) * 10)
    order = np.argsort(d["g"][rows])
    rows = rows[order]
    expected = dict(
        F=np.load(args.dataset / "training/F.npy", mmap_mode="r")[rows],
        base=d["base"][rows],
        D=d["D"][[args.group]],
        g=d["g"][rows],
        h=d["h"][[args.group]],
    )
    rebuilt = dict(np.load(query, allow_pickle=False))
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        model = Predictor(args.model)
        a = model.predict_all(expected)
        b = model.predict_all(rebuilt)
    descriptor_error = float(abs(expected["F"] - rebuilt["F"]).max())
    score_error = float(abs(a["scores"] - b["scores"]).max())
    passed = (
        np.array_equal(expected["D"], rebuilt["D"])
        and np.array_equal(expected["base"], rebuilt["base"])
        and score_error < 1e-5
    )
    report = {
        "group_index": args.group,
        "pairs": 10,
        "structures": 30,
        "passed": passed,
        "descriptor_max_abs_error": descriptor_error,
        "score_max_abs_error": score_error,
        "score_tolerance": 1e-5,
        "fitting_performed": False,
        "note": "Historical float32 spread cancellation can affect descriptor bitwise equality; inspect reported error. This is one group, not all structures.",
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Structure reconstruction disagrees with the frozen reference")


if __name__ == "__main__":
    main()
