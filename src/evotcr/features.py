"""Resumable structure → descriptor → fixed-panel query pipeline."""

import json
from pathlib import Path
import numpy as np
from .artifacts import atomic_json, digest, fingerprint, verify
from .pipeline import Pipeline, Stage
from .integrations.nettcrfold import extract, reduce_confidence
from .sequence import paired_distances
from .structure import build_structural_features


def build_query(tasks_path: Path, archives: Path, model: Path, run_root: Path) -> Path:
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if not tasks or len(tasks) % 10:
        raise ValueError("Provide complete ten-peptide query groups")
    verify(model, json.loads((model / "MANIFEST.json").read_text()))
    panel = json.loads((model / "panel.json").read_text())
    target = {(p["hla"], p["peptide"]): p["target_id"] for p in panel}
    source = dict(np.load(model / "train.npz", allow_pickle=False))
    g = np.array(
        [target[(t["record"]["metadata"]["hla"], t["record"]["chains"]["C"])] for t in tasks]
    )
    order = np.concatenate(
        [start + np.argsort(g[start : start + 10]) for start in range(0, len(tasks), 10)]
    )
    tasks = [tasks[i] for i in order]
    g = g[order]
    h = source["HL"][g.reshape(-1, 10)[:, 0]]
    if not np.all(np.diff(g.reshape(-1, 10), axis=1) > 0) or not np.all(
        source["HL"][g.reshape(-1, 10)] == h[:, None]
    ):
        raise ValueError("Each query group must contain the complete same-HLA panel")
    for start in range(0, len(tasks), 10):
        records = [t["record"] for t in tasks[start : start + 10]]
        for chain in ["A", "D", "E"]:
            if len({r["chains"][chain] for r in records}) != 1:
                raise ValueError("TCR and MHC chains must agree within a query group")
        if (
            len(
                {
                    json.dumps(
                        r["metadata"]["cdr3_chain_ranges_zero_based_halfopen"], sort_keys=True
                    )
                    for r in records
                }
            )
            != 1
        ):
            raise ValueError("CDR3 ranges must agree within a query group")

    def extract_stage(out, deps):
        for i, task in enumerate(tasks):
            np.savez_compressed(out / f"{i:06d}.npz", **extract(task, archives))

    def descriptor_stage(out, deps):
        F = np.lib.format.open_memmap(
            out / "F.npy", mode="w+", dtype=np.float32, shape=(len(tasks), 6250)
        )
        scalars = []
        a = []
        b = []
        for start in range(0, len(tasks), 100):
            block = [
                dict(np.load(deps["extract"] / f"{i:06d}.npz", allow_pickle=False))
                for i in range(start, min(start + 100, len(tasks)))
            ]
            banks = build_structural_features(
                np.stack([r["qa"] for r in block]),
                np.stack([r["qb"] for r in block]),
                np.stack([r["length"] for r in block]),
                [str(r["cdr3a"]) for r in block],
                [str(r["cdr3b"]) for r in block],
                [str(r["peptide"]) for r in block],
            )
            F[start : start + len(block)] = np.concatenate(
                [banks[k][0] for k in ["F1", "F3", "F5"]], axis=1
            )
            scalars.extend(r["values"] for r in block)
            a.extend(str(r["cdr3a"]) for r in block)
            b.extend(str(r["cdr3b"]) for r in block)
        F.flush()
        base = reduce_confidence(np.stack(scalars))
        D = paired_distances(a[::10], b[::10], source["cdr3a"], source["cdr3b"])
        np.savez_compressed(out / "query.npz", F=F, base=base, D=D, g=g, h=h)
        atomic_json(
            out / "row_order.json", {"query_row_to_input_task_row": order.tolist(), "index_base": 0}
        )

    code_root = Path(__file__).parent
    code = fingerprint(
        {str(p.relative_to(code_root)): digest(p) for p in sorted(code_root.rglob("*.py"))}
    )
    pipeline = Pipeline(
        [
            Stage("extract", "1", (), extract_stage),
            Stage("query", "1", ("extract",), descriptor_stage),
        ],
        run_root,
    )
    out = pipeline.execute(
        "query",
        {"tasks_sha256": digest(tasks_path), "model_manifest": digest(model / "MANIFEST.json")},
        {"code_sha256": code, "descriptor_schema": "F1-F3-F5-6250"},
    )
    return out / "query.npz"
