"""Small command surface; heavy numerical and optional dependencies load lazily."""

import json
from pathlib import Path
import typer
from .artifacts import atomic_json, manifest, verify

app = typer.Typer(
    no_args_is_help=True, help="EvoTCR: registered factors and frozen TCR–pMHC inference"
)


@app.command("build-query")
def build_query(tasks: Path, archives: Path, model: Path, run_root: Path = Path("runs/query")):
    from .features import build_query as build

    typer.echo(build(tasks, archives, model, run_root))


data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
factor_app = typer.Typer(no_args_is_help=True)
app.add_typer(factor_app, name="factors")
net_app = typer.Typer(no_args_is_help=True)
app.add_typer(net_app, name="nettcrfold")


@data_app.command("register")
def register_data(spec: Path, registry: Path = Path("registries/data")):
    from .data import DatasetSpec, register

    typer.echo(register(DatasetSpec.model_validate_json(spec.read_text()), registry))


@data_app.command("validate")
def validate_data(spec: Path, root: Path):
    from .data import DatasetSpec

    value = DatasetSpec.model_validate_json(spec.read_text())
    value.verify(root)
    typer.echo(value.fingerprint)


@app.command("download")
def download(repo: str, revision: str, destination: Path, kind: str = "dataset"):
    from .hub import download as fetch

    if kind not in {"dataset", "model"}:
        raise typer.BadParameter("kind must be dataset or model")
    typer.echo(fetch(repo, revision, destination, kind))


@factor_app.command("list")
def list_factors(plugins: bool = False):
    from .factors.builtin import registry

    r = registry()
    if plugins:
        r.load_plugins()
    typer.echo(json.dumps(r.catalog(), ensure_ascii=False, indent=2))


@factor_app.command("diagnose")
def diagnose(features: Path, output: Path):
    import numpy as np
    from .factors.builtin import diagnostics

    atomic_json(output, diagnostics(np.load(features)["raw17"]))


@factor_app.command("build")
def build_features(packed: Path, output: Path):
    import numpy as np
    from .structure import build_structural_features

    d = np.load(packed, allow_pickle=False)
    banks = build_structural_features(
        **{k: d[k] for k in ["qa", "qb", "length", "cdr3a", "cdr3b", "peptides"]}
    )
    F = np.concatenate([banks[k][0] for k in ["F1", "F3", "F5"]], axis=1)
    names = sum([list(banks[k][1]) for k in ["F1", "F3", "F5"]], [])
    np.savez_compressed(output, F=F, names=np.array(names))


@app.command("predict")
def predict(model: Path, query: Path, output: Path, seed: int = 1109):
    import numpy as np
    from .inference import Predictor

    result = Predictor(model, seed).predict(dict(np.load(query, allow_pickle=False)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    typer.echo(f"{len(result['score'])} pair scores written; complete-group ranking only")


@app.command("evaluate")
def evaluate(predictions: Path, labels: Path, output: Path):
    import numpy as np
    from .metrics import evaluate as metric

    p = np.load(predictions, allow_pickle=False)
    d = np.load(labels, allow_pickle=False)
    if not np.array_equal(p["g"], d["g"]):
        raise ValueError("Target order mismatch")
    if "scores" in p:
        from .metrics import evaluate_five

        result = evaluate_five(d["y"], p["scores"], d["g"], p["seeds"])
    else:
        result = metric(d["y"], p["score"], d["g"])
    atomic_json(output, result)


@app.command("verify")
def verify_artifacts(root: Path):
    count = verify(root, json.loads((root / "MANIFEST.json").read_text()))
    typer.echo(f"Verified {count} files")


@net_app.command("install")
def install_nettcrfold(destination: Path = Path("third_party/NetTCRfold-1.0")):
    from .integrations.nettcrfold import install

    install(destination)


@net_app.command("extract")
def extract_nettcrfold(task: Path, root: Path, output: Path):
    import numpy as np
    from .integrations.nettcrfold import extract

    result = extract(json.loads(task.read_text()), root)
    np.savez_compressed(output, **result)


@app.command("seal")
def seal(root: Path):
    atomic_json(root / "MANIFEST.json", manifest(root))


@app.command("predict-all")
def predict_all(model: Path, query: Path, output: Path):
    import numpy as np
    from .inference import Predictor

    result = Predictor(model).predict_all(dict(np.load(query, allow_pickle=False)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    typer.echo("Five predefined seeds saved separately; no ensemble or fitting")
