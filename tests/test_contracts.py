import json
import numpy as np
import pytest
from evotcr.artifacts import digest, manifest, verify, safe_path
from evotcr.data import DatasetSpec, register, require_training
from evotcr.factors.contracts import FitView
from evotcr.factors.registry import Factor, Registry
from evotcr.pipeline import Pipeline, Stage
from evotcr.evolution import Candidate, promote


def test_dataset_is_immutable_and_integrity_checked(tmp_path):
    (tmp_path / "rows").write_bytes(b"observations")
    spec = DatasetSpec(
        name="example",
        version="1",
        role="train",
        label_semantics="synthetic",
        groups=1,
        provenance={},
        assets=[dict(path="rows", sha256=digest(tmp_path / "rows"), bytes=12)],
    )
    spec.verify(tmp_path)
    register(spec, tmp_path / "registry")
    with pytest.raises(ValueError, match="immutable"):
        register(spec.model_copy(update={"groups": 2}), tmp_path / "registry")
    (tmp_path / "rows").write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        spec.verify(tmp_path)
    with pytest.raises(ValueError, match="training"):
        require_training(spec.model_copy(update={"role": "benchmark"}))


def test_manifest_and_path_escape(tmp_path):
    (tmp_path / "a").write_text("verified")
    receipt = manifest(tmp_path)
    assert verify(tmp_path, receipt) == 1
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")
    (tmp_path / "a").write_text("tampered")
    with pytest.raises(ValueError):
        verify(tmp_path, receipt)


def test_fit_view_does_not_expose_validation_labels():
    x = np.arange(30).reshape(10, 3)
    y = np.arange(10)
    view = FitView.subset(x, y, [0, 1, 2], [3, 4], "training-fingerprint")
    assert view.labels.tolist() == [0, 1, 2]
    y[:] = 99
    assert view.labels.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        view.features[0, 0] = 1
    with pytest.raises(ValueError):
        FitView.subset(x, y, [0, 3], [3, 4], "x")


def test_plugin_dependencies_are_lazy_and_cycle_checked():
    r = Registry()
    calls = []

    def base(c):
        calls.append("base")
        return np.array([2.0, 3.0])

    r.register(Factor("base", "1", "custom", "base", (), False, base))
    r.register(
        Factor(
            "double",
            "1",
            "custom",
            "twice",
            ("base",),
            False,
            lambda c: 2 * c["dependencies"]["base"],
        )
    )
    r.register(Factor("unused", "1", "custom", "unused", (), False, lambda c: 1 / 0))
    np.testing.assert_array_equal(r.build(["double", "base"], {}), [[4, 2], [6, 3]])
    assert calls == ["base"]
    with pytest.raises(ValueError):
        r.register(r._factors["base"])
    r.register(Factor("cycle", "1", "custom", "cycle", ("cycle",), False, base))
    with pytest.raises(ValueError, match="cycle"):
        r.build(["cycle"], {})


def test_pipeline_resumes_only_verified_results(tmp_path):
    calls = []

    def first(out, deps):
        calls.append(1)
        (out / "a").write_text("one")

    def second(out, deps):
        calls.append(2)
        (out / "b").write_text((deps["a"] / "a").read_text())

    p = Pipeline([Stage("a", "1", (), first), Stage("b", "1", ("a",), second)], tmp_path)
    result = p.execute("b", {"dataset": "hash"}, {})
    assert p.execute("b", {"dataset": "hash"}, {}) == result and calls == [1, 2]
    (result / "b").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        p.execute("b", {"dataset": "hash"}, {})


def test_failed_stage_does_not_publish_partial_outputs(tmp_path):
    calls = []

    def run(out, deps):
        (out / "partial").write_text("pending")
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("interruption")

    p = Pipeline([Stage("s", "1", (), run)], tmp_path)
    with pytest.raises(RuntimeError):
        p.execute("s", {}, {})
    assert not list(tmp_path.rglob("MANIFEST.json"))
    result = p.execute("s", {}, {})
    assert verify(result, json.loads((result / "MANIFEST.json").read_text())) == 1


def test_promotion_requires_same_protocol_and_improvement(tmp_path):
    c = Candidate(
        model_sha256="a" * 64,
        validation_protocol_sha256="b" * 64,
        validation_score=0.7,
        invariance_passed=True,
        leakage_audit_passed=True,
    )
    assert promote(c, tmp_path)
    assert not promote(c.model_copy(update={"validation_score": 0.6}), tmp_path)
    with pytest.raises(ValueError):
        promote(c.model_copy(update={"selection_split": "benchmark"}), tmp_path)
    with pytest.raises(ValueError):
        promote(c.model_copy(update={"validation_protocol_sha256": "c" * 64}), tmp_path)
