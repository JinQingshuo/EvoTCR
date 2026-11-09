import json
from pathlib import Path
import numpy as np
import pytest
from typer.testing import CliRunner
from evotcr.cli import app
from evotcr.parameters import factor_parameters, frozen_head
from evotcr.metrics import evaluate_five


def test_release_parameters_are_immutable():
    root = Path(__file__).resolve().parents[1]
    factors = json.loads((root / "configs/frozen_factors.json").read_text())
    head = frozen_head()
    selected = {
        "selected_factors": factors["selected_indices"],
        "config": head["config"],
        "final_seeds": head["final_seeds"],
        "selection": {"epoch": head["epoch"], "mode": head["mode"]},
    }
    actual = factor_parameters(selected)
    assert actual["7"]["bandwidth"] == 1.0
    assert actual["14"]["dim"] == 256
    assert frozen_head()["final_seeds"] == [1109, 1110, 1111, 1112, 1113]
    selected["selected_factors"]["7"] += 1
    with pytest.raises(ValueError, match="Factor configuration"):
        factor_parameters(selected)


def test_search_and_training_commands_are_absent():
    runner = CliRunner()
    for command in ["search", "fit", "prepare-training"]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_five_seed_statistics_are_not_an_ensemble():
    y = np.array([0, 0, 1, 1])
    g = np.zeros(4, int)
    scores = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 1, 3], [1, 1, 1, 1], [0, 0, 1, 1]], float)
    report = evaluate_five(y, scores, g, [1109, 1110, 1111, 1112, 1113])
    values = [r["macro_partial_auc"] for r in report["per_seed"]]
    assert report["macro_partial_auc_mean"] == pytest.approx(np.mean(values))
    assert report["macro_partial_auc_sd"] == pytest.approx(np.std(values, ddof=1))
    assert not report["ensemble_predictions_evaluated"]
    with pytest.raises(ValueError):
        evaluate_five(y, scores, g, [1110, 1109, 1111, 1112, 1113])


def test_geometry_channels_reconstruct_distances(monkeypatch):
    from evotcr.integrations import distances as d

    left = [
        {
            "comp": "ALA",
            "atoms": {"CA": ("C", np.array([0, 0, 0]), 90), "N": ("N", np.array([1, 0, 0]), 90)},
        }
    ]
    right = [
        {
            "comp": "ALA",
            "atoms": {"O": ("O", np.array([4, 0, 0]), 90), "CB": ("C", np.array([5, 0, 0]), 90)},
        }
    ]
    monkeypatch.setattr(d.chem, "_parse_residues", lambda cif, row: (left, right, 1, [0, 0]))
    result, na, _ = d.compute(b"", {})
    assert result.shape == (1, 1, 33) and len(d.CHANNELS) == 33 and na == 1
    assert result[0, 0, 0] == 3.0
    assert result[0, 0, 22] == 3.0  # N donor → O acceptor
    assert np.isnan(result[0, 0, 26])  # no positively charged atoms in alanine


def test_compact_archive_uses_exact_saved_confidence(monkeypatch, tmp_path):
    import io
    import tarfile
    import hashlib
    import zstandard
    from evotcr.integrations import distances
    from evotcr.integrations.nettcrfold import extract

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, blob in [("model.cif", b"fixture"), ("summary_confidences.json", b"{}")]:
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    blob = zstandard.ZstdCompressor().compress(buffer.getvalue())
    (tmp_path / "one.tar.zst").write_bytes(blob)
    monkeypatch.setattr(
        distances, "compute", lambda cif, row: (np.ones((2, 9, 33), np.float32), 1, [0, 0])
    )
    values = np.arange(21, dtype=float).reshape(3, 7) / 10
    task = {
        "record": {
            "chains": {"A": "A", "C": "ACDEFGHIK", "D": "A", "E": "C"},
            "metadata": {"cdr3_chain_ranges_zero_based_halfopen": {"D": [0, 1], "E": [0, 1]}},
        },
        "refs": [
            {
                "seed": i,
                "path": "one.tar.zst",
                "sha256": hashlib.sha256(blob).hexdigest(),
                "confidence_values": v.tolist(),
            }
            for i, v in enumerate(values)
        ],
    }
    output = extract(task, tmp_path)
    np.testing.assert_array_equal(output["values"], values)
    assert output["qa"].shape == (3, 20, 9, 33)
    task["refs"][0]["confidence_values"] = [1]
    with pytest.raises(ValueError, match="seven unchanged"):
        extract(task, tmp_path)
