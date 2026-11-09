"""Scientific invariants of the portable packed-distance feature banks."""

import json
from importlib.resources import files

import numpy as np
import pytest

from evotcr.structure import build_structural_features, validate_packed_distances


def fixture_arrays(n=2, code=36):
    # 36 * 0.125 Å = 4.5 Å, at which the c45 transform is exactly 0.5.
    shape = (n, 3, 20, 9, 33)
    return np.full(shape, code, dtype=np.uint8), np.full(shape, code, dtype=np.uint8)


def one(values, names, name):
    return values[:, names.index(name)]


def test_analytic_contacts_coordinates_and_sequence_identity():
    qa, qb = fixture_arrays()
    banks = build_structural_features(
        qa,
        qb,
        np.array([[9, 9], [12, 12]]),
        ["A" * 9, "C" * 12],
        ["A" * 9, "C" * 12],
        ["A" * 9] * 2,
    )
    f1, n1 = banks["F1"]
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|c45|all|p0"), [4.5, 6])
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|c45|apex|p0"), [1.5, 3])
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|dmin|all|MIN"), [4.5, 4.5])
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|seedsd|all"), [0, 0])
    f3, n3 = banks["F3"]
    np.testing.assert_array_equal(one(f3, n3, "a|heavy|c45|g0|p0"), [1.5, 2])
    np.testing.assert_allclose(one(f3, n3, "a|heavy|c45|shape|cent"), [4, 4])
    f5, n5 = banks["F5"]
    np.testing.assert_array_equal(one(f5, n5, "a|aaA|c45|pos|p0"), [4.5, 0])
    np.testing.assert_array_equal(one(f5, n5, "a|aaC|c45|pos|p0"), [0, 6])
    np.testing.assert_array_equal(one(f5, n5, "a|aaA|comp|all|w"), [40.5, 0])
    np.testing.assert_array_equal(one(f5, n5, "b|aaC|comp|all|w"), [0, 54])


def test_names_match_audited_historical_names_and_counts():
    qa, qb = fixture_arrays(n=1)
    banks = build_structural_features(qa, qb, np.array([[9, 9]]), ["A" * 9], ["A" * 9], ["A" * 9])
    catalog = json.loads(files("evotcr").joinpath("resources/raw_factor_catalog.json").read_text())
    expected = {b["name"]: b for b in catalog["structural_banks"]}
    assert sum(values.shape[1] for values, _ in banks.values()) == 6250
    for family, (values, names) in banks.items():
        assert list(names) == expected[family]["coordinate_names"]
        assert values.shape[1] == expected[family]["count"]
        assert values.dtype == np.float32


def test_missing_distances_and_padding_do_not_become_contacts():
    qa, qb = fixture_arrays(n=1, code=255)
    # Bytes beyond the declared loop are ignored even if they encode a zero distance.
    qa[:, :, 9:] = 0
    qb[:, :, 9:] = 0
    banks = build_structural_features(qa, qb, np.array([[9, 9]]), ["A" * 9], ["A" * 9], ["A" * 9])
    f1, n1 = banks["F1"]
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|c45|all|SUM"), [0])
    np.testing.assert_array_equal(one(f1, n1, "a|heavy|dmin|all|MIN"), [32])
    assert not banks["F3"][0].any()
    assert not banks["F5"][0].any()


def test_row_and_batch_independence_without_input_mutation():
    rng = np.random.default_rng(17)
    qa = rng.integers(0, 256, (3, 3, 20, 9, 33), dtype=np.uint8)
    qb = rng.integers(0, 256, qa.shape, dtype=np.uint8)
    before_a, before_b = qa.copy(), qb.copy()
    length = np.array([[9, 12], [12, 10], [13, 9]])
    a = ["A" * n for n in length[:, 0]]
    b = ["C" * n for n in length[:, 1]]
    p = ["D" * 9] * 3
    first = build_structural_features(qa, qb, length, a, b, p, batch_size=1)
    perm = [2, 0, 1]
    second = build_structural_features(
        qa[perm],
        qb[perm],
        length[perm],
        [a[i] for i in perm],
        [b[i] for i in perm],
        p,
        batch_size=2,
    )
    for family in first:
        np.testing.assert_allclose(first[family][0][perm], second[family][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(qa, before_a)
    np.testing.assert_array_equal(qb, before_b)


def test_nonsequence_banks_do_not_require_sequences():
    qa, qb = fixture_arrays(n=1)
    result = build_structural_features(qa, qb, np.array([[8, 8]]), families=("F1", "F3"))
    assert set(result) == {"F1", "F3"}


@pytest.mark.parametrize(
    "kind", ["dtype", "shape", "empty", "length_shape", "length_type", "length_range"]
)
def test_invalid_packed_inputs_fail_before_computation(kind):
    qa, qb = fixture_arrays(n=1)
    length = np.array([[9, 9]])
    if kind == "dtype":
        qa = qa.astype(float)
    elif kind == "shape":
        qb = qb[:, :, :, :8]
    elif kind == "empty":
        qa, qb = qa[:0], qb[:0]
        length = length[:0]
    elif kind == "length_shape":
        length = length[:, :1]
    elif kind == "length_type":
        length = length.astype(float)
    else:
        length[0, 0] = 21
    with pytest.raises(ValueError):
        validate_packed_distances(qa, qb, length)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"families": ("F2",)},
        {"families": ()},
        {"families": ("F1", "F1")},
        {"families": "F1"},
        {"batch_size": 0},
        {"batch_size": True},
        {"cdr3a": ["X" * 9]},
        {"cdr3a": ["A" * 8]},
        {"peptides": ["A" * 8]},
    ],
)
def test_invalid_options_or_sequences_fail(kwargs):
    qa, qb = fixture_arrays(n=1)
    params = dict(cdr3a=["A" * 9], cdr3b=["C" * 9], peptides=["D" * 9])
    params.update(kwargs)
    with pytest.raises(ValueError):
        build_structural_features(qa, qb, np.array([[9, 9]]), **params)
