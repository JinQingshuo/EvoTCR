import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from evotcr.metrics import evaluate
from evotcr.metrics_reference import auc
from evotcr.models.head import Model, encode, normalize
from evotcr.parameters import frozen_head
from evotcr.integrations.nettcrfold import reduce_confidence


def test_partial_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(22)
    y = np.r_[np.ones(23), np.zeros(81)].astype(int)
    scores = np.round(rng.normal(size=len(y)), 1)
    assert auc(y, scores) == pytest.approx(roc_auc_score(y, scores, max_fpr=0.1), abs=1e-12)
    result = evaluate(np.tile(y, 2), np.tile(scores, 2), np.repeat([0, 1], len(y)))
    assert result["macro_partial_auc"] == pytest.approx(auc(y, scores))


def test_fixed_mlp_parameter_count_and_group_invariance():
    torch.set_num_threads(1)
    h = frozen_head()
    raw = np.random.default_rng(3).normal(size=(40, 17)).astype(np.float32)
    x, mean, sd = encode(raw, h["config"])
    model = Model(x, h["config"], 1109)
    assert sum(p.numel() for p in model.parameters()) == 467
    # Nonzero output weights make invariance meaningful before training.
    with torch.no_grad():
        model.out.weight.fill_(0.01)
        model.linear.weight.fill_(0.01)
    model.eval()
    with torch.no_grad():
        s = model(torch.tensor(x)).numpy()
    subset = np.r_[np.arange(20, 30), np.arange(0, 10)]
    xx, _, _ = encode(raw[subset], h["config"], mean, sd)
    with torch.no_grad():
        ss = model(torch.tensor(xx)).numpy()
    np.testing.assert_allclose(
        normalize(s, h["mode"], 1)[subset], normalize(ss, h["mode"], 1), rtol=0, atol=2e-8
    )


def test_confidence_reduction_is_group_local():
    raw = np.random.default_rng(9).random((30, 3, 7))
    base = reduce_confidence(raw)
    np.testing.assert_array_equal(base[10:20], reduce_confidence(raw[10:20]))
    np.testing.assert_array_equal(base[:, 2], raw[:, 0, 2])
