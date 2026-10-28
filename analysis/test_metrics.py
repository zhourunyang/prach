"""Numeric sanity checks for the open-set metrics and scorers."""

import numpy as np

from . import metrics, scorers


def test_separation():
    rng = np.random.default_rng(0)
    known = rng.normal(1.0, 1.0, 4000)
    unknown = rng.normal(-1.0, 1.0, 4000)
    auc = metrics.auroc(known, unknown)
    assert auc > 0.85, auc
    _, _, oscr = metrics.oscr_curve(known, np.ones(len(known), int), unknown)
    assert oscr > 0.80, oscr
    e, _ = metrics.eer(known, unknown)
    assert e < 0.25, e
    # For a 2-sigma separation the theory gives FPR@95TPR = 1 - Phi(0.355) = 0.36
    # and TPR@5FPR = 1 - Phi(-0.355) = 0.64.
    assert metrics.fpr_at_tpr(known, unknown, 0.95) < 0.45
    assert metrics.tpr_at_fpr(known, unknown, 0.05) > 0.55
    print(f"separation: auroc={auc:.3f} oscr={oscr:.3f} eer={e:.3f}")


def test_null():
    rng = np.random.default_rng(1)
    known = rng.normal(0.0, 1.0, 4000)
    unknown = rng.normal(0.0, 1.0, 4000)
    auc = metrics.auroc(known, unknown)
    assert 0.45 < auc < 0.55, auc
    print(f"null: auroc={auc:.3f}")


def test_threshold():
    rng = np.random.default_rng(2)
    validation = rng.normal(0.0, 1.0, 10000)
    threshold = metrics.threshold_at_acceptance(validation, 0.95)
    acceptance = float((validation >= threshold).mean())
    assert 0.93 < acceptance < 0.97, acceptance
    print(f"threshold: tau={threshold:.3f} acceptance={acceptance:.3f}")


def test_evaluate_consistency():
    rng = np.random.default_rng(3)
    known = rng.normal(1.0, 1.0, 300)
    unknown = rng.normal(-1.0, 1.0, 200)
    labels = rng.integers(0, 7, 300)
    result = metrics.evaluate(known, labels, labels, unknown, np.zeros(200, int),
                              metrics.threshold_at_acceptance(known, 0.95))
    assert result["closed_set_accuracy"] == 1.0
    assert 0.0 <= result["unknown_detection_rate"] <= 1.0
    assert result["open_set_accuracy"] <= 1.0
    assert abs(sum(result["unknown_accepted_as"].values())
               + int(result["unknown_detection_rate"] * 200) - 200) <= 1
    print(f"evaluate: auroc={result['auroc']:.3f} os_acc={result['open_set_accuracy']:.3f}")


def test_scorers_shapes():
    rng = np.random.default_rng(4)
    classes, dim = 7, 16
    labels = rng.integers(0, classes, 700)
    centres = rng.normal(0, 3, (classes, dim))
    logits = centres[labels] + rng.normal(0, 0.5, (700, dim))
    embedding = logits / np.linalg.norm(logits, axis=1, keepdims=True)
    fitted = scorers.fit_all(logits, embedding, labels)
    for name, scorer in fitted.items():
        knownness, prediction = scorer.score(logits[:50], embedding[:50])
        assert knownness.shape == (50,), name
        assert prediction.shape == (50,), name
        assert np.all(np.isfinite(knownness)), name
    print("scorers: " + ", ".join(sorted(fitted)))


if __name__ == "__main__":
    test_separation()
    test_null()
    test_threshold()
    test_evaluate_consistency()
    test_scorers_shapes()
    print("all metric/scorer checks passed")
