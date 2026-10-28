"""Open-set metrics with a blind threshold protocol.

Orientation is fixed once and everywhere: a sample is "positive" when it is KNOWN,
and the score is `knownness` (higher = more likely known). So
    TPR = fraction of known samples accepted
    FPR = fraction of unknown samples accepted
and AUROC = P(knownness of a known sample > knownness of an unknown sample).

The rejection threshold is calibrated on the *known validation split* only, at a
target known-acceptance rate; unknown data is scored exactly once.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def _stack(knownness_known, knownness_unknown):
    labels = np.r_[np.ones(len(knownness_known), int), np.zeros(len(knownness_unknown), int)]
    scores = np.r_[np.asarray(knownness_known, float), np.asarray(knownness_unknown, float)]
    return labels, scores


def auroc(knownness_known, knownness_unknown):
    labels, scores = _stack(knownness_known, knownness_unknown)
    return float(roc_auc_score(labels, scores))


def threshold_at_acceptance(validation_knownness, target=0.95):
    """Cut tau such that at least `target` of the known validation set is accepted."""
    return float(np.quantile(validation_knownness, 1.0 - target, method="lower"))


def eer(knownness_known, knownness_unknown):
    labels, scores = _stack(knownness_known, knownness_unknown)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    miss = 1.0 - tpr                      # known rejected
    index = int(np.argmin(np.abs(fpr - miss)))
    return float((fpr[index] + miss[index]) / 2.0), float(thresholds[index])


def fpr_at_tpr(knownness_known, knownness_unknown, tpr_target=0.95):
    labels, scores = _stack(knownness_known, knownness_unknown)
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= tpr_target)
    return float(fpr[valid[0]]) if len(valid) else 1.0


def tpr_at_fpr(knownness_known, knownness_unknown, fpr_target=0.05):
    labels, scores = _stack(knownness_known, knownness_unknown)
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(fpr <= fpr_target)
    return float(tpr[valid[-1]]) if len(valid) else 0.0


def oscr_curve(knownness_known, correct, knownness_unknown):
    """CCR vs. FPR while the acceptance threshold sweeps from strict to loose."""
    score = np.r_[np.asarray(knownness_known, float), np.asarray(knownness_unknown, float)]
    correct = np.r_[np.asarray(correct, int), np.zeros(len(knownness_unknown), int)]
    unknown = np.r_[np.zeros(len(knownness_known), int), np.ones(len(knownness_unknown), int)]
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    endpoints = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(order) - 1]
    ccr = np.r_[0.0, np.cumsum(correct[order])[endpoints] / len(knownness_known)]
    fpr = np.r_[0.0, np.cumsum(unknown[order])[endpoints] / len(knownness_unknown)]
    return fpr, ccr, float(np.trapezoid(ccr, fpr))


def evaluate(knownness_known, prediction_known, labels_known,
             knownness_unknown, prediction_unknown, threshold):
    knownness_known = np.asarray(knownness_known, float)
    knownness_unknown = np.asarray(knownness_unknown, float)
    accepted_known = knownness_known >= threshold
    accepted_unknown = knownness_unknown >= threshold
    labels_known = np.asarray(labels_known)
    correct = np.asarray(prediction_known) == labels_known

    final_known = np.where(accepted_known, prediction_known, -1)
    final_unknown = np.where(accepted_unknown, prediction_unknown, -1)
    truth = np.r_[labels_known, np.full(len(knownness_unknown), -1)]
    final = np.r_[final_known, final_unknown]

    equal_error, _ = eer(knownness_known, knownness_unknown)
    fpr_curve, ccr_curve, oscr = oscr_curve(knownness_known, correct, knownness_unknown)
    classes = sorted(np.unique(np.r_[labels_known, -1]).tolist())

    return {
        "auroc": auroc(knownness_known, knownness_unknown),
        "oscr": oscr,
        "eer": equal_error,
        "fpr_at_tpr95": fpr_at_tpr(knownness_known, knownness_unknown, 0.95),
        "tpr_at_fpr05": tpr_at_fpr(knownness_known, knownness_unknown, 0.05),
        "threshold": float(threshold),
        "closed_set_accuracy": float(correct.mean()),
        "known_acceptance": float(accepted_known.mean()),
        "known_correct_and_accepted": float((correct & accepted_known).mean()),
        "unknown_detection_rate": float((~accepted_unknown).mean()),
        "open_set_accuracy": float((final == truth).mean()),
        "confusion": confusion_matrix(truth, final, labels=classes).tolist(),
        "confusion_labels": classes,
        "per_class": {
            str(int(c)): {
                "n": int((labels_known == c).sum()),
                "accuracy": float(correct[labels_known == c].mean()),
                "false_rejection": float((~accepted_known)[labels_known == c].mean()),
            }
            for c in np.unique(labels_known)
        },
        "unknown_accepted_as": {
            str(int(c)): int((final_unknown == c).sum())
            for c in np.unique(prediction_known) if c >= 0
        },
        "curves": {"fpr": fpr_curve.tolist(), "ccr": ccr_curve.tolist()},
    }
