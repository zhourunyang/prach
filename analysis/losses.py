"""Training objectives: cross-entropy plus open-set regularisers."""

import torch
from torch.nn import functional as F


def supervised_contrastive(embedding, labels, temperature=0.1):
    """Supervised contrastive loss; every other same-class file is a positive.

    Anchors whose batch contains no same-class partner are skipped rather than
    raising: with random shuffling a 7-class batch occasionally holds a single
    sample of one class, and dropping those anchors keeps training stable.
    """
    features = F.normalize(embedding.float(), dim=1)
    batch = len(labels)
    diagonal = torch.eye(batch, dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    count = positives.sum(1)
    valid = count > 0
    if not bool(valid.any()):
        return torch.zeros((), device=features.device, requires_grad=True)
    similarity = features @ features.t() / temperature
    normaliser = torch.logsumexp(similarity.masked_fill(diagonal, -torch.inf), dim=1)
    log_prob = similarity - normaliser[:, None]
    per_anchor = -((log_prob * positives).sum(1) / count.clamp_min(1))
    return per_anchor[valid].mean()


def maxmin(embedding, labels, num_classes):
    """d_max / d_min from Yin et al. (TCCN 2024, eq. 19-22).

    d_max is the class-averaged worst intra-class radius, d_min the closest pair of
    class means, so the ratio rewards simultaneously tight and separated classes.
    """
    means, radii = [], []
    for c in range(num_classes):
        group = embedding[labels == c]
        if len(group) == 0:
            continue
        mean = group.mean(0, keepdim=True)
        means.append(mean)
        radii.append(torch.linalg.vector_norm(group - mean, dim=1).max())
    if len(means) < 2:
        return torch.zeros((), device=embedding.device)
    d_max = torch.stack(radii).mean()
    stacked = torch.cat(means, 0)
    pairwise = torch.cdist(stacked, stacked)
    pairwise = pairwise + torch.eye(len(stacked), device=pairwise.device) * 1e6
    d_min = pairwise.min()
    return d_max / d_min.clamp_min(1e-6)


def combined(logits, embedding, labels, num_classes,
             ce_weight=1.0, maxmin_weight=0.0, supcon_weight=0.0,
             temperature=0.1, label_smoothing=0.0):
    loss = ce_weight * F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    terms = {"ce": float(loss.detach())}
    if maxmin_weight:
        mm = maxmin(embedding, labels, num_classes)
        loss = loss + maxmin_weight * mm
        terms["maxmin"] = float(mm.detach())
    if supcon_weight:
        sc = supervised_contrastive(embedding, labels, temperature)
        loss = loss + supcon_weight * sc
        terms["supcon"] = float(sc.detach())
    return loss, terms
