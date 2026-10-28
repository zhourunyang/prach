"""Open-set scorers: map a test sample to a "knownness" score and a known class.

Every scorer is fitted on *known training data only*. `score()` returns
(knownness, predicted_class) with knownness oriented so that larger values mean
"more likely to belong to a known device" - the rejection rule is knownness < tau.
"""

import numpy as np
from scipy.stats import weibull_min
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


def _softmax(logits):
    shifted = logits - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(1, keepdims=True)


def _unit(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


class Scorer:
    name = "base"

    def fit(self, logits, embedding, labels):
        return self

    def score(self, logits, embedding):
        raise NotImplementedError


class MaxSoftmax(Scorer):
    """Hendrycks & Gimpel baseline: maximum softmax probability."""

    name = "msp"

    def score(self, logits, embedding):
        probability = _softmax(logits)
        return probability.max(1), probability.argmax(1)


class Energy(Scorer):
    """Free energy over logits: E = -T logsumexp(z); knownness = -E."""

    name = "energy"

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def score(self, logits, embedding):
        shifted = logits / self.temperature
        maximum = shifted.max(1, keepdims=True)
        lse = maximum[:, 0] + np.log(np.exp(shifted - maximum).sum(1))
        return self.temperature * lse, logits.argmax(1)


class Prototype(Scorer):
    """Cosine similarity to the per-class mean embedding."""

    name = "prototype"

    def fit(self, logits, embedding, labels):
        unit = _unit(embedding)
        self.prototypes = _unit(np.stack([unit[labels == c].mean(0)
                                          for c in range(len(np.unique(labels)))]))
        return self

    def score(self, logits, embedding):
        similarity = _unit(embedding) @ self.prototypes.T
        return similarity.max(1), similarity.argmax(1)


class Mahalanobis(Scorer):
    """Per-class Mahalanobis distance in a PCA-reduced embedding (TCCN 2024, eq. 24).

    The paper computes it in the K-dimensional head space; we keep a configurable
    low dimension because a 128-d per-class covariance is singular with 480 samples.
    """

    name = "mahalanobis"

    def __init__(self, dim=32, shrinkage="auto"):
        self.dim = dim
        self.shrinkage = shrinkage

    def fit(self, logits, embedding, labels):
        classes = np.unique(labels)
        dim = int(min(self.dim, embedding.shape[1], len(labels) - 1))
        self.pca = PCA(n_components=dim, whiten=False).fit(embedding)
        projected = self.pca.transform(embedding)
        self.means, self.precisions = [], []
        for c in classes:
            group = projected[labels == c]
            self.means.append(group.mean(0))
            self.precisions.append(LedoitWolf().fit(group).precision_)
        self.classes = classes
        return self

    def score(self, logits, embedding):
        projected = self.pca.transform(embedding)
        distances = np.empty((len(projected), len(self.means)))
        for index, (mean, precision) in enumerate(zip(self.means, self.precisions)):
            difference = projected - mean
            distances[:, index] = np.einsum("ij,jk,ik->i", difference, precision, difference)
        distances = np.sqrt(np.clip(distances, 0, None))
        return -distances.min(1), self.classes[distances.argmin(1)]


class OpenMax(Scorer):
    """Meta-recognition with a Weibull tail fit on the activation-vector distance."""

    name = "openmax"

    def __init__(self, tail=20, alpha=3):
        self.tail = tail
        self.alpha = alpha

    def fit(self, logits, embedding, labels):
        classes = np.unique(labels)
        predicted = logits.argmax(1)
        self.means, self.weibulls = [], []
        for c in classes:
            mask = (labels == c) & (predicted == c)
            group = logits[mask] if mask.sum() >= self.tail else logits[labels == c]
            mean = group.mean(0)
            self.means.append(mean)
            distances = np.linalg.norm(group - mean, axis=1)
            tail = np.sort(distances)[-min(self.tail, len(distances)):]
            try:
                shape, loc, scale = weibull_min.fit(tail, floc=0)
            except Exception:
                shape, loc, scale = 1.0, 0.0, float(np.std(distances) + 1e-6)
            self.weibulls.append((shape, loc, scale))
        self.classes = classes
        return self

    def score(self, logits, embedding):
        distances = np.stack([np.linalg.norm(logits - mean, axis=1)
                              for mean in self.means], axis=1)
        revised = logits.copy()
        unknown_mass = np.zeros(len(logits))
        top = np.argsort(-distances, axis=1)[:, :self.alpha]
        for row in range(len(logits)):
            for column in top[row]:
                shape, loc, scale = self.weibulls[column]
                weight = float(weibull_min.sf(distances[row, column], shape, loc, scale))
                weight = min(max(weight, 0.0), 1.0)
                revised[row, column] = logits[row, column] * (1.0 - weight)
                unknown_mass[row] += weight
        probability = _softmax(revised)
        total = probability.sum(1) + unknown_mass
        known_probability = probability / np.clip(total[:, None], 1e-12, None)
        return known_probability.max(1), known_probability.argmax(1)


REGISTRY = {
    "msp": MaxSoftmax,
    "energy": Energy,
    "prototype": Prototype,
    "mahalanobis": Mahalanobis,
    "openmax": OpenMax,
}


def make(name, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown scorer {name}; available {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def fit_all(logits, embedding, labels, names=None):
    names = names or list(REGISTRY)
    fitted = {}
    for name in names:
        fitted[name] = make(name).fit(logits, embedding, labels)
    return fitted
