"""Feature-space visualisations for a trained run.

Produces
--------
tsne_features.png         t-SNE of the 128-d embedding, coloured by device
class_similarity.png      cosine similarity between per-device mean embeddings
feature_distributions.png score distributions: known test vs. the unknown device
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from . import config, models, scorers, train

DEVICE_COLORS = {
    "UE9": "#1f77b4", "UE12": "#ff7f0e", "UE17": "#2ca02c",
    "model1": "#d62728", "model2": "#9467bd", "model5": "#8c564b",
    "model9": "#e377c2", config.UNKNOWN_NAME: "#111111",
}


def load_model(directory):
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    unknown = summary.get("unknown_devices")
    if not unknown:
        single = summary.get("unknown_device", config.UNKNOWN_NAME)
        unknown = [single] if isinstance(single, str) else list(single)
    model = models.build(cfg["embedding_dim"], cfg["num_classes"], cfg["head"])
    state = __import__("torch").load(directory / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg, unknown


def embeddings(model, data):
    """Row-aligned embeddings, devices and split tags over every split."""
    pooled = []
    for split in config.SPLITS:
        _logits, embedding = train.infer(model, data[split]["X"], device="cpu")
        pooled.append((embedding, data[split]["device"], np.full(len(embedding), split,
                                                                dtype=object)))
    embedding = np.concatenate([p[0] for p in pooled])
    device = np.concatenate([p[1] for p in pooled])
    split = np.concatenate([p[2] for p in pooled])
    return embedding, device, split


def figure_tsne(embedding, device, out, unknown_names=None, seed=42):
    unknown_names = unknown_names or [config.UNKNOWN_NAME]
    projected = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                     max_iter=1200, random_state=seed).fit_transform(embedding)

    devices = [d for d in config.CLASS_NAMES + [config.UNKNOWN_NAME]
               if (device == d).any()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    for name in devices:
        mask = device == name
        axes[0].scatter(projected[mask, 0], projected[mask, 1], s=5, alpha=0.55,
                        color=DEVICE_COLORS[name], label=name)
    axes[0].set_title(f"t-SNE of the 128-d embedding ({len(embedding)} captures)")
    axes[0].legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)

    is_unknown = np.isin(device, unknown_names)
    axes[1].scatter(projected[~is_unknown, 0], projected[~is_unknown, 1], s=4, alpha=0.18,
                    color="#9aa7b8", label="known devices")
    for index, name in enumerate(unknown_names):
        mask = device == name
        axes[1].scatter(projected[mask, 0], projected[mask, 1], s=16, alpha=0.9,
                        color=["#111111", "#d62728", "#9467bd"][index % 3],
                        marker="x", label=f"{name} (unknown)")
    axes[1].set_title("Same projection, unknown devices highlighted")
    axes[1].legend(markerscale=2, fontsize=9, loc="best", framealpha=0.9)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_similarity(embedding, device, out):
    devices = config.CLASS_NAMES + [config.UNKNOWN_NAME]
    means = []
    for name in devices:
        mean = embedding[device == name].mean(0)
        means.append(mean / np.linalg.norm(mean))
    means = np.stack(means)
    similarity = means @ means.T

    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(similarity, cmap="RdYlBu_r", vmin=0.2, vmax=1.0)
    axis.set_xticks(range(len(devices)), devices, rotation=35, ha="right")
    axis.set_yticks(range(len(devices)), devices)
    for i in range(len(devices)):
        for j in range(len(devices)):
            axis.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                      fontsize=8,
                      color="white" if abs(similarity[i, j] - 0.6) > 0.25 else "black")
    axis.set_title("Cosine similarity between per-device mean embeddings")
    fig.colorbar(image, ax=axis, shrink=0.82, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    nearest = {}
    for index, name in enumerate(devices):
        order = np.argsort(-similarity[index])
        others = [j for j in order if j != index]
        nearest[name] = (devices[others[0]], float(similarity[index, others[0]]))
    return out, nearest


def figure_distributions(model, data, out, acceptance=0.95, unknown_names=None):
    unknown_names = unknown_names or [config.UNKNOWN_NAME]
    train_logits, train_embedding = train.infer(model, data["train"]["X"], device="cpu")
    test_logits, test_embedding = train.infer(model, data["test"]["X"], device="cpu")
    unknown_logits, unknown_embedding = train.infer(model, data["unknown_test"]["X"],
                                                    device="cpu")
    val_logits, val_embedding = train.infer(model, data["val"]["X"], device="cpu")
    unknown_device = data["unknown_test"]["device"]

    prototype = scorers.Prototype().fit(train_logits, train_embedding, data["train"]["y"])
    mahalanobis = scorers.Mahalanobis().fit(train_logits, train_embedding, data["train"]["y"])

    def cosine_distance(logits, embedding):
        knownness, _ = prototype.score(logits, embedding)
        return 1.0 - knownness

    colours = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
    known_distance = cosine_distance(test_logits, test_embedding)
    unknown_distance = cosine_distance(unknown_logits, unknown_embedding)

    knownness_val, _ = mahalanobis.score(val_logits, val_embedding)
    threshold = float(np.quantile(knownness_val, 1.0 - acceptance, method="lower"))
    knownness_test, _ = mahalanobis.score(test_logits, test_embedding)
    knownness_unknown, _ = mahalanobis.score(unknown_logits, unknown_embedding)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins_distance = np.linspace(0, max(known_distance.max(), unknown_distance.max()) * 1.02, 60)
    axes[0].hist(known_distance, bins=bins_distance, density=True, alpha=0.5,
                 color="#1f77b4", label="known test (5 devices)")
    for index, name in enumerate(unknown_names):
        mask = unknown_device == name
        axes[0].hist(unknown_distance[mask], bins=bins_distance, density=True,
                     histtype="step", linewidth=2.0, color=colours[index % len(colours)],
                     label=f"{name} (unknown)")
    axes[0].set_xlabel("cosine distance to nearest class mean")
    axes[0].set_ylabel("density")
    axes[0].set_title("Prototype distance")
    axes[0].legend(fontsize=8)

    low = min(knownness_test.min(), knownness_unknown.min())
    high = max(knownness_test.max(), knownness_unknown.max())
    bins_score = np.linspace(low, high, 70)
    axes[1].hist(knownness_test, bins=bins_score, density=True, alpha=0.5,
                 color="#1f77b4", label="known test")
    for index, name in enumerate(unknown_names):
        mask = unknown_device == name
        axes[1].hist(knownness_unknown[mask], bins=bins_score, density=True,
                     histtype="step", linewidth=2.0, color=colours[index % len(colours)],
                     label=f"{name} (unknown)")
    axes[1].axvline(threshold, color="#111111", linestyle="--",
                    label=f"$\\tau$ ({acceptance:.0%} known acceptance)")
    axes[1].set_xlabel("Mahalanobis knownness  (higher = more likely known)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Mahalanobis score")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    stats = {
        "prototype_distance_median_known": float(np.median(known_distance)),
        "mahalanobis_threshold": threshold,
    }
    for name in unknown_names:
        mask = unknown_device == name
        stats[f"prototype_distance_median_{name}"] = float(np.median(unknown_distance[mask]))
        stats[f"rejection_rate_{name}"] = float((knownness_unknown[mask] < threshold).mean())
    return out, stats


def main(directory):
    directory = Path(directory)
    figures = Path(config.OUTPUT_DIR) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    model, _cfg, unknown = load_model(directory)
    data, _known = train.load_arrays(unknown)
    embedding, device, _split = embeddings(model, data)
    suffix = "" if unknown == [config.UNKNOWN_NAME] else "_" + "_".join(unknown)

    print(figure_tsne(embedding, device, figures / f"tsne_features{suffix}.png", unknown))
    path, nearest = figure_similarity(embedding, device,
                                      figures / f"class_similarity{suffix}.png")
    print(path)
    for name, (other, value) in nearest.items():
        print(f"  nearest mean to {name:6s}: {other:6s} ({value:.3f})")
    path, stats = figure_distributions(model, data,
                                       figures / f"feature_distributions{suffix}.png",
                                       unknown_names=unknown)
    print(path)
    for key, value in stats.items():
        print(f"  {key} = {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(Path(config.OUTPUT_DIR) / "main" / "ce_seed42"))
    main(parser.parse_args().run)
