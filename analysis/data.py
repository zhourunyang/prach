"""Loaders for the preprocessed cache and the frozen split manifests."""

import json
from pathlib import Path

import numpy as np

from . import config


def load_meta():
    return json.loads(config.CACHE_META.read_text(encoding="utf-8"))


def load_cache(mmap=True):
    cache = np.load(config.CACHE_NPY, mmap_mode="r" if mmap else None)
    if cache.ndim != 3 or cache.shape[1] != 2 or cache.shape[2] != config.BODY:
        raise ValueError(f"Unexpected cache shape {cache.shape}")
    return cache


def read_manifest(split):
    path = config.split_dir(split)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index():
    """Column arrays aligned with cache rows, plus a split mask helper."""
    items = load_meta()["items"]
    columns = {
        "source": np.array([row["source"] for row in items], dtype=object),
        "device": np.array([row["device"] for row in items], dtype=object),
        "split": np.array([row["split"] for row in items], dtype=object),
        "class_id": np.array([int(row["class_id"]) for row in items], dtype=np.int64),
        "known": np.array([bool(row["known"]) for row in items], dtype=bool),
        "root_u": np.array([int(row.get("root_u", -1)) for row in items], dtype=np.int64),
        "cp_corr": np.array([float(row.get("cp_corr", np.nan)) for row in items], dtype=np.float64),
        "cfo_hz": np.array([float(row.get("cfo_hz", np.nan)) for row in items], dtype=np.float64),
        "sync_start": np.array([int(row.get("sync_start", -1)) for row in items], dtype=np.int64),
    }
    return columns


def mask(split, columns=None):
    columns = columns or index()
    return columns["split"] == split


def arrays(splits, columns=None, cache=None):
    """Return (X, class_id, device, split_tags, row_indices) for the given splits."""
    columns = columns or index()
    cache = cache if cache is not None else load_cache()
    selected = np.zeros(len(columns["split"]), dtype=bool)
    for split in splits:
        selected |= columns["split"] == split
    rows = np.flatnonzero(selected)
    X = np.asarray(cache[rows], dtype=np.float32)
    return X, columns["class_id"][rows], columns["device"][rows], columns["split"][rows], rows


def class_weights(class_id, num_known=config.NUM_KNOWN):
    counts = np.bincount(class_id[class_id >= 0], minlength=num_known).astype(np.float64)
    weights = np.zeros(num_known, dtype=np.float64)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights
