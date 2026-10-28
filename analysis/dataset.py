"""Build and load the preprocessed tensor cache for the frozen split.

One row per manifest entry, aligned with the row order stored in meta.json:
    X[i] : float32 [2, 8192]  CFO-corrected, RMS-normalised PRACH body
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config, preprocess


def _one(args):
    index, source, compensate_cfo = args
    values, metrics = preprocess.prepare(config.ROOT / source, compensate_cfo)
    metrics = {k: v for k, v in metrics.items() if k != "origin"}
    return index, values, metrics


def build(compensate_cfo=True, workers=8, force=False):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if config.CACHE_NPY.exists() and not force:
        raise FileExistsError(f"{config.CACHE_NPY} exists; pass --force to rebuild")

    items = []
    for split in config.SPLITS:
        with config.split_path(split).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(json.loads(line))
    print(f"preprocessing {len(items)} captures with {workers} workers", flush=True)

    X = np.zeros((len(items), 2, config.BODY), dtype=np.float32)
    tasks = [(i, row["source"], compensate_cfo) for i, row in enumerate(items)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (index, values, metrics) in enumerate(pool.map(_one, tasks, chunksize=64), 1):
            X[index] = values
            items[index] = {**items[index], **metrics}
            if done % 1000 == 0:
                print(f"  {done}/{len(items)}", flush=True)

    np.save(config.CACHE_NPY, X)
    meta = {
        "n": len(items),
        "shape": list(X.shape),
        "compensate_cfo": bool(compensate_cfo),
        "cp": config.CP,
        "period": config.PERIOD,
        "body": config.BODY,
        "items": items,
    }
    config.CACHE_META.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {config.CACHE_NPY} {X.shape} {X.nbytes / 2**20:.0f} MiB")
    return meta


def load(mmap=True):
    if not config.CACHE_NPY.exists():
        raise FileNotFoundError("Run `python -m analysis.dataset` first")
    cache = np.load(config.CACHE_NPY, mmap_mode="r" if mmap else None)
    meta = json.loads(config.CACHE_META.read_text(encoding="utf-8"))
    if cache.shape[0] != meta["n"]:
        raise ValueError("Cache and meta disagree on row count")
    return cache, meta


def index(meta=None):
    if meta is None:
        _, meta = load()
    items = meta["items"]
    return {
        "source": np.array([row["source"] for row in items], dtype=object),
        "device": np.array([row["device"] for row in items], dtype=object),
        "split": np.array([row["split"] for row in items], dtype=object),
        "class_id": np.array([int(row["class_id"]) for row in items], dtype=np.int64),
        "known": np.array([bool(row["known"]) for row in items], dtype=bool),
        "root_u": np.array([int(row.get("root_u", 0)) for row in items], dtype=np.int64),
        "cp_corr": np.array([float(row.get("cp_corr", np.nan)) for row in items]),
        "cfo_hz": np.array([float(row.get("cfo_hz", np.nan)) for row in items]),
        "sync_start": np.array([int(row.get("sync_start", -1)) for row in items]),
    }


def view(columns, split):
    rows = np.flatnonzero(columns["split"] == split)
    return rows


def tensors(splits):
    """Return X, y, device, root for the union of the requested splits."""
    cache, meta = load()
    columns = index(meta)
    rows = np.concatenate([view(columns, s) for s in splits])
    return (np.asarray(cache[rows], dtype=np.float32), columns["class_id"][rows],
            columns["device"][rows], columns["root_u"][rows], columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cfo", action="store_true", help="skip CFO compensation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(compensate_cfo=not args.no_cfo, workers=args.workers, force=args.force)
