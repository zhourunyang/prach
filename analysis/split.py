"""Build the balanced, preamble-root-stratified 2 m open-set split.

Policy
------
* Known devices: 800 captures each, split 480 / 160 / 160 (train / val / test).
* Unknown device (UE20): 800 captures, test only - never used for training or for
  threshold calibration.
* Every allocation is *proportional within each preamble root*, so train, val and
  test share the same root mixture. This removes "which ZC sequence was sent" as a
  confound: a model cannot score by memorising the preamble index.

Only manifests are written; the raw .bin files are never modified or copied.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from . import config, preprocess


def estimate_root(frequency):
    """Estimate the physical ZC root from the quadratic spectral phase.

    For OAI's frequency-domain PRACH sequence, adjacent-bin products carry a phase
    slope of 2*pi*inv(u)/139, so the FFT peak of those products is the modular
    inverse of the root. Returned folded into 1..69 (min(u, 139-u)).
    """
    size = config.FREQ_BINS
    adjacent = np.roll(frequency, -1) * np.conj(frequency)
    energy = float(np.sum(np.abs(adjacent) ** 2))
    if energy == 0:
        return 0, 0.0
    spectrum = np.abs(np.fft.fft(adjacent)) ** 2
    peak = int(np.argmax(spectrum))
    root = pow(peak, -1, size) if peak else 0
    folded = min(root, size - root) if root else 0
    return folded, float(np.clip(spectrum[peak] / (size * energy), 0.0, 1.0))


def allocate(total, fractions):
    """Split `total` into integer buckets by largest remainder, exact sum."""
    raw = np.array(fractions, dtype=np.float64) * total
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base.tolist()


def root_of(path):
    """Read only the trailing 139 DFT bins, which is where the root evidence is."""
    with path.open("rb") as handle:
        handle.seek(-4 * config.FREQ_BINS, 2)
        raw = handle.read()
    bins = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float64)
    frequency = bins[:, 0] + 1j * bins[:, 1]
    return estimate_root(frequency)


def scan():
    """Inventory the pool with root estimates and SHA-256 digests."""
    pool = {}
    for device in config.CLASS_NAMES + [config.UNKNOWN_NAME]:
        directory = config.ROOT / config.DEVICE_DIRS[device]
        files = sorted(directory.glob("*.bin"))
        if not files:
            raise FileNotFoundError(f"{device}: no captures in {directory}")
        entries = []
        for path in files:
            root, score = root_of(path)
            entries.append({
                "path": path,
                "source": path.relative_to(config.ROOT).as_posix(),
                "root_u": root,
                "root_score": score,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        pool[device] = entries
        print(f"scanned {device:8s} {len(entries):6d} files", flush=True)
    return pool


def stratified_pick(entries, count, rng):
    """Pick `count` entries with a per-root quota proportional to the pool."""
    by_root = defaultdict(list)
    for entry in entries:
        by_root[entry["root_u"]].append(entry)
    roots = sorted(by_root)
    weights = np.array([len(by_root[r]) for r in roots], dtype=np.float64)
    quotas = allocate(count, weights / weights.sum())
    picked = []
    for root, quota in zip(roots, quotas):
        group = by_root[root]
        if quota > len(group):
            raise ValueError(f"root {root}: quota {quota} exceeds {len(group)}")
        indices = rng.permutation(len(group))[:quota]
        picked.extend(group[i] for i in indices)
    return picked


def largest_remainder(values, total, caps):
    """Integer allocation of `total` proportional to `values`, exact and capped."""
    values = np.asarray(values, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.int64)
    base = np.minimum(np.floor(values).astype(np.int64), caps)
    deficit = int(total - base.sum())
    if deficit < 0:
        raise ValueError("proportional floors already exceed the target")
    fraction = values - np.floor(values)
    order = np.argsort(-fraction, kind="stable")
    cursor = 0
    while deficit > 0:
        if cursor >= len(order):
            raise ValueError("cannot satisfy the allocation under the caps")
        index = order[cursor]
        if base[index] < caps[index]:
            base[index] += 1
            deficit -= 1
        cursor += 1
    return base


def stratified_split(entries, totals, rng):
    """Assign train/val/test labels with per-root proportions and exact totals.

    Train is allocated first against its quota, then val from the remainder, and test
    takes whatever is left; the three quotas sum to the device total, so every split
    lands on its exact target while staying proportional within each preamble root.
    """
    by_root = defaultdict(list)
    for entry in entries:
        by_root[entry["root_u"]].append(entry)
    roots = sorted(by_root)
    available = np.array([len(by_root[r]) for r in roots], dtype=np.int64)
    total = int(available.sum())
    if total != sum(totals):
        raise ValueError(f"{total} captures do not match quotas {totals}")

    train = largest_remainder(available * totals[0] / total, totals[0], available)
    rest = available - train
    val = largest_remainder(rest * totals[1] / max(rest.sum(), 1), totals[1], rest)
    test = rest - val

    named = {"train": [], "val": [], "test": []}
    for index, root in enumerate(roots):
        group = by_root[root]
        shuffled = [group[i] for i in rng.permutation(len(group))]
        cursor = 0
        for name, count in zip(("train", "val", "test"), (train[index], val[index], test[index])):
            named[name].extend(shuffled[cursor:cursor + int(count)])
            cursor += int(count)
    return named


def build(seed=config.SEED, output=config.SPLIT_DIR, force=False):
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to rebuild")
    rng = np.random.default_rng(seed)
    totals = tuple(int(round(config.PER_CLASS_KNOWN * f)) for f in config.KNOWN_FRACTIONS)
    if sum(totals) != config.PER_CLASS_KNOWN:
        raise ValueError("known split quotas must sum to the per-class budget")

    pool = scan()
    rows = []
    for class_id, device in enumerate(config.CLASS_NAMES):
        picked = stratified_pick(pool[device], config.PER_CLASS_KNOWN, rng)
        split = stratified_split(picked, totals, rng)
        for name, entries in split.items():
            for entry in entries:
                rows.append({**{k: v for k, v in entry.items() if k != "path"},
                             "device": device, "class_id": class_id, "known": True,
                             "split": name, "distance_m": 2})

    unknown = stratified_pick(pool[config.UNKNOWN_NAME], config.UNKNOWN_TEST, rng)
    for entry in unknown:
        rows.append({**{k: v for k, v in entry.items() if k != "path"},
                     "device": config.UNKNOWN_NAME, "class_id": config.UNKNOWN_LABEL,
                     "known": False, "split": "unknown_test", "distance_m": 2})

    # Validate before touching disk.
    counts = Counter(row["split"] for row in rows)
    expected = {"train": 7 * 480, "val": 7 * 160, "test": 7 * 160,
                "unknown_test": config.UNKNOWN_TEST}
    assert dict(counts) == expected, (dict(counts), expected)
    assert len({row["source"] for row in rows}) == len(rows)
    assert len({row["sha256"] for row in rows}) == len(rows)
    for row in rows:
        path = config.ROOT / row["source"]
        assert path.stat().st_size in (36556, 38956), path
        assert row["root_u"] > 0, row

    per_device = {device: dict(Counter(r["split"] for r in rows if r["device"] == device))
                  for device in config.CLASS_NAMES + [config.UNKNOWN_NAME]}

    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row)
    for name, entries in grouped.items():
        with config.split_path(name).open("w", encoding="utf-8") as handle:
            for row in sorted(entries, key=lambda r: (r["device"], r["source"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    root_by_split = {
        name: dict(sorted(Counter(str(r["root_u"]) for r in entries).items()))
        for name, entries in grouped.items()
    }
    summary = {
        "seed": seed,
        "pool": "2 m captures only",
        "known_classes": config.CLASS_NAMES,
        "unknown_class": config.UNKNOWN_NAME,
        "class_ids": {**{d: i for i, d in enumerate(config.CLASS_NAMES)},
                      config.UNKNOWN_NAME: config.UNKNOWN_LABEL},
        "per_class_known": config.PER_CLASS_KNOWN,
        "known_fractions": config.KNOWN_FRACTIONS,
        "known_split": {"train": 480, "val": 160, "test": 160},
        "unknown_split": {"unknown_test": config.UNKNOWN_TEST},
        "counts": dict(counts),
        "per_device": per_device,
        "root_distribution_by_split": root_by_split,
        "root_stratified": True,
        "available": {d: len(pool[d]) for d in pool},
        "storage": "manifest_only",
        "total_files": len(rows),
        "referenced_binary_bytes": sum((config.ROOT / r["source"]).stat().st_size for r in rows),
        "verification": "unique sources and SHA-256; every capture 36556 or 38956 bytes",
        "caveats": [
            "File-level split, not session-isolated: each 2 m directory is one capture "
            "session, so a device's train/val/test share channel, gain and receiver state.",
            "Known class identity is the directory group, not a verified physical handset.",
            "Distance is not uniform across the known classes: the four model* "
            "directories were recorded at roughly 1 m, not 2 m (owner clarification), "
            "while the phones were recorded at 2 m. Distance is not recorded inside "
            "the capture files, so this cannot be corrected from metadata alone.",
            "Blind protocol: unknown data never enters training or threshold calibration; "
            "the rejection threshold is set on the known val split only.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(seed=args.seed, force=args.force)
