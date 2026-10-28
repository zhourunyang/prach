"""Independent integrity audit: split manifests, cache alignment, PRACH layout.

Run after `python -m analysis.split` and `python -m analysis.dataset`.
"""

import hashlib
import json
from collections import Counter

import numpy as np

from . import config, dataset, preprocess


def audit(sample=40, seed=0):
    report = {}
    rng = np.random.default_rng(seed)

    manifests = {}
    for split in config.SPLITS:
        with config.split_path(split).open(encoding="utf-8") as handle:
            manifests[split] = [json.loads(line) for line in handle if line.strip()]
    report["manifest_counts"] = {k: len(v) for k, v in manifests.items()}

    # 1. hash spot-check against the raw captures
    bad = []
    for split, rows in manifests.items():
        picks = rng.choice(len(rows), size=min(10, len(rows)), replace=False)
        for i in picks:
            row = rows[i]
            digest = hashlib.sha256((config.ROOT / row["source"]).read_bytes()).hexdigest()
            if digest != row["sha256"]:
                bad.append(row["source"])
    report["hash_mismatches"] = bad

    # 2. cache / meta / manifest alignment
    cache, meta = dataset.load()
    columns = dataset.index(meta)
    report["cache_shape"] = list(cache.shape)
    report["cache_split_counts"] = dict(Counter(columns["split"]))
    report["manifest_matches_cache"] = all(
        report["cache_split_counts"].get(k, 0) == len(manifests[k]) for k in config.SPLITS
    )

    # 3. recompute a random sample
    picks = rng.choice(cache.shape[0], size=sample, replace=False)
    failures = []
    for i in picks:
        values, _ = preprocess.prepare(config.ROOT / columns["source"][i])
        delta = float(np.max(np.abs(values - np.asarray(cache[i], dtype=np.float32))))
        if not np.isfinite(delta) or delta > 1e-5:
            failures.append({"row": int(i), "max_abs_delta": delta})
    report["recompute_failures"] = failures

    # 4. layout evidence: the CP-coherence peak proves 576 + 4*2048, but it is a
    #    *plateau* rather than a spike because the four 2048-sample blocks are
    #    identical, so a wide peak is expected and only a low one would be a fault.
    peaks, widths, coherence = [], [], []
    for i in picks:
        x, _freq, _cap = preprocess.load_capture(config.ROOT / columns["source"][i])
        metrics = preprocess.analyze(x)
        profile = preprocess.cp_profile(x)
        start = int(np.argmax(profile))
        peaks.append(float(profile[start]))
        widths.append(int(np.sum(profile >= 0.9 * profile[start])))
        coherence.append(metrics["min_repeat_corr"])
    report["cp_peak_median"] = float(np.median(peaks))
    report["cp_plateau_width_median"] = int(np.median(widths))
    report["repeat_coherence_median"] = float(np.median(coherence))
    report["layout_ok"] = bool(report["cp_peak_median"] > 0.9
                               and report["repeat_coherence_median"] > 0.9)

    # 5. independence of the unknown device
    report["unknown_only_in_test"] = set(columns["split"][columns["device"] == config.UNKNOWN_NAME]) == {"unknown_test"}
    report["unknown_absent_from_train"] = not bool(
        ((columns["device"] == config.UNKNOWN_NAME) & (columns["split"] == "train")).any()
    )

    report["all_checks_passed"] = bool(
        not bad and not failures and report["layout_ok"]
        and report["manifest_matches_cache"]
        and report["unknown_only_in_test"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    audit()
