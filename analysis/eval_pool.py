"""Sample captures from every device directory and score them with a trained run.

Draws from files that were NOT used for training (the frozen train split is
excluded), so known-device accuracy is not inflated by memorised samples.
For a known device "correct" means the predicted class matches its directory;
for the unknown device it means the capture was rejected.

    python -m analysis.eval_pool --per-device 200
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, dataset, predict


def training_sources():
    with config.split_path("train").open(encoding="utf-8") as handle:
        return {json.loads(line)["source"] for line in handle if line.strip()}


def device_directory(device, distance):
    """Resolve the capture directory for a device at a given distance.

    Only some devices have 4 m / 6 m siblings; those without return None and are
    skipped rather than silently falling back to 2 m.
    """
    base = config.DEVICE_DIRS[device]
    if distance == "2m":
        return base
    candidate = base.replace("_2m", f"_{distance}")
    if candidate == base:
        return None
    return candidate if (config.ROOT / candidate).is_dir() else None


def sample_paths(directory, count, exclude, rng):
    candidates = [path for path in sorted((config.ROOT / directory).glob("*.bin"))
                  if path.relative_to(config.ROOT).as_posix() not in exclude]
    count = min(count, len(candidates))
    picked = rng.choice(len(candidates), count, replace=False)
    return [candidates[i] for i in sorted(picked)]


def summarise(device, records, unknown=False):
    predicted = np.array([record["device"] for record in records])
    accepted = np.array([record["is_known"] for record in records])
    confidence = np.array([record["confidence"] for record in records])
    knownness = np.array([record["knownness"] for record in records])

    if unknown:
        correct = ~accepted                      # rejecting is the correct behaviour
        accuracy = float(correct.mean())
        destination = {}
        for name in predicted[accepted]:
            destination[name] = destination.get(name, 0) + 1
    else:
        correct = predicted == device
        accuracy = float(correct.mean())
        destination = {}
        for name in predicted[~correct]:
            destination[name] = destination.get(name, 0) + 1

    top = sorted(destination.items(), key=lambda kv: -kv[1])[:2]
    return {
        "device": device,
        "n": len(records),
        "accuracy": accuracy,
        "accepted": float(accepted.mean()),
        "correct_and_accepted": float((correct & accepted).mean()),
        "median_knownness": float(np.median(knownness)),
        "median_confidence": float(np.median(confidence)),
        "destination": ", ".join(f"{k} {v}" for k, v in top) or "-",
        "unknown": unknown,
    }


def main(per_device=200, run=None, seed=0, distance="2m"):
    predictor = predict.Predictor(run)
    exclude = training_sources()
    print(f"distance {distance} | excluding {len(exclude)} training captures\n")

    rng = np.random.default_rng(seed)
    rows = []
    for device in config.CLASS_NAMES + [config.UNKNOWN_NAME]:
        directory = device_directory(device, distance)
        if directory is None:
            print(f"skip {device:8s} (no {distance} directory)")
            continue
        unknown = device == config.UNKNOWN_NAME
        paths = sample_paths(directory, per_device, exclude, rng)
        records = predictor.predict_many(paths)
        rows.append(summarise(device, records, unknown=unknown))
        print(f"scored {device:8s} n={len(records):4d}  <- {directory}")

    header = (f"{'device':8s} {'n':>4s} {'acc':>7s} {'accept':>7s} "
              f"{'ok&acc':>7s} {'med_known':>10s}   destination")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        label = row["device"] + (" *" if row["unknown"] else "")
        print(f"{label:8s} {row['n']:4d} {row['accuracy']:7.3f} {row['accepted']:7.3f} "
              f"{row['correct_and_accepted']:7.3f} {row['median_knownness']:10.3f}  "
              f"{row['destination']}")

    known = [row for row in rows if not row["unknown"]]
    unknown_rows = [row for row in rows if row["unknown"]]
    print(f"\nknown-device mean accuracy {np.mean([r['accuracy'] for r in known]):.4f} "
          f"| mean accept {np.mean([r['accepted'] for r in known]):.4f}"
          + (f" | UE20 rejection {unknown_rows[0]['accuracy']:.4f}" if unknown_rows else ""))
    print("* UE20: acc = correct rejection rate, accept = false-accept rate")

    out = Path(config.OUTPUT_DIR) / f"eval_pool_{distance}.json"
    out.write_text(json.dumps({"per_device": per_device, "seed": seed,
                               "distance": distance, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-device", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", default="2m", choices=["2m", "4m", "6m"])
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    main(args.per_device, args.run, args.seed, args.distance)
