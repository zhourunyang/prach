"""SNR robustness sweep: does the classifier degrade the same way as at 4 m?

The 4 m captures are ~6 dB weaker than 2 m but still clean (CP coherence 0.94-0.98).
If adding controlled noise to 2 m captures reproduces the 4 m error pattern, then the
collapse is an SNR effect rather than a distance-specific one.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, dataset, predict, train


def add_noise(X, snr_db, rng):
    """X is RMS-normalised, so signal power is 1 and sigma follows from the SNR."""
    sigma = np.sqrt(10.0 ** (-snr_db / 10.0))
    return X + rng.normal(0.0, sigma, X.shape).astype(np.float32)


def main(per_device=100, run=None):
    predictor = predict.Predictor(run)
    data, _known = train.load_arrays()
    rng = np.random.default_rng(0)

    rows = []
    for device in ["UE9", "UE12", "UE17"]:
        index = np.flatnonzero(data["test"]["device"] == device)
        picked = index[rng.choice(len(index), min(per_device, len(index)), replace=False)]
        X = data["test"]["X"][picked]
        truth = device

        for snr in (30, 25, 20, 15, 12, 10, 8, 6):
            records = predictor.predict_array(add_noise(X, snr, rng))
            predicted = np.array([r["device"] for r in records])
            accepted = np.array([r["is_known"] for r in records])
            knownness = np.array([r["knownness"] for r in records])
            destination = {}
            for name in predicted[predicted != truth]:
                destination[name] = destination.get(name, 0) + 1
            top = sorted(destination.items(), key=lambda kv: -kv[1])[:2]
            rows.append({
                "device": device,
                "snr_db": snr,
                "accuracy": float((predicted == truth).mean()),
                "accepted": float(accepted.mean()),
                "median_knownness": float(np.median(knownness)),
                "destination": ", ".join(f"{k} {v}" for k, v in top) or "-",
            })

    print(f"{'device':7s} {'SNR':>4s} {'acc':>7s} {'accept':>7s} "
          f"{'med_known':>10s}   misclassified as")
    print("-" * 74)
    for row in rows:
        print(f"{row['device']:7s} {row['snr_db']:4d} {row['accuracy']:7.3f} "
              f"{row['accepted']:7.3f} {row['median_knownness']:10.2f}  "
              f"{row['destination']}")

    out = Path(config.OUTPUT_DIR) / "snr_robustness.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-device", type=int, default=100)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    main(args.per_device, args.run)
