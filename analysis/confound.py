"""How much of the 7-class separable does non-fingerprint information explain?

The capture distances are not uniform: the four module directories were recorded at
roughly 1 m while the phones were recorded at 2 m. Per-sample RMS normalisation
removes absolute amplitude, but not the signal-to-noise ratio, so distance can leak
into the representation as an SNR cue.

This script estimates an SNR proxy straight from each capture and asks how well that
proxy alone predicts (a) the exact device and (b) phone-vs-module.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, preprocess

PHONES = ["UE9", "UE12", "UE17", "UE20"]
MODULES = ["model1", "model2", "model5", "model9"]


def physical_features(path):
    x, _frequency, _capture = preprocess.load_capture(path)
    metrics = preprocess.analyze(x)
    start = metrics["sync_start"]
    body = x[start + config.CP:start + config.CP + config.BODY]
    quiet = np.concatenate((x[:max(start - 32, 0)],
                            x[start + config.PREAMBLE:]))
    if len(quiet) < 64:
        return None
    # Exponential-distributed power: the median is robust to transients.
    noise = float(np.median(np.abs(quiet) ** 2) / np.log(2.0))
    signal = float(np.mean(np.abs(body) ** 2))
    if noise <= 0 or signal <= 0:
        return None
    return {
        "snr_db": 10.0 * np.log10(signal / noise),
        "log_rms": float(np.log10(np.sqrt(signal))),
        "abs_cfo_hz": abs(metrics["cfo_hz"]),
        "cp_corr": metrics["cp_corr"],
        "quiet_samples": len(quiet),
    }


def sample(device, count, rng):
    directory = config.ROOT / config.DEVICE_DIRS[device]
    files = sorted(directory.glob("*.bin"))
    picked = [files[i] for i in rng.choice(len(files), min(count, len(files)), replace=False)]
    rows = []
    for path in picked:
        values = physical_features(path)
        if values:
            rows.append({**values, "device": device})
    return rows


def main(per_device=150):
    rng = np.random.default_rng(0)
    rows = []
    for device in PHONES + MODULES:
        rows.extend(sample(device, per_device, rng))

    devices = np.array([row["device"] for row in rows])
    snr = np.array([row["snr_db"] for row in rows])
    features = np.array([[row["snr_db"], row["log_rms"], row["abs_cfo_hz"], row["cp_corr"]]
                         for row in rows])

    print(f"{'device':8s} {'n':>4s} {'SNR dB p10':>11s} {'p50':>7s} {'p90':>7s}")
    for device in PHONES + MODULES:
        mask = devices == device
        print(f"{device:8s} {mask.sum():4d} {np.quantile(snr[mask], 0.1):11.1f} "
              f"{np.median(snr[mask]):7.1f} {np.quantile(snr[mask], 0.9):7.1f}")

    is_module = np.isin(devices, MODULES)
    print(f"\nphones median SNR {np.median(snr[~is_module]):.1f} dB | "
          f"modules {np.median(snr[is_module]):.1f} dB")

    def evaluate(X, y, label):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"  {label:44s} {scores.mean():.3f} +/- {scores.std():.3f}")
        return float(scores.mean())

    print("\n5-fold CV accuracy on physical features")
    summary = {}
    summary["device_4features"] = evaluate(features, devices, "7-class device, 4 features")
    summary["device_snr_only"] = evaluate(snr[:, None], devices, "7-class device, SNR only")
    summary["class_4features"] = evaluate(features, is_module.astype(int),
                                          "phone vs module, 4 features")
    summary["class_snr_only"] = evaluate(snr[:, None], is_module.astype(int),
                                         "phone vs module, SNR only")
    summary["class_depth"] = evaluate(np.abs(features[:, :1]), is_module.astype(int),
                                      "phone vs module, |SNR| only")

    out = config.OUTPUT_DIR / "confound_analysis.json"
    out.write_text(json.dumps({"per_device": per_device, "summary": summary,
                               "snr_by_device": {
                                   d: np.round(snr[devices == d], 2).tolist()
                                   for d in PHONES + MODULES}},
                              ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-device", type=int, default=150)
    args = parser.parse_args()
    main(args.per_device)
