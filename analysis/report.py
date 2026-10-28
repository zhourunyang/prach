"""Compact report of one experiment directory: scorer table + open-set failure modes."""

import json
import sys
from pathlib import Path

from . import config


def main(directory):
    directory = Path(directory)
    summary = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    results = summary["results"]
    print(f"run: {directory}")
    print(f"unknown device: {summary.get('unknown_device')}   "
          f"known: {', '.join(summary.get('known_devices', []))}")
    print(f"train: {summary['training']}")

    header = f"{'scorer':13s} {'AUROC':>7s} {'OSCR':>7s} {'EER':>7s} {'FPR@95':>7s} " \
             f"{'TPR@5':>7s} {'closed':>7s} {'unk.det':>8s} {'OS-acc':>7s}"
    print("\n" + header)
    best = sorted(results.items(), key=lambda kv: -kv[1]["oscr"])
    for name, r in best:
        print(f"{name:13s} {r['auroc']:7.4f} {r['oscr']:7.4f} {r['eer']:7.4f} "
              f"{r['fpr_at_tpr95']:7.4f} {r['tpr_at_fpr05']:7.4f} "
              f"{r['closed_set_accuracy']:7.4f} {r['unknown_detection_rate']:8.4f} "
              f"{r['open_set_accuracy']:7.4f}")

    name, r = best[0]
    print(f"\nbest scorer: {name}")
    print("unknown accepted as: " + json.dumps(r["unknown_accepted_as"]))
    print("per-class (known):")
    for label, stats in r["per_class"].items():
        print(f"  class {label}: n={stats['n']:4d} acc={stats['accuracy']:.4f} "
              f"false_rejection={stats['false_rejection']:.4f}")
    print("confusion (truth x predicted, -1 = rejected as unknown):")
    labels = r["confusion_labels"]
    print("        " + " ".join(f"{lab:>5d}" for lab in labels))
    for label, row in zip(labels, r["confusion"]):
        print(f"  {label:5d} " + " ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.OUTPUT_DIR / "main" / "ce_seed42")
