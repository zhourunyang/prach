"""Experiment driver: closed-set baseline, open-set scorer sweep, ablations, LODO.

Stages
------
smoke     3 epochs, proves the pipeline end to end.
main      cross-entropy backbone, several seeds, all open-set scorers.
losses    CE / +maxmin / +supcon / cosface, all scorers, several seeds.
lodo      leave-one-device-out: each known device becomes the unknown in turn.
summarize collect every outputs/**/results.json into one table.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import config, train

OUT = config.OUTPUT_DIR
SCORERS = ("msp", "energy", "prototype", "mahalanobis", "openmax")


def base_config(**overrides):
    cfg = train.TrainConfig(epochs=60, patience=12, batch_size=64, seed=42,
                            scorer_names=SCORERS)
    return replace(cfg, **overrides)


def run_one(cfg, directory, unknown_device=config.UNKNOWN_NAME, verbose=True):
    summary, model, data = train.run(cfg, directory, unknown_device=unknown_device,
                                     verbose=verbose)
    best = max(summary["results"].items(), key=lambda kv: kv[1]["oscr"])
    print(f"[{directory.name}] closed={summary['results']['msp']['closed_set_accuracy']:.4f} "
          f"best_scorer={best[0]} auroc={best[1]['auroc']:.4f} oscr={best[1]['oscr']:.4f}",
          flush=True)
    return summary


def stage_smoke():
    cfg = base_config(epochs=3, patience=3, tag="smoke")
    summary = run_one(cfg, OUT / "smoke")
    print(json.dumps({k: v["auroc"] for k, v in summary["results"].items()}, indent=2))


def stage_main(seeds=(42,)):
    for seed in seeds:
        cfg = base_config(seed=seed, tag=f"ce_s{seed}")
        run_one(cfg, OUT / "main" / f"ce_seed{seed}")


def stage_losses(seeds=(42,), only=None):
    # Cross-entropy is already covered by `main`; this stage adds the open-set
    # regularisers and the angular-margin head on top of the same backbone.
    variants = {
        "maxmin": {"maxmin_weight": 0.5},
        "supcon": {"supcon_weight": 0.5},
        "cosface": {},
    }
    if only:
        variants = {k: v for k, v in variants.items() if k in only}
    for name, overrides in variants.items():
        for seed in seeds:
            cfg = base_config(seed=seed, tag=f"{name}_s{seed}", **overrides)
            if name == "cosface":
                cfg = replace(cfg, head="cosface")
            run_one(cfg, OUT / "losses" / f"{name}_seed{seed}")


def stage_lodo(seeds=(42,)):
    for device in config.CLASS_NAMES:
        for seed in seeds:
            cfg = base_config(seed=seed, tag=f"lodo_{device}_s{seed}")
            run_one(cfg, OUT / "lodo" / f"{device}_seed{seed}", unknown_device=device)


def stage_robust(noise=0.20):
    """Retrain with much stronger noise augmentation to survive the 4 m condition."""
    cfg = base_config(seed=42, tag="snr_aug", augment_noise=noise)
    run_one(cfg, OUT / "robust" / f"snr_aug_seed42")


def stage_open53(unknowns=("UE20", "model5", "model9")):
    """5-known / 3-unknown protocol: three devices are held out at once."""
    cfg = base_config(seed=42, tag="open53")
    run_one(cfg, OUT / "open53" / "ce_seed42", unknown_device=list(unknowns))


def summarize(root=OUT):
    rows = []
    for path in sorted(root.rglob("results.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        cfg = summary["config"]
        unknown = summary.get("unknown_device", "UE20")
        for scorer, result in summary["results"].items():
            rows.append({
                "run": path.parent.relative_to(root).as_posix(),
                "unknown": unknown,
                "seed": cfg["seed"],
                "head": cfg["head"],
                "maxmin": cfg["maxmin_weight"],
                "supcon": cfg["supcon_weight"],
                "closed_acc": result["closed_set_accuracy"],
                "auroc": result["auroc"],
                "oscr": result["oscr"],
                "eer": result["eer"],
                "fpr_tpr95": result["fpr_at_tpr95"],
                "scorer": scorer,
            })
    if not rows:
        print("no results found")
        return []
    rows.sort(key=lambda r: (r["run"], r["scorer"]))
    table = OUT / "summary.csv"
    header = list(rows[0])
    with table.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(f"{row[h]:.6g}" if isinstance(row[h], float) else str(row[h])
                                  for h in header) + "\n")
    print(f"wrote {table} ({len(rows)} rows)")

    pivot = {}
    for row in rows:
        pivot.setdefault(row["run"], {})[row["scorer"]] = (
            row["auroc"], row["oscr"], row["closed_acc"])
    print(f"\n{'run':34s} " + " ".join(f"{s:>16s}" for s in SCORERS))
    for run, values in pivot.items():
        cells = " ".join(f"{values.get(s, (float('nan'),) * 3)[0]:.3f}/"
                         f"{values.get(s, (0, float('nan')))[1]:.3f}" for s in SCORERS)
        print(f"{run:34s} {cells}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["smoke", "main", "losses", "lodo", "robust", "open53",
                                 "summarize"])
    parser.add_argument("--variants", help="comma separated loss variants to run")
    parser.add_argument("--noise", type=float, default=0.20,
                        help="noise sigma for --stage robust (0.20 is about 14 dB SNR)")
    parser.add_argument("--unknowns", help="comma separated devices for --stage open53")
    args = parser.parse_args()
    if args.stage == "losses":
        only = args.variants.split(",") if args.variants else None
        stage_losses(only=only)
    elif args.stage == "robust":
        stage_robust(args.noise)
    elif args.stage == "open53":
        stage_open53(args.unknowns.split(",") if args.unknowns
                     else ("UE20", "model5", "model9"))
    else:
        {"smoke": stage_smoke, "main": stage_main,
         "lodo": stage_lodo, "summarize": summarize}[args.stage]()
