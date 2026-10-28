"""Figures for the open-set report: scorer comparison, OSCR curves, LODO, losses."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

SCORERS = ("msp", "energy", "prototype", "mahalanobis", "openmax")
COLORS = {"msp": "#4C72B0", "energy": "#DD8452", "prototype": "#55A868",
          "mahalanobis": "#C44E52", "openmax": "#8172B3"}


def collect(root=None):
    root = Path(root or config.OUTPUT_DIR)
    rows = []
    for path in sorted(root.rglob("results.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        config_used = summary["config"]
        for scorer, result in summary["results"].items():
            rows.append({
                "run": path.parent.relative_to(root).as_posix(),
                "unknown": summary.get("unknown_device", "UE20"),
                "head": config_used["head"],
                "maxmin": config_used["maxmin_weight"],
                "supcon": config_used["supcon_weight"],
                "scorer": scorer,
                "auroc": result["auroc"],
                "oscr": result["oscr"],
                "eer": result["eer"],
                "fpr95": result["fpr_at_tpr95"],
                "closed": result["closed_set_accuracy"],
                "curves": result["curves"],
            })
    return rows


def figure_scorers(rows, out):
    run = [r for r in rows if r["run"] == "main/ce_seed42"]
    if not run:
        return None
    order = sorted(run, key=lambda r: -r["auroc"])
    names = [r["scorer"] for r in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [r["auroc"] for r in order], 0.4, label="AUROC",
                color="#4C72B0")
    axes[0].bar(x + 0.2, [r["oscr"] for r in order], 0.4, label="OSCR",
                color="#C44E52")
    axes[0].set_xticks(x, names, rotation=15)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Open-set separation (UE20 unknown, CE backbone)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for r in order:
        axes[1].plot(r["curves"]["fpr"], r["curves"]["ccr"], label=r["scorer"],
                     color=COLORS.get(r["scorer"]))
    axes[1].set_xlabel("FPR (unknown accepted)")
    axes[1].set_ylabel("CCR (known correct & accepted)")
    axes[1].set_title("OSCR curves")
    axes[1].set_xlim(0, 0.6)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_lodo(rows, out):
    lodo = [r for r in rows if r["run"].startswith("lodo/")]
    main = [r for r in rows if r["run"] == "main/ce_seed42"]
    if not lodo:
        return None
    lookup = {(r["unknown"], r["scorer"]): r for r in lodo + main}
    devices = [d for d in config.CLASS_NAMES if (d, "mahalanobis") in lookup]
    if (config.UNKNOWN_NAME, "mahalanobis") in lookup:
        devices.append(config.UNKNOWN_NAME)
    tick_labels = [d if d != config.UNKNOWN_NAME else "UE20 *" for d in devices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.16
    base = np.arange(len(devices))
    for index, scorer in enumerate(SCORERS):
        for axis, key in zip(axes, ("auroc", "oscr")):
            values = [lookup[(device, scorer)][key] if (device, scorer) in lookup else np.nan
                      for device in devices]
            axis.bar(base + index * width, values, width, label=scorer, color=COLORS[scorer])
        for axis in axes:
            axis.set_xticks(base + 2 * width, tick_labels, rotation=20)
            axis.set_ylim(0.5, 1.02)
            axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Leave-one-device-out: AUROC per held-out device")
    axes[1].set_ylabel("OSCR")
    axes[1].set_title("Leave-one-device-out: OSCR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.text(0.5, 0.005, "* UE20 is the main protocol (trained on all 7 known devices)",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_losses(rows, out):
    variants = {"ce": "main/ce_seed42", "maxmin": "losses/maxmin_seed42",
                "supcon": "losses/supcon_seed42", "cosface": "losses/cosface_seed42"}
    available = {name: run for name, run in variants.items()
                 if any(r["run"] == run for r in rows)}
    if len(available) < 2:
        return None
    scorer = "mahalanobis"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(available)
    x = np.arange(len(names))
    auroc = []
    oscr = []
    for name, run in available.items():
        match = [r for r in rows if r["run"] == run and r["scorer"] == scorer]
        auroc.append(match[0]["auroc"] if match else np.nan)
        oscr.append(match[0]["oscr"] if match else np.nan)
    axes[0].bar(x - 0.2, auroc, 0.4, label="AUROC", color="#4C72B0")
    axes[0].bar(x + 0.2, oscr, 0.4, label="OSCR", color="#C44E52")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title(f"Loss ablation with {scorer}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for name, run in available.items():
        match = [r for r in rows if r["run"] == run and r["scorer"] == scorer]
        if match:
            axes[1].plot(match[0]["curves"]["fpr"], match[0]["curves"]["ccr"], label=name)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("CCR")
    axes[1].set_title("OSCR curves by loss")
    axes[1].set_xlim(0, 0.6)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    figures = Path(config.OUTPUT_DIR) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        print("no results yet")
        return
    for name, function in (("scorers.png", figure_scorers),
                           ("lodo.png", figure_lodo),
                           ("losses.png", figure_losses)):
        path = function(rows, figures / name)
        print(f"{name}: {'written' if path else 'skipped'}")


if __name__ == "__main__":
    main()
