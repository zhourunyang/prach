"""Draw confusion matrices with absolute counts (how many captures each cell).

Rows are ground truth, columns are predictions; the last column is the rejection
bucket, so the last row is the unknown device(s) that should land there.

    python -m analysis.confusion --run outputs/main/ce_seed42
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_labels(summary, confusion_labels):
    known = summary["known_devices"]
    unknown = summary.get("unknown_devices")
    if not unknown:
        single = summary.get("unknown_device", config.UNKNOWN_NAME)
        unknown = [single] if isinstance(single, str) else list(single)
    unknown_text = (f"未知({len(unknown)}台)" if len(unknown) > 1
                    else f"未知({unknown[0]})")
    row_labels = [unknown_text if label == -1 else known[label]
                  for label in confusion_labels]
    col_labels = ["拒识" if label == -1 else known[label] for label in confusion_labels]
    return row_labels, col_labels, unknown


def draw(run_dir, scorer=None, output=None):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    results = summary["results"]
    scorer = scorer or max(results, key=lambda name: results[name]["oscr"])
    result = results[scorer]
    matrix = np.array(result["confusion"], dtype=int)
    row_labels, col_labels, _unknown = build_labels(summary, result["confusion_labels"])

    row_totals = matrix.sum(1, keepdims=True)
    share = matrix / np.clip(row_totals, 1, None)

    figure, axis = plt.subplots(figsize=(1.05 * len(col_labels) + 3.2,
                                          0.85 * len(row_labels) + 2.2))
    image = axis.imshow(share, cmap="Blues", vmin=0, vmax=1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            if count == 0:
                axis.text(j, i, "0", ha="center", va="center", fontsize=9, color="#b0b0b0")
                continue
            colour = "white" if share[i, j] > 0.55 else "#1a202c"
            axis.text(j, i, f"{count}", ha="center", va="center", fontsize=11,
                      fontweight="bold", color=colour)
            axis.text(j, i + 0.30, f"{share[i, j]:.1%}", ha="center", va="center",
                      fontsize=7, color=colour)

    axis.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel("预测")
    axis.set_ylabel("真实")
    known_labels = len(summary["known_devices"])
    unknown_devices = summary.get("unknown_devices")
    if not unknown_devices:
        single = summary.get("unknown_device", config.UNKNOWN_NAME)
        unknown_devices = [single] if isinstance(single, str) else list(single)
    axis.set_title(f"{known_labels} 已知 + {len(unknown_devices)} 未知"
                   f"（{'+'.join(unknown_devices)}）· {scorer}\n"
                   f"数字 = 条数，小字 = 占该行比例；最后一列为拒识", fontsize=11)

    for edge in range(len(col_labels) + 1):
        axis.axvline(edge - 0.5, color="white", linewidth=2)
        axis.axhline(edge - 0.5, color="white", linewidth=2)
    figure.colorbar(image, ax=axis, shrink=0.8, label="行内占比")
    figure.tight_layout()

    output = Path(output or (Path(config.OUTPUT_DIR) / "figures" /
                             f"confusion_{run_dir.parent.name}_{run_dir.name}.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    print(f"{scorer}: {output}")
    print(f"{'真实\\预测':16s}" + "".join(f"{label:>14s}" for label in col_labels) +
          f"{'合计':>8s}")
    for i, label in enumerate(row_labels):
        cells = "".join(f"{value:>14d}" for value in matrix[i])
        print(f"{label:16s}{cells}{row_totals[i, 0]:>8d}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append",
                        default=[str(Path(config.OUTPUT_DIR) / "main" / "ce_seed42"),
                                 str(Path(config.OUTPUT_DIR) / "open53" / "ce_seed42")])
    parser.add_argument("--scorer", default=None)
    args = parser.parse_args()
    for run in args.run:
        draw(run, args.scorer)
        print()
