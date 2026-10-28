"""Draw the end-to-end training pipeline as a single figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

STAGES = [
    {
        "title": "① 数据池",
        "color": "#E8F0FE", "edge": "#3B6FB6",
        "lines": [
            "8 个 5G 模块，只取 2 m",
            "UE 36556 B / model 38956 B",
            "记录 = int16 I/Q + 139 频域点",
            "候选 35,650 条",
            "记录未对齐，必须搜索同步点",
        ],
    },
    {
        "title": "② 划分（均衡 + 分层）",
        "color": "#E6F4EA", "edge": "#2E7D46",
        "lines": [
            "已知 7 类 × 800：480 / 160 / 160",
            "未知 UE20 800（仅测试）",
            "先按 ZC 根分层，再切 60/20/20",
            "三个 split 的根分布一致",
            "SHA-256 去重与完整性校验",
        ],
    },
    {
        "title": "③ 预处理与缓存",
        "color": "#FDF2E3", "edge": "#B26B1F",
        "lines": [
            "CP 相关同步（CP 576 + 4×2048）",
            "4 个重复块估计 CFO 并补偿",
            "单样本 RMS 归一（不跨集统计）",
            "张量 [2, 8192] → 6400 条缓存",
            "时序抖动约 ±4 样点（CP 平台）",
        ],
    },
    {
        "title": "④ 训练",
        "color": "#FDE8E8", "edge": "#B03A3A",
        "lines": [
            "增强：循环移位 / 相位旋转 / 增益 / 噪声",
            "1D ResNet-18 → 512 → 128 维嵌入",
            "7 类头；CE（+maxmin / SupCon / CosFace）",
            "AdamW + 余弦退火，batch 64",
            "val 准确率早停，恢复最佳权重",
        ],
    },
    {
        "title": "⑤ 开集打分与阈值",
        "color": "#EEEAF8", "edge": "#5B4B9E",
        "lines": [
            "冻结骨干，只用已知训练集拟合",
            "MSP / Energy / Prototype / 马氏 / OpenMax",
            "马氏：PCA 32 维 + Ledoit-Wolf 收缩",
            "τ = val 已知 95% 接受分位数",
            "未知既不参与拟合，也不参与定 τ",
        ],
    },
    {
        "title": "⑥ 评估",
        "color": "#E8F6F8", "edge": "#1F7A8C",
        "lines": [
            "test 1120（7 类闭集）",
            "unknown_test 800（UE20）",
            "AUROC · OSCR · EER · FPR@95",
            "混淆矩阵 + 未知被误判成谁",
            "留一设备：7 台轮流当未知",
        ],
    },
]

# Snake order: top row left->right, bottom row right->left.
POSITIONS = [(0.30, 4.75), (5.55, 4.75), (10.80, 4.75),
             (10.80, 0.55), (5.55, 0.55), (0.30, 0.55)]
WIDTH, HEIGHT = 3.90, 2.15


def draw_box(axis, x, y, stage):
    axis.add_patch(FancyBboxPatch(
        (x, y), WIDTH, HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.8, facecolor=stage["color"], edgecolor=stage["edge"], zorder=2))
    axis.text(x + 0.18, y + HEIGHT - 0.30, stage["title"], fontsize=13,
              fontweight="bold", color=stage["edge"], va="center", zorder=3)
    body = "\n".join(f"· {line}" for line in stage["lines"])
    axis.text(x + 0.18, y + HEIGHT - 0.68, body, fontsize=9.8, color="#2b2b2b",
              va="top", linespacing=1.66, zorder=3)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.0,
        color="#4a5568", shrinkA=0, shrinkB=0, zorder=1))


def main():
    figure, axis = plt.subplots(figsize=(15.2, 7.8))
    axis.set_xlim(0, 15.0)
    axis.set_ylim(0, 7.75)
    axis.axis("off")

    for (x, y), stage in zip(POSITIONS, STAGES):
        draw_box(axis, x, y, stage)

    top_middle = 4.75 + HEIGHT / 2
    bottom_middle = 0.55 + HEIGHT / 2
    arrow(axis, (4.26, top_middle), (5.49, top_middle))        # 1 -> 2
    arrow(axis, (9.51, top_middle), (10.74, top_middle))       # 2 -> 3
    arrow(axis, (12.75, 4.70), (12.75, 2.76))                  # 3 -> 4
    arrow(axis, (10.74, bottom_middle), (9.51, bottom_middle)) # 4 -> 5
    arrow(axis, (5.49, bottom_middle), (4.26, bottom_middle))  # 5 -> 6

    axis.text(0.30, 7.50, "PRACH 射频指纹开集识别 · 训练流程", fontsize=16.5,
              fontweight="bold", color="#1a202c")
    axis.text(0.30, 7.14, "盲测协议：UE20 全程不参与训练，也不参与阈值标定；"
              "所有开集指标只在最后评估一次", fontsize=10, color="#4a5568")

    figure.tight_layout()
    out = config.OUTPUT_DIR / "figures" / "pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(out)


if __name__ == "__main__":
    main()
