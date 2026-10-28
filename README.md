# 5G NR PRACH 射频指纹 · 开集识别

8 个 5G 模块的 PRACH 采集，7 个已知 + `UE20` 未知，只取 2 m 数据。
本仓库包含完整的划分、预处理、训练、开集打分与评估流水线，以及实验设计与结果文档。

## 先说结论

| | 数值 |
|---|---:|
| 闭集（7 类）准确率 | **98.30%** |
| 未知识别 AUROC | **0.9795** |
| 未知识别 OSCR | **0.9661** |
| 95% 已知接受率下 UE20 拒识率 | **88.62%** |
| 留一设备 AUROC 区间 | 0.8005 – 0.9876（均值 0.9295） |

最佳开集打分器是**低维马氏距离**（PCA 32 + Ledoit-Wolf）；maxmin / SupCon / CosFace
三个开集正则在本题数据上都没有带来增益。详见 [docs/RESULTS.md](docs/RESULTS.md)。

## 目录

```
analysis/            流水线代码
  config.py          全局常量（路径、类别、PRACH 几何、划分参数）
  split.py           均衡 + 按前导根分层的划分生成
  preprocess.py      同步、CFO 补偿、裁剪、归一化
  dataset.py         预处理张量缓存
  verify_data.py     数据完整性审计
  models.py          一维 ResNet-18（linear / cosface 头）
  losses.py          CE / maxmin / SupCon
  scorers.py         MSP / Energy / Prototype / Mahalanobis / OpenMax
  metrics.py         AUROC / OSCR / EER / FPR@95 + 盲测阈值
  train.py           训练与单次运行评估
  experiments.py     实验编排（smoke / main / losses / lodo / summarize）
  report.py          单次运行详细报告
  confusion.py       混淆矩阵图（绝对条数）
  features.py        特征分布图（t-SNE、类相似度、打分分布）
  predict.py         推理：特征向量 + 类别 + 置信度（含开集判决）
  diagram.py         训练流程图
  plots.py           图表
  diag_timing.py     时序对齐诊断
  test_metrics.py    指标自检
docs/
  EXPERIMENT_DESIGN.md   实验流程设计（数据、划分、方法、实验矩阵）
  RESULTS.md             实验结果与分析
data/
  UE9 UE12 UE17 UE20 model1 model2 model5 model9   原始 .bin
  cache/                 预处理张量（400 MiB）+ 逐条元数据
  splits/                划分清单（jsonl）+ summary.json
outputs/               runs、summary.csv、figures/
```

> **`data/` 不入库**（见 `.gitignore`）：原始采集 3.5 GB、缓存 400 MB、模型权重 210 MB 都在本地。
> 仓库里保留了代码、文档、图表和全部 `results.json`。只要把原始 `.bin` 放回 `data/` 的相同目录结构，
> 划分是确定性的（种子 42），`python -m analysis.split` 能逐字节重建清单。
> 论文 PDF 与其抽取文本同样不入库。

## 复现

所有命令在项目根目录执行，使用 `rff` 环境：

```bash
python -m analysis.split          # 生成划分清单
python -m analysis.dataset        # 预处理缓存
python -m analysis.verify_data    # 审计
python -m analysis.test_metrics   # 指标自检
python -m analysis.experiments --stage main
python -m analysis.experiments --stage losses
python -m analysis.experiments --stage lodo
python -m analysis.experiments --stage summarize
python -m analysis.plots
python -m analysis.features       # 特征分布图（t-SNE / 相似度 / 打分分布）
```

## 推理接口

```python
from analysis.predict import Predictor

predictor = Predictor("outputs/main/ce_seed42")
out = predictor.predict("data/UE9/UE9_honor_2m/prach_671.bin")

out["feature"]      # np.ndarray, shape (128,)   特征向量
out["device"]       # 'UE9'                       分类（已知类里最近的一个）
out["confidence"]   # 0.907                      置信度（softmax 最大概率）
out["knownness"]    # -4.38                      开集分数（马氏，越大越像已知）
out["is_known"]     # True                        是否接受为已知设备
out["device_if_known"]  # 'UE9'；若拒绝则为 'UNKNOWN'
```

批量与命令行（`--per-dir N --sample` 每个目录随机抽 N 条）：

```bash
python -m analysis.predict data/UE20 --per-dir 5 --sample --top-k 3 \
    --json outputs/predictions.json
```

### 为什么要同时给 confidence 和 knownness

softmax 置信度是**闭集**量，对未知设备会给出很高的值。演示中 UE20 的两条被预测成
UE9 且置信度 0.72 / 0.89，只有 knownness（−8.93 / −7.36，低于阈值 −6.75）把它们拒掉了。
只用置信度做开集判决会失效。

## 数据格式

每条记录是 int16 小端交织 I/Q：`4 × (时域点数 + 139)` 字节。
UE 家族 9000 时域点（36556 B），model 家族 9600 时域点（38956 B）。
时域是长格式 NR PRACH：**CP 576 + 4 × 2048**，记录未对齐，须搜索同步点。
尾部 139 点是频域 PRACH 样点，用于估计 ZC 根序列。
