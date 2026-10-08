# Hard Hat Detection — YOLOv8 目标检测全流程项目

> 状态：进行中。本 README 随实验推进补全，最终版本以提交日为准。

## 1. 任务与难点

- 数据集：[Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)，
  5000 张工地场景图片，PASCAL VOC 标注，3 类：helmet / head / person。
- 难点：① 目标小（工人通常占画面很小比例）；② 类别不均衡（helmet 实例远多于 head）；
  ③ `person` 类存在已知标注不全问题（社区多处提及）；④ helmet 与 head 语义相邻，天然混淆对。

## 2. 我做了什么（三步）

1. 数据工程：VOC XML → YOLO 格式转换（固定类别映射 helmet=0/head=1/person=2）、
   seed=42 固定 80/10/10 划分、五项数据体检 + 50 张标注叠加人工抽检。
2. 基线对齐：YOLOv8n @640px，50 epoch，cos_lr，lr0=0.01→1e-4。
3. 控制变量消融（2 组，每次只变一个变量）：
   - A. 关闭 mosaic 增强（验证增强对过拟合的影响）
   - B. lr0 减半至 5e-3（学习率敏感性）

## 3. 结果总表

| 实验 | 改动 | best_epoch | mAP50 | mAP50-95 | 备注 |
|---|---|---|---|---|---|
| base | — | 待填 | 待填 | 待填 | |
| ablation_no_mosaic | mosaic=0 | 待填 | 待填 | 待填 | |
| ablation_lr5e3 | lr0=5e-3 | 待填 | 待填 | 待填 | |

## 4. 过拟合诊断

（证据：`runs/analysis/<name>/loss_curves.png` + `train_vs_val_map.png` + `conclusion.txt`）
待填：判定 + 证据 + 对策。

## 5. 学习率诊断

（证据：`runs/analysis/<name>/lr_schedule.png`）
待填：调度形态、两组 lr0 对比、最终选择理由。

## 6. 错误分析（人工）

固定 conf 阈值，分 FP / FN / 错分类 / 定位误差 四类人工看图归类。
待填：3~5 个失效模式，每模式配一张图 + 一句"我认为是 X，因为 Y"。

## 7. 失败记录

| 时间 | 尝试 | 预期 | 实际 | 原因 |
|---|---|---|---|---|
| 待填 | | | | |

## 7.5 过程中的疑问与思考（学习记录）

> 本节记录项目进行中的真实疑问及澄清后的理解，体现学习过程。

1. **"怎么确认有没有过拟合？"** → 不能只看单一曲线：val loss 拐点 + mAP 平台期双证据，
   再用 best.pt 分别在 train/val 上评估，差距 >10 个点才算真过拟合。注意检测任务因训练集有增强，
   val loss 低于 train loss 属正常现象。
2. **"最终学习率不能太高或太低"** → cosine 衰减保证末轮 lr = lr0×lrf = 1e-4；
    warmup 3 epoch 防止冷启动打飞权重。lr 曲线应呈"爬升—平台—平滑衰减到近零"。
3. **"调试总不能手调梯度和偏导吧？"** → 反向传播全自动；实践调的是四类旋钮：
   数据 / 优化超参 / 模型容量与输入分辨率 / 后处理阈值（零训练成本）。诊断驱动，一次一变量。
4. **"每次优化都要重训一遍岂不是很慢？"** → 用 200 张×10 epoch 的"探针"验证方向（5~10 分钟），
   方向对才上全量；patience 早停自动止损；每轮实验必须先写明"这轮要回答什么问题"。
5. **"消融是什么意思？"** → 对照实验：每次拆掉/改掉唯一一个部件看成绩变化，
   差异可归因到该变量。
6. **"SGD 和 Adam 有什么区别，这里用哪个？"** → Adam 赢在收敛快、SGD 赢在终点泛化好；
   本项目 optimizer=auto 按迭代数自动选择（冒烟用 AdamW，正式训练用 SGD+momentum）。
7. **"模型结构是什么样的？"** → 实测 YOLOv8n 共 3,157,200 参数：backbone 40% / neck 31% / head 28%，
   64 层卷积（39×3×3 + 25×1×1），SiLU 激活，损失 = CIoU + DFL + BCE 三项。

## 8. 局限与未覆盖项（诚实交代）

因 28 小时交付窗口，以下项目主动放弃并在评审时如实说明：

- 未做 OOD（跨域）独立测试集评估
- 未做 ONNX 导出与部署一致性校验（接口已预留）
- 消融仅 2 组（原计划 4 组），未覆盖 imgsz 与 weight_decay 维度
- 错误分析样本量 50 张（原计划 100 张）

## 9. 复现

```bash
# 解释器: torch314 conda 环境（torch 2.14.0+cu130, ultralytics 8.4.155）
python scripts/convert_voc_to_yolo.py --raw data/raw --out data/yolo
python scripts/check_data.py
python scripts/visualize_annotations.py --n 50
python scripts/train.py --config configs/base.yaml
python scripts/analyze.py --run runs/detect/base
```

环境版本与种子：seed=42，requirements.txt 锁定。
