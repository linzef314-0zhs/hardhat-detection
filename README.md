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

## 7.5 学习路径与认知递进（学习记录）

> 本节按认知层次而非时间顺序整理项目中的真实疑问。四层的递进本身就是学习成效的证明：
> 先让东西跑起来（工程），再学会判断好坏（诊断），再学会高效改进（方法），最后理解为什么（原理）。

### 第一层：工程层 —— "怎么让流程跑通"

- 驱动问题：数据集选哪个？格式怎么转？训练怎么挂机不中断？
- 收获：VOC→YOLO 转换与类别映射、固定 seed 划分防泄漏、训练进程与交互会话解耦、
  后台训练 + 日志轮询的工作模式
- 代价换来的经验：校园网限速下用多线程 Range 断点续传；Windows 后台进程必须 workers=0
  （详见失败记录）

### 第二层：诊断层 —— "怎么知道模型状态好不好"

- 驱动问题："怎么确认有没有过拟合？""学习率怎么才算合适？"
- 收获：过拟合判定需要 val loss 拐点 + mAP 平台期**双证据**，再以 best.pt 做 train/val 对比定量
  （差距 >10 点才算真过拟合）；检测任务因训练集有增强，val loss 低于 train loss 属正常，
  判过拟合看走势分化而非绝对高低
- 关键转变：从"看一个数"升级为"看一组相互印证的证据"

### 第三层：方法层 —— "怎么科学地迭代"

- 驱动问题："调试总不能手调梯度和偏导吧？""每次优化都重训一遍岂不是很慢？"
- 收获：① 实践调的是四类旋钮——数据/优化超参/模型容量与输入分辨率/后处理阈值
  （后者零训练成本）；② 消融实验纪律：一次只变一个变量，每轮实验先写明"要回答什么问题"；
  ③ 成本控制：200 张×10 epoch 探针先验证方向（5~10 分钟），方向对才上全量，
  patience 早停自动止损
- 关键转变：从"盲目试错"升级为"诊断驱动的假设验证循环"

### 第四层：原理层 —— "为什么这样设计"

- 驱动问题："模型的 CNN 架构、卷积核、参数量是什么样的？""损失函数和激活函数用了什么？"
  "SGD 和 Adam 有何优劣，这里适配有何区别？"
- 收获（全部基于对实际权重的实测，非文档转述）：YOLOv8n 共 3,157,200 参数
  （backbone 40%/neck 31%/head 28%），64 层卷积（39×3×3+25×1×1），SiLU 激活；
  损失 = CIoU（重叠+中心距+宽高比）+ DFL（框边距离的概率分布预测）+ BCE 三项分工；
  优化器 auto 规则按迭代数自动选择——冒烟测试用 AdamW（收敛快），正式训练用
  SGD+momentum（终点泛化好），一个项目内亲身体会了"Adam 赢前半程、SGD 赢终点线"
- 关键转变：从"会用 API"升级为"能解释每个设计决策的动机"

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
