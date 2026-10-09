# RESUME — 项目交接备忘录（给下一个会话的 AI / 给自己的恢复指南）

更新时间：2026-10-09 09:45

## 项目是什么
Hard Hat Detection（Kaggle andrewmvd/hard-hat-detection，5000 张，helmet/head/person）
YOLOv8n 全流程投名状项目。截止：2026-10-09 24:00。用户 ML 理论扎实、实践少。
评审看重：可复现、真实过程、失败记录、错误分析（人工）、学习路径分层。

## 环境
- 解释器: C:\Users\什亭之匣\.conda\envs\torch314\python.exe（torch 2.14.0+cu130, ultralytics 8.4.155）
- GPU: RTX 5060 Laptop 8.5GB；训练必须后台独立进程 + workers=0 + python.exe（不要用 pythonw，stdout=None 会崩）
- 数据: data/yolo（seed=42，4000/500/500，25502 框；person 仅 751 实例且标注不全 → AP50≈0.03 是数据问题）

## 当前进度（截至 09:45）
- ✅ 基线 base：best_epoch=47，mAP50=0.648，mAP50-95=0.425，train/val gap=0.063，未见明显过拟合（欠拟合方向）
- ✅ 消融A ablation_no_mosaic：mAP50=0.643，gap 拉宽到 0.089 → mosaic 起正则作用
- 🏃 消融B ablation_lr_half（AdamW lr0=0.000714，单变量修正版）：09:25 启动，约 10:30 完成+自动 analyze
- 🏃 消融C ablation_imgsz960（960px batch8）：由 scripts/run_queue2.py 排队，B 完成后自动接力，约 12:15 完
- ⚠️ 重要事实：optimizer=auto 实际选的是 AdamW(0.001429) 而非 SGD，且静默忽略手动 lr0。
  旧 ablation_lr5e3 已在 results.tsv 标记 INVALID。configs/ablation_lr5e3.yaml 已删除，以 ablation_lr_half.yaml 为准。
- bad case 已生成：runs/analysis/base/badcases/，conf=0.25 下 FP=63 FN=40 CLS=4 LOC=2

## 用户回来后三步（约 12:30）
1. 汇总四组结果（base / no_mosaic / lr_half / imgsz960）→ 用户下最终判定（过拟合、学习率、分辨率结论）
2. 用户人工看 badcases 归类 3~5 个失效模式（"我认为是 X，因为 Y"）
3. 填 README：第 3 节结果总表、4/5 节诊断、6 节错误分析；用户改写第 7 节失败记录
  （素材 logs/failure_facts.md，共 7 条，第 7 条 optimizer=auto 必写）和 7.5 节口吻 → 定稿交付

## 关键纪律（AGENTS.md）
- 所有结论基于 best.pt/best_epoch，禁止 last.pt
- 失败记录与学习记录要用户自己的话，AI 只提供素材
- README 第 8 节局限已写明砍项原因；若 imgsz960 成功则划掉"未覆盖 imgsz"那条
- 单变量对照纪律：所有消融必须显式对齐 AdamW(lr=0.001429, momentum=0.9)，只改目标变量
