# RESUME — 项目交接备忘录（给下一个会话的 AI / 给明早的自己）

更新时间：2026-10-08 23:28

## 项目是什么
Hard Hat Detection（Kaggle andrewmvd/hard-hat-detection，5000 张，helmet/head/person）
YOLOv8n 全流程投名状项目。截止：2026-10-09 24:00。用户 ML 理论扎实、实践少，
评审看重：可复现、真实过程、失败记录、错误分析（人工）、学习路径。

## 环境
- 解释器: C:\Users\什亭之匣\.conda\envs\torch314\python.exe（torch 2.14.0+cu130, ultralytics 8.4.155）
- GPU: RTX 5060 Laptop 8.5GB；训练必须后台独立进程 + workers=0（DETACHED 下 dataloader 崩溃已踩过）
- 数据: data/yolo（已转换，seed=42，4000/500/500，25502 框，person 类仅 751 实例且标注不全）

## 当前进度
- ✅ 下载/转换/体检/抽检图/冒烟测试（mAP50=0.519）
- ✅ 基线 50 epoch 完成（末轮 mAP50=0.646/mAP50-95=0.424，best 见 results.tsv）
- 🏃 run_all.py 后台链路运行中：ablation_no_mosaic（23:14 启动）→ ablation_lr5e3 → 各自 analyze
  预计 01:20 前全部完成。查 logs\run_all.log 尾部出现 "ALL DONE" 即全完。
- 若链路中断：直接重跑 `python scripts/run_all.py`（已完成的实验会自动跳过）

## 明早三步
1. 读 runs\analysis\*\conclusion.txt + metrics.json，带用户下过拟合/学习率判定（用户拍板）
2. 用户人工看 runs\analysis\base\badcases\{FP,FN,CLS,LOC}，归类 3~5 个失效模式
3. 填 README：第 3 节结果总表（数据在 logs\results.tsv）、4/5 节诊断、6 节错误分析、
   7 节失败记录（素材在 logs\failure_facts.md，必须用户自己改写）、7.5 节口吻润色

## 关键纪律（AGENTS.md）
- 所有结论基于 best.pt/best_epoch，禁止 last.pt
- 失败记录与学习记录要用户自己的话，AI 只提供素材
- README 第 8 节局限已写明砍项原因，不许删
