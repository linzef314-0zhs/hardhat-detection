# AGENTS.md — 项目硬约束（所有改动必须遵守）

## 项目目标
Hard Hat Detection（5000 张，helmet/head/person 三类）YOLOv8n 目标检测全流程项目，
交付物为可复现代码 + 实验记录 + README，用于团队评审。

## 硬约束
1. 训练一律以后台独立进程运行，日志写 `logs/`，禁止在前台阻塞会话。
2. OOM 自动降级：batch 8→4→2，重试并在 results.tsv 的 note 列记录；仍失败则 imgsz 640→512。
3. 每轮训练结束必须向 `logs/results.tsv` 追加一行（配置、指标、note）。
4. **所有评估与结论一律基于 `best.pt` 与 `best_epoch`，禁止用 `last.pt` 出结论**；
   `analyze.py` 必须断言 best.pt 存在且 best_epoch < epochs。
5. 禁止 ensemble / TTA / pseudo-labeling；禁止私自下载其他数据混入训练。
6. 数据划分固定 seed=42，划分文件生成后不许重新随机划分（防止 val 泄漏进 train）。
7. 消融实验每次只变一个变量，对照 `configs/base.yaml`。
8. 结论必须附证据（曲线图 / 数值），禁止空口判断。

## 环境
- Python: C:\Users\什亭之匣\.conda\envs\torch314\python.exe（torch 2.14.0+cu130, ultralytics 8.4.155）
- GPU: RTX 5060 Laptop 8.5GB；imgsz=640, batch≤8, AMP 开
