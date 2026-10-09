# RESUME — 项目交接备忘录（新会话开场必读）

更新时间：2026-10-09 15:05

## 项目是什么
Hard Hat Detection（Kaggle andrewmvd/hard-hat-detection，5000 张，helmet/head/person）
YOLOv8n 全流程投名状项目。**截止：今天 2026-10-09 24:00**。
用户 ML 理论扎实、实践少；评审看重：可复现、真实过程、失败记录、人工错误分析、学习路径分层。
项目根目录：D:\KimiData\kimi\tasks\2026-10-08\20-14-45-56fe26c6\hardhat-detection

## 环境
- 解释器: C:\Users\什亭之匣\.conda\envs\torch314\python.exe（torch 2.14.0+cu130, ultralytics 8.4.155）
- GPU: RTX 5060 Laptop 8.5GB；训练规则：后台独立进程 + workers=0 + 用 python.exe（pythonw 会崩）
- 数据: data/yolo（seed=42，4000/500/500，25502 框）

## 已完成（全部）
- 四组实验（best_epoch / mAP50 / mAP50-95 / train-val gap）：
  - base: 47 / 0.6480 / 0.4248 / 0.063（AdamW auto lr=0.001429）
  - ablation_no_mosaic: 49 / 0.6427 / 0.4211 / 0.089（mosaic 起正则作用）
  - ablation_lr_half: 47 / 0.6449 / 0.4244 / 0.102（lr 减半几乎不敏感；gap 越线但非病态过拟合）
  - ablation_imgsz960: 50(顶格) / 0.6439 / 0.4244 / 0.035（960 无收益且未收敛足；person 仍崩 → 瓶颈在标注不在分辨率）
- 旧 ablation_lr5e3 已在 results.tsv 标记 INVALID（optimizer=auto 静默忽略 lr0 的坑，failure_facts 第 7 条）
- 测试集评估（best.pt, 500 张）：mAP50=0.6233, mAP50-95=0.4203, P=0.952/R=0.580,
  各类 AP50: helmet 0.951 / head 0.898 / person 0.021 → runs/analysis/demo/test_metrics.txt
- 预测叠加图 12 张：runs/analysis/demo/test_pred/
- CNN 特征图 4 张：runs/analysis/feature_maps/（L1 边缘→L9 语义，教学素材）
- bad case：runs/analysis/base/badcases/（FP=63 FN=40 CLS=4 LOC=2，conf=0.25）
- 诊断图每组 5 张 + conclusion.txt 在 runs/analysis/<name>/
- README 第 2/3/4/5/6/8/9 节已由 AI 填入全部真实数据并定稿（结果总表含 INVALID 行、
  测试集终评表、过拟合/lr 诊断、错误分析量化部分、局限与下一步、复现命令已核对脚本参数）
- 第 6 节失效模式留了">（待补）"占位等用户手写；第 7 节失败记录仍是空表等用户改写
- git 12 次提交

## 待办（剩余约 2 小时工作量，截止 24:00）
1. 【用户】翻 badcases FP/FN 文件夹，写 3~5 条失效模式（"我认为是 X，因为 Y"）
   填入 README 第 6 节占位处（runs/analysis/base/badcases/，summary: FP=63 FN=40 CLS=4 LOC=2）
2. 【用户】对三个判定表态：① 过拟合（README 第 4 节已按"拟合差异加大但非病态过拟合"写入，
   认可即可）② lr ±50% 稳健（第 5 节）③ 960 排除分辨率假设（第 3 节备注）
3. 【可选】100 epoch 加长跑验证欠拟合判定（约 2h 挂机；若跑，22:00 前必须启动）
4. 【用户】改写第 7 节失败记录为自己口吻（素材 logs/failure_facts.md 共 7 条，挑 3 条以上；
   第 7.5 节如需也可润色）
5. 【AI】git 最终整理 + 交付

## 用户风格备忘
- 喜欢追问原理，回答可带教学性但要落地到他的数据
- 表达偶有错别字/语音输入，理解意图即可
- 重视"自己的话"和真实学习轨迹，README 里不要 AI 腔
