# RESUME — 项目交接备忘录（新会话开场必读）

更新时间：2026-10-09 16:55

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

## 项目已交付（2026-10-09 17:30 收工）

- README 全文定稿：12 张配图（docs/images/）、5 条失效模式（用户人工读图）、
  第 7 节详记 2/7/1 + 略记 4 条、base_long100 推翻欠拟合假设
- 新增 00-导读.md（项目说明书/文件夹地图）+ docs/git_history.txt（git 历史导出）
- 桌面交付两份：C:\Users\什亭之匣\Desktop\hardhat-detection（完整版 3.9G，无 .git）、
  hardhat-detection-评审版（59MB 轻量发送版，含评审版说明.md）
- git 22 次提交，工作区干净
- 遗留可选：用户若改第 7 节口吻，直接在 README 编辑后补一次 commit 即可

## 15:08~16:55 进展（base_long100 已完成）
- base_long100：66 轮早停（best_epoch=56），mAP50=0.6461 / mAP50-95=0.4248 与 base 持平，
  gap=0.043 → **推翻"欠拟合"假设，50 epoch 已收敛**；README 第 4 节已改写为完整反转记录
  （含 cosine 尾巴未走完的诚实备注）
- 已修复 feature_maps.py 中文乱码（雅黑字体），analyze/check_data 加同款保险；其余图全英文无乱码
- 镜像图来源已查实：原始 Kaggle 包自带（水印都镜像），非我方 pipeline
- git 16 次提交

## 用户风格备忘
- 喜欢追问原理，回答可带教学性但要落地到他的数据
- 表达偶有错别字/语音输入，理解意图即可
- 重视"自己的话"和真实学习轨迹，README 里不要 AI 腔
