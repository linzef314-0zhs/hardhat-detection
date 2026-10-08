# 失败记录原始素材（今晚真实发生，供你改写进 README 第 7 节）

> 规则：README 里的失败记录必须是你自己的话。以下只是事实清单，你挑 3 条以上改写。

1. **Kaggle 下载被校园网限速到 0.03 MB/s**
   - 尝试：单连接直下 → 0.2 MB/s；8 线程 Range 并行 → 冲到 2 MB/s 后被 QoS 压回
   - 解决：重启下载进程换新连接 + 分块断点续传校验（按分块字节数校验而非计数器）
   - 附带 bug：续传后进度计数器只算新增字节，误报 "incomplete download"，改为逐块校验大小

2. **DataLoader worker 集体崩溃（Windows 特有）**
   - 现象：`DataLoader worker exited unexpectedly`
   - 原因：后台分离进程（DETACHED_PROCESS）没有控制台，spawn 出的 dataloader 子进程无法存活
   - 解决：`workers: 0`，代价是数据加载串行化、训练变慢约 20~30%

3. **AMP 混合精度校验失败**
   - 现象：AMP check 需要下载 yolo26n.pt，github.com 间歇性超时（校园网）
   - 后果：自动回退 FP32 训练，显存占用翻倍、速度下降
   - 解决：趁网络窗口手动 curl 下 yolo26n.pt 放到项目根目录，后续 run 的 AMP check 直接通过

4. **Ultralytics 输出路径嵌套 bug**
   - 现象：`project="runs/detect"` 相对路径被保存到 `runs/detect/runs/detect/<name>`
   - 解决：改为传绝对路径 `Path.cwd()/runs/detect`

5. **Git 误把 1.3GB 数据集加入暂存**
   - 现象：`git add -A` 卡死超时，10000 个 XML/图片全被 stage
   - 解决：清索引锁 + `.gitignore` 排除 data/runs/weights 后重提交（git 仓库最终只含 20 个代码文件）

6. **（数据本身的问题，可写进"数据工作"一节）**
   - person 类只有 751 个实例且社区公认标注不全；helmet:person = 25:1 严重长尾
   - 68.8% 的框面积小于图像 1%（小目标主导）

7. **消融B无声失效：optimizer=auto 静默忽略 lr0**（最有价值的一条，建议必写）
   - 现象：ablation_lr5e3 与 base 的结果逐位相同（best_epoch、mAP 四位小数全同）
   - 排查：同 seed + 结果完全相同 → 怀疑配置未生效 → 翻日志发现 auto 模式提示
     "ignoring lr0 and determining best optimizer automatically"
   - 根因：Ultralytics 的 optimizer=auto 会忽略用户传入的 lr0/momentum 自行决定
   - 修复：显式 optimizer: SGD 后重跑；无效行在 results.tsv 中标记 INVALID 保留审计痕迹
   - 教训：框架的"智能默认"会静默覆盖用户意图；消融实验必须先验证"变量真的变了"
