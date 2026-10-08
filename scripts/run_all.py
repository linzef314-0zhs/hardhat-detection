"""run_all.py — 夜间挂机链路：基线 → 消融A → 消融B → 逐个分析

用法（后台）: python scripts/run_all.py
日志: logs/run_all.log
"""
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("logs/run_all.log")

STEPS = [
    ("configs/base.yaml", "runs/detect/base"),
    ("configs/ablation_no_mosaic.yaml", "runs/detect/ablation_no_mosaic"),
    ("configs/ablation_lr5e3.yaml", "runs/detect/ablation_lr5e3"),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log("STDOUT tail:\n" + (r.stdout or "")[-2000:])
        log("STDERR tail:\n" + (r.stderr or "")[-2000:])
        raise SystemExit(f"命令失败: {' '.join(cmd)}")
    return r


def main():
    py = sys.executable
    for cfg, run_dir in STEPS:
        if (Path(run_dir) / "weights" / "best.pt").exists():
            log(f"跳过已完成: {run_dir}")
        else:
            log(f"=== 开始训练 {cfg} ===")
            t0 = time.time()
            sh([py, "scripts/train.py", "--config", cfg])
            log(f"=== 完成 {cfg}，耗时 {(time.time()-t0)/60:.1f} min ===")
        log(f"=== 分析 {run_dir} ===")
        sh([py, "scripts/analyze.py", "--run", run_dir])
        if cfg.endswith("base.yaml"):
            log("=== 生成 bad case 叠加图 ===")
            sh([py, "scripts/gen_badcases.py", "--run", run_dir])
    log("ALL DONE — 全部实验与分析完成")


if __name__ == "__main__":
    main()
