"""run_queue2.py — 上午排队器：等消融B完成后自动接 imgsz960 消融

用法（后台）: python scripts/run_queue2.py
日志: logs/run_queue2.log
"""
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("logs/run_queue2.log")
WAIT_FOR = Path("runs/detect/ablation_lr_half/weights/best.pt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"命令失败: {' '.join(cmd)}")


def main():
    py = sys.executable
    log("等待消融B完成...")
    t0 = time.time()
    while not WAIT_FOR.exists():
        if time.time() - t0 > 3 * 3600:
            raise SystemExit("等待超时 3h，消融B可能失败")
        time.sleep(60)
    log(f"消融B已就绪，启动 imgsz960 消融")
    sh([py, "scripts/train.py", "--config", "configs/ablation_imgsz960.yaml"])
    sh([py, "scripts/analyze.py", "--run", "runs/detect/ablation_imgsz960"])
    log("QUEUE2 DONE — imgsz960 消融与分析完成")


if __name__ == "__main__":
    main()
