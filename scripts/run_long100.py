"""run_long100.py — 后台启动 100 epoch 加长训练（验证欠拟合判定）

用法: python scripts/run_long100.py
日志: logs/run_long100.log
规则: DETACHED_PROCESS + workers=0（Windows 后台进程约束，见失败记录 2）
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable  # 用调用它的同一个解释器（torch314 的 python.exe，不用 pythonw）

def main():
    log_path = ROOT / "logs" / "run_long100.log"
    lf = open(log_path, "ab")
    p = subprocess.Popen(
        [PY, "scripts/train.py", "--config", "configs/base_long100.yaml"],
        cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(f"已后台启动 base_long100 训练, pid={p.pid}, 日志: {log_path}")

if __name__ == "__main__":
    main()
