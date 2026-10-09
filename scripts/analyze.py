"""analyze.py — 过拟合/学习率诊断：出图 + 判定结论

用法: python scripts/analyze.py --run runs/detect/base
产出（runs/analysis/<name>/）:
  loss_curves.png / map_curves.png / lr_schedule.png /
  train_vs_val_map.png / per_class_ap.png / conclusion.txt
硬规则（AGENTS.md 第 4 条）：所有结论基于 best.pt 与 best_epoch。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# 中文字体保险：即使图注含中文也不乱码
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd
import yaml

CLASSES = ["helmet", "head", "person"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/detect/<name>")
    args = ap.parse_args()
    run = Path(args.run)
    name = run.name
    outdir = Path("runs/analysis") / name
    outdir.mkdir(parents=True, exist_ok=True)

    best_pt = run / "weights" / "best.pt"
    assert best_pt.exists(), f"违反硬约束: best.pt 不存在于 {run}"
    csv_path = run / "results.csv"
    assert csv_path.exists(), f"results.csv 不存在: {run}"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    best_idx = df["metrics/mAP50-95(B)"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    n_epochs = int(df["epoch"].max())

    conclusion = []
    def say(s):
        print(s)
        conclusion.append(s)

    say(f"# 诊断结论 — {name}")
    say(f"实际训练 {n_epochs} epoch，best_epoch={best_epoch}（依据 metrics/mAP50-95(B) 最大值）")

    # ---- 1. loss 曲线 + 拐点 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tr, va, ttl) in zip(axes, [
        ("train/box_loss", "val/box_loss", "box_loss"),
        ("train/cls_loss", "val/cls_loss", "cls_loss")]):
        ax.plot(df["epoch"], df[tr], label="train")
        ax.plot(df["epoch"], df[va], label="val")
        ax.axvline(best_epoch, color="red", ls="--", alpha=0.7, label=f"best_epoch={best_epoch}")
        ax.set_title(ttl); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "loss_curves.png", dpi=120); plt.close(fig)

    # 过拟合判定：best_epoch 之后 val loss 是否持续回升
    tail = df[df["epoch"] > best_epoch]
    overfit_verdicts = []
    for col in ("val/box_loss", "val/cls_loss"):
        if len(tail) >= 5:
            rise = tail[col].iloc[-1] - tail[col].iloc[0]
            rel = rise / max(abs(tail[col].iloc[0]), 1e-9)
            if rel > 0.10:
                overfit_verdicts.append(f"{col} 在 best_epoch 后回升 {rel*100:.0f}%（过拟合证据）")
    if overfit_verdicts:
        say("## 过拟合判定: 存在过拟合迹象")
        for v in overfit_verdicts:
            say(f"- {v}")
        say("- 对策建议: ① 以 best.pt 为准（已自动满足）② 加大 weight_decay ③ 提前 close_mosaic ④ 增加数据")
    else:
        say("## 过拟合判定: 50 epoch 内未见明显过拟合（val loss 未持续回升）")
        say("- 说明: 训练更多受欠拟合/收敛限制，可考虑增加 epoch 或模型容量")

    # ---- 2. mAP 曲线 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    ax.axvline(best_epoch, color="red", ls="--", alpha=0.7)
    ax.set_xlabel("epoch"); ax.set_title("mAP curves"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "map_curves.png", dpi=120); plt.close(fig)

    # ---- 3. 学习率曲线 ----
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    if lr_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in lr_cols:
            ax.plot(df["epoch"], df[c], label=c)
        ax.set_xlabel("epoch"); ax.set_title("LR schedule"); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / "lr_schedule.png", dpi=120); plt.close(fig)
        pg0 = lr_cols[0]
        say(f"## 学习率: 峰值 {df[pg0].max():.2e} → 末轮 {df[pg0].iloc[-1]:.2e}"
            f"（{'cosine 衰减正常' if df[pg0].iloc[-1] < df[pg0].max()*0.05 else '⚠️ 末轮 lr 偏高，衰减不充分'}）")

    # ---- 4. best.pt 在 train vs val 上对比（过拟合定量） ----
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    data_yaml = yaml.safe_load(open(run / "args.yaml", encoding="utf-8"))["data"] \
        if (run / "args.yaml").exists() else "data/yolo/data.yaml"
    if isinstance(data_yaml, dict):
        data_yaml = "data/yolo/data.yaml"
    val_res = model.val(data=data_yaml, split="val", plots=False, verbose=False)
    train_res = model.val(data=data_yaml, split="train", plots=False, verbose=False)
    gap = float(train_res.box.map50) - float(val_res.box.map50)
    say(f"## train/val 对比（best.pt）: train mAP50={train_res.box.map50:.4f} "
        f"val mAP50={val_res.box.map50:.4f} 差距={gap:.4f}"
        f"（{'>0.10 显著过拟合' if gap > 0.10 else '≤0.10 正常范围'}）")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["train", "val"], [float(train_res.box.map50), float(val_res.box.map50)],
           color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("mAP50"); ax.set_title("best.pt: train vs val")
    fig.tight_layout(); fig.savefig(outdir / "train_vs_val_map.png", dpi=120); plt.close(fig)

    # ---- 5. 每类 AP ----
    ap50 = val_res.box.ap50  # ndarray per class
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASSES[: len(ap50)], [float(x) for x in ap50], color="#55A868")
    ax.set_ylabel("AP50"); ax.set_title(f"per-class AP50 (val, best.pt)")
    fig.tight_layout(); fig.savefig(outdir / "per_class_ap.png", dpi=120); plt.close(fig)
    say("## 各类别 AP50: " + ", ".join(f"{CLASSES[i]}={float(v):.3f}" for i, v in enumerate(ap50)))
    worst = int(pd.Series([float(v) for v in ap50]).idxmin())
    say(f"- 最弱类别: {CLASSES[worst]}，建议检查其实例数与标注质量（已知问题: person 类标注不全）")

    (outdir / "conclusion.txt").write_text("\n".join(conclusion), encoding="utf-8")
    metrics = {"name": name, "best_epoch": best_epoch, "epochs_run": n_epochs,
               "val_mAP50": float(val_res.box.map50), "val_mAP50_95": float(val_res.box.map),
               "train_mAP50": float(train_res.box.map50), "gap": gap,
               "per_class_ap50": {CLASSES[i]: float(v) for i, v in enumerate(ap50)}}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"\n产物已输出到 {outdir}")


if __name__ == "__main__":
    main()
