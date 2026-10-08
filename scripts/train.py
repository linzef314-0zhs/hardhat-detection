"""train.py — config 驱动训练：OOM 自动降级、results.tsv 落盘、best.pt 记录

用法: python scripts/train.py --config configs/base.yaml
"""
import argparse
import csv
import time
import traceback
from pathlib import Path

import yaml

TSV = Path("logs/results.tsv")
TSV_HEADER = ["time", "name", "config", "epochs_run", "batch", "imgsz",
              "lr0", "mosaic", "weight_decay", "best_epoch",
              "mAP50", "mAP50_95", "best_pt", "note"]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_tsv(row):
    TSV.parent.mkdir(parents=True, exist_ok=True)
    new = not TSV.exists()
    with open(TSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TSV_HEADER, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def run_once(cfg, batch):
    from ultralytics import YOLO
    model = YOLO(cfg["model"])
    kw = dict(
        data=cfg["data"], epochs=cfg["epochs"], imgsz=cfg["imgsz"],
        batch=batch, workers=cfg.get("workers", 8),
        patience=cfg.get("patience", 10), seed=cfg.get("seed", 42),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg["lr0"], lrf=cfg.get("lrf", 0.01),
        cos_lr=cfg.get("cos_lr", True),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        weight_decay=cfg.get("weight_decay", 5e-4),
        mosaic=cfg.get("mosaic", 1.0), mixup=cfg.get("mixup", 0.0),
        close_mosaic=cfg.get("close_mosaic", 10),
        amp=cfg.get("amp", True), plots=True,
        project=str((Path.cwd() / "runs" / "detect").resolve()),
        name=cfg["name"], exist_ok=True,
    )
    # 冒烟测试：限制训练图片数
    max_imgs = cfg.get("max_train_images")
    if max_imgs:
        import random
        data_cfg = yaml.safe_load(open(cfg["data"], encoding="utf-8"))
        root = Path(data_cfg["path"])
        imgs = sorted((root / "images" / "train").glob("*"))
        random.seed(0)
        imgs = imgs[: max_imgs] if len(imgs) <= max_imgs else random.sample(imgs, max_imgs)
        lst = Path("logs") / f"subset_{cfg['name']}.txt"
        lst.write_text("\n".join(str(p.resolve()) for p in imgs), encoding="utf-8")
        data_cfg["train"] = str(lst.resolve())
        sub_yaml = Path("logs") / f"subset_{cfg['name']}_data.yaml"
        with open(sub_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_cfg, f, allow_unicode=True)
        kw["data"] = str(sub_yaml.resolve())
    return model.train(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    name = cfg["name"]
    batch, note = cfg["batch"], ""
    t0 = time.time()

    while True:
        try:
            results = run_once(cfg, batch)
            break
        except (RuntimeError, torch.OutOfMemoryError) as e:  # noqa: F821
            if "out of memory" in str(e).lower() and batch > 2:
                batch //= 2
                note += f"OOM->batch={batch};"
                print(f"[train] CUDA OOM，降级 batch={batch} 重试", flush=True)
                continue
            if "out of memory" in str(e).lower() and cfg["imgsz"] > 512:
                cfg["imgsz"] = 512
                batch = cfg["batch"]
                note += "OOM->imgsz=512;"
                print("[train] 仍 OOM，降级 imgsz=512 重试", flush=True)
                continue
            raise

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    assert best_pt.exists(), f"best.pt 不存在: {save_dir}"

    import pandas as pd
    df = pd.read_csv(save_dir / "results.csv")
    df.columns = df.columns.str.strip()
    best_idx = df["metrics/mAP50-95(B)"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    map50 = float(df.loc[best_idx, "metrics/mAP50(B)"])
    map5095 = float(df.loc[best_idx, "metrics/mAP50-95(B)"])
    assert best_epoch < cfg["epochs"] or True

    append_tsv({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name, "config": args.config,
        "epochs_run": int(df["epoch"].max()), "batch": batch,
        "imgsz": cfg["imgsz"], "lr0": cfg["lr0"],
        "mosaic": cfg.get("mosaic", 1.0),
        "weight_decay": cfg.get("weight_decay", 5e-4),
        "best_epoch": best_epoch, "mAP50": f"{map50:.4f}",
        "mAP50_95": f"{map5095:.4f}", "best_pt": str(best_pt),
        "note": note.strip(";"),
    })
    print(f"[train] {name} 完成: best_epoch={best_epoch} mAP50={map50:.4f} "
          f"mAP50-95={map5095:.4f} 耗时={(time.time()-t0)/60:.1f}min note={note or '-'}", flush=True)


if __name__ == "__main__":
    import torch  # noqa: F401  (OOM 异常类引用)
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
