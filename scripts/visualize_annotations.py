"""visualize_annotations.py — 标注叠加抽检图（人工质检用）

用法: python scripts/visualize_annotations.py --data data/yolo --split train --n 50
输出: runs/analysis/label_check/<split>/*.jpg
"""
import argparse
import random
from pathlib import Path

import cv2

CLASSES = ["helmet", "head", "person"]
COLORS = [(0, 200, 0), (0, 140, 255), (255, 60, 60)]  # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="runs/analysis/label_check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    imgs = sorted((root / "images" / args.split).glob("*.png")) + \
           sorted((root / "images" / args.split).glob("*.jpg"))
    random.seed(args.seed)
    random.shuffle(imgs)
    imgs = imgs[: args.n]
    outdir = Path(args.out) / args.split
    outdir.mkdir(parents=True, exist_ok=True)

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print(f"跳过无法读取: {ip.name}")
            continue
        h, w = img.shape[:2]
        lp = root / "labels" / args.split / (ip.stem + ".txt")
        if lp.exists():
            for ln in lp.read_text().strip().splitlines():
                cid, xc, yc, bw, bh = ln.split()[:5]
                cid, xc, yc, bw, bh = int(cid), float(xc), float(yc), float(bw), float(bh)
                x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cid], 2)
                cv2.putText(img, CLASSES[cid], (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cid], 1)
        cv2.imwrite(str(outdir / ip.name), img)
    print(f"已输出 {len(imgs)} 张抽检图到 {outdir}")


if __name__ == "__main__":
    main()
