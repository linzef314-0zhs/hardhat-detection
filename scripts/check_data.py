"""check_data.py — 数据体检：5 项关键统计，输出到 logs/data_check.txt

用法: python scripts/check_data.py --data data/yolo
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["helmet", "head", "person"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--log", default="logs/data_check.txt")
    args = ap.parse_args()
    root = Path(args.data)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    all_areas, all_counts, cls_counter = [], [], Counter()
    oob = 0
    for split in ("train", "val", "test"):
        imgs = list((root / "images" / split).glob("*"))
        lbls = list((root / "labels" / split).glob("*.txt"))
        emit(f"[{split}] 图片 {len(imgs)} / 标注 {len(lbls)}")
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        missing_lbl = img_stems - lbl_stems
        if missing_lbl:
            emit(f"  ⚠️ {len(missing_lbl)} 张图片无标注文件: {sorted(missing_lbl)[:5]}...")
        split_counts = []
        for lp in lbls:
            lines = lp.read_text().strip().splitlines()
            split_counts.append(len(lines))
            for ln in lines:
                parts = ln.split()
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                cls_counter[cid] += 1
                all_areas.append(bw * bh)
                split_counts and all_counts.append(0)
                # 越界检测（归一化坐标应全在 [0,1]）
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    oob += 1
        if split_counts:
            emit(f"  每图框数: min={min(split_counts)} max={max(split_counts)} "
                 f"avg={sum(split_counts)/len(split_counts):.1f}")
    emit()
    emit("== 类别实例分布 ==")
    for cid, cname in enumerate(CLASSES):
        emit(f"  {cname}: {cls_counter.get(cid, 0)}")
    emit(f"\n越界/非法框: {oob}")
    if all_areas:
        sa = sorted(all_areas)
        emit(f"框面积占比(归一化): p50={sa[len(sa)//2]:.4f} p90={sa[int(len(sa)*0.9)]:.4f} "
             f"小目标(<0.01): {sum(1 for a in all_areas if a < 0.01)/len(all_areas)*100:.1f}%")

    # 尺寸分布图
    Path("runs/analysis").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(all_areas, bins=60, color="#4C72B0")
    plt.xlabel("bbox area (normalized)"); plt.ylabel("count")
    plt.title("BBox area distribution"); plt.tight_layout()
    plt.savefig("runs/analysis/bbox_area_dist.png", dpi=120)
    emit("\n尺寸分布图: runs/analysis/bbox_area_dist.png")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text("\n".join(out_lines), encoding="utf-8")
    emit(f"体检报告已写入 {args.log}")


if __name__ == "__main__":
    main()
