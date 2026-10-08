"""gen_badcases.py — 自动生成 bad case 叠加图，供人工错误分析归类

用法: python scripts/gen_badcases.py --run runs/detect/base --conf 0.25 --n 60
输出: runs/analysis/<name>/badcases/{FP,FN,CLS,LOC}/<img>.jpg
分类规则（IoU≥0.5 匹配）:
  FP  : 预测框无 GT 匹配
  FN  : GT 框无预测匹配
  CLS : 匹配上但类别不一致
  LOC : 匹配且同类但 0.3≤IoU<0.5（定位偏差）
每张图同时画 GT（实线）与预测（虚线框+置信度），文件名标注错误类型计数。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["helmet", "head", "person"]
GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 80, 255)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_gt(label_path, w, h):
    boxes = []
    if label_path.exists():
        for ln in label_path.read_text().strip().splitlines():
            cid, xc, yc, bw, bh = ln.split()[:5]
            cid, xc, yc, bw, bh = int(cid), float(xc), float(yc), float(bw), float(bh)
            boxes.append((cid, [(xc - bw / 2) * w, (yc - bh / 2) * h,
                                (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    run = Path(args.run)
    best_pt = run / "weights" / "best.pt"
    assert best_pt.exists(), "必须使用 best.pt"

    from ultralytics import YOLO
    model = YOLO(str(best_pt))

    data_root = Path("data/yolo")
    imgs = sorted((data_root / "images" / "val").glob("*.png")) + \
           sorted((data_root / "images" / "val").glob("*.jpg"))

    outbase = Path("runs/analysis") / run.name / "badcases"
    for cat in ("FP", "FN", "CLS", "LOC"):
        (outbase / cat).mkdir(parents=True, exist_ok=True)

    stats = {"FP": 0, "FN": 0, "CLS": 0, "LOC": 0}
    saved = 0
    for ip in imgs:
        if saved >= args.n:
            break
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gt = load_gt(data_root / "labels" / "val" / (ip.stem + ".txt"), w, h)
        pred = model.predict(str(ip), conf=args.conf, verbose=False)[0]
        pboxes = [(int(b.cls[0]), b.xyxy[0].tolist(), float(b.conf[0])) for b in pred.boxes]

        matched_gt, matched_pred = set(), set()
        img_err = {"FP": 0, "FN": 0, "CLS": 0, "LOC": 0}
        pairs = []
        for pi, (pc, pb, pconf) in enumerate(pboxes):
            best_iou, best_gi = 0, -1
            for gi, (gc, gb) in enumerate(gt):
                if gi in matched_gt:
                    continue
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi); matched_pred.add(pi)
                if pc != gt[best_gi][0]:
                    img_err["CLS"] += 1; pairs.append(("CLS", pc, pb, pconf))
            elif best_iou >= 0.3 and best_gi >= 0:
                matched_gt.add(best_gi); matched_pred.add(pi)
                if pc == gt[best_gi][0]:
                    img_err["LOC"] += 1; pairs.append(("LOC", pc, pb, pconf))
                else:
                    img_err["CLS"] += 1; pairs.append(("CLS", pc, pb, pconf))
            else:
                img_err["FP"] += 1; pairs.append(("FP", pc, pb, pconf))
        img_err["FN"] += len(gt) - len(matched_gt)
        for k in stats:
            stats[k] += img_err[k]
        if sum(img_err.values()) == 0:
            continue

        for gc, gb in gt:
            cv2.rectangle(img, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), GT_COLOR, 2)
            cv2.putText(img, "GT:" + CLASSES[gc], (int(gb[0]), max(int(gb[1]) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1)
        for tag, pc, pb, pconf in pairs:
            cv2.rectangle(img, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), PRED_COLOR, 1)
            cv2.putText(img, f"{tag}:{CLASSES[pc]} {pconf:.2f}",
                        (int(pb[0]), min(int(pb[3]) + 12, h - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1)
        main_err = max(img_err, key=img_err.get)
        cv2.imwrite(str(outbase / main_err / f"{main_err}{img_err[main_err]}_{ip.stem}.jpg"), img)
        saved += 1

    summary = ", ".join(f"{k}={v}" for k, v in stats.items())
    print(f"错误统计(val 全集, conf={args.conf}): {summary}")
    print(f"叠加图已输出 {saved} 张到 {outbase}")
    (outbase / "summary.txt").write_text(
        f"conf={args.conf}\n{summary}\n图片数={saved}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
