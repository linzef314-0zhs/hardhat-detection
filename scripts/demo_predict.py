"""demo_predict.py — 用 best.pt 在测试集上推理并保存可视化图 + 测试集评估

用法: python scripts/demo_predict.py
产出: runs/analysis/demo/test_pred/*.jpg（预测框叠加图）
      runs/analysis/demo/test_metrics.txt（测试集指标）
"""
from pathlib import Path

import cv2

CLASSES = ["helmet", "head", "person"]
COLORS = [(0, 200, 0), (0, 140, 255), (255, 60, 60)]


def main():
    from ultralytics import YOLO
    model = YOLO("runs/detect/base/weights/best.pt")
    outdir = Path("runs/analysis/demo/test_pred")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 测试集整体评估（从未参与训练和调参的 500 张）
    res = model.val(data="data/yolo/data.yaml", split="test", plots=True, verbose=False)
    lines = [
        f"测试集(500张, best.pt):",
        f"  mAP50    = {res.box.map50:.4f}",
        f"  mAP50-95 = {res.box.map:.4f}",
        f"  precision= {res.box.mp:.4f}  recall = {res.box.mr:.4f}",
        "  各类别 AP50: " + ", ".join(
            f"{CLASSES[i]}={float(v):.3f}" for i, v in enumerate(res.box.ap50)),
    ]
    print("\n".join(lines))
    Path("runs/analysis/demo").mkdir(parents=True, exist_ok=True)
    Path("runs/analysis/demo/test_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    # 2) 挑 12 张测试图，画预测框（绿色=helmet 橙=head 红=person）
    imgs = sorted((Path("data/yolo/images/test")).glob("*.png"))[::40][:12]
    for ip in imgs:
        img = cv2.imread(str(ip))
        pred = model.predict(str(ip), conf=0.25, verbose=False)[0]
        for b in pred.boxes:
            c = int(b.cls[0]); x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2)
            cv2.putText(img, f"{CLASSES[c]} {conf:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1)
        cv2.imwrite(str(outdir / ip.name), img)
    print(f"预测叠加图已输出 {len(imgs)} 张到 {outdir}")


if __name__ == "__main__":
    main()
