"""feature_maps.py — 把 CNN 的"所见"画出来：各层特征图可视化

用法: python scripts/feature_maps.py --img data/yolo/images/test/<任意一张.png>
产出: runs/analysis/feature_maps/<层名>_fmap.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2
    import numpy as np

    model = YOLO("runs/detect/base/weights/best.pt")
    net = model.model.model  # 内部 nn.Sequential

    img0 = cv2.imread(args.img)
    img = cv2.resize(img0, (640, 640))[:, :, ::-1] / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    feats = {}
    hooks = []
    # 选几个代表层：浅层(边缘纹理)、中层、P3/P4/P5 输出前
    pick = {1: "L1_浅层边缘", 4: "L4_中浅层", 6: "L6_中层语义", 9: "L9_深层SPPF"}
    for idx, name in pick.items():
        hooks.append(net[idx].register_forward_hook(
            lambda m, i, o, n=name: feats.__setitem__(n, o)))

    with torch.no_grad():
        _ = model.model(x)  # DetectionModel 自带跨层路由（Concat/跳跃连接）
    for h in hooks:
        h.remove()

    outdir = Path("runs/analysis/feature_maps")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, f in feats.items():
        f = f[0]  # [C, H, W]
        # 通道均值热度图：该层整体"关注哪里"
        heat = f.mean(0).numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")
        im = axes[1].imshow(heat, cmap="viridis")
        axes[1].set_title(f"{name}  shape={list(f.shape)}")
        axes[1].axis("off"); fig.colorbar(im, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"{name}: {list(f.shape)} -> {outdir/name}.png")


if __name__ == "__main__":
    main()
