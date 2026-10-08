"""convert_voc_to_yolo.py — PASCAL VOC XML → Ultralytics YOLO 格式转换 + 固定划分

用法: python scripts/convert_voc_to_yolo.py --raw data/raw --out data/yolo
输入: data/raw/hardhat_full.zip 解压后的 images/ 与 annotations/
输出: data/yolo/{images,labels}/{train,val,test}/ 与 data/yolo/data.yaml

关键点:
- 类别映射固定 helmet=0, head=1, person=2（写入 data.yaml，与训练一致）
- 划分 seed=42 固定，生成 split_manifest.json 供审计，重复运行结果一致
- XML 中 unknown 类别不静默跳过，计数并打印警告
"""
import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

CLASSES = ["helmet", "head", "person"]
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def find_raw_dirs(raw: Path):
    """兼容 zip 解压后的各种嵌套结构，找到 images/ 与 annotations/。"""
    imgs = ann = None
    for p in raw.rglob("*"):
        if p.is_dir() and p.name.lower() in ("images", "jpegimages"):
            imgs = p
        if p.is_dir() and p.name.lower() in ("annotations",):
            ann = p
    if not imgs or not ann:
        raise SystemExit(f"未找到 images/annotations 目录，请检查 {raw} 的解压结构")
    return imgs, ann


def convert_box(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    xc = ((xmin + xmax) / 2) / w
    yc = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xc, yc, bw, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/yolo")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    # 若 zip 未解压则就地解压
    zips = list(raw.glob("*.zip"))
    imgs_dir = ann_dir = None
    try:
        imgs_dir, ann_dir = find_raw_dirs(raw)
    except SystemExit:
        if zips:
            print(f"解压 {zips[0].name} ...")
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(raw)
            imgs_dir, ann_dir = find_raw_dirs(raw)
        else:
            raise

    xmls = sorted(ann_dir.glob("*.xml"))
    assert xmls, f"{ann_dir} 中没有 xml 文件"
    print(f"共 {len(xmls)} 个标注文件")

    random.seed(SEED)
    order = xmls[:]
    random.shuffle(order)
    n = len(order)
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])
    splits = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }

    stats = {"unknown_class": {}, "empty_images": 0, "converted_boxes": 0}
    manifest = {k: [x.stem for x in v] for k, v in splits.items()}

    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml_path in files:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            w, h = int(size.find("width").text), int(size.find("height").text)
            img_name = root.find("filename").text
            img_src = imgs_dir / img_name
            if not img_src.exists():
                # 兼容文件名不一致
                cands = list(imgs_dir.glob(xml_path.stem + ".*"))
                assert cands, f"找不到图片 {img_name}"
                img_src = cands[0]
            lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name not in CLASSES:
                    stats["unknown_class"][name] = stats["unknown_class"].get(name, 0) + 1
                    continue
                bb = obj.find("bndbox")
                box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                xc, yc, bw, bh = convert_box((w, h), box)
                xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
                bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
                lines.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["converted_boxes"] += 1
            if not lines:
                stats["empty_images"] += 1
            dst_img = out / "images" / split / img_src.name
            shutil.copy2(img_src, dst_img)
            (out / "labels" / split / (xml_path.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_text = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: helmet\n  1: head\n  2: person\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(
        {"seed": SEED, "ratios": RATIOS, "splits": manifest}, indent=1), encoding="utf-8")

    print(f"划分: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"转换框数: {stats['converted_boxes']}, 无框图片: {stats['empty_images']}")
    if stats["unknown_class"]:
        print(f"⚠️ 未知类别（已跳过并计数）: {stats['unknown_class']}")
    print(f"data.yaml 已写入 {out/'data.yaml'}")


if __name__ == "__main__":
    main()
