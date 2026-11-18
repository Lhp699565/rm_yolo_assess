#!/usr/bin/env python
"""生成一个合成小数据集，用来验证整条流水线能跑通。

!!! 这不是考核数据 !!!
真实数据还没标完时，先用它确认 train / val / detect 三件套没坏，
免得标完 300 张才发现脚本报错。

用法：
    python tools/make_smoke.py                       # 生成 40 张到 datasets/smoke
    python tools/make_smoke.py --n 80 --imgsz 416

生成后：
    python train.py --data datasets/smoke/smoke.yaml --epochs 3 --imgsz 320 \
        --batch 8 --model weights/yolov8n.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_one(rng: random.Random, size: int) -> tuple[np.ndarray, list[tuple]]:
    """画一张随机背景 + 1~2 个目标矩形。返回 (图, [(cls, cx, cy, w, h)])。"""
    # 随机背景：底色 + 噪声，避免网络只学到「背景是纯色」
    base = np.array([rng.randint(20, 90) for _ in range(3)], dtype=np.uint8)
    img = np.full((size, size, 3), base, dtype=np.uint8)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 12, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 类别 0 = 蓝（BGR），类别 1 = 红（BGR）
    palette = [(200, 60, 20), (20, 40, 200)]
    boxes = []
    for _ in range(rng.randint(1, 2)):
        cls_id = rng.randint(0, 1)
        w = rng.randint(size // 8, size // 3)
        h = rng.randint(size // 8, size // 3)
        x = rng.randint(4, size - w - 4)
        y = rng.randint(4, size - h - 4)

        color = palette[cls_id]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        # 加一点内部纹理，别让模型只学「找色块」
        cv2.rectangle(img, (x + w // 6, y + h // 6), (x + 5 * w // 6, y + 5 * h // 6),
                      tuple(int(c * 0.45) for c in color), -1)

        boxes.append((cls_id, (x + w / 2) / size, (y + h / 2) / size, w / size, h / size))
    return img, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成小数据集（仅用于流水线自测）")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "smoke"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    n_val = max(2, int(args.n * args.val_ratio))
    splits = {"train": args.n - n_val, "val": n_val}

    total_boxes = 0
    for split, count in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, boxes = make_one(rng, args.imgsz)
            name = f"{split}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            total_boxes += len(boxes)

    yaml_path = out / "smoke.yaml"
    yaml_path.write_text(
        "# 合成数据集，仅用于验证流水线\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names:\n"
        "  0: blue_armor\n"
        "  1: red_armor\n",
        encoding="utf-8",
    )

    print(f"[ ok ] 合成数据集：{out}")
    print(f"       train {splits['train']} 张 / val {splits['val']} 张，共 {total_boxes} 个框")
    print(f"       配置：{yaml_path}")
    print("\n下一步自测：")
    print(f"  python train.py --data {yaml_path} --epochs 3 --imgsz 320 --batch 8")


if __name__ == "__main__":
    main()
