#!/usr/bin/env python
"""把 YOLO Pose 数据集导成纯检测数据集（丢掉关键点）。

为什么需要
----------
魔搭上这个 RM 装甲板数据集（P1ne4pp1e/armor-pose-blender-allcls-5k-1109）的
标签是 YOLO Pose，每个目标一行 17 列：

    class cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    └──── 前 5 列就是检测框 ────┘  └────── 4 个关键点，考核用不到 ──────┘

考核只要求矩形框检测，所以取前 5 列即可。

为什么不重新随机划分
--------------------
数据集自带 train.txt / val.txt，是「按类别出现情况 + 采样类别分层」分好的。
Blender 渲染时同一场景会出多个视角，随机重分很容易让同一场景落进 train 和 val，
指标虚高（数据泄漏）。这里尊重原划分，只从 val 里切一部分当 test。

用法
----
    python tools/import_pose.py ^
        --src D:\\rm_datasets\\armor-pose ^
        --out datasets\\armor ^
        --rename "2=armor_2,3=armor_3,4=armor_4"

产物
----
    datasets/armor/
    ├── images/{train,val,test}/*.png
    ├── labels/{train,val,test}/*.txt     ← 已转成 5 列检测格式
    ├── data.yaml
    └── import_info.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(src: Path) -> dict[int, str]:
    """从 dataset.yaml 读类别名。"""
    cfg = src / "dataset.yaml"
    if not cfg.exists():
        return {}
    import yaml

    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    names = data.get("names") or {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def convert_label(text: str) -> tuple[str, int, int]:
    """17 列 pose -> 5 列 det。返回 (新内容, 保留目标数, 丢弃行数)。"""
    out, kept, dropped = [], 0, 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            dropped += 1
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            dropped += 1
            continue
        out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        kept += 1
    return ("\n".join(out) + "\n" if out else ""), kept, dropped


def read_split(src: Path, list_file: str) -> list[Path]:
    f = src / list_file
    if not f.exists():
        return []
    items = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        items.append(p if p.is_absolute() else (src / p))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO Pose -> 纯检测 数据集导入")
    ap.add_argument("--src", required=True, help="原始数据集根目录（含 images/ labels/ train.txt）")
    ap.add_argument("--out", required=True, help="输出到 datasets/armor 这种 YOLO 检测结构")
    ap.add_argument("--test-ratio", type=float, default=0.5,
                    help="从 val 里切多少比例做 test（0 表示不切）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rename", default="", help='改类别名，例 "2=armor_2,3=armor_3"')
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"源目录不存在：{src}")

    names = load_names(src)
    for pair in filter(None, (p.strip() for p in args.rename.split(","))):
        if "=" in pair:
            old, new = pair.split("=", 1)
            for k, v in list(names.items()):
                if v == old.strip():
                    names[k] = new.strip()

    train_items = read_split(src, "train.txt")
    val_items = read_split(src, "val.txt")
    if not train_items:
        raise SystemExit(f"{src} 里没读到 train.txt，确认下载完整了吗？")

    rng = random.Random(args.seed)
    rng.shuffle(val_items)
    n_test = int(len(val_items) * args.test_ratio)
    buckets = {
        "train": train_items,
        "val": val_items[n_test:],
        "test": val_items[:n_test],
    }

    stats = {}
    cls_counter: Counter = Counter()
    dropped_total = 0

    for split, items in buckets.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        kept_imgs = boxes = empty = missing_lbl = 0
        for img in items:
            if not img.exists() or img.suffix.lower() not in IMG_EXT:
                continue
            lbl = Path(str(img).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/"))
            lbl = lbl.with_suffix(".txt")

            new_text, kept, dropped = "", 0, 0
            if lbl.exists():
                new_text, kept, dropped = convert_label(lbl.read_text(encoding="utf-8", errors="ignore"))
            else:
                missing_lbl += 1

            dropped_total += dropped
            if kept == 0:
                empty += 1

            shutil.copy2(img, img_dir / img.name)
            (lbl_dir / f"{img.stem}.txt").write_text(new_text, encoding="utf-8")
            for line in new_text.splitlines():
                cls_counter[int(line.split()[0])] += 1
            kept_imgs += 1
            boxes += kept

        stats[split] = {
            "images": kept_imgs,
            "boxes": boxes,
            "empty_labels": empty,
            "missing_label_files": missing_lbl,
        }

    nc = (max(names) + 1) if names else 5
    names_block = "\n".join(f"  {i}: {n}" for i, n in sorted(names.items()))

    (out / "data.yaml").write_text(
        "# 由 tools/import_pose.py 生成\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )

    info = {
        "source": str(src),
        "test_ratio": args.test_ratio,
        "splits": stats,
        "classes": {str(k): v for k, v in sorted(names.items())},
        "class_instances": {str(k): v for k, v in sorted(cls_counter.items())},
        "dropped_keypoint_rows": dropped_total,
    }
    (out / "import_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 56)
    print(f"导入完成 -> {out}")
    print("=" * 56)
    for split in ("train", "val", "test"):
        s = stats.get(split)
        if s:
            print(f"  {split:<6} {s['images']:>5} 张  {s['boxes']:>6} 个框  "
                  f"空标签 {s['empty_labels']:>4}")
    total_box = sum(s["boxes"] for s in stats.values())
    print("-" * 56)
    print("每类实例数：")
    for k, v in sorted(cls_counter.items()):
        print(f"  {k} {names.get(k, '?'):<12} {v:>6}")
    if dropped_total:
        print(f"\n[note] 丢弃了 {dropped_total} 行格式不对的标签")
    print(f"\n数据集配置：{out / 'data.yaml'}")
    print("train.py 默认读 configs/armor.yaml，记得把里面 nc / names 改成上面这份")
    print("=" * 56)


if __name__ == "__main__":
    main()
