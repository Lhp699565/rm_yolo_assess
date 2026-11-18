#!/usr/bin/env python
"""数据集工具：划分数据集 + 体检。

两个子命令
-----------
split  把一堆平铺的「图片 + 同名 .txt 标签」按比例分到 images|labels/{train,val,test}
check  体检：标签格式、类别越界、坐标越界、孤儿文件、train/val 泄漏、类别不平衡

用法
----
    # 划分（默认 8:1:1）
    python tools/prepare.py split --src raw --out datasets/armor

    # 体检（强烈建议训练前先跑）
    python tools/prepare.py check --root datasets/armor --nc 2

为什么必须先体检
----------------
YOLO 训练时对坏标签的处理是「静默跳过」，你只会看到 mAP 上不去，
查半天查不出原因。10 秒钟跑一遍 check 能省两小时。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- split


def cmd_split(args: argparse.Namespace) -> int:
    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        print(f"[error] 源目录不存在：{src}")
        return 1

    img_root = src / "images" if (src / "images").is_dir() else src
    lbl_root = src / "labels" if (src / "labels").is_dir() else img_root

    images = sorted(p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        print(f"[error] {img_root} 下没找到图片")
        return 1

    pairs, missing = [], []
    for img in images:
        rel = img.relative_to(img_root)
        cand = [lbl_root / rel.with_suffix(".txt"), img.with_suffix(".txt")]
        lbl = next((c for c in cand if c.exists()), None)
        if lbl is None:
            missing.append(img.name)
        else:
            pairs.append((img, lbl))

    print(f"[info] 找到 {len(images)} 张图，{len(pairs)} 张有标签")
    if missing:
        print(f"[warn] {len(missing)} 张图没有标签（当作纯背景图，或先补标）：")
        for name in missing[:10]:
            print(f"        {name}")
        if len(missing) > 10:
            print(f"        ... 另外 {len(missing) - 10} 张")

    if not pairs:
        print("[error] 没有任何「图片 + 标签」配对，先标注再划分")
        return 1

    random.Random(args.seed).shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.ratio[0])
    n_val = int(n * args.ratio[1])
    buckets = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    used: set[str] = set()
    for split, items in buckets.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            stem = img.stem
            if stem in used:  # 不同子目录里的同名图，加序号避免覆盖
                stem = f"{stem}_{len(used):04d}"
            used.add(stem)
            shutil.copy2(img, img_dir / f"{stem}{img.suffix.lower()}")
            shutil.copy2(lbl, lbl_dir / f"{stem}.txt")

    total = sum(len(v) for v in buckets.values())
    print("\n" + "=" * 46)
    for split in SPLITS:
        c = len(buckets[split])
        print(f"  {split:<6} {c:>5} 张  ({c / total * 100:5.1f}%)")
    print(f"  {'合计':<5} {total:>5} 张")
    print("=" * 46)

    info = {
        "source": str(src),
        "output": str(out),
        "seed": args.seed,
        "ratio": args.ratio,
        "counts": {s: len(buckets[s]) for s in SPLITS},
        "images_without_label": missing,
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ ok ] 已写入 {out}/dataset_info.json")
    print("[next] python tools/prepare.py check --root " + str(out) + " --nc <你的类别数>")
    return 0


# ----------------------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"[error] 数据集目录不存在：{root}")
        return 1

    errors: list[str] = []
    warns: list[str] = []
    split_stems: dict[str, set[str]] = defaultdict(set)
    per_split_stats: dict[str, dict] = {}
    class_counter: Counter = Counter()

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}

        n_boxes, n_empty = 0, 0
        for stem, lbl in labels.items():
            if stem not in images:
                errors.append(f"[{split}] 孤儿标签（没有对应图片）：{lbl.name}")

            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines()]
            lines = [ln for ln in lines if ln]
            if not lines:
                n_empty += 1
                continue

            for lineno, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"[{split}] {lbl.name}:{lineno} 字段数={len(parts)}，应为 5")
                    continue
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    errors.append(f"[{split}] {lbl.name}:{lineno} 含非数字：{line[:40]}")
                    continue

                if not 0 <= cls_id < args.nc:
                    errors.append(f"[{split}] {lbl.name}:{lineno} 类别 id={cls_id} 超出 0..{args.nc - 1}")
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    errors.append(f"[{split}] {lbl.name}:{lineno} 坐标未归一化到 0~1：{line[:40]}")
                if w <= 0 or h <= 0:
                    errors.append(f"[{split}] {lbl.name}:{lineno} 宽或高 <= 0")
                class_counter[cls_id] += 1
                n_boxes += 1

        n_no_label = len(set(images) - set(labels))
        split_stems[split] = set(images)

        per_split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": n_boxes,
            "empty_label_files": n_empty,
            "images_without_label": n_no_label,
        }
        if n_no_label:
            warns.append(f"[{split}] {n_no_label} 张图没有标签（会被当成背景图）")

    # 泄漏检查：同一张图既在 train 又在 val，指标会虚高
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            dup = split_stems[SPLITS[a]] & split_stems[SPLITS[b]]
            if dup:
                errors.append(
                    f"数据泄漏：{len(dup)} 个文件同时出现在 {SPLITS[a]} 和 {SPLITS[b]}"
                    f"（例：{sorted(dup)[:3]}）"
                )

    # 类别不平衡
    if len(class_counter) < args.nc:
        missing_cls = sorted(set(range(args.nc)) - set(class_counter))
        warns.append(f"以下类别在数据集里一个实例都没有：{missing_cls}")
    if class_counter:
        lo = min(class_counter.values())
        hi = max(class_counter.values())
        if lo and hi / lo > args.imbalance:
            warns.append(f"类别不平衡，最多/最少 = {hi / lo:.1f}x（阈值 {args.imbalance}）")

    # ---- 输出 ----
    print("\n" + "=" * 58)
    print(f"数据集体检：{root}")
    print("=" * 58)
    print(f"{'split':<8}{'图片':>7}{'标签':>7}{'框':>8}{'空标签':>8}{'缺标签':>8}")
    for split in SPLITS:
        s = per_split_stats.get(split)
        if not s:
            continue
        print(f"{split:<8}{s['images']:>7}{s['labels']:>7}{s['boxes']:>8}"
              f"{s['empty_label_files']:>8}{s['images_without_label']:>8}")
    print("-" * 58)
    total_img = sum(s["images"] for s in per_split_stats.values())
    total_box = sum(s["boxes"] for s in per_split_stats.values())
    print(f"总图片 {total_img}   总框 {total_box}")
    if total_img:
        print(f"平均每张 {total_box / total_img:.2f} 个目标")
    if class_counter:
        print("每类实例数：")
        for cls_id, cnt in sorted(class_counter.items()):
            print(f"  类别 {cls_id}: {cnt:>6}  ({cnt / total_box * 100:5.1f}%)")

    if errors:
        print(f"\n[ERROR] {len(errors)} 个错误，必须修：")
        for e in errors[:25]:
            print(f"  x {e}")
        if len(errors) > 25:
            print(f"  ... 另外 {len(errors) - 25} 条")
    if warns:
        print(f"\n[WARN] {len(warns)} 个警告：")
        for w in warns:
            print(f"  ! {w}")
    if not errors and not warns:
        print("\n[PASS] 数据集没问题，可以开训")
    elif not errors:
        print("\n[PASS] 没有致命错误，可以开训（警告建议看一眼）")
    print("=" * 58)

    return 0 if not errors or not args.strict else 1


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集划分与体检")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="把数据集划分成 train/val/test")
    sp.add_argument("--src", required=True, help="原始目录（含 images/ 和 labels/，或平铺）")
    sp.add_argument("--out", required=True, help="输出目录，如 datasets/armor")
    sp.add_argument("--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_split)

    cp = sub.add_parser("check", help="体检数据集")
    cp.add_argument("--root", required=True)
    cp.add_argument("--nc", type=int, required=True, help="类别数，和 data.yaml 里一致")
    cp.add_argument("--imbalance", type=float, default=5.0, help="类别不平衡告警倍数")
    cp.add_argument("--strict", action="store_true", help="有错误时返回非 0")
    cp.set_defaults(func=cmd_check)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
