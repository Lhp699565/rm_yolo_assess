"""构建"渲染图 + 真实帧"混合数据集，用于域适配微调。

设计
----
不复制那 4000 张渲染图（6 GB），而是在 data.yaml 里直接用绝对路径
指过去；只把 110 张真实帧复制过来（约 11 MB）。

真实帧按文件名（= 时间顺序）切分：
    前 80 张 -> real_train
    后 30 张 -> real_val   （留作域适配效果的客观评估，不参与训练）

这样"真实数据的验证集"是时间上靠后的片段，和训练片段不重叠，
比随机切分更可信（相邻帧长得几乎一样，随机切会泄漏）。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "raw" / "real_small"
OUT = ROOT / "datasets" / "armor_v2"
RENDER = ROOT / "datasets" / "armor"

N_TRAIN = 80


def main() -> None:
    imgs = sorted((SRC / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"没找到真实帧：{SRC / 'images'}")
    print(f"[info] 真实帧 {len(imgs)} 张")

    train_imgs, val_imgs = imgs[:N_TRAIN], imgs[N_TRAIN:]
    print(f"[info] real_train {len(train_imgs)} 张 / real_val {len(val_imgs)} 张")

    for split, items in (("real_train", train_imgs), ("real_val", val_imgs)):
        idir = OUT / "images" / split
        ldir = OUT / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        n_box = 0
        for img in items:
            shutil.copy2(img, idir / img.name)
            lbl = SRC / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, ldir / lbl.name)
                n_box += len([ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()])
            else:
                (ldir / f"{img.stem}.txt").write_text("", encoding="utf-8")
        print(f"[ ok ] {split}: {len(items)} 张, {n_box} 个框")

    # data.yaml：渲染集用绝对路径引用，真实帧用相对路径
    yaml_path = ROOT / "configs" / "armor_v2.yaml"
    render_img = (RENDER / "images").as_posix()
    yaml_path.write_text(
        "# 混合域数据集：渲染图（4000+500） + 真实场景帧（80 训 / 30 验）\n"
        "# 由 tools/build_mixed_dataset.py 生成\n"
        f"path: {(OUT).as_posix()}\n"
        "train:\n"
        f"  - {render_img}/train\n"
        "  - images/real_train\n"
        "val:\n"
        f"  - {render_img}/val\n"
        "  - images/real_val\n"
        "test:\n"
        f"  - {render_img}/test\n"
        "nc: 6\n"
        "names:\n"
        "  0: armor_2\n"
        "  1: armor_3\n"
        "  2: armor_4\n"
        "  3: sentry\n"
        "  4: outpost\n"
        "  5: base\n",
        encoding="utf-8",
    )
    print(f"[ ok ] 配置写入 {yaml_path}")
    print()
    print(yaml_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
