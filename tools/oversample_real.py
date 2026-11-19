"""把真实训练帧复制多份，提升它在总训练数据里的占比。

为什么需要
----------
第一轮域适配（v0.5）失败，诊断结果是欠拟合：
  真实训练帧 mAP50-95 = 0.2678（模型在自己训练过的图上都学不会）
原因不是方法错，是配比错——80 张真实图只占 4080 张训练数据的 2%，
梯度里真实信号的比重太低，推不动 314 万参数的模型。

标准做法：过采样少数域。把 80 张复制 N 份（改文件名），
占比从 2% 提到 ~19%，让每个 batch 里都能见到真实数据。

注意：验证集 real_val 保持原样（30 张，不复制、不参与训练），
否则评估就失去意义。

用法：
    python tools/oversample_real.py --times 12
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "datasets" / "armor_v2"
SRC_IMG = BASE / "images" / "real_train"
SRC_LBL = BASE / "labels" / "real_train"


def main() -> None:
    ap = argparse.ArgumentParser(description="过采样真实训练帧")
    ap.add_argument("--times", type=int, default=12, help="复制份数")
    args = ap.parse_args()

    imgs = sorted(SRC_IMG.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"没找到真实训练帧：{SRC_IMG}")

    dst_img = BASE / "images" / f"real_train_x{args.times}"
    dst_lbl = BASE / "labels" / f"real_train_x{args.times}"
    if dst_img.exists():
        shutil.rmtree(dst_img)
    if dst_lbl.exists():
        shutil.rmtree(dst_lbl)
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in imgs:
        for k in range(args.times):
            new_name = f"{img.stem}_c{k:02d}"
            shutil.copy2(img, dst_img / f"{new_name}.jpg")
            lbl = SRC_LBL / f"{img.stem}.txt"
            target = dst_lbl / f"{new_name}.txt"
            if lbl.exists():
                shutil.copy2(lbl, target)
            else:
                target.write_text("", encoding="utf-8")
            n += 1

    print(f"[ ok ] {len(imgs)} 张 × {args.times} = {n} 张 -> {dst_img}")

    # 生成对应的 data.yaml
    render = ROOT / "datasets" / "armor" / "images"
    yaml_path = ROOT / "configs" / "armor_v2_oversample.yaml"
    yaml_path.write_text(
        f"# 混合域 + 真实数据过采样 {args.times}x（由 tools/oversample_real.py 生成）\n"
        f"path: {BASE.as_posix()}\n"
        "train:\n"
        f"  - {render.as_posix()}/train\n"
        f"  - images/real_train_x{args.times}\n"
        "val:\n"
        f"  - {render.as_posix()}/val\n"
        "  - images/real_val\n"
        "test:\n"
        f"  - {render.as_posix()}/test\n"
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
    total = 4000 + n
    print(f"[info] 训练集总数约 {total} 张，真实数据占比 {n / total * 100:.1f}%")


if __name__ == "__main__":
    main()
