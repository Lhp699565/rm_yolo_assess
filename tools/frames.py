#!/usr/bin/env python
"""从视频里抽帧，用来攒数据集。

自己拍 / 录屏比赛视频都是最省事的数据来源：
找一个能拍到目标的角度，录 2~3 分钟，抽 200~300 帧，就够训一个能看的模型。

用法：
    python tools/frames.py --src match.mp4 --out raw/images
    python tools/frames.py --src videos/ --out raw/images --every 20 --max 400
    python tools/frames.py --src match.mp4 --out raw/images --resize 1280 --dedup
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def extract_one(
    video: Path, out_dir: Path, every: int, max_n: int, resize: int, dedup: bool,
    start: int, prefix: str, counter: list[int],
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[warn] 打不开 {video}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    saved = 0
    idx = start
    prev_small = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % every == 0:
            if resize and frame.shape[1] > resize:
                scale = resize / frame.shape[1]
                frame = cv2.resize(frame, (resize, int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)

            # 抽帧去重：相邻帧几乎一样的话没必要都存
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
            if dedup and prev_small is not None:
                diff = float(cv2.absdiff(small, prev_small).mean())
                if diff < 2.0:
                    idx += 1
                    continue
            prev_small = small

            name = f"{prefix}_{counter[0]:05d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            counter[0] += 1
            saved += 1
            if saved % 50 == 0:
                print(f"  已抽 {saved} 帧 ...")
            if max_n and saved >= max_n:
                break

        idx += 1

    cap.release()
    print(f"[ ok ] {video.name}: 共 {total_frames} 帧，抽取 {saved} 张")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="视频抽帧")
    ap.add_argument("--src", required=True, help="视频文件或目录")
    ap.add_argument("--out", required=True, help="输出目录，如 raw/images")
    ap.add_argument("--every", type=int, default=15, help="每 N 帧抽一张")
    ap.add_argument("--max", type=int, default=0, help="每个视频最多抽多少张，0 不限")
    ap.add_argument("--resize", type=int, default=1280, help="长边缩到这个宽度，0 表示不缩放")
    ap.add_argument("--dedup", action="store_true", help="跳过几乎重复的相邻帧")
    ap.add_argument("--start", type=int, default=0, help="从第几帧开始")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VID_EXT)
    else:
        videos = [src]

    if not videos:
        raise SystemExit(f"{src} 下没有视频文件")

    counter = [0]
    total = 0
    for i, video in enumerate(videos):
        print(f"[{i + 1}/{len(videos)}] {video}")
        total += extract_one(
            video, out_dir, args.every, args.max, args.resize, args.dedup,
            args.start, f"f{i:02d}", counter,
        )

    print("\n" + "=" * 46)
    print(f"共抽 {total} 帧 -> {out_dir}")
    print("下一步：用 labelImg / X-AnyLabeling 逐张标注，标签存成同名 .txt")
    print("=" * 46)


if __name__ == "__main__":
    main()
