#!/usr/bin/env python
"""用一张图生成一段"镜头平移"的演示视频，用来验证视频检测链路。

为什么需要它
------------
考核演示通常要一段视频。如果摄像头打不开（很多笔记本有 Fn 摄像头开关 /
设备管理器显示正常但枚举不到），可以用这个脚本先把视频链路跑通：

    一张 1440x1080 的渲染图
        -> 裁一个 1280x720 的窗口，按正弦轨迹在上面平移
        -> 得到 250 帧 mp4，画面像摄像机在缓慢扫视
        -> 交给 detect.py，逐帧检测、画框、统计 FPS

它生成的是**模拟镜头运动**，不是真实拍摄。用于验证代码没问题可以，
正式答辩建议还是用手机拍一段真实场景。

用法
----
    python tools/make_demo_video.py
    python tools/make_demo_video.py --image datasets/armor/images/test/000037_view_targets.png ^
        --out raw/demo_pan.mp4 --seconds 10 --fps 25
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="生成镜头平移演示视频")
    ap.add_argument("--image", default=None, help="底图，默认自动挑测试集里目标最多的一张")
    ap.add_argument("--out", default=str(ROOT / "raw" / "demo_pan.mp4"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    # ---- 选底图：优先挑标注目标最多的那张 ----
    img_path = args.image
    if img_path is None:
        lbl_dir = ROOT / "datasets" / "armor" / "labels" / "test"
        img_dir = ROOT / "datasets" / "armor" / "images" / "test"
        best, best_n = None, -1
        for lbl in lbl_dir.glob("*.txt"):
            n = len([ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()])
            if n > best_n:
                best, best_n = lbl.stem, n
        if best is None:
            raise SystemExit(f"{lbl_dir} 里没有标签文件，先导入数据集")
        img_path = str(img_dir / f"{best}.png")
        print(f"[info] 自动选中目标最多的图：{best}（{best_n} 个标注）")

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"读不到图片：{img_path}（路径含中文时 OpenCV 会静默失败）")

    H, W = img.shape[:2]
    cw, ch = args.width, args.height
    if cw > W or ch > H:
        raise SystemExit(f"底图 {W}x{H} 比裁剪窗口 {cw}x{ch} 还小，请调小 --width/--height")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cw, ch)
    )
    if not writer.isOpened():
        raise SystemExit(f"无法创建视频写入器：{out_path}")

    # 平移范围：留出窗口宽高，让中心点在剩余空间里走正弦轨迹
    span_x = W - cw
    span_y = H - ch

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)          # 0 -> 1
        # 两个不同频率的正弦，路径不重复，看起来像慢速扫视
        x = int(span_x * (0.5 - 0.5 * math.cos(2 * math.pi * t)))
        y = int(span_y * (0.5 - 0.5 * math.cos(2 * math.pi * t * 0.7 + 1.0)))
        frame = img[y:y + ch, x:x + cw]
        writer.write(frame)

    writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ ok ] {out_path}  {cw}x{ch}  {args.fps}fps  {n_frames} 帧  {size_mb:.1f} MB")
    print()
    print("下一步跑检测：")
    print(f"  python detect.py --weights runs\\train\\v030_baseline\\weights\\best.pt "
          f"--source {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
