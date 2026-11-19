"""从真实场景视频里抽帧，并用当前模型生成预标注，给人工修正用。

目的
----
训练集全是 Blender 渲染图，真实场景有域差距。用真实视频抽一批帧、
人工修正后混入训练集，是最有效的改进手段。

为什么要预标注
--------------
从零画 100 张图的框要两三个小时。用当前模型先跑一遍，把框和类别填进去，
人工只需要"改错的、补漏的、删多的"，通常能省 70% 时间。

注意：模型在这个域上类别判错严重，所以**类别基本都要改**，但框的位置
大体可用。别盲信预标注，它只是省了"从空白开始画"。

用法
----
    python tools/make_real_frames.py
    python tools/make_real_frames.py --video "C:\\path\\to\\video.mp4" --n 100 --conf 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="抽真实场景帧并生成预标注")
    ap.add_argument("--video", default=r"C:\Users\Lenovo\Downloads\Screenrecording_20261119_134925.mp4")
    ap.add_argument("--out", default=str(ROOT / "raw" / "real"))
    ap.add_argument("--n", type=int, default=110, help="最多抽多少张")
    ap.add_argument("--every", type=int, default=25, help="每 N 帧取一个候选")
    ap.add_argument("--max-second", type=float, default=100.0,
                    help="只取前 N 秒（后面可能有网页 UI，会引入误检）")
    ap.add_argument("--conf", type=float, default=0.15, help="预标注置信度阈值")
    ap.add_argument("--weights", default=str(ROOT / "runs/train/v030_baseline/weights/best.pt"))
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"视频不存在：{video}")

    out = Path(args.out)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(fps * args.max_second)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] {total} 帧，只取前 {limit} 帧（{args.max_second:.0f} 秒）")

    # ---------- 1. 抽帧 + 去重 ----------
    saved = []
    prev = None
    idx = 0
    while idx < min(limit, total) and len(saved) < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
            if prev is None or float(np.abs(small - prev).mean()) > 3.0:
                name = f"real_{len(saved):04d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(name)
                prev = small
        idx += 1
    cap.release()
    print(f"[ ok ] 抽帧完成：{len(saved)} 张 -> {img_dir}")

    # ---------- 2. 预标注 ----------
    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"[info] 用 {args.weights} 生成预标注（conf={args.conf}）...")

    n_box = 0
    n_empty = 0
    for name in saved:
        img_path = img_dir / f"{name}.jpg"
        r = model.predict(str(img_path), conf=args.conf, imgsz=640, device=0, verbose=False)[0]
        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            h, w = r.orig_shape
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_dir / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        n_box += len(lines)
        if not lines:
            n_empty += 1

    print(f"[ ok ] 预标注完成：{n_box} 个框，{n_empty} 张空标签")
    print()
    print("=" * 56)
    print("  下一步：人工修正预标注")
    print("=" * 56)
    print(f"  图片目录: {img_dir}")
    print(f"  标签目录: {lbl_dir}")
    print()
    print("  打开标注工具后：")
    print("    1. 逐个检查框的位置（大体可用，可能偏大偏小）")
    print("    2. **类别基本都要改**（模型在这个域上判错严重）")
    print("    3. 漏掉的装甲板补框，误检的删掉")
    print("    4. 空标签文件保留（当负样本）")
    print()


if __name__ == "__main__":
    main()
