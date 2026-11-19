#!/usr/bin/env python
"""推理与可视化入口：图片 / 视频 / 摄像头。

考核要求的「矩形框检测」就在这里输出：
  - 每帧画 axis-aligned bbox + 类别名 + 置信度（自己用 OpenCV 画，不调 result.plot()）
  - 统计单帧推理耗时和 FPS
  - 把每帧检测结果写进 CSV，方便统计、复盘、写报告

用法：
    python detect.py --weights runs/train/exp_xxx/weights/best.pt --source test.jpg --show
    python detect.py --weights runs/train/exp_xxx/weights/best.pt --source match.mp4
    python detect.py --weights runs/train/exp_xxx/weights/best.pt --source 0        # 摄像头
    python detect.py --weights ... --source 0 --classes 1                           # 只看红装甲板
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [
    (255, 170, 0),   # BGR 蓝
    (0, 60, 255),    # BGR 红
    (0, 200, 0),
    (200, 0, 200),
    (0, 220, 220),
    (180, 120, 255),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO 推理 / 画框 / 统计")
    ap.add_argument("--weights", default=None, help="训练好的 best.pt（--list-cameras 时可省略）")
    ap.add_argument("--source", default="0", help="图片/视频路径，或摄像头序号（0）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="auto | 0 | cpu")
    ap.add_argument("--out", default=None, help="输出目录，默认 runs/detect/det_<时间>")
    ap.add_argument("--classes", default=None, help="只保留这些类别，逗号分隔")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限")
    ap.add_argument("--no-save", action="store_true", help="不落盘，只在窗口看")
    ap.add_argument("--show", action="store_true", help="显示窗口（摄像头默认开启）")
    ap.add_argument("--no-show", action="store_true", help="强制不显示窗口")
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--backend", default="any", choices=["any", "msmf", "dshow"],
                    help="摄像头后端。Windows 上 dshow 找不到设备时换 msmf 试试")
    ap.add_argument("--list-cameras", action="store_true",
                    help="只探测有哪些可用摄像头，不做检测")
    return ap.parse_args()


BACKENDS = {"any": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}


def list_cameras() -> None:
    """把所有后端 × 索引 0~4 都试一遍，打印能用的组合。

    Windows 上同一个摄像头在不同后端下的可用性经常不一样：
    dshow 枚举不到、msmf 能开，或者反过来。OpenCV 默认只试一个，
    失败就报 "Camera index out of range"，看着像没有摄像头，其实只是后端不对。
    """
    try:
        cv2.setLogLevel(0)  # 关掉 FFMPEG/dshow 的噪音日志
    except Exception:  # noqa: BLE001
        pass

    found = []
    for backend_name in ("msmf", "dshow", "any"):
        for idx in range(5):
            cap = cv2.VideoCapture(idx, BACKENDS[backend_name])
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"  [OK] 后端={backend_name:<6} 索引={idx}  {w}x{h}")
                        found.append((backend_name, idx))
            finally:
                cap.release()

    print()
    if found:
        backend_name, idx = found[0]
        print(f"  建议命令：--source {idx} --backend {backend_name}")
    else:
        print("  没找到任何可用摄像头。可选方案：")
        print("    1) 手机装 Iriun Webcam / DroidCam，把手机当电脑摄像头（零成本，最快）")
        print("    2) 用手机录一段视频传到电脑，改成 --source 视频路径.mp4")
        print("    3) 检查：设置 → 蓝牙和其他设备 → 摄像头；设备管理器 → 照相机")
        print("    4) 笔记本有些型号有物理摄像头开关/遮挡片，确认已打开")


# ----------------------------------------------------------------------------- 推理


def infer(model, frame: np.ndarray, args, device, cls_filter):
    """返回 (检测结果列表, 推理耗时 ms)。"""
    t0 = time.perf_counter()
    result = model.predict(
        frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        classes=cls_filter,
        verbose=False,
    )[0]
    infer_ms = (time.perf_counter() - t0) * 1000.0

    dets: list[dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            dets.append(
                {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "conf": float(conf), "cls": int(cls_id),
                }
            )
    return dets, infer_ms


# ----------------------------------------------------------------------------- 画图


def draw_boxes(frame: np.ndarray, dets: list[dict], names: dict, line_width: int) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        color = COLORS[d["cls"] % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

        label = f"{names.get(d['cls'], d['cls'])} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 6), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return frame


# ----------------------------------------------------------------------------- 主流程


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        print("正在探测摄像头（每个后端试 0~4 号索引）...\n")
        list_cameras()
        return

    if not args.weights:
        raise SystemExit("缺少 --weights，例如：\n"
                         "  python detect.py --weights runs/train/v030_baseline/weights/best.pt --source 0")

    import torch
    from ultralytics import YOLO

    if args.device == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = YOLO(args.weights)
    names = model.names
    cls_filter = [int(c) for c in args.classes.split(",")] if args.classes else None

    run_dir = (
        Path(args.out) if args.out
        else ROOT / "runs" / "detect" / f"det_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    src = args.source
    is_cam = src.isdigit()
    show = args.show or (is_cam and not args.no_show)

    print(f"[info] weights={args.weights}  device={device}  conf={args.conf}")
    print(f"[info] source ={'摄像头 ' if is_cam else ' '}{src}   out={run_dir}")

    csv_path = run_dir / "detections.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["frame", "time_s", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2", "infer_ms"]
    )

    fps_hist: list[float] = []
    ms_hist: list[float] = []
    cls_counter: dict[str, int] = {}
    total_dets = 0
    frames_done = 0
    t_start = time.perf_counter()

    try:
        # ---------------- 单张图片 ----------------
        if not is_cam and Path(src).suffix.lower() in IMG_EXT:
            frame = cv2.imread(str(src))
            if frame is None:
                raise SystemExit(f"读不到图片：{src}")
            dets, infer_ms = infer(model, frame, args, device, cls_filter)
            draw_boxes(frame, dets, names, args.line_width)
            draw_hud(frame, [
                f"file: {Path(src).name}",
                f"infer: {infer_ms:.1f} ms",
                f"objects: {len(dets)}",
            ])
            for d in dets:
                name = names.get(d["cls"], d["cls"])
                cls_counter[name] = cls_counter.get(name, 0) + 1
                writer.writerow([0, 0.0, d["cls"], name, f"{d['conf']:.4f}",
                                 f"{d['x1']:.1f}", f"{d['y1']:.1f}", f"{d['x2']:.1f}", f"{d['y2']:.1f}",
                                 f"{infer_ms:.2f}"])
            total_dets = len(dets)
            frames_done = 1

            if not args.no_save:
                out_img = run_dir / f"{Path(src).stem}_pred.jpg"
                cv2.imwrite(str(out_img), frame)
                print(f"[ ok ] 结果图：{out_img}")
            if show and not args.no_show:
                cv2.imshow("detect", frame)
                cv2.waitKey(0)
            for d in dets:
                print(f"  {names.get(d['cls'], d['cls']):<12} conf={d['conf']:.3f} "
                      f"box=({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f})")

        # ---------------- 视频 / 摄像头 ----------------
        else:
            if is_cam:
                cap = cv2.VideoCapture(int(src), BACKENDS[args.backend])
            else:
                cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise SystemExit(
                    f"打不开视频源：{src}\n"
                    f"  摄像头：先跑 --list-cameras 看有没有可用设备，再试 --backend msmf\n"
                    f"  视频文件：确认路径是英文且文件存在"
                )

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[info] {width}x{height} @ {src_fps:.1f} fps")

            writer_video = None
            video_path = None
            if not args.no_save:
                video_path = run_dir / "pred.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_video = cv2.VideoWriter(str(video_path), fourcc, src_fps, (width, height))
                if not writer_video.isOpened():
                    video_path = run_dir / "pred.avi"
                    writer_video = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"XVID"), src_fps, (width, height)
                    )
                print(f"[info] 录制输出 -> {video_path}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                dets, infer_ms = infer(model, frame, args, device, cls_filter)
                ms_hist.append(infer_ms)
                fps_hist.append(1000.0 / infer_ms if infer_ms > 0 else 0.0)
                fps = float(np.mean(fps_hist[-30:]))

                draw_boxes(frame, dets, names, args.line_width)
                draw_hud(frame, [
                    f"FPS(infer): {fps:5.1f}",
                    f"infer: {infer_ms:5.1f} ms",
                    f"objects: {len(dets)}",
                ])

                ts = time.perf_counter() - t_start
                for d in dets:
                    name = names.get(d["cls"], d["cls"])
                    cls_counter[name] = cls_counter.get(name, 0) + 1
                    writer.writerow([frames_done, f"{ts:.2f}", d["cls"], name, f"{d['conf']:.4f}",
                                     f"{d['x1']:.1f}", f"{d['y1']:.1f}", f"{d['x2']:.1f}", f"{d['y2']:.1f}",
                                     f"{infer_ms:.2f}"])

                total_dets += len(dets)
                frames_done += 1

                if writer_video is not None:
                    writer_video.write(frame)

                if show and not args.no_show:
                    cv2.imshow("detect", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break

                if args.max_frames and frames_done >= args.max_frames:
                    break

            cap.release()
            if writer_video is not None:
                writer_video.release()
                print(f"[ ok ] 结果视频：{video_path}")
    finally:
        csv_file.close()
        if show and not args.no_show:
            cv2.destroyAllWindows()

    # ---------------- 汇总 ----------------
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 52)
    print(f"处理帧数     : {frames_done}")
    print(f"检测目标总数 : {total_dets}")
    if frames_done:
        print(f"平均端到端   : {frames_done / elapsed:.1f} FPS")
    if ms_hist:
        print(f"平均推理耗时 : {np.mean(ms_hist):.1f} ms   (P95 {np.percentile(ms_hist, 95):.1f} ms)")
    if cls_counter:
        print("-" * 52)
        for name, cnt in sorted(cls_counter.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<14} {cnt}")
    print(f"检测明细     : {csv_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()
