#!/usr/bin/env python
"""验证/评估入口 —— 输出可写进报告的指标表。

用法：
    python val.py --weights runs/train/exp_xxx/weights/best.pt
    python val.py --weights ... --split test

产物：
    runs/val/<name>/                 ultralytics 的混淆矩阵、PR 曲线等
    runs/val/<name>/metrics.json     本次评估的指标（含每类）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="评估训练好的模型")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "configs" / "armor.yaml"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.001, help="评估阈值，一般保持默认")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=0,
                    help="数据加载进程数。受限环境（沙箱）里必须为 0，否则 CreateFile 命名管道被拒")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if args.device == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    name = args.name or f"val_{datetime.now():%Y%m%d_%H%M%S}"
    model = YOLO(args.weights)

    # 和 train.py 用同一套路径解析，避免换个目录执行就找不到数据集
    sys.path.insert(0, str(ROOT))
    from train import prepare_data_config

    data_path = prepare_data_config(args.data)

    m = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        conf=args.conf,
        iou=args.iou,
        workers=args.workers,
        project=str(ROOT / "runs" / "val"),
        name=name,
        plots=True,
        verbose=False,
    )

    # 每类指标：ap_class_index 是实际出现在验证集里的类别下标
    per_class = {}
    try:
        for cls_idx, ap in zip(m.box.ap_class_index.tolist(), m.box.maps.tolist()):
            per_class[m.names[int(cls_idx)]] = round(float(ap), 4)
    except Exception:  # noqa: BLE001
        per_class = {}

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "split": args.split,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "mAP75": round(float(m.box.map75), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class_mAP50_95": per_class,
        "speed_ms": {k: round(v, 2) for k, v in (m.speed or {}).items()},
    }

    save_dir = Path(m.save_dir)
    (save_dir / "metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 56)
    print(f"split        : {args.split}")
    print(f"mAP@0.5      : {record['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 : {record['mAP50_95']:.4f}")
    print(f"mAP@0.75     : {record['mAP75']:.4f}")
    print(f"precision    : {record['precision']:.4f}")
    print(f"recall       : {record['recall']:.4f}")
    if per_class:
        print("-" * 56)
        for k, v in per_class.items():
            print(f"  {k:<14} mAP50-95 = {v:.4f}")
    print("-" * 56)
    print(f"产物目录     : {save_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
