#!/usr/bin/env python
"""导出 ONNX —— 证明模型能脱离训练框架部署。

考核本身不要求，但「能导出 + 能用 OpenCV DNN 跑」是答辩加分项，
也是以后上车的必经步骤（车上一般不用 PyTorch 推理）。

用法：
    python export.py --weights runs/train/exp_xxx/weights/best.pt
    python export.py --weights ... --imgsz 640 --simplify
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 ONNX 模型")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--simplify", action="store_true", help="需要额外安装 onnxsim")
    ap.add_argument("--dynamic", action="store_true", help="动态 batch/尺寸")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
    print(f"[ ok ] 导出完成：{out}")

    # 用 onnxruntime 跑一遍，确认文件没坏
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        outs = sess.run(None, {inp.name: dummy})
        print(f"[ ok ] onnxruntime 自检通过，输入 {inp.shape} -> 输出 {[o.shape for o in outs]}")
    except ImportError:
        print("[warn] 未安装 onnxruntime，跳过自检（pip install onnxruntime）")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] onnxruntime 自检失败：{exc}")


if __name__ == "__main__":
    main()
