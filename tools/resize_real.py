"""把真实场景帧缩小一份，供标注使用（标签是归一化的，缩放后无需改动）。"""
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC_IMG = ROOT / "raw" / "real" / "images"
SRC_LBL = ROOT / "raw" / "real" / "labels"
OUT = ROOT / "raw" / "real_small"
OUT_IMG = OUT / "images"
OUT_LBL = OUT / "labels"

TARGET_W = 960

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

n = 0
for img in sorted(SRC_IMG.glob("*.jpg")):
    frame = cv2.imread(str(img))
    if frame is None:
        print(f"[warn] 读不到 {img.name}")
        continue
    h, w = frame.shape[:2]
    if w > TARGET_W:
        scale = TARGET_W / w
        frame = cv2.resize(frame, (TARGET_W, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_IMG / img.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    n += 1

# 标签原样复制（归一化坐标与分辨率无关）
m = 0
for lbl in SRC_LBL.glob("*.txt"):
    shutil.copy2(lbl, OUT_LBL / lbl.name)
    m += 1

print(f"[ ok ] 缩放 {n} 张图 -> {OUT_IMG}  (宽 {TARGET_W}px)")
print(f"[ ok ] 复制 {m} 个标签 -> {OUT_LBL}")
if n:
    sample = next(OUT_IMG.glob("*.jpg"))
    h, w = cv2.imread(str(sample)).shape[:2]
    print(f"[info] 新尺寸示例: {w}x{h}")
