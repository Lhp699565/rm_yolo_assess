#!/usr/bin/env python
"""给 labelImg 1.8.6 打 PyQt5 5.15 兼容补丁（幂等）。

问题
----
labelImg 1.8.6 是 2021 年的代码，作者当年假设 `QPointF.x()` 返回 int。
PyQt5 5.15 开始它返回 float，而 `QPainter.drawLine` / `drawText` 的
整数重载不接受 float，于是在鼠标画框的瞬间抛：

    TypeError: drawLine(...) argument 1 has unexpected type 'float'

表现就是：界面能打开、图能看，一画框立刻闪退。

补丁
----
1. libs/canvas.py: drawRect / drawLine 坐标显式 int() 转换
2. libs/shape.py : drawText 坐标显式 int() 转换

用法
----
    python tools/patch_labelimg.py
    python tools/patch_labelimg.py --src .tools/labelImg-src
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / ".tools" / "labelImg-src"

PATCHES = [
    (
        "libs/canvas.py",
        "            p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
        "            p.drawRect(int(left_top.x()), int(left_top.y()),\n"
        "                       int(rect_width), int(rect_height))",
    ),
    (
        "libs/canvas.py",
        "            p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())\n"
        "            p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
        "            px, py = int(self.prev_point.x()), int(self.prev_point.y())\n"
        "            p.drawLine(px, 0, px, self.pixmap.height())\n"
        "            p.drawLine(0, py, self.pixmap.width(), py)",
    ),
    (
        "libs/shape.py",
        "                    painter.drawText(min_x, min_y, self.label)",
        "                    painter.drawText(int(min_x), int(min_y), self.label)",
    ),
    # 滚轮滚动：units 是 float，QScrollBar.setValue 只接受 int
    (
        "labelImg.py",
        "        bar.setValue(bar.value() + bar.singleStep() * units)",
        "        bar.setValue(int(bar.value() + bar.singleStep() * units))",
    ),
    # 缩放：QSpinBox.setValue 只接受 int
    (
        "labelImg.py",
        "        self.zoom_widget.setValue(value)",
        "        self.zoom_widget.setValue(int(value))",
    ),
    # 缩放时调整滚动条位置，move_x/move_y 是 float
    (
        "labelImg.py",
        "        h_bar.setValue(new_h_bar_value)\n        v_bar.setValue(new_v_bar_value)",
        "        h_bar.setValue(int(new_h_bar_value))\n        v_bar.setValue(int(new_v_bar_value))",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="给 labelImg 打 PyQt5 兼容补丁")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()

    src = Path(args.src)
    if not (src / "labelImg.py").exists():
        raise SystemExit(f"没找到 labelImg 源码：{src}")

    changed = 0
    for rel, old, new in PATCHES:
        path = src / rel
        if not path.exists():
            print(f"[skip] {rel} 不存在")
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"[ ok ] {rel} 已打过补丁")
            continue
        if old not in text:
            print(f"[warn] {rel} 里找不到要改的代码，版本可能不同，跳过")
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"[fix ] {rel} 已修补")
        changed += 1

    print()
    if changed:
        print(f"完成，修补了 {changed} 处。重新打开 labelImg 即可。")
    else:
        print("没有需要修补的地方（可能已经打过补丁）。")


if __name__ == "__main__":
    main()
