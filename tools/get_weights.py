#!/usr/bin/env python
"""预训练权重下载器（GitHub 直连不通时的国内镜像方案）。

ultralytics 默认从 github.com/ultralytics/assets 拉权重，国内经常超时。
本脚本按顺序尝试几个镜像，下载完成后再交给 YOLO 加载。

用法：
    python tools/get_weights.py yolov8n.pt
    python tools/get_weights.py yolov8s.pt yolo11n.pt
    python tools/get_weights.py --list
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"

# 每个镜像都套同一个 GitHub 地址，tag 对应 ultralytics 的权重发布版本
MIRRORS = [
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/{tag}/{name}",
    "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/{tag}/{name}",
    "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/{tag}/{name}",
    "https://github.com/ultralytics/assets/releases/download/{tag}/{name}",
]
TAG = "v8.3.0"

KNOWN = {
    "yolov8n.pt": "6.2 MB  最轻，CPU 也能训",
    "yolov8s.pt": "22 MB   小模型，精度更高一点",
    "yolov8m.pt": "50 MB   中等模型，需要显存",
    "yolo11n.pt": "5.4 MB  v11 的 nano 版",
    "yolo11s.pt": "19 MB   v11 的 small 版",
}


def download(name: str, out_dir: Path = WEIGHTS_DIR, tag: str = TAG, timeout: int = 60) -> Path:
    out = out_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[skip] 已存在 {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out

    for mirror in MIRRORS:
        url = mirror.format(tag=tag, name=name)
        tmp = out.with_suffix(out.suffix + ".part")
        try:
            print(f"[try ] {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                with open(tmp, "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            print(f"\r        {got / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="", flush=True)
            print()
            # 校验：长度必须与 Content-Length 一致，且是 zip 魔数（.pt 本质是 zip）
            # 只判 "> 1MB" 不够 —— 镜像断流时会给一个截断文件，照样超过 1MB
            size = tmp.stat().st_size
            if total and size != total:
                raise OSError(f"下载不完整：{size} / {total} bytes")
            if size < 1_000_000:
                raise OSError(f"下载文件过小：{size} bytes")
            with open(tmp, "rb") as fh:
                if fh.read(4) != b"PK\x03\x04":
                    raise OSError("文件头不是 zip，镜像大概返回了错误页")
            tmp.replace(out)
            print(f"[ ok ] {out}")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"\n[fail] {type(exc).__name__}: {str(exc)[:100]}")
            tmp.unlink(missing_ok=True)

    raise SystemExit(
        f"所有镜像均失败。请手动下载 {name} 放到 {out_dir}/，或让队友传一份。"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 YOLO 预训练权重")
    ap.add_argument("names", nargs="*", default=["yolov8n.pt"], help="权重文件名")
    ap.add_argument("--out", default=str(WEIGHTS_DIR))
    ap.add_argument("--list", action="store_true", help="列出常用权重后退出")
    args = ap.parse_args()

    if args.list:
        print(f"{'文件名':<14}{'大小':<10}说明")
        for k, v in KNOWN.items():
            size, _, desc = v.partition(" ")
            print(f"{k:<14}{size:<10}{desc.strip()}")
        return

    for name in args.names:
        download(name, Path(args.out))


if __name__ == "__main__":
    main()
