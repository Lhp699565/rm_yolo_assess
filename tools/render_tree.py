"""把 git 版本树渲染成一张 PNG，方便直接放进报告。

终端截图在报告里往往很糊，而且中文字体容易乱码。这里用 Pillow 按等宽字体
重绘一遍，深色背景 + 标签高亮，清晰度可控。

用法：
    python tools/render_tree.py
    python tools/render_tree.py --out docs/figures/version_tree.png
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def find_git() -> str | None:
    import glob
    import os
    import shutil

    exe = shutil.which("git")
    if exe:
        return exe
    patterns = [
        os.path.expandvars(r"%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe"),
        r"C:\Program Files\Git\cmd\git.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe"),
    ]
    for p in patterns:
        hits = sorted(glob.glob(p), reverse=True)
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染 git 版本树为 PNG")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "version_tree.png"))
    ap.add_argument("--font-size", type=int, default=20)
    args = ap.parse_args()

    git = find_git()
    if not git:
        raise SystemExit("找不到 git 可执行文件")

    out = subprocess.run(
        [git, "log", "--graph", "--oneline", "--all", "--decorate", "--date=short"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    lines = [ln.rstrip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("git log 没有输出")

    # 字体：优先中文等宽（覆盖提交信息里的中文），退到英文等宽
    font_candidates = [
        (r"C:\Windows\Fonts\consola.ttf", "Consolas"),
        (r"C:\Windows\Fonts\msyh.ttc", "Microsoft YaHei"),
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
    ]
    font = None
    for path, _ in font_candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, args.font_size)
                break
            except Exception:  # noqa: BLE001
                continue
    if font is None:
        font = ImageFont.load_default()

    # 中文提交信息需要 CJK 字体，纯 Consolas 画不出中文 → 中文用雅黑
    cjk_font = None
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(path).exists():
            try:
                cjk_font = ImageFont.truetype(path, args.font_size)
                break
            except Exception:  # noqa: BLE001
                continue

    def has_cjk(s: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in s)

    line_h = int(args.font_size * 1.7)
    pad = 28
    header_h = 56

    # 估算宽度
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    max_w = 0
    for ln in lines:
        f = cjk_font if (has_cjk(ln) and cjk_font) else font
        w = probe.textlength(ln, font=f)
        max_w = max(max_w, w)

    width = int(max_w) + pad * 2
    height = header_h + line_h * len(lines) + pad

    bg = (30, 32, 38)
    fg = (220, 223, 228)
    tag_color = (255, 200, 80)
    head_color = (120, 220, 160)

    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)

    # 标题
    title = f"Git version tree  -  {len(lines)} commits"
    d.text((pad, 16), title, font=font, fill=(150, 200, 255))
    d.line([(pad, header_h - 12), (width - pad, header_h - 12)], fill=(70, 76, 90), width=1)

    y = header_h
    for ln in lines:
        f = cjk_font if (has_cjk(ln) and cjk_font) else font
        # 标签 (tag: ...) 高亮；HEAD 也高亮
        if "tag:" in ln:
            d.text((pad, y), ln, font=f, fill=tag_color)
        elif "HEAD" in ln:
            d.text((pad, y), ln, font=f, fill=head_color)
        else:
            d.text((pad, y), ln, font=f, fill=fg)
        y += line_h

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[ ok ] {out_path}  {width}x{height}  {len(lines)} 行")
    print()
    print(out.stdout)


if __name__ == "__main__":
    main()
