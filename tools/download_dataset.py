#!/usr/bin/env python
"""一键下载 + 导入 RM 装甲板数据集。

平时不用手动敲这个脚本，直接双击项目根目录的 download_dataset.cmd 就行。

它按顺序做三件事：
  1. 从魔搭下载数据集到 D 盘（断点续传，中断了再跑一次会接着下）
  2. 把 YOLO Pose 标签转成纯检测标签，导进 datasets/armor
  3. 体检

手动用法：
    python tools/download_dataset.py
    python tools/download_dataset.py --dest E:\\datasets\\armor-pose
    python tools/download_dataset.py --dry-run        # 只打印计划，不下载
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "P1ne4pp1e/armor-pose-blender-allcls-5k-1109"
DEFAULT_DEST = Path(r"D:\rm_datasets\armor-pose")
OUT_DIR = "datasets/armor"
RENAME = "2=armor_2,3=armor_3,4=armor_4"


def hr(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62, flush=True)


def run(cmd: list[str]) -> int:
    print("  $ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    return subprocess.call([str(c) for c in cmd])


def main() -> None:
    ap = argparse.ArgumentParser(description="一键下载并导入 RM 装甲板数据集")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="下载到哪个目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = ap.parse_args()

    dest = Path(args.dest)
    py = sys.executable

    hr("RM 装甲板数据集 · 一键下载")
    print(f"  数据集 : {DATASET_ID}")
    print(f"  保存到 : {dest}")
    print(f"  导入到 : {ROOT / OUT_DIR}")
    print(f"  体积   : 图片约 7 GB，全量含附加目录约 15~20 GB")
    print()
    print("  说明：下载耗时取决于网速，通常 10~40 分钟。")
    print("        中途关掉也没关系，重新双击会断点续传。")

    # 盘符检查
    drive = dest.drive or "D:"
    if not Path(drive + "\\").exists():
        print(f"\n[error] {drive} 不存在，请用 --dest 指定别的目录")
        sys.exit(1)
    try:
        import ctypes

        free = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive + "\\"), None, None, ctypes.byref(free)
        )
        print(f"\n  {drive} 剩余空间: {free.value / 1024**3:.1f} GB")
        if free.value < 20 * 1024**3:
            print("  [warn] 剩余空间不足 20 GB，可能下不完（可用 --exclude 跳过附加目录）")
    except Exception:  # noqa: BLE001
        pass

    if args.dry_run:
        print("\n[dry-run] 计划确认无误，退出（未下载）")
        return

    # ---------- 1. 下载 ----------
    hr("1/3  下载数据集（可中断，重跑会续传）")
    code = run([py, "-m", "modelscope.cli.cli", "download", DATASET_ID,
                "--repo-type", "dataset", "--local-dir", str(dest)])
    if code != 0:
        print(f"\n[error] 下载失败（退出码 {code}）")
        print("  常见原因：网络中断、D 盘满了、镜像超时")
        print("  直接重新双击本文件即可续传")
        sys.exit(code)

    # ---------- 2. 导入 ----------
    hr("2/3  转成纯检测格式并导入")
    code = run([py, ROOT / "tools" / "import_pose.py",
                "--src", str(dest), "--out", str(ROOT / OUT_DIR), "--rename", RENAME])
    if code != 0:
        print(f"\n[error] 导入失败（退出码 {code}）")
        sys.exit(code)

    # ---------- 3. 体检 ----------
    hr("3/3  数据集体检")
    code = run([py, ROOT / "tools" / "prepare.py", "check",
                "--root", str(ROOT / OUT_DIR), "--nc", "6"])

    hr("完成")
    print("  下一步训练（先跑 30 轮看趋势）：")
    print("    python train.py --epochs 30 --imgsz 640 --batch 16 --device 0")
    print()
    print("  然后提交这一次迭代：")
    print('    git add -A')
    print('    git commit -m "feat(data): 导入 5000 张 RM26 装甲板数据集，6 类"')
    print('    git tag -a v0.3-dataset -m "数据集 v1"')
    print()


if __name__ == "__main__":
    main()
