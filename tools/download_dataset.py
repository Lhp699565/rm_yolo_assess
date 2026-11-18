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
    """跑子进程，实时打印输出，同时留最后 30 行 —— 失败时把真正的报错甩到最下面。"""
    from collections import deque

    print("  $ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    tail: deque[str] = deque(maxlen=30)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    proc.wait()

    if proc.returncode != 0:
        print("\n" + "-" * 62)
        print(f"  子进程退出码 {proc.returncode}，最后几行输出：")
        print("-" * 62)
        for line in tail:
            if line.strip():
                print("  " + line)
        print("-" * 62)
    return proc.returncode


def ensure_writable_dir(dest: Path) -> None:
    """提前创建并试写目标目录，权限问题在这里就说清楚，别等下载到一半才炸。"""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        probe = dest / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except PermissionError:
        drive = dest.drive or "?"
        print("\n" + "!" * 62)
        print(f"  没有权限在 {dest} 创建/写入文件夹")
        print("!" * 62)
        print(f"\n  原因：当前账号（非管理员）不能直接在 {drive} 盘根目录新建文件夹。")
        print("\n  两个解决办法，任选其一：\n")
        print(f"  【办法 A】手动把上一级文件夹建出来（推荐，只需建一层）")
        print(f"    1. 打开「此电脑」→ 进入 {drive} 盘")
        print(f"    2. 右键 → 新建 → 文件夹，命名 {dest.parent.name}")
        print(f"    3. 如果弹出「需要管理员权限」，点「继续」")
        print(f"    4. 再双击本脚本；里面的 {dest.name} 它会自己建\n")
        print(f"  【办法 B】换一个你肯定能写的目录（比如 C 盘的用户目录）")
        print(f"    打开 shell.cmd，运行：")
        print(f"      python tools\\download_dataset.py --dest C:\\Users\\{Path.home().name}\\rm_datasets\\armor-pose")
        print(f"    C 盘当前剩余 95 GB 左右，够用。\n")
        print("!" * 62)
        sys.exit(2)


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

    ensure_writable_dir(dest)

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
