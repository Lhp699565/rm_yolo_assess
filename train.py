#!/usr/bin/env python
"""训练入口 —— 在自定义类别数据集上微调 YOLO。

考核要的是「YOLO 基本分类之外的物体」，所以这里从预训练权重出发做迁移学习，
只替换最后的检测头类别数，而不是从零训练。

用法：
    # 先确认数据集没问题（不训练，只校验）
    python train.py --dry-run

    # 正式训练（有 N 卡）
    python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device 0

    # 纯 CPU fallback
    python train.py --device cpu --imgsz 416 --batch 8 --epochs 60

产物：
    runs/train/<name>/weights/best.pt     最终权重
    runs/train/<name>/results.png         loss / mAP 曲线（放进报告）
    runs/train/<name>/confusion_matrix.png
    runs/train/<name>/summary.json        本次实验的完整记录（含 git commit）
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))


# ----------------------------------------------------------------------------- 工具


def find_git() -> str | None:
    """找 git 可执行文件。PATH 里没有时，兜底找 GitHub Desktop / 常见安装位置。

    很多人是用 GitHub Desktop 传代码的，它的 git 不在 PATH 上，
    这里兜一下，免得 summary.json 里 commit 永远是 unknown。
    """
    exe = shutil.which("git")
    if exe:
        return exe

    import glob
    import os

    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe"),
        os.path.expandvars(r"%ProgramFiles%\Git\cmd\git.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Git\cmd\git.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe"),
    ]
    for pattern in candidates:
        for hit in sorted(glob.glob(pattern), reverse=True):
            if Path(hit).exists():
                return hit
    return None


def git_info() -> dict:
    """记录本次实验对应的代码版本 —— 版本管理树的一部分。"""
    info = {"commit": "unknown", "branch": "unknown", "dirty": None}
    exe = find_git()
    if not exe:
        return info
    try:
        def run(*a: str) -> str:
            return subprocess.check_output(
                [exe, *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()

        info["commit"] = run("rev-parse", "--short", "HEAD")
        info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
        info["dirty"] = bool(run("status", "--porcelain"))
    except Exception:  # noqa: BLE001
        pass
    return info


def resolve_model(model: str) -> str:
    """权重不存在时自动从镜像下载到 weights/。"""
    p = Path(model)
    if p.exists():
        return str(p)

    if p.suffix != ".pt" or p.parent != Path("."):
        return model  # 交给 ultralytics 自己处理（会报错，但信息更清楚）

    from get_weights import download

    return str(download(p.name))


def prepare_data_config(data_arg: str) -> Path:
    """把 data.yaml 里的相对 path 解析成绝对路径，另存一份给 ultralytics。

    原因：ultralytics 解析相对 `path` 时以【运行时工作目录】为基准，
    换个目录执行就找不到数据集。这里统一按仓库根目录解析，
    不管你在哪个目录敲命令，结果都一样。
    """
    import yaml

    p = Path(data_arg)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise SystemExit(f"找不到数据集配置：{p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    cfg["path"] = str(root)

    out = ROOT / "runs" / "_cfg" / p.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def count_images(folder) -> int:
    """统计图片数量。混合数据集时 ultralytics 会返回目录列表，这里一并支持。"""
    if folder is None:
        return 0
    if isinstance(folder, (list, tuple)):
        return sum(count_images(f) for f in folder)
    p = Path(folder)
    if p.is_file():
        return 1
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def summarize_run(save_dir: Path) -> dict:
    """从 results.csv 里挑出最好的一轮，避免手抄指标抄错。"""
    import pandas as pd

    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    key = "metrics/mAP50(B)"
    if key not in df.columns or df.empty:
        return {}

    best = df.loc[df[key].idxmax()]
    last = df.iloc[-1]
    return {
        "epochs_done": int(last["epoch"]),
        "best_epoch": int(best["epoch"]),
        "mAP50": round(float(best[key]), 4),
        "mAP50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
        "precision": round(float(best["metrics/precision(B)"]), 4),
        "recall": round(float(best["metrics/recall(B)"]), 4),
    }


# ----------------------------------------------------------------------------- 主流程


def main() -> None:
    ap = argparse.ArgumentParser(description="训练自定义类别 YOLO 模型")
    ap.add_argument("--data", default=str(ROOT / "configs" / "armor.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="预训练权重，如 yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | 0 | cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="mAP 多久不涨就早停")
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（小数据集可设 10）")
    ap.add_argument("--lr0", type=float, default=None,
                    help="初始学习率。微调已有模型时建议调小（如 0.0005），不填用框架默认")
    ap.add_argument("--optimizer", default="auto",
                    help="auto | SGD | Adam | AdamW。微调建议 AdamW")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据集，不训练")
    args = ap.parse_args()

    # ---- 设备 ----
    import torch

    if args.device == "auto":
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 68)
    print(f"torch      : {torch.__version__}")
    print(f"cuda       : {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"device     : {device}")
    print(f"data       : {args.data}")
    print(f"model      : {args.model}")
    print("=" * 68)

    # ---- 数据集配置：先解析成绝对路径，再交给 ultralytics ----
    data_path = prepare_data_config(args.data)
    print(f"[data] resolved yaml = {data_path}")

    from ultralytics.data.utils import check_det_dataset

    try:
        info = check_det_dataset(str(data_path))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"数据集校验失败：{exc}\n"
            f"排查顺序：1) configs/*.yaml 的 path 指向的目录是否存在 "
            f"2) images/labels 子目录是否存在 3) 标签是否为 YOLO txt 格式"
        ) from exc

    print(f"[data] train = {info['train']}")
    print(f"[data] val   = {info.get('val')}")
    print(f"[data] names = {info['names']}")

    if args.dry_run:
        print(f"[dry-run] train {count_images(info['train'])} 张 / "
              f"val {count_images(info.get('val'))} 张")
        print("[dry-run] 数据集没问题，退出（未训练）")
        return

    # ---- 训练 ----
    from ultralytics import YOLO

    model_path = resolve_model(args.model)
    name = args.name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
    project = ROOT / "runs" / "train"

    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        freeze=args.freeze,
        optimizer=args.optimizer,
        **({"lr0": args.lr0} if args.lr0 else {}),
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=name,
        exist_ok=args.resume,
        plots=True,
        val=True,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir)
    metrics = summarize_run(save_dir)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "args": vars(args),
        "device": str(device),
        "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__,
        "save_dir": str(save_dir),
        "metrics": metrics,
    }
    (save_dir / "summary.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print(f"训练完成，产物目录：{save_dir}")
    if metrics:
        print(f"best epoch : {metrics['best_epoch']} / {metrics['epochs_done']}")
        print(f"mAP@0.5    : {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:95 : {metrics['mAP50_95']:.4f}")
        print(f"precision  : {metrics['precision']:.4f}")
        print(f"recall     : {metrics['recall']:.4f}")
        print("\n把上面 4 个数字和 runs/train/<name>/results.png 一起贴进 docs/results.md")
    print("=" * 68)


if __name__ == "__main__":
    main()
