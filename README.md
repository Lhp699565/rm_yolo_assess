# RM 视觉组梯队考核 —— 自定义目标检测

> 考核要求：**实现对 YOLO 基本分类（COCO 80 类）之外物体的目标矩形框检测，并做好每次代码迭代的版本管理树。**

这个仓库就是为了交这一份东西：自己造数据集 → 训练自定义类别 → 输出矩形框 → 全过程进 git。
预训练权重直接识别 `bus`、`person` 不算数，那只是装环境的验证。

---

## 1. 我训的是什么

| 项目 | 内容 |
|---|---|
| 类别 | `blue_armor` 蓝色装甲板 / `red_armor` 红色装甲板（2 类） |
| 为什么选它 | 不在 COCO 80 类里；和 RM 赛场直接相关；框是矩形、标注成本低 |
| 模型 | YOLOv8n / YOLO11n（迁移学习，只换检测头类别数） |
| 数据量 | 见 `docs/results.md` 每次实验的记录 |

想换成别的目标？只改 `configs/armor.yaml` 里的 `nc` 和 `names`，重新标数据即可，代码不用动。

---

## 2. 目录结构

```
rm_yolo_assess/
├── configs/armor.yaml      数据集配置（类别、路径）
├── datasets/               数据集（不进 git，见 README 第 4 节交接方式）
├── tools/
│   ├── prepare.py          split 划分 + check 体检
│   ├── frames.py           视频抽帧（攒数据用）
│   ├── get_weights.py      预训练权重镜像下载（GitHub 被墙时用）
│   └── make_smoke.py       合成小数据集，验证流水线没坏
├── train.py                训练
├── val.py                  评估
├── detect.py               推理 + 画框 + FPS + 检测明细 CSV
├── export.py               导出 ONNX
├── runs/                   训练/评估产物（不进 git，只把图挑进 docs/figures）
└── docs/
    ├── checklist.md        考核项对照表（交作业时从这看）
    ├── dataset.md          数据集怎么来、怎么标
    └── results.md          实验结果记录表
```

---

## 3. 环境

### 3.1 Windows + N 卡（推荐，本仓库验证过的组合）

RTX **50 系是 Blackwell 架构，必须 cu128 及以上**。网上大部分教程写的 cu118 会报
`no kernel image is available for execution on the device`，不是你装错了，是版本不支持。

```powershell
# 1) 建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) 先装 torch（一定要先装，否则 ultralytics 会顺手拉一个 CPU 版）
#    镜像按自己网络选：上海交大 / 阿里云；版本号不存在就去掉 == 后面的部分
pip install torch torchvision `
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/ `
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 再装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 自检
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 2.8 GB 的 CUDA 包在国内镜像上经常被限速到几百 KB/s。实测可用：
> 上海交大 `mirror.sjtu.edu.cn/pytorch-wheels/cu128/`（多线程能到 4 MB/s）、
> 阿里云 `mirrors.aliyun.com/pytorch-wheels/cu128/`（单线程约 0.3 MB/s，慢但不限量）。

### 3.2 没有独显 / 只有核显

不用改代码，训练时加 `--device cpu`：

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python train.py --device cpu --imgsz 416 --batch 8 --epochs 60
```

CPU 训练就三点：**nano 模型 + 416 尺寸 + 60~80 轮**，300 张数据大约几小时，挂着跑。

> 想快就借 GPU：魔搭 Notebook 有免费算力，Kaggle 每周 30 小时 T4。

### 3.3 Ubuntu（队伍环境）

```bash
sudo apt install -y python3-venv git
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

没有显示器时跑 `detect.py` 记得加 `--no-show`，否则 `cv2.imshow` 会崩。

### 3.4 两台机器怎么配合（本项目的实际工作方式）

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  A 机：Windows + RTX 5060    │        │  B 机：Ubuntu（虚拟机里）      │
│  ─────────────────────────  │        │  ───────────────────────────  │
│  · 标注（labelImg）          │        │  · Linux 命令 / 环境练习       │
│  · 训练（CUDA，快）          │  git   │  · 跑 detect.py 验证跨平台     │
│  · 调参 / 对比实验           │ ─────► │  · OpenCV C++ 编译与调试       │
│  · 导出 ONNX                 │  网盘  │  · 以后接 ROS / 上车代码       │
└─────────────────────────────┘        └──────────────────────────────┘
```

**为什么这么分**：虚拟机没有 GPU 直通，在里面训练会慢到不可用；但队伍最终跑在
Linux 上，所以 Linux 侧的验证必须在 B 机做，不能只在 Windows 上"应该没问题"。

交接规则：

| 传什么 | 怎么传 | 注意 |
|---|---|---|
| 代码 / 配置 / 文档 | `git push` | B 机 `git pull` 即可，版本树天然同步 |
| 数据集（几百 MB） | 网盘 / 局域网上传 | **不要进 git** |
| 权重 `best.pt`（6 MB） | 网盘，或 Git LFS | 同上 |
| 导出模型 `best.onnx` | 随权重一起 | B 机用来做 OpenCV DNN 验证 |
| 实验结果 | 写进 `docs/results.md` 一起提交 | 两台机器看到同一份结论 |

A→B 的具体流程：

```bash
# A 机：训练完，导出
python export.py --weights runs/train/exp_xxx/weights/best.pt
git add -A && git commit -m "feat(train): v1.0 模型 mAP50=0.8x" && git push

# B 机：拉代码 + 接权重，验证 Linux 侧能跑
git pull
python detect.py --weights weights/best.pt --source test.jpg --no-show
```

跨机器最容易翻车的几件事：

1. **路径分隔符**——代码里一律用 `pathlib.Path`，别手拼 `"a\\b"`；
2. **文件名不要用中文**——Ubuntu 的 OpenCV 读中文路径会静默失败（`imread` 返回 `None`）；
3. **无显示器**——B 机加 `--no-show`，否则 `cv2.imshow` 直接崩；
4. **torch 版本要对齐**——两台机器写同一个版本号，不然权重加载报 key 不匹配；
5. **显存不够时**——A 机 `--batch` 从 16 降到 8 或 4，B 机纯 CPU 就别训练了，只推理。

### 3.5 电脑上没有 Python？用自带的启动器

如果你系统里敲 `python` 提示"不是内部或外部命令"，**不用去装 Python**。
项目里有两个 `.cmd`，直接调用 `.venv` 里的解释器：

| 文件 | 作用 |
|---|---|
| `shell.cmd` | 打开一个已经配好环境的命令行窗口，进去后 `python` / `pip` 直接能用 |
| `run.cmd` | 单次执行：`run.cmd train.py --epochs 100 --device 0` |

双击 `shell.cmd` 就能开始干活。等价于每次手敲：

```cmd
.venv\Scripts\python.exe train.py --epochs 100 --imgsz 640 --batch 16 --device 0
```

> 注意：`.venv` 是绑在这台机器上的，**不能复制到别的电脑用**（里面记着解释器绝对路径）。
> 换机器要按 3.1 / 3.3 重新建环境。

---

## 4. 数据

### 4.1 从哪来（三条路，按性价比排）

1. **公开数据集**（最快，考核允许）
   - [魔搭 ModelScope](https://www.modelscope.cn/home) 搜「装甲板 / RoboMaster / armor」
   - Roboflow Universe 搜 `robomaster armor`，可以直接导 YOLO 格式
   - 注意看 license，并且**自己跑一遍 `check`**，公开数据的标签质量参差
2. **自己拍**（最稳，答辩有话说）
   用手机/摄像头绕装甲板拍 2~3 分钟不同角度光照的视频，抽帧：
   ```bash
   python tools/frames.py --src videos/armor.mp4 --out raw/images --every 20 --dedup
   ```
3. **两者混合**（推荐）：公开数据打底 200 张 + 自拍 100 张做域适配，
   顺带就形成了两轮迭代的实验对比。

数据集**不进 git**（几百 MB 会把仓库搞坏）。README 里写清来源和网盘链接，别人能拿到就行。

### 4.2 标注

用 `labelImg` / `X-AnyLabeling`，导出 **YOLO 格式**，每张图一个同名 `.txt`：

```
<class_id> <x_center> <y_center> <width> <height>     # 全部归一化到 0~1
0 0.512 0.443 0.083 0.121
```

三条硬规矩：
- 类别 id 从 0 开始连续，和 `configs/armor.yaml` 的 `names` 顺序一致；
- 坐标归一化，不是像素；
- 一张图里没有目标也要建空 `.txt`（当作负样本，能显著降误检）。

### 4.3 划分 + 体检

```bash
# 8:1:1 划分
python tools/prepare.py split --src raw --out datasets/armor --ratio 0.8 0.1 0.1

# 体检（训练前必跑）
python tools/prepare.py check --root datasets/armor --nc 2
```

`check` 会查：字段数、类别越界、坐标未归一化、宽高为 0、孤儿标签、
**train/val 同名泄漏**（这个最容易被忽略，会让 mAP 虚高十几点）、类别不平衡。

---

## 5. 训练

```bash
# 先确认数据集能读（不训练）
python train.py --dry-run

# 正式训练
python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device 0

# 小数据量防过拟合：冻结骨干前 10 层
python train.py --freeze 10 --epochs 80
```

产物在 `runs/train/<name>/`，其中这几样要贴进报告：

| 文件 | 用途 |
|---|---|
| `results.png` | loss + mAP 曲线，证明「确实训练了」 |
| `confusion_matrix.png` | 哪两类容易混 |
| `weights/best.pt` | 最终权重 |
| `summary.json` | 本次实验的完整参数 + git commit |

---

## 6. 评估

```bash
python val.py --weights runs/train/exp_xxx/weights/best.pt --split val
python val.py --weights ... --split test        # 最后一轮才看 test
```

输出 mAP@0.5、mAP@0.5:0.95、P、R 和每类指标，同时写 `metrics.json`。
抄到 `docs/results.md` 的表格里。

---

## 7. 推理（考核演示就在这一步）

```bash
# 单张图
python detect.py --weights runs/train/exp_xxx/weights/best.pt --source test.jpg --show

# 视频
python detect.py --weights ... --source match.mp4

# 笔记本前置摄像头
python detect.py --weights ... --source 0

# 只看红装甲板
python detect.py --weights ... --source 0 --classes 1
```

窗口里按 `q` 退出。每次运行都会在 `runs/detect/det_<时间>/` 下落：
- `pred.mp4` / `*_pred.jpg` —— 画好框的结果
- `detections.csv` —— 每帧每个框的坐标、置信度、推理耗时

**演示时这么讲**：CSV 里能算出「平均单帧耗时 / 目标数 / 置信度分布」，
不是「我跑了一下能出框」，而是有量化数据。

---

## 8. 导出 ONNX（加分项）

```bash
python export.py --weights runs/train/exp_xxx/weights/best.pt
```

导出后 `export.py` 会自动用 onnxruntime 跑一遍自检，确认文件没坏。
这一步证明模型能脱离 PyTorch 部署——以后上车的必经路径。

---

## 9. 版本管理

考核要的是**能看出迭代过程的树**，不是「Push 过一次」。

### 分支

```
main                 只放能跑的版本
└── dev              日常开发
    ├── feature/dataset    数据采集与标注
    ├── feature/train      训练与调参
    └── feature/detect     推理脚本
```

### Commit 规范

```
feat(data): 采集并标注 320 张装甲板图像，2 类
feat(train): yolov8n baseline，416 输入，mAP50=0.72
feat(train): 输入提到 640，mAP50 提升到 0.86
fix(detect): 修复摄像头释放后窗口不关闭
docs: 补充数据集来源与复现步骤
```

### 打标签

```bash
git tag -a v0.1-baseline -m "环境 + 官方 bus 推理跑通"
git tag -a v0.2-dataset  -m "数据集 v1：320 张，2 类"
git tag -a v0.3-train    -m "首个自定义模型 mAP50=0.72"
git tag -a v1.0          -m "考核版：mAP50=0.86，含推理与文档"

git push origin --all --tags
```

### 交作业时截这个图

```bash
git log --graph --oneline --all --decorate
```

再配一张 Gitee/GitHub 的 Network 图。这张图就是「版本管理树」的证据。

### 不要提交的东西

`.gitignore` 已经挡住 `datasets/` `runs/` `*.pt` `*.onnx`。
权重用网盘或 Git LFS，README 里给链接。别为了"完整"把 2GB 权重塞进仓库。

---

## 10. 踩过的坑

| 现象 | 原因 | 解法 |
|---|---|---|
| `Dataset not found` / images not found | ultralytics 解析相对 `path` 以**运行时工作目录**为基准，不是 yaml 所在目录 | `train.py`/`val.py` 已自动把相对路径按仓库根目录转成绝对路径；直接用 `yolo` 命令行时才必须在仓库根目录执行 |
| `no kernel image is available` | RTX 50 系用了 cu118 的 torch | 换 cu128：见 3.1 |
| 训练 mAP 一直是 0 | 标签坐标是像素没归一化 | `prepare.py check` 会直接报出来 |
| 下载权重超时 | github.com 被墙 | `python tools/get_weights.py yolov8n.pt` 走镜像 |
| val 指标高得离谱 | train/val 有重复图片 | `prepare.py check` 的泄漏检查 |
| `cv2.imshow` 崩溃 | 无显示器 / SSH 环境 | 加 `--no-show` |

### 关于 `.devtools/`

这个目录是**给受限沙箱环境用的临时补丁**（Windows 上 `tempfile.mkdtemp` 建的目录
会让沙箱进程自己进不去，导致 pip 报 `Permission denied`）。正常电脑上装依赖不需要它，
可以直接忽略或删掉，不影响本项目任何代码。
