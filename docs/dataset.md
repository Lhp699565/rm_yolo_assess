# 数据集说明

这份文档要写进仓库，因为**数据集不进 git**——别人复现时得知道数据长什么样、
从哪来、标了什么规矩。

---

## 1. 规模

| split | 图片数 | 框数 | 来源 |
|---|---|---|---|
| train | 待填 | 待填 | 自采 / 公开 |
| val | 待填 | 待填 | 同上 |
| test | 待填 | 待填 | 同上 |
| 合计 | 待填 | 待填 | |

> 填完表后，把 `python tools/prepare.py check` 的完整输出截图放到 `docs/figures/`。

---

## 2. 来源

### 2.1 公开数据集

- 魔搭 ModelScope：https://www.modelscope.cn/home
- Roboflow Universe：搜 `robomaster armor`，导出格式选 **YOLOv8**
- 公开数据一定要做两件事：
  1. 看 license，比赛/发表要注明来源；
  2. 跑一遍 `prepare.py check`，公开数据里坐标没归一化、类别 id 错位很常见。

### 2.2 自采

```bash
# 1. 录视频：手机或摄像头，绕目标转一圈，覆盖不同角度/光照/距离，2~3 分钟
# 2. 抽帧
python tools/frames.py --src videos/armor.mp4 --out raw/images --every 20 --dedup
# 3. 标注（labelImg / X-AnyLabeling），导出 YOLO 格式
# 4. 划分
python tools/prepare.py split --src raw --out datasets/armor
```

自采的价值：**域适配**。公开数据集往往和你的相机、光照、背景不一样，
混入 30% 自采数据通常能明显降误检，而且正好构成一轮"为什么涨了"的实验。

---

## 3. 标注规范

### 3.1 格式

```
<class_id> <x_center> <y_center> <width> <height>
```

- 全部归一化到 `0~1`（除以图片宽高），不是像素；
- 每张图一个同名 `.txt`，和图放同级目录；
- 没有目标的图也要建**空 txt**（负样本）。

### 3.2 框怎么画

| 规矩 | 原因 |
|---|---|
| 框贴紧目标外沿，不预留边距 | 边距会让回归目标偏移，定位变差 |
| 被遮挡超过 70% 的目标不标 | 标了反而制造噪声，模型学到模糊特征 |
| 同一目标只标一个框 | 重复框会被 NMS 丢掉，还会拉低 recall |
| 小目标（<16px）尽量补，或裁图放大 | YOLO 对小目标本来就弱 |
| 类别定义写死，别中途改 | 改了要全部重标，版本也对不上 |

### 3.3 类别定义（本项目）

| id | 名称 | 定义 |
|---|---|---|
| 0 | `blue_armor` | 蓝色方灯条 + 中间装甲板整体 |
| 1 | `red_armor` | 红色方灯条 + 中间装甲板整体 |

> 如果你要测的是"灯条"而不是"装甲板"，类别定义要重写，别混着标。

---

## 4. 质检清单

训练前跑：

```bash
python tools/prepare.py check --root datasets/armor --nc 2
```

它查这些：

- [x] 标签字段数是否为 5
- [x] 类别 id 是否在 `0..nc-1`
- [x] 坐标是否归一化到 0~1
- [x] 宽高是否 > 0
- [x] 有没有孤儿标签（有 txt 没图）
- [x] 有多少图没标签（会被当背景）
- [x] **train/val/test 之间有没有同名文件（数据泄漏）**
- [x] 类别是否严重不平衡（默认 5x 告警）

有 ERROR 就别训，先修。YOLO 遇到坏标签是静默跳过，你只会看到 mAP 上不去。

---

## 5. 交接方式

数据集不进 git，用这个方式给队友：

1. 打包：`datasets/armor` 压成 zip（通常 200~800 MB）；
2. 传网盘 / 队内 NAS；
3. **链接写进本文件第 6 节**，并写清 md5，避免传坏；
4. 队友解压到 `datasets/armor`，就能直接 `python train.py`。

---

## 6. 下载链接

| 内容 | 链接 | md5 | 大小 |
|---|---|---|---|
| datasets/armor.zip | 待填 | 待填 | 待填 |
