# 原创片头两段：同频 / 升维

这两段是**新做的原创片头**（不是重制）：素材以原片画面作**参考图**生成，沿用原片的角色、
渲染语言、遥测 UI 语言，但**配色由本次重新设计**，与原片四段都不重复。

## 交付物

| 文件 | 规格 | 时长/帧数 | 大小 |
| --- | --- | --- | --- |
| `deepseek-resonance-intro-1080p.mp4` | 1920×1080 · 24fps · H.264 High@4.1 · CRF15 | 8.00s / 192 帧 | 13.9 MB |
| `deepseek-resonance-intro-2k.mp4` | 2560×1440 · High@5.1 · CRF15 | 8.00s / 192 帧 | 36.4 MB |
| `deepseek-ascension-intro-1080p.mp4` | 1920×1080 · 24fps · High@4.1 · CRF15 | 7.60s / 182 帧 | 18.9 MB |
| `deepseek-ascension-intro-2k.mp4` | 2560×1440 · High@5.1 · CRF15 | 7.60s / 182 帧 | 39.6 MB |

* 全部 `yuv420p` + bt709(tv) + **faststart**，浏览器 `<video>` 与片库路由都直接可用。
* 音频是**合成的原创音效**（次低音涌动 + 噪声拉升 + 冲击 + 铺底 pad + 收尾铃声），
  AAC 32 kHz 立体声 128 kb/s；实测 RMS −19.7 dBFS、峰值 −2.6 dBFS。
* 编码忠实度（1080p 交付 vs 无损 2K 母版）：同频 **51.2 dB**、升维 **47.5 dB**（PSNR avg）。

`preview-*.png` 是每段抽帧；`art/contact_sheet.png` 是全部素材一览；
`art/refine_compare.png` 是《同频》优化前后的对照。

### 优化轮（按反馈调整）

第二轮针对《同频》做了三处调整（都在 `art/q1_*.png`，逐帧比对见 `art/refine_compare.png`）：

* **肖像**（`q1_portrait`）：暖金主光从左侧扫过脸颊、冷青补光在右，主角移到画面右侧，
  标题区留出左侧暗区，字标不再压在脸上。
* **数据墙**（`q1_wall`）：去掉头顶光环，改成左上落下的暖金光束 + 青色面板阵列，
  强化"人站在数据面前"的构图。
* **眼中数据流**（`q1_eye`）：减少横穿画面的杂线，保留 2–3 条细流，加了眼里的高光点，
  朱砂色/青色两道粒子流对比更干净。

镜头顺序（8.00s）：数据厅建立 → 数据墙（暖金光束）→ 触碰面板 → **肖像** → **眼中数据流**（闪光切）
→ 回到数据厅收尾 + 字标。

## 两段内容

**《同频 · 数据共鸣》**（沿用原片角色，配色：**墨蓝 + 青 + 暖金**）
巨型数据厅建立镜头（暖金地面光带）→ 数据墙背影推近 → 触碰全息面板（暖金边光）→
肖像（暖金 vs 冷青对打）→ 眼中数据流特写 → 闪光回到数据厅收尾，字标浮现。

**《升维 · 思考的跃迁》**（沿用原片 cyberpunk 的线稿语言，配色：**紫罗兰 + 品红 + 暖橙**）
上升轨迹 → 线稿人形伸手 → 线框立方核心 → 光爆环（闪光切）→ 光门 →
回到立方核心，字标浮现。

两段都带原片同款 UI：左上/右上 `DEEPSEEK V1.0`、左下 `YOUR AI COMPANION / FOR A
SMARTER FUTURE`、右下 `READY`、左侧 `ANALYSIS/THINKING/SEARCHING/GENERATING`
逐行打字 + 进度条；字标是从 brand 片段提取的**真实字形**，按各段配色重新上色。

## 做法（`scripts/make-original-intro.py`）

```
media/original/art/*.png        以原片 1080p 帧作参考图生成（1376×768 或 1365×768）
  → media/original/sr/*_x4.png   Real-ESRGAN realesr-general-x4v3 放大 4×（镜头推进才不糊）
  → 2560×1440 画布上按镜头插值裁切（推/拉/横移，smoothstep 缓动）
  → 叠化 0.30s + 切点闪光 + 粒子（按各段配色着色）+ 数据流字符 + 阈值泛光
     + 暗角 + 胶片颗粒（±3 级）+ 渐变遮罩
  → UI/字标图层（淡入 + 打字机 + 进度条）
  → 无损 libx264rgb 母版 → 2K / 1080p 交付编码
```

单帧合成约 1.2 s（2 核），整段 8s ≈ 5 分钟；音频用 numpy 直接合成。

复现：

```bash
# 素材（可选，仓库里已含），再放大 + 渲染
python3 scripts/make-original-intro.py --clip resonance --work /tmp/orig --crf 15
python3 scripts/make-original-intro.py --clip ascension --work /tmp/orig --crf 15
```

## 手法边界（重要，先说清楚）

成片是**静帧画面 + 电影级运动**：推拉摇移、叠化/闪光转场、粒子/光晕/颗粒、UI 与字标浮现。
**做不到**角色连续表演、口型、镜头内大变形的动画 —— 这里没有视频生成模型可用。
因此镜头靠"多镜头剪辑 + 运镜 + 转场"来讲节奏，和原片那种连续运镜的观感会有差别。

## 放进片库

```bash
mkdir -p ~/.dsh/boot-animation/videos
cp media/original/deepseek-*-intro-1080p.mp4 ~/.dsh/boot-animation/videos/
```

`lib/clips.data.js` 未改动（内置片源仍是原来四段）；若要把它俩设成内置片源，
把 `scripts/embed-clips.mjs` 的清单加上这两段再 `npm run embed-clips`。
