# HQ 重制：四段片头 1080p / 2K

把 `media/*.mp4`（全部是 1280×720 / 24fps）用 AI 超分重建为 1080p 与 2K。
**内容、时长、帧率、运镜、音轨与原片完全一致** —— 每一帧都是同一画面的重建，
不是重新生成的新画面。

## 交付物（`media/hq/`）

| 片段 | 原始 | 1080p (1920×1080) | 2K (2560×1440) | 1080p PSNR | 2K PSNR |
| --- | --- | --- | --- | --- | --- |
| `deepseek-startup-intro` | 8.05s / 193 帧 | 20.8 MB · 21.6 Mb/s | 33.6 MB · 35.0 Mb/s | 48.3 dB | 49.1 dB |
| `deepseek-brand-intro` | 7.05s / 169 帧 | 8.5 MB · 10.1 Mb/s | 14.5 MB · 17.2 Mb/s | 52.7 dB | 53.0 dB |
| `deepseek-cyberpunk-intro` | 7.05s / 169 帧 | 11.2 MB · 13.3 Mb/s | 18.7 MB · 22.2 Mb/s | 51.3 dB | 52.0 dB |
| `deepseek-awakening-intro` | 7.05s / 169 帧 | 16.3 MB · 19.4 Mb/s | 26.9 MB · 32.0 Mb/s | 48.0 dB | 48.8 dB |

* 全部 **H.264 High**（1080p→Level 4.1，2K→Level 5.1），CRF 11 + `preset slow`，
  `yuv420p` + bt709（tv range），`+faststart`（moov 前置，满足片库的 faststart 检查）。
* 音频：原 AAC（32 kHz 立体声 128 kb/s）**原样 copy**，无重编码、无响度处理。
* PSNR 是相对于无损 5120×2880 母版（同尺寸同重采样）的实测值；
  含 sRGB↔YUV 往返约 3 dB 的固定损失，所以真实还原度比数字更好。
  每个片段的 `manifest-*.json` 记录了源文件与模型的 sha256、逐档 PSNR、时间戳。

`compare-<片段>-full.png`（整幅三帧）和 `compare-<片段>-detail.png`（细节 200%，
自动挑细节最丰富的帧与区域）是直观的新旧对比。

## 内容验证（`scripts/verify-hq-content.py` → `VERIFY.md`）

不是只看参数：每一帧都拿来和 720p 源比过。

| 片段 | 帧数 | 帧偏移 | 运动序列相关 | 冻结/黑帧 | 结构 PSNR(中位) | 对源 PSNR(中位) | 对照* PSNR | ΔY | ΔU | ΔV | 时域抖动 | 音频 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| startup | 193 | 0 | 0.9994 | 0 / 0 | 40.2 dB | 32.3 dB | 47.5 dB | −0.03 | −1.67 | +2.22 | 1.18× | 逐位相同 |
| brand | 169 | 0 | 0.9906 | 0 / 0 | 40.4 dB | 33.8 dB | 48.9 dB | −0.98 | −0.11 | +0.43 | 1.18× | 逐位相同 |
| cyberpunk | 169 | 0 | 0.9955 | 0 / 0 | 39.3 dB | 32.4 dB | 44.8 dB | −1.11 | −0.20 | +0.71 | 1.08× | 逐位相同 |
| awakening | 169 | 0 | 0.9982 | 0 / 0 | 39.3 dB | 32.0 dB | 47.9 dB | −0.82 | −0.77 | +1.19 | 1.19× | 逐位相同 |

\* 对照 = 源片自身做 Lanczos 720→1080→720 往返（即今天的播放观感）。SR 版对源的
PSNR 必然低于它 —— 差值就是模型重建出的细节；内容是否改变要看**结构 PSNR、
帧偏移、运动序列相关性**。结论：**PASS**。

补充验证与归因（都在固定矩阵 / 直接读码流 Y 平面的前提下做的）：

* **编码链路对母版忠实**：交付版码流里的 Y 与母版（5120×2880 RGB）按 bt709
  转出的 Y 相差 ≤ 0.05 级；x264 用同样参数重编源片，亮度偏差 +0.006 级 —— 编码不改变亮度。
* **模型亮度中性**：母版 vs 源在固定矩阵下相差 ±0.3 级（四段有正有负，无系统性偏移）。
* **ΔY ≤ 1.1 级（0.4%）的来源是色彩约定**：源片**没有任何色彩标签**，
  交付版显式写了 bt709/limited；把源按 601 解释、交付按 709 解释，
  差值就是这个量级，且集中在色度（ΔU/ΔV ~0.1–2.2/128）。肉眼不可见。
  若用 RGB 反推亮度来做这个比较，会看到 −1.5~−1.9 级的伪差异（我最初就踩了这个坑）。
* **最差帧**：8× 差分图（`verify-worst-frames.jpg`）显示差异只集中在物体边缘与微纹理，
  几何、位置、明暗分布完全一致 —— 没有画面被换掉、没有变形。
* **时域稳定性**：静止区域内高频分量的时域标准差 = 对照的 1.08–1.19×
  （绝对值 0.12–0.37/255，低于编码噪声底），逐帧超分没有造成"沸腾/闪烁"。

## 做法

```
media/deepseek-*-intro.mp4       1280x720 24fps
  → ffmpeg rawvideo rgb24
  → Real-ESRGAN realesr-general-x4v3（SRVGGNetCompact，4x，ncnn，float32）
  → 5120x2880 每帧（无损 libx264rgb 中间母版）
  → Lanczos + accurate_rnd → 1920x1080 / 2560x1440
  → x264 CRF 11 slow，bt709，faststart，音频 copy
```

* **模型**：`realesr-general-x4v3`（Real-ESRGAN 的紧凑版 SRVGGNetCompact，
  权重 2.4 MB）。这台机器 2 核 CPU、无 GPU，网络只通 GitHub，
  更大的 RRDB（`RealESRGAN_x4plus`，64 MB）跑 700 帧不现实；紧凑模型在这段
  动漫/特效素材上质量已经很稳（见 brand / awakening 的角色脸部、毛发细节）。
* **必须 fp32**：`use_fp16_storage/arithmetic` 在高光爆闪帧上会溢出
  （紧凑模型末层无 clamp，实测出现 ±1e9）。ncnn 从 Python 读的本来就是
  float32，fp16 只影响层间精度，改 fp32 后输出干净（每帧都断言无 NaN/Inf）。
* 中间祖是无损 RGB，所以同一片段的 1080p / 2K 是从同一母版分别重采样，
  两者间没有二次损失；`--crf` 只影响交付编码。
* 运行成本：4x 超分约 **10 s/帧**（2 线程），四段 700 帧合计约 2 小时；
  编码 1080p ≈ 0.5–1 min、2K ≈ 1–1.5 min/段。

## 复现

```bash
# 1. 模型（ncnn 权重；ONNX 权重可用同样的导出流程）
git clone --depth 1 https://github.com/IbrahimGhadre/realesrgan-mobile
M=realesrgan-mobile/android/ncnn-app/app/src/main/assets

# 2. 渲染 + 编码（可重复执行，已完成的块自动跳过）
python3 scripts/render-hq.py \
  --src media/deepseek-startup-intro.mp4 \
  --work /tmp/hq-startup --out media/hq \
  --model-param $M/realesr_general_x4v3.ncnn.param \
  --model-bin   $M/realesr_general_x4v3.ncnn.bin \
  --targets 1920x1080,2560x1440 --crf 11 --preset slow \
  --manifest media/hq/manifest-startup.json

# 3. 对比图
python3 scripts/make-compare-sheets.py
```

依赖：`ncnn`（pip teewheel，CPU 后端）、`numpy`、`pillow`、`ffmpeg`
（无系统 ffmpeg 时自动用 `imageio-ffmpeg` 的静态构建，或读 `$FFMPEG`）。
文件名后缀：1080p → `-1080p.mp4`，1440p → `-2k.mp4`。

## 和 npm 包的关系（重要）

`media/` 只是嵌入脚本的**输入**，不随包发布：`scripts/embed-clips.mjs` 会把
`media/*.mp4` 转成 `lib/clips.data.js` 里的 base64（当前 4 段合计 12 MB）。
HQ 版本体积是原片的 3~5 倍，base64 后会接近 100 MB —— 这个取舍请自行决定；
本次**没有**改动 `lib/clips.data.js`，插件行为、路由、片库列表全部保持原样。

想让插件直接播 HQ 版而不动包体，丢进用户片库即可（见 README「换自己的视频」）：

```bash
mkdir -p ~/.dsh/boot-animation/videos
cp media/hq/deepseek-*-intro-1080p.mp4 ~/.dsh/boot-animation/videos/
```

如果确实要把 HQ 版设为内置片源，把 `scripts/embed-clips.mjs` 里的 `file:`
改指向 `media/hq/...` 再 `npm run embed-clips` 即可（体积见上表）。
