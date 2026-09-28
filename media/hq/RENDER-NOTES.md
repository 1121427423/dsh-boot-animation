# HQ 重制：deepseek-startup-intro

把 `media/deepseek-startup-intro.mp4`（1280×720 / 24fps / 8.05s，193 帧）用 AI 超分
重建为 1080p 与 2K 版本。**内容、时长、帧率、运镜、音轨与原片完全一致**，
每一帧都是同一画面的重建，不是重新生成的新画面。

## 交付物

| 文件 | 分辨率 | 码率 | 大小 | 说明 |
| --- | --- | --- | --- | --- |
| `deepseek-startup-intro-1080p.mp4` | 1920×1080 | 11.6 Mb/s | 11.8 MB | H.264 High @ L4.1，CRF 15，faststart |
| `deepseek-startup-intro-2k.mp4` | 2560×1440 | 17.7 Mb/s | 17.9 MB | H.264 High @ L5.1，CRF 15，faststart |

两版均为 `yuv420p` + bt709（tv range），音频直接从原封装的 AAC（32 kHz 立体声
128 kb/s）**原样 copy**，没有任何重新编码或响度处理。

```
9809bc13d6d1dfa58abd…  deepseek-startup-intro-1080p.mp4
d1de0c298d5a4dc38b38…  deepseek-startup-intro-2k.mp4
ba72c501021444cd4bd8…  media/deepseek-startup-intro.mp4   (源文件)
```

同目录下的 `compare-startup-full.png`（整幅三帧对比）和
`compare-startup-detail.png`（细节 200% 对比）就是新旧的直观差别。

## 做法

```
media/deepseek-startup-intro.mp4   1280x720
  → ffmpeg rawvideo rgb24
  → Real-ESRGAN realesr-general-x4v3（SRVGGNetCompact，4x，ncnn，float32）
  → 5120x2880 每帧（无损 libx264rgb 中间件，可断点续跑）
  → Lanczos + accurate_rnd 缩放到 1920x1080 / 2560x1440
  → x264 CRF 15 slow，yuv420p bt709，faststart
```

* 模型：`realesr-general-x4v3`（Real-ESRGAN 的紧凑版 SRVGGNetCompact，
  官方 `realesr-general-x4v3.pth` 系列权重，2.4 MB）。
  选它是因为这台机器只有 2 核 CPU、无 GPU、且网络只能访问 GitHub：
  更大的 RRDB（`RealESRGAN_x4plus`，64 MB）在这里跑 193 帧不现实。
* **必须用 fp32**：`use_fp16_storage/arithmetic` 在这段素材上会溢出
  （画面里有大量高光爆闪，紧凑模型的最后一层没有 clamp，
  实测输出出现 ±1e9 的数值）。ncnn 从 Python 读出的是 float32，
  fp16 只影响层间精度，改成 fp32 后输出干净（无 NaN/Inf）。
* 中间件是无损 RGB 母版，所以 1080p / 2K 两版是从同一个母版分别重采样的，
  两者之间没有二次损失。

## 客观质量

对照"无损 5120×2880 母版→同一目标尺寸"的 PSNR（数值越高越接近无损）：

| 交付 | PSNR (Y) | PSNR (avg) |
| --- | --- | --- |
| 1080p | 45.5 dB | 46.0 dB |
| 2K | 46.5 dB | 47.0 dB |

*实测* PSNR 是 1080p 45.5 dB / 2K 46.5 dB（含 sRGB↔YUV 往返，约 3 dB 的
天花板损失），`scripts/render-hq.py` 会在编码后自动打印这一项并写进 manifest。

## 复现

```bash
# 1. 取模型（ncnn 权重；ONNX 权重也能用同样的导出流程）
git clone --depth 1 https://github.com/IbrahimGhadre/realesrgan-mobile
M=realesrgan-mobile/android/ncnn-app/app/src/main/assets

# 2. 渲染 + 编码（可重复执行，已完成的块会跳过）
python3 scripts/render-hq.py \
  --src media/deepseek-startup-intro.mp4 \
  --work /tmp/hq-startup --out media/hq \
  --model-param $M/realesr_general_x4v3.ncnn.param \
  --model-bin   $M/realesr_general_x4v3.ncnn.bin \
  --targets 1920x1080,2560x1440 \
  --manifest media/hq/manifest.json
```

依赖：`ncnn`（pip 有 teewheel，含 CPU 后端）、`numpy`、`ffmpeg`
（没有系统 ffmpeg 时会自动用 `imageio-ffmpeg` 里的静态构建，或读 `$FFMPEG`）。
单帧 4x 超分约 13 s（2 线程），193 帧总共约 40 分钟；编码 1080p ≈ 1 min、
2K ≈ 1.5 min。

## 另外三段

`deepseek-brand-intro` / `deepseek-cyberpunk-intro` / `deepseek-awakening-intro`
用同一条流水线即可（都是 1280×720 / 24fps / 7.05s，169 帧），
把 `--src` 换掉、`--work` 换个目录就行：

```bash
python3 scripts/render-hq.py --src media/deepseek-brand-intro.mp4 \
  --work /tmp/hq-brand --out media/hq …      # 其余参数相同
```

## 和 npm 包的关系（重要）

`media/` 只是嵌入脚本的**输入**，不随包发布：`scripts/embed-clips.mjs` 会把
`media/*.mp4` 转成 `lib/clips.data.js` 里的 base64（现在 4 段合计 12 MB）。
HQ 版本体积是原片的 3.5~5 倍，如果要用 HQ 版做内置片源，
base64 后的 `lib/clips.data.js` 会接近 40 MB —— 这个取舍请自行决定；
本次**没有**改动 `lib/clips.data.js`，插件行为、路由、片库列表都保持原样。

想要插件播到 HQ 版而不动包体，最省事的办法是把文件丢进
`~/.dsh/boot-animation/videos/`（见 README 的片库一节）：

```bash
cp media/hq/deepseek-startup-intro-1080p.mp4 ~/.dsh/boot-animation/videos/
```
