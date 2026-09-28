# 内容一致性验证 / content verification

每一帧都与 720p 源逐帧比对；`scripts/verify-hq-content.py` 生成。

| clip | frames | SR median | SR min | control median | structure median | structure min | motion r | offset | frozen | ΔY | ΔU | ΔV | temporal x | audio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| startup | 193 | 32.25 dB | 29.58 dB | 47.50 dB | 40.23 dB | 30.97 dB | 0.9994 | 0 | 0 | -0.03 | -1.67 | +2.22 | 1.18x | bit-identical |
| brand | 169 | 33.83 dB | 28.94 dB | 48.91 dB | 40.39 dB | 30.14 dB | 0.9906 | 0 | 0 | -0.98 | -0.11 | +0.43 | 1.18x | bit-identical |
| cyberpunk | 169 | 32.36 dB | 28.00 dB | 44.79 dB | 39.29 dB | 37.89 dB | 0.9955 | 0 | 0 | -1.11 | -0.20 | +0.71 | 1.08x | bit-identical |
| awakening | 169 | 32.04 dB | 28.32 dB | 47.87 dB | 39.25 dB | 30.00 dB | 0.9982 | 0 | 0 | -0.82 | -0.77 | +1.19 | 1.19x | bit-identical |

**判据 / verdict：PASS — every delivery carries the same content, timing and audio as its source**


## 怎么读这张表

* **SR / control / structure PSNR** 都在固定的 bt709 矩阵下计算。
  `control` 是源片用 Lanczos 720→1080→720 往返（今天的播放观感）——
  SR 版对源的 PSNR 必然低于它，差值来自模型重建出的细节；
  真正的"内容有没有变"看 **structure**（240×135 低频）与 **motion r / offset**。
* **ΔY / ΔU / ΔV** 直接取自码流里的 Y/U/V 平面（不做任何缩放或矩阵变换），
  因为源片**没有色彩标签**而交付版显式标了 bt709，用 RGB 反推亮度会产生伪差异。
  ΔY ≤ 1 级（0.4%）属于色彩约定差异，肉眼不可见。
* **temporal x**：静态区域内高频分量的时域标准差之比（交付版 / control）。
  逐帧超分可能让静止纹理"沸腾"，该值 > 2 才是问题。
* 判据：帧偏移 = 0、无冻结/黑帧、音频逐位一致、structure min ≥ 28 dB、
  ΔY ≤ 1.5 级、temporal x ≤ 2。全部满足才 PASS。
