#!/usr/bin/env python3
"""verify-hq-content.py — did the rebuild keep the same content?

Technical checks alone (frame count, bitrate, faststart) say nothing about
whether the *picture* survived.  This script compares every delivered frame
against its 720p source and answers four questions:

  1. same picture?      per-frame PSNR delivery-vs-source, plus a *control*:
                        the source resampled 720->1080->720 with plain Lanczos,
                        which is what a player shows today.  A super-resolution
                        pass is expected to differ from the source (it invents
                        detail); what matters is that the difference stays
                        bounded and structural.
  2. same structure?    PSNR of the low-frequency component only (240x135).
  3. same timing?       frame-to-frame motion series of both, correlated, plus
                        a scan for the best temporal offset (+-4 frames).
  4. same audio?        decoded PCM md5 of source vs delivery (audio is copied,
                        so it must be bit-identical).
  plus: frozen / black frame census, highlight census, and a temporal-stability
  test (per-frame SR can "boil" on static texture: high-frequency temporal
  standard deviation of the delivery vs the control inside static regions).

Usage: python3 scripts/verify-hq-content.py [--maps DIR] [--report FILE]
Exit code 0 = PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(REPO, 'media')
HQ = os.path.join(MEDIA, 'hq')
SRC_W, SRC_H = 1280, 720
CLIPS = ['startup', 'brand', 'cyberpunk', 'awakening']
TIME_WINDOW, TIME_N = 40, 48       # frames used for the temporal-stability test


def ffmpeg() -> str:
    exe = os.environ.get('FFMPEG')
    if exe and os.path.exists(exe):
        return exe
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def raw_frames(path: str, w: int, h: int, skip: int = 0, count: int | None = None):
    """Yield HxWx3 uint8 frames at w x h (ffmpeg scales: swscale/lanczos)."""
    cmd = [ffmpeg(), '-v', 'error']
    if skip:
        cmd += ['-ss', f'{skip / 24:.6f}']
    cmd += ['-i', path, '-map', '0:v', '-vf', f'scale={w}:{h}:flags=lanczos']
    if count:
        cmd += ['-frames:v', str(count)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if not buf or len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def pcm_md5(path: str) -> str:
    out = subprocess.run([ffmpeg(), '-v', 'error', '-i', path, '-map', '0:a',
                          '-f', 's16le', '-acodec', 'pcm_s16le', '-'],
                         stdout=subprocess.PIPE).stdout
    return hashlib.md5(out).hexdigest()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def rs(f: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS))


def highpass(f: np.ndarray) -> np.ndarray:
    blur = np.asarray(Image.fromarray(f).filter(ImageFilter.GaussianBlur(1.2)))
    return f.astype(np.float32) - blur.astype(np.float32)


def temporal_stability(new_path: str, ctl_path: str) -> tuple[float, float]:
    """HF temporal std in static areas: delivery vs control, and % static pixels."""
    new = list(raw_frames(new_path, 1920, 1080, skip=TIME_WINDOW, count=TIME_N))
    ctl = list(raw_frames(ctl_path, 1920, 1080, skip=TIME_WINDOW, count=TIME_N))
    mot = np.zeros((1080, 1920), np.float32)
    for i in range(1, len(ctl)):
        d = np.abs(ctl[i].astype(np.int16) - ctl[i - 1].astype(np.int16)).mean(axis=2)
        mot += d
    static = mot / max(len(ctl) - 1, 1) < 0.6

    def tstd(frames):
        s = s2 = None
        for f in frames:
            h = highpass(f)
            s = h if s is None else s + h
            s2 = h * h if s2 is None else s2 + h * h
        m = s / len(frames)
        return np.sqrt(np.maximum(s2 / len(frames) - m * m, 0)).mean(axis=2)

    a, b = tstd(ctl), tstd(new)
    denom = max(float(a[static].mean()), 1e-6)
    return float(b[static].mean() / denom), float(100 * static.mean())


def stored_planes_stats(path: str, w: int, h: int) -> dict:
    """Mean stored Y/U/V + Y range, decoded at the file's own resolution.

    Y is read straight out of the bitstream (no scaling, no matrix involved), so
    comparing stored Y of source and delivery is not confounded by the source
    being untagged while the delivery is tagged bt709.
    """
    p = subprocess.Popen([ffmpeg(), '-v', 'error', '-i', path, '-map', '0:v',
                          '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'],
                         stdout=subprocess.PIPE, bufsize=1 << 20)
    n = w * h * 3 // 2
    ys = us = vs = 0.0
    lo, hi = 255, 0
    frames = 0
    while True:
        b = p.stdout.read(n)
        if not b or len(b) < n:
            break
        a = np.frombuffer(b, np.uint8)
        y = a[:w * h]
        ys += float(y.sum())
        lo, hi = min(lo, int(y.min())), max(hi, int(y.max()))
        us += float(a[w * h:w * h + w * h // 4].sum())
        vs += float(a[w * h + w * h // 4:].sum())
        frames += 1
    p.stdout.close()
    p.wait()
    npix = frames * w * h
    return {'Y': ys / npix, 'U': us / (npix / 4), 'V': vs / (npix / 4),
            'y_min': lo, 'y_max': hi, 'frames': frames}


def verify_clip(cid: str, maps_dir: str | None) -> dict:
    src = os.path.join(MEDIA, f'deepseek-{cid}-intro.mp4')
    new = os.path.join(HQ, f'deepseek-{cid}-intro-1080p.mp4')
    r: dict = {'clip': cid, 'problems': [], 'src': src, 'new': new}

    p_sr, p_ctl, p_lf = [], [], []
    ds_list, dn_list = [], []
    prev_s = prev_n = None
    frozen = black = 0
    frames = 0
    for fs, fn in zip(raw_frames(src, SRC_W, SRC_H), raw_frames(new, SRC_W, SRC_H)):
        frames += 1
        p_sr.append(psnr(fs, fn))
        p_ctl.append(psnr(fs, rs(rs(fs, (1920, 1080)), (SRC_W, SRC_H))))
        p_lf.append(psnr(rs(fs, (240, 135)), rs(fn, (240, 135))))
        black += int(fn.max() < 8)
        if prev_s is not None:
            ds = float(np.mean(np.abs(fs.astype(np.int16) - prev_s.astype(np.int16))))
            dn = float(np.mean(np.abs(fn.astype(np.int16) - prev_n.astype(np.int16))))
            ds_list.append(ds)
            dn_list.append(dn)
            if ds > 1.0 and dn < 0.05:
                frozen += 1
        prev_s, prev_n = fs, fn

    r.update(frames=frames,
             psnr_median=round(float(np.median(p_sr)), 2), psnr_min=round(min(p_sr), 2),
             ctl_median=round(float(np.median(p_ctl)), 2), ctl_min=round(min(p_ctl), 2),
             lf_median=round(float(np.median(p_lf)), 2), lf_min=round(min(p_lf), 2),
             frozen_frames=frozen, black_frames=black,
             audio_identical=(pcm_md5(src) == pcm_md5(new)))

    a, b = np.array(ds_list), np.array(dn_list)
    r['motion_corr'] = round(float(np.corrcoef(a, b)[0, 1]), 4)
    best_off, best_corr = 0, -2.0
    for off in range(-4, 5):
        x, y = (a[:len(a) - off or None], b[off:]) if off >= 0 else (a[-off:], b[:len(b) + off])
        if len(x) < 20:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_corr:
            best_off, best_corr = off, c
    r['best_offset'], r['best_offset_corr'] = best_off, round(best_corr, 4)

    ss, sn = stored_planes_stats(src, SRC_W, SRC_H), stored_planes_stats(new, 1920, 1080)
    r['dY'] = round(sn['Y'] - ss['Y'], 3)
    r['dU'] = round(sn['U'] - ss['U'], 3)
    r['dV'] = round(sn['V'] - ss['V'], 3)
    r['src_Y_range'] = f"{ss['y_min']}-{ss['y_max']}"
    r['new_Y_range'] = f"{sn['y_min']}-{sn['y_max']}"

    ratio, static_pct = temporal_stability(new, src)
    r['temporal_ratio'], r['static_pct'] = round(ratio, 3), round(static_pct, 1)

    if best_off != 0:
        r['problems'].append(f'best motion correlation at offset {best_off} frame(s)')
    if frozen:
        r['problems'].append(f'{frozen} frozen frame(s)')
    if black:
        r['problems'].append(f'{black} black frame(s)')
    if not r['audio_identical']:
        r['problems'].append('audio is not bit-identical to the source')
    if r['lf_min'] < 28:
        r['problems'].append(f'low-frequency (structure) PSNR {r["lf_min"]} dB < 28 dB')
    if r['temporal_ratio'] > 2.0:
        r['problems'].append(f'temporal-stability ratio {r["temporal_ratio"]} > 2.0')
    if abs(r['dY']) > 1.5:
        r['problems'].append(f'mean stored luma differs by {r["dY"]} levels')
    if max(abs(r['dU']), abs(r['dV'])) > 4.0:
        r['problems'].append(f'mean chroma differs by U{r["dU"]} V{r["dV"]}')

    if maps_dir:
        os.makedirs(maps_dir, exist_ok=True)
        worst = int(np.argmin(p_sr))
        it_s = list(raw_frames(src, SRC_W, SRC_H))
        it_n = list(raw_frames(new, SRC_W, SRC_H))
        s, n = it_s[worst], it_n[worst]
        Image.fromarray(s).save(f'{maps_dir}/{cid}_worst_src.png')
        Image.fromarray(n).save(f'{maps_dir}/{cid}_worst_new.png')
        Image.fromarray(np.clip(np.abs(n.astype(np.int16) - s.astype(np.int16)) * 8, 0, 255)
                        .astype(np.uint8)).save(f'{maps_dir}/{cid}_worst_diff_x8.png')
        r['worst_frame'] = worst
        r['worst_psnr'] = round(p_sr[worst], 2)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='')
    ap.add_argument('--report', default=os.path.join(HQ, 'VERIFY.md'))
    args = ap.parse_args()

    rows = [verify_clip(c, args.maps or None) for c in CLIPS]

    hdr = (f'{"clip":10s} {"frames":>6s} {"SR med":>7s} {"SR min":>7s} {"ctl med":>8s} '
           f'{"LF med":>7s} {"LF min":>7s} {"motion r":>9s} {"off":>4s} {"frozen":>6s} '
           f'{"dY":>6s} {"dU":>6s} {"dV":>6s} {"temp x":>6s} {"audio":>6s}')
    print(hdr)
    lines = ['| clip | frames | SR median | SR min | control median | structure median | '
             'structure min | motion r | offset | frozen | ΔY | ΔU | ΔV | temporal x | audio |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    ok = True
    for r in rows:
        print(f'{r["clip"]:10s} {r["frames"]:6d} {r["psnr_median"]:7.2f} {r["psnr_min"]:7.2f} '
              f'{r["ctl_median"]:8.2f} {r["lf_median"]:7.2f} {r["lf_min"]:7.2f} '
              f'{r["motion_corr"]:9.4f} {r["best_offset"]:4d} {r["frozen_frames"]:6d} '
              f'{r["dY"]:6.2f} {r["dU"]:6.2f} {r["dV"]:6.2f} {r["temporal_ratio"]:6.2f} '
              f'{"same" if r["audio_identical"] else "DIFF":>6s}')
        lines.append(f'| {r["clip"]} | {r["frames"]} | {r["psnr_median"]:.2f} dB | '
                     f'{r["psnr_min"]:.2f} dB | {r["ctl_median"]:.2f} dB | {r["lf_median"]:.2f} dB | '
                     f'{r["lf_min"]:.2f} dB | {r["motion_corr"]:.4f} | {r["best_offset"]} | '
                     f'{r["frozen_frames"]} | {r["dY"]:+.2f} | {r["dU"]:+.2f} | {r["dV"]:+.2f} | '
                     f'{r["temporal_ratio"]:.2f}x | '
                     f'{"bit-identical" if r["audio_identical"] else "DIFFERENT"} |')
        for p in r['problems']:
            print(f'   !! {p}')
            lines.append(f'| | | | | | | | | | | | | **{p}** |')
            ok = False

    verdict = ('PASS — every delivery carries the same content, timing and audio as its source'
               if ok else 'FAIL — see the flagged rows above')
    print('\nverdict:', verdict)

    with open(args.report, 'w') as fh:
        fh.write('# 内容一致性验证 / content verification\n\n')
        fh.write('每一帧都与 720p 源逐帧比对；`scripts/verify-hq-content.py` 生成。\n\n')
        fh.write('\n'.join(lines) + '\n\n')
        fh.write('**判据 / verdict：%s**\n\n' % verdict)
        fh.write("""
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
""")

    print('report ->', args.report)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
