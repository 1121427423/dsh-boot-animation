#!/usr/bin/env python3
"""make-original-intro.py — compose brand-new intro clips in the house style.

These are *original* compositions (not rebuilds): AI keyframes generated with the
existing clips as image references, then driven by procedural camera moves,
cross-dissolves and flashes, a palette-aware particle field and telemetry UI, the
real DeepSeek wordmark (shape extracted from the brand clip, re-inked per clip),
and a synthesised sound design.

    media/original/art/*.png       AI keyframes (reference-image generated)
      -> media/original/sr/*_x4.png  4x Real-ESRGAN upscale (so pushes stay sharp)
      -> per-shot camera interpolation on a 2560x1440 canvas
      -> dissolve / flash transitions, particles, data streaks, bloom,
         vignette, grain, UI + wordmark fade-in
      -> lossless libx264rgb master -> 2K + 1080p H.264 (faststart, bt709)
      -> synthesised stereo audio (sub, riser, impact, pad, bell) -> AAC

Usage: python3 scripts/make-original-intro.py --clip resonance
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_SR = os.path.join(REPO, 'media', 'original', 'sr')
FONTS = '/tmp/fonts/ofl'
F_CN = f'{FONTS}/notosanssc/NotoSansSC[wght].ttf'
F_UI = f'{FONTS}/rajdhani/Rajdhani-Bold.ttf'
BRAND_FRAME = os.path.join(REPO, 'media', 'original', 'art', 'ref_brand_hero.png')
FPS = 24
BASE_W, BASE_H = 2560, 1440
XFADE = 0.30


def ffmpeg() -> str:
    exe = os.environ.get('FFMPEG')
    if exe and os.path.exists(exe):
        return exe
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


# --------------------------------------------------------------------- palettes
PALETTES = {
    # cyan + warm gold: same family as the originals, but the floor light and the
    # rim light are gold so it reads as a different clip at a glance
    'resonance': dict(
        accent=(126, 228, 255), accent2=(255, 198, 126), ui=(150, 208, 236),
        glow=(128, 216, 255), wordmark=((16, 52, 78), (124, 202, 240), (244, 253, 255)),
        under=(255, 198, 126),
    ),
    # violet / magenta + amber: deliberately away from every existing clip
    'ascension': dict(
        accent=(236, 150, 255), accent2=(255, 176, 104), ui=(226, 178, 246),
        glow=(226, 150, 250), wordmark=((52, 18, 72), (206, 130, 236), (255, 246, 255)),
        under=(255, 176, 104),
    ),
}


def mix(c1, c2, k):
    return tuple(int(round(a + (b - a) * k)) for a, b in zip(c1, c2))


# --------------------------------------------------------------------- art / cam
class Shot:
    def __init__(self, art: str, t0: float, t1: float,
                 cam0: tuple[float, float, float], cam1: tuple[float, float, float],
                 particles: float = 1.0, streaks: int = 0, flash_in: float = 0.0,
                 fade_in: float = 0.0):
        self.art, self.t0, self.t1 = art, t0, t1
        self.cam0, self.cam1 = cam0, cam1
        self.particles, self.streaks = particles, streaks
        self.flash_in, self.fade_in = flash_in, fade_in
        self.img: Image.Image | None = None
        self.box: tuple[float, float] = (0.0, 0.0)
        self.art_size: tuple[int, int] = (0, 0)

    def bind(self, art: Image.Image, margin: float = 0.09) -> None:
        """Keep only the region the camera can reach — 10 shots of 5460x3072 art
        do not fit in RAM, a single union box per shot does."""
        aw, ah = art.size
        self.art_size = (aw, ah)
        boxes = []
        for cam in (self.cam0, self.cam1):
            fx, fy, sc = cam
            w, h = sc * aw, sc * ah
            cx = min(max(fx * aw, w / 2), aw - w / 2)
            cy = min(max(fy * ah, h / 2), ah - h / 2)
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        x0 = max(0.0, min(b[0] for b in boxes) - margin * aw)
        y0 = max(0.0, min(b[1] for b in boxes) - margin * ah)
        x1 = min(float(aw), max(b[2] for b in boxes) + margin * aw)
        y1 = min(float(ah), max(b[3] for b in boxes) + margin * ah)
        self.img = art.crop((int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))))
        self.box = (x0, y0)

    def cam_at(self, t: float) -> tuple[float, float, float]:
        local = (t - self.t0) / max(self.t1 - self.t0, 1e-6)
        return tuple(a + (b - a) * ease(local) for a, b in zip(self.cam0, self.cam1))

    def frame(self, t: float, size: tuple[int, int]) -> Image.Image:
        aw, ah = self.art_size
        fx, fy, sc = self.cam_at(t)
        w, h = sc * aw, sc * ah
        cx = min(max(fx * aw, w / 2), aw - w / 2)
        cy = min(max(fy * ah, h / 2), ah - h / 2)
        x0 = cx - w / 2 - self.box[0]
        y0 = cy - h / 2 - self.box[1]
        return self.img.resize(size, Image.LANCZOS, box=(x0, y0, x0 + w, y0 + h))


# ------------------------------------------------------------------- particles
class Particles:
    """Drifting motes, tinted to the clip palette (accent + accent2 mix)."""

    def __init__(self, n: int, seed: int, size: tuple[int, int], palette: dict):
        rng = np.random.default_rng(seed)
        self.size = size
        self.x = rng.random(n) * size[0]
        self.y = rng.random(n) * size[1]
        self.z = rng.random(n) ** 2 * 0.7 + 0.3
        self.vx = (rng.random(n) - 0.5) * 26.0
        self.vy = -(8.0 + rng.random(n) * 42.0)
        self.tw = rng.random(n) * math.tau
        base = self._sprite()
        cols = [mix(palette['accent2'], palette['accent'], k) for k in (0.0, 0.45, 1.0)]
        self.sprites = [self._tint(base, a, c)
                        for a, c in zip((0.35, 0.55, 0.85), (cols[0], cols[1], cols[2]))]

    @staticmethod
    def _sprite() -> Image.Image:
        r = 24
        s = Image.new('L', (r * 2, r * 2), 0)
        d = ImageDraw.Draw(s)
        for i in range(r, 0, -1):
            d.ellipse((r - i, r - i, r + i, r + i), fill=int(255 * (1 - i / r) ** 2.4))
        return s

    @staticmethod
    def _tint(sprite: Image.Image, alpha: float, colour) -> Image.Image:
        t = Image.new('RGBA', sprite.size, tuple(colour) + (0,))
        t.putalpha(sprite.point(lambda v: int(v * alpha)))
        return t

    def draw(self, frame: Image.Image, t: float, intensity: float = 1.0) -> None:
        if intensity <= 0.01:
            return
        w, h = self.size
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t) % w
            y = (self.y[i] + self.vy[i] * t) % h
            z = self.z[i]
            r = int(6 + 26 * z)
            tw = 0.65 + 0.35 * math.sin(t * 2.4 + self.tw[i])
            sp = self.sprites[2 if z > 0.8 else (1 if z > 0.55 else 0)]
            k = intensity * tw
            if k < 0.99:
                sp = sp.point(lambda v, kk=k: int(v * kk))
            frame.alpha_composite(sp.resize((r, r)), (int(x), int(y)))


def draw_streaks(frame: Image.Image, t: float, seed: int, count: int, alpha: int,
                 colour) -> None:
    """Faint columns of drifting digits, the clips' shared telemetry texture."""
    if count <= 0:
        return
    w, h = frame.size
    rng = np.random.default_rng(seed)
    layer = Image.new('RGBA', frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fs = max(12, int(w / 120))
    f = ImageFont.truetype(F_UI, fs)
    for c in rng.integers(0, w, count):
        phase = (t * 0.35 + rng.random()) % 1.0
        for k in range(14):
            y = int((phase * h + k * fs * 2.1) % h)
            ch = '1' if (k + int(t * 3)) % 3 else '0'
            d.text((int(c), y), ch, font=f,
                   fill=tuple(colour) + (int(alpha * (0.25 + 0.75 * k / 14)),))
    layer.putalpha(layer.getchannel('A').point(lambda v: int(v * 0.72)))
    frame.alpha_composite(layer)


# ------------------------------------------------------------------- wordmark
_wordmark_cache: dict[tuple, Image.Image] = {}


def wordmark(palette: dict) -> Image.Image:
    """Real DeepSeek lettering from the brand clip, re-inked in the clip's palette.

    Located by projection, which is fussy in that corner of the frame: rows
    qualify when their ink is wide, and the *longest contiguous run* of such rows
    is kept (rejects the 1-2px UI rules); columns qualify when filled over most
    of the band (letter stems), and the largest column gap cuts detached specks
    (hair strands at the edge of the crop).
    """
    key = tuple(palette['wordmark'][1])
    if key in _wordmark_cache:
        return _wordmark_cache[key]
    from PIL import ImageOps

    im = Image.open(BRAND_FRAME).convert('RGB')
    crop = im.crop((int(im.width * 0.05), int(im.height * 0.26),
                    int(im.width * 0.60), int(im.height * 0.57)))
    lum = np.asarray(crop.convert('L'), dtype=np.int16)
    ink = lum > 42
    spans = []
    for r in range(ink.shape[0]):
        xs = np.where(ink[r])[0]
        spans.append(0 if xs.size == 0 else xs.max() - xs.min())
    wide = np.array([sp > crop.width * 0.30 for sp in spans])
    best = (0, 0, 0)
    r = 0
    while r < len(wide):
        if not wide[r]:
            r += 1
            continue
        j = r
        while j + 1 < len(wide) and (wide[j + 1] or (j + 2 < len(wide) and wide[j + 2])):
            j += 1
        if j - r > best[0]:
            best = (j - r, r, j)
        r = j + 1
    _, y0, y1 = best
    band = ink[y0:y1 + 1]
    fill = band.mean(axis=0)
    cols = np.where(fill > 0.12)[0]
    gaps = np.diff(cols)
    if gaps.size and gaps.max() > crop.width * 0.04:
        cols = cols[cols <= cols[int(np.argmax(gaps))]]
    pad = 3
    x0, x1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad, crop.width)
    word = crop.crop((x0, max(y0 - pad, 0), x1, min(y1 + pad, crop.height)))

    gl = word.convert('L')
    shape = gl.point(lambda v: 0 if v < 30 else min(255, int((v - 30) * 2.6)))
    alpha = shape.point(lambda v: min(255, int(v * 2.0)))
    rgb = ImageOps.colorize(shape, black=palette['wordmark'][0], mid=palette['wordmark'][1],
                            white=palette['wordmark'][2])
    mark = rgb.convert('RGBA')
    mark.putalpha(alpha)
    bb = alpha.getbbox()
    if bb:
        mark = mark.crop(bb)
    _wordmark_cache[key] = mark
    return mark


# ------------------------------------------------------------------------- text
def text_layer(size: tuple[int, int], clip: dict, progress: float, t_show: float) -> Image.Image:
    pal = clip['palette']
    W, H = size
    s = W / 1920.0
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    k = ease(min(t_show / 0.8, 1.0))

    scrim = Image.new('RGBA', size, (0, 0, 0, 0))
    sw_, sh_ = int(1180 * s), int(660 * s)
    ramp = np.zeros((sh_, sw_, 4), np.uint8)
    gx = np.linspace(1.0, 0.0, sw_) ** 1.7
    gy = np.clip(1.0 - np.abs(np.linspace(-1, 1, sh_)) ** 3, 0, 1)
    a = (gx[None, :] * gy[:, None] * 74 * k).astype(np.uint8)
    ramp[..., 3] = a
    ramp[..., 0:3] = (3, 6, 14)
    scrim.alpha_composite(Image.fromarray(ramp, 'RGBA'), (int(96 * s), int(300 * s)))
    layer.alpha_composite(scrim)

    mark = wordmark(pal)
    mw = int(620 * s)
    mark = mark.resize((mw, max(1, int(mark.height * mw / mark.width))), Image.LANCZOS)
    halo = Image.new('RGBA', (mark.width + 40, mark.height + 40), (0, 0, 0, 0))
    sil = Image.new('RGBA', mark.size, (2, 4, 10, 210))
    sil.putalpha(mark.getchannel('A').point(lambda v: min(255, int(v * 0.95))))
    halo.alpha_composite(sil, (20, 20))
    halo = halo.filter(ImageFilter.GaussianBlur(13))
    layer.alpha_composite(halo, (int(196 * s) - 20, int(356 * s) - 20))
    layer.alpha_composite(mark, (int(196 * s), int(356 * s)))

    d = ImageDraw.Draw(layer)
    f_cn = ImageFont.truetype(F_CN, int(52 * s))
    f_en = ImageFont.truetype(F_UI, int(23 * s))
    x, y = int(200 * s), int(560 * s)
    d.text((x + 2, y + 2), clip['taglines'][0], font=f_cn, fill=(6, 16, 26, int(200 * k)))
    d.text((x, y), clip['taglines'][0], font=f_cn, fill=(232, 244, 255, int(255 * k)))
    d.text((x + 3, y + int(74 * s)), ' '.join(clip['taglines'][1]), font=f_en,
           fill=tuple(pal['ui']) + (int(215 * k),))
    uw = int(300 * s * k)
    d.rectangle((x, y + int(66 * s), x + uw, y + int(68 * s)),
                fill=tuple(pal['under']) + (int(200 * k),))

    f_top = ImageFont.truetype(F_UI, int(20 * s))
    f_lbl = ImageFont.truetype(F_UI, int(17 * s))
    d.polygon([(int(62 * s), int(52 * s)), (int(70 * s), int(62 * s)),
               (int(62 * s), int(72 * s)), (int(54 * s), int(62 * s))],
              fill=tuple(pal['accent']) + (int(230 * k),))
    d.text((int(84 * s), int(50 * s)), 'DEEPSEEK', font=f_top, fill=(220, 238, 250, int(225 * k)))
    d.text((int(86 * s), int(76 * s)), 'V1.0', font=f_lbl, fill=tuple(pal['ui']) + (int(175 * k),))
    d.text((int(1230 * s), int(50 * s)), 'DEEPSEEK', font=f_lbl, fill=(200, 224, 240, int(190 * k)))
    d.text((int(1232 * s), int(74 * s)), 'V1.0', font=f_lbl, fill=tuple(pal['ui']) + (int(160 * k),))
    d.text((int(56 * s), int(1010 * s)), 'YOUR AI COMPANION', font=f_lbl,
           fill=tuple(pal['ui']) + (int(175 * k),))
    d.text((int(56 * s), int(1034 * s)), 'FOR A SMARTER FUTURE', font=f_lbl,
           fill=tuple(pal['ui']) + (int(175 * k),))
    d.polygon([(int(1240 * s), int(1012 * s)), (int(1247 * s), int(1020 * s)),
               (int(1240 * s), int(1028 * s)), (int(1233 * s), int(1020 * s))],
              fill=tuple(pal['accent']) + (int(190 * k),))
    d.text((int(1252 * s), int(1010 * s)), 'READY', font=f_lbl,
           fill=(198, 226, 244, int(185 * k)))

    f_st = ImageFont.truetype(F_UI, int(19 * s))
    y0 = int(560 * s)
    for i, line in enumerate(clip['status']):
        rev = min(max((t_show - (1.1 + i * 0.42)) / 0.5, 0.0), 1.0)
        vis = line[:int(len(line) * rev)]
        if vis:
            d.text((int(58 * s), y0 + i * int(26 * s)), vis, font=f_st,
                   fill=tuple(pal['ui']) + (int(210 * k),))
    bx = int(58 * s)
    by = y0 + len(clip['status']) * int(26 * s) + int(10 * s)
    d.rectangle((bx, by, bx + int(150 * s), by + int(3 * s)), fill=(60, 70, 96, int(160 * k)))
    d.rectangle((bx, by, bx + int(150 * s * progress), by + int(3 * s)),
                fill=tuple(pal['accent']) + (int(225 * k),))
    return layer


# -------------------------------------------------------------------- vignette
def vignette(size: tuple[int, int], strength: float = 0.55) -> Image.Image:
    W, H = size
    y, x = np.mgrid[0:H, 0:W]
    nx = (x - W / 2) / (W / 2)
    ny = (y - H / 2) / (H / 2)
    r = np.sqrt(nx ** 2 * 0.92 + ny ** 2)
    v = np.clip(1.0 - strength * np.clip(r - 0.35, 0, None) ** 1.7, 0, 1)
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    layer.putalpha(Image.fromarray(((1 - v) * 255).astype(np.uint8)))
    return layer


# ----------------------------------------------------------------------- audio
def synth_audio(duration: float, clip_id: str, sr: int = 32000) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7 if clip_id == 'resonance' else 11)
    noise = rng.normal(0, 1, n)

    def lowpass(x, cutoff):
        a = math.exp(-2 * math.pi * cutoff / sr)
        y = np.empty_like(x)
        acc = 0.0
        for i in range(len(x)):
            acc = a * acc + (1 - a) * x[i]
            y[i] = acc
        return y

    def bp(x, lo, hi):
        return lowpass(x, hi) - lowpass(x, lo)

    warm = clip_id == 'resonance'
    f0 = 48.0 if warm else 58.0
    sweep = f0 * (1 - 0.32 * t / duration)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    sub = np.sin(phase) * (0.5 * np.clip(t / 1.6, 0, 1) * np.exp(-t * 0.22))
    amb = bp(noise, 90, 700) * 0.16
    pad_f = [174.6, 261.6, 349.2] if warm else [220.0, 330.0, 440.0]
    pad = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate(pad_f)) / 3.0
    pad *= 0.10 * np.clip(t / 2.0, 0, 1) * np.clip((duration - t) / 1.4, 0, 1)
    imp = duration * (0.63 if warm else 0.58)
    riser_env = np.clip((t - imp + 1.5) / 1.5, 0, 1) ** 2.2 * (t < imp)
    riser = bp(noise, 300, 5000) * riser_env * 0.34
    riser_tone = np.sin(2 * np.pi * (300 + 900 * riser_env) * t) * riser_env * 0.07
    di = np.clip(t - imp, 0, None)
    hit = (np.sin(2 * np.pi * 62 * di) * np.exp(-di * 7.0) * 0.55 +
           bp(noise, 400, 6000) * np.exp(-di * 22.0) * 0.30)
    db = np.clip(t - (duration - 2.1), 0, None)
    bell = (np.sin(2 * np.pi * 523.25 * db) * np.exp(-db * 2.4) * 0.16 +
            np.sin(2 * np.pi * 784.88 * db) * np.exp(-db * 3.1) * 0.09 +
            np.sin(2 * np.pi * 1046.5 * db) * np.exp(-db * 3.6) * 0.05)
    if warm:      # gold chime right after the hit
        dc = np.clip(t - imp - 0.12, 0, None)
        bell = bell + (np.sin(2 * np.pi * 1318.5 * dc) * np.exp(-dc * 3.4) * 0.10)
    mix = sub + amb + pad + riser + riser_tone + hit + bell
    mix *= np.clip(t / 0.25, 0, 1) * np.clip((duration - t) / 0.5, 0, 1)
    mix = mix / (float(np.max(np.abs(mix))) or 1.0) * 0.78
    return np.stack([mix, mix * 0.97], axis=1)


def write_wav(path: str, data: np.ndarray, sr: int = 32000) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------- clips
CLIPS: dict[str, dict] = {
    'resonance': dict(
        duration=8.0,
        taglines=('同频 · 数据共鸣', 'RESONANCE  ·  IN SYNC WITH YOU'),
        status=['ANALYSIS', 'THINKING', 'SEARCHING', 'GENERATING'],
        palette=PALETTES['resonance'],
        shots=[
            Shot('q3_hall', 0.0, 1.7, (0.50, 0.47, 0.96), (0.50, 0.44, 0.76), streaks=2, fade_in=0.6),
            Shot('q3_hall_b', 1.7, 3.1, (0.50, 0.44, 0.92), (0.50, 0.40, 0.72), streaks=2),
            Shot('r1_hand', 3.1, 4.3, (0.47, 0.50, 0.42), (0.50, 0.50, 0.32), streaks=2),
            Shot('q1_portrait', 4.3, 5.5, (0.64, 0.46, 0.82), (0.64, 0.46, 0.68), streaks=0),
            Shot('q3_eye', 5.5, 6.3, (0.50, 0.50, 0.60), (0.50, 0.50, 0.44), streaks=0),
            Shot('q3_hall', 6.3, 8.0, (0.50, 0.44, 0.74), (0.50, 0.47, 0.96), flash_in=0.13, streaks=2),
        ],
    ),
    'ascension': dict(
        duration=7.6,
        taglines=('升维 · 思考的跃迁', 'ASCENSION  ·  A LEAP IN THINKING'),
        status=['ANALYSIS', 'THINKING', 'SEARCHING', 'GENERATING'],
        palette=PALETTES['ascension'],
        shots=[
            Shot('r2_trail', 0.0, 1.4, (0.52, 0.58, 0.44), (0.46, 0.48, 0.34), streaks=2, fade_in=0.6),
            Shot('r2_figure', 1.4, 3.0, (0.50, 0.52, 0.44), (0.50, 0.48, 0.34), streaks=3),
            Shot('r2_core', 3.0, 4.4, (0.50, 0.50, 0.46), (0.50, 0.50, 0.32), streaks=2),
            Shot('r2_burst', 4.4, 5.3, (0.50, 0.50, 0.52), (0.50, 0.50, 0.34), flash_in=0.18, streaks=5),
            Shot('r2_gate', 5.3, 6.3, (0.50, 0.50, 0.40), (0.50, 0.50, 0.28), streaks=2),
            Shot('r2_core', 6.3, 7.6, (0.50, 0.50, 0.30), (0.50, 0.50, 0.42), flash_in=0.13, streaks=2),
        ],
    ),
}


def render_frame(t: float, clip: dict, parts: Particles, vg: Image.Image,
                 grain_seed: int) -> Image.Image:
    shots: list[Shot] = clip['shots']
    pal = clip['palette']
    active = [s for s in shots if s.t0 - 1e-6 <= t < s.t1] or [shots[-1]]
    sh = active[0]
    frame = sh.frame(t, (BASE_W, BASE_H)).convert('RGBA')

    idx = shots.index(sh)
    if idx + 1 < len(shots) and t > sh.t1 - XFADE:
        nxt = shots[idx + 1]
        a = ease((t - (sh.t1 - XFADE)) / XFADE)
        frame = Image.blend(frame, nxt.frame(t, (BASE_W, BASE_H)).convert('RGBA'), a)

    # bloom: threshold the brights, blur, add back
    small = frame.convert('RGB').resize((BASE_W // 4, BASE_H // 4), Image.BILINEAR)
    lum = small.convert('L').point(lambda v: 0 if v < 112 else int((v - 112) * 1.55))
    glow = small.copy()
    glow.putalpha(lum)
    glow = glow.resize(frame.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(22))
    glow.putalpha(glow.getchannel('A').point(lambda v: int(v * 0.85)))
    frame.alpha_composite(glow)

    parts.draw(frame, t, sh.particles * min(1.0, 0.35 + t * 0.5))
    draw_streaks(frame, t, seed=hash(sh.art) % 9973, count=sh.streaks, alpha=64,
                 colour=pal['accent'])
    frame.alpha_composite(vg, (0, 0))

    rng = np.random.default_rng(grain_seed)
    arr = np.asarray(frame.convert('RGB')).astype(np.int16)
    arr += rng.integers(-3, 4, arr.shape, dtype=np.int16)
    frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert('RGBA')

    if sh.flash_in and t - sh.t0 < sh.flash_in * 2:
        k = 1.0 - (t - sh.t0) / (sh.flash_in * 2)
        col = mix((255, 255, 255), pal['accent'], 0.35)
        frame.alpha_composite(Image.new('RGBA', frame.size, tuple(col) + (int(205 * k ** 1.6),)))
    if sh.fade_in:
        k = min((t - sh.t0) / sh.fade_in, 1.0)
        frame.alpha_composite(Image.new('RGBA', frame.size, (3, 5, 12, int(255 * (1 - k)))))
    dur = clip['duration']
    if t > dur - 0.5:
        k = (t - (dur - 0.5)) / 0.5
        frame.alpha_composite(Image.new('RGBA', frame.size, (2, 4, 10, int(235 * k))))
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True, choices=sorted(CLIPS))
    ap.add_argument('--work', default='/tmp/orig')
    ap.add_argument('--out', default=os.path.join(REPO, 'media', 'original'))
    ap.add_argument('--crf', type=float, default=15)
    args = ap.parse_args()

    clip = CLIPS[args.clip]
    dur = clip['duration']
    total = int(round(dur * FPS))
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    base = f'deepseek-{args.clip}-intro'

    for sh in clip['shots']:
        path = os.path.join(ART_SR, f'{sh.art}_x4.png')
        sh.bind(Image.open(path).convert('RGB'))
        print(f'  bound {sh.art} -> {sh.img.size}', flush=True)
    parts = Particles(120, seed=3 if args.clip == 'resonance' else 5,
                      size=(BASE_W, BASE_H), palette=clip['palette'])
    vg = vignette((BASE_W, BASE_H), strength=0.58)

    wav = os.path.join(args.work, f'{base}.wav')
    write_wav(wav, synth_audio(dur, args.clip))

    master = os.path.join(args.work, f'{base}-master.mkv')
    p = subprocess.Popen([ffmpeg(), '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                          '-s', f'{BASE_W}x{BASE_H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'ultrafast',
                          '-g', str(FPS), '-f', 'matroska', master], stdin=subprocess.PIPE)
    for i in range(total):
        t = i / FPS
        lay = text_layer((BASE_W, BASE_H), clip, ease_out(min(t / (dur - 0.4), 1.0)), t)
        frame = render_frame(t, clip, parts, vg, grain_seed=1000 + i)
        frame.alpha_composite(lay)
        p.stdin.write(frame.convert('RGB').tobytes())
        if i % 48 == 0:
            print(f'  frame {i}/{total}', flush=True)
    p.stdin.close()
    p.wait()
    print(f'master -> {master} ({os.path.getsize(master)/1e6:.1f} MB)')

    for tag, (w, h), level, ref in (('2k', (2560, 1440), '5.1', 5), ('1080p', (1920, 1080), '4.1', 4)):
        out = os.path.join(args.out, f'{base}-{tag}.mp4')
        vf = 'format=yuv420p' if (w, h) == (BASE_W, BASE_H) else \
            f'scale={w}:{h}:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p'
        subprocess.run([ffmpeg(), '-hide_banner', '-v', 'warning', '-y', '-i', master,
                        '-i', wav, '-map', '0:v:0', '-map', '1:a:0', '-vf', vf,
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
                        '-profile:v', 'high', '-level', level,
                        '-x264-params', f'ref={ref}:bframes=4:aq-mode=3:rc-lookahead=40:me=umh:subme=8',
                        '-c:a', 'aac', '-b:a', '128k', '-ar', '32000', '-ac', '2',
                        '-movflags', '+faststart',
                        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                        '-metadata:s:v:0',
                        f'title=DeepSeek {args.clip} intro {w}x{h} (original composition)',
                        out], check=True)
        print(f'  {out} ({os.path.getsize(out)/1e6:.1f} MB)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
