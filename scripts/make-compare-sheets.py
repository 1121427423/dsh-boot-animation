#!/usr/bin/env python3
"""Build the before/after sheets for every rebuilt clip.

Left column of each sheet is the 720p source resampled to 1080p (what a player
shows today), right column is the delivered 1080p rebuild — same frames, so the
comparison is honest.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

FF = os.environ.get('FFMPEG') or 'ffmpeg'
REPO = '/home/user/dsh-boot-animation'
MEDIA = f'{REPO}/media'
OUT = f'{MEDIA}/hq'
TMP = '/tmp/hqwork/sheets'
F_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

CLIPS = [
    ('startup', 'startup', '3.0|6.6|7.6'),
    ('brand', 'brand', '1.2|3.5|6.4'),
    ('cyberpunk', 'cyberpunk', '1.2|3.5|6.4'),
    ('awakening', 'awakening', '1.2|3.5|6.4'),
]


def frame(path, t, w, h, out):
    subprocess.run([FF, '-hide_banner', '-v', 'error', '-y', '-ss', str(t), '-i', path,
                    '-frames:v', '1', '-vf', f'scale={w}:{h}:flags=lanczos', out], check=True)
    return out


def main():
    os.makedirs(TMP, exist_ok=True)
    fb = ImageFont.truetype(F_BOLD, 26)
    fs = ImageFont.truetype(F_BOLD, 20)
    for cid, _, times in CLIPS:
        src = f'{MEDIA}/deepseek-{cid}-intro.mp4'
        new = f'{OUT}/deepseek-{cid}-intro-1080p.mp4'
        if not os.path.exists(new):
            print(f'skip {cid}: {new} missing')
            continue
        cell = (620, 349)
        ts = times.split('|')
        rows = [('old', 'OLD  1280x720 source, upscaled to 1080p (what a player shows today)'),
                ('new', 'NEW  1920x1080 rebuilt by Real-ESRGAN 4x + Lanczos')]
        W = cell[0] * len(ts) + 12 * (len(ts) + 1)
        H = 52 + (cell[1] + 34) * 2 + 12
        sheet = Image.new('RGB', (W, H), (12, 14, 20))
        d = ImageDraw.Draw(sheet)
        for r, (kind, label) in enumerate(rows):
            y = 52 + r * (cell[1] + 34)
            d.text((14, y - 24), label, font=fs,
                   fill=(200, 200, 200) if r == 0 else (160, 205, 255))
            for i, t in enumerate(ts):
                p = f'{TMP}/{cid}_{kind}_{t}.png'
                if kind == 'old':
                    frame(src, t, 1920, 1080, p)
                else:
                    frame(new, t, 1920, 1080, p)
                sheet.paste(Image.open(p).convert('RGB').resize(cell, Image.LANCZOS),
                            (12 + i * (cell[0] + 12), y))
        sheet.save(f'{OUT}/compare-{cid}-full.png')
        print(f'wrote compare-{cid}-full.png')


if __name__ == '__main__':
    sys.exit(main())
