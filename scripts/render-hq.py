#!/usr/bin/env python3
"""render-hq.py — rebuild 1080p / 2K masters from a 1280x720 source clip.

Pipeline (one pass, resumable):

    source .mp4
      -> ffmpeg rawvideo (rgb24, full range)
      -> realesr-general-x4v3 (SRVGGNetCompact, 4x, ncnn, fp32)  5120x2880 per frame
      -> lossless libx264rgb chunks (matroska), one chunk per N frames
      -> concat -> Lanczos + accurate-rounding resample to each target size
         -> yuv420p tv/bt709, x264 CRF, faststart mp4, source audio copied

Every step is reproducible: same source, same model, same parameters give the
same bytes (libx264 is deterministic for a fixed build/binary, the SR model is
fixed-point-free float32).

Model files are not shipped in this repo.  Fetch a Real-ESRGAN *compact*
(SRVGGNetCompact) 4x model and point --model-param/--model-bin at it, e.g.

    git clone --depth 1 https://github.com/IbrahimGhadre/realesrgan-mobile
    file: android/ncnn-app/app/src/main/assets/realesr_general_x4v3.ncnn.{param,bin}

The ONNX weights from xinntao/Real-ESRGAN-v0.2.5.0
(realesr-general-x4v3.pth / realesr-animevideov3.pth) can be exported to the
same ncnn layout; the compact models are the ones that fit a CPU-only box.

Usage
-----
    python3 scripts/render-hq.py \\
        --src media/deepseek-startup-intro.mp4 \\
        --work /tmp/hq-startup \\
        --model-param $MODEL/realesr_general_x4v3.ncnn.param \\
        --model-bin   $MODEL/realesr_general_x4v3.ncnn.bin \\
        --out media/hq \\
        --targets 1920x1080,2560x1440

Re-running resumes: chunks already on disk are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

SR_SCALE = 4
FPS_DEFAULT = 24


# --------------------------------------------------------------------------- tools
def ffmpeg_exe() -> str:
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # optional, pure-python wheel that carries a static build
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f'no ffmpeg found ({exc}); set $FFMPEG or install ffmpeg') from exc


def probe(path: str) -> dict:
    """Minimal ffprobe replacement: parse `ffmpeg -i` output."""
    ff = ffmpeg_exe()
    out = subprocess.run([ff, '-hide_banner', '-i', path], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True).stdout
    info: dict = {'path': path, 'bytes': os.path.getsize(path)}
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', out)
    if m:
        h, mi, s = m.groups()
        info['duration'] = int(h) * 3600 + int(mi) * 60 + float(s)
    for line in out.splitlines():
        if 'Video:' in line and 'Stream #' in line:
            vm = re.search(r'(\d{2,5})x(\d{2,5})', line)
            if vm:
                info['width'], info['height'] = int(vm.group(1)), int(vm.group(2))
            fm = re.search(r'([\d.]+) fps', line)
            if fm:
                info['fps'] = float(fm.group(1))
            info['video_codec'] = re.search(r'Video: (\S+)', line).group(1).rstrip(',')
        elif 'Audio:' in line and 'Stream #' in line:
            info['audio_codec'] = re.search(r'Audio: (\S+)', line).group(1).rstrip(',')
    return info


def sha256(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------- super-resolution
def make_net(param: str, bin_: str, threads: int):
    import ncnn

    net = ncnn.Net()
    o = net.opt
    o.num_threads = threads
    o.use_packing_layout = True
    # fp16 *storage* overflows on this material (specular flares blow past 65504,
    # the compact model has no clamping).  fp32 costs ~12% speed and stays exact.
    o.use_fp16_packed = False
    o.use_fp16_storage = False
    o.use_fp16_arithmetic = False
    net.load_param(param)
    net.load_model(bin_)
    return net


def frame_reader(ff: str, src: str, w: int, h: int, start: int, count: int):
    """Yield (frame_index, HxWx3 uint8) for `count` frames starting at `start`."""
    cmd = [ff, '-v', 'error', '-i', src, '-map', '0:v', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)
    n = w * h * 3
    idx = 0
    produced = 0
    try:
        while produced < count:
            buf = p.stdout.read(n)
            if not buf or len(buf) < n:
                break
            if idx >= start:
                produced += 1
                yield idx, buf
            idx += 1
    finally:
        p.stdout.close()
        p.wait()


def render_chunk(net, ff: str, src: str, w: int, h: int, start: int, count: int,
                 out: str, log, fps: int) -> int:
    import numpy as np

    tmp = out + '.part.mkv'
    big_w, big_h = w * SR_SCALE, h * SR_SCALE
    cmd = [ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{big_w}x{big_h}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'ultrafast',
           '-g', str(fps), '-f', 'matroska', tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    done = 0
    try:
        for idx, raw in frame_reader(ff, src, w, h, start, count):
            t = time.time()
            arr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            chw = arr.astype(np.float32).transpose(2, 0, 1) / 255.0
            with net.create_extractor() as ex:
                import ncnn
                ex.input('in0', ncnn.Mat(np.ascontiguousarray(chw)).clone())
                _, o = ex.extract('out0')
            big = np.array(o)
            if big.dtype != np.float32 or not np.isfinite(big).all():
                raise RuntimeError(f'frame {idx}: bad SR output (dtype={big.dtype})')
            np.clip(big, 0.0, 1.0, out=big)
            p.stdin.write((big * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0).tobytes())
            done += 1
            log(f'  frame {idx} ({done}/{count}) {time.time()-t:5.1f}s  '
                f'elapsed {time.time()-t0:6.1f}s')
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'lossless writer failed rc={p.returncode}')
    if done != count:
        raise RuntimeError(f'wanted {count} frames, got {done}')
    os.replace(tmp, out)
    log(f'  wrote {os.path.basename(out)} ({os.path.getsize(out)/1e6:.1f} MB, '
        f'{done} frames, {time.time()-t0:.1f}s)')
    return done


# ----------------------------------------------------------------------------- encode
def encode_target(ff: str, master_list: str, audio: str, target: tuple[int, int],
                  out: str, crf: float, preset: str, log) -> dict:
    w, h = target
    level = '4.1' if h <= 1080 else '5.1'
    ref = 4 if h <= 1080 else 5
    cmd = [ff, '-hide_banner', '-v', 'warning', '-y',
           '-f', 'concat', '-safe', '0', '-i', master_list]
    if audio:
        cmd += ['-i', audio, '-map', '1:a:0']
    cmd += ['-map', '0:v:0',
            '-vf', (f'scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int:'
                    'out_color_matrix=bt709:out_range=tv,format=yuv420p'),
           '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
           '-profile:v', 'high', '-level', level,
           '-x264-params', (f'ref={ref}:bframes=4:aq-mode=3:rc-lookahead=40:'
                            'me=umh:subme=8:psy-rd=1.0,0.15'),
           '-c:a', 'copy', '-movflags', '+faststart',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-metadata:s:v:0', f'title=DeepSeek startup intro {w}x{h} (super-resolved from source)',
           out]
    subprocess.run(cmd, check=True)
    info = probe(out)
    log(f'  {out}: {info["bytes"]/1e6:.1f} MB, {info["width"]}x{info["height"]}, '
        f'{info.get("video_codec")} @ {info.get("duration", 0):.2f}s')
    return info


def psnr_vs_master(ff: str, master_list: str, encoded: str, w: int, h: int) -> str:
    vf = (f'[0:v]scale={w}:{h}:flags=lanczos+accurate_rnd:out_color_matrix=bt709:'
          'out_range=tv,format=yuv420p[a];[a][1:v]psnr=stats_file=-')
    out = subprocess.run([ff, '-hide_banner', '-v', 'error', '-f', 'concat', '-safe', '0',
                          '-i', master_list, '-i', encoded, '-lavfi', vf, '-f', 'null', '-'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    hits = re.findall(r'psnr_avg:([\d.]+)', out)
    return hits[-1] if hits else ''


# ------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--work', required=True, help='scratch dir (master chunks, audio, logs)')
    ap.add_argument('--out', required=True, help='directory for the encoded masters')
    ap.add_argument('--model-param', required=True)
    ap.add_argument('--model-bin', required=True)
    ap.add_argument('--targets', default='1920x1080,2560x1440')
    ap.add_argument('--chunk-frames', type=int, default=30)
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--crf', type=float, default=15)
    ap.add_argument('--preset', default='slow')
    ap.add_argument('--fps', type=int, default=FPS_DEFAULT)
    ap.add_argument('--max-frames', type=int, default=0, help='debug: stop early')
    ap.add_argument('--manifest', default='', help='write a provenance json here')
    args = ap.parse_args()

    ff = ffmpeg_exe()
    src = os.path.abspath(args.src)
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(work, 'render.log'), 'a', buffering=1)

    def log(msg: str) -> None:
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')

    sinfo = probe(src)
    w, h = sinfo['width'], sinfo['height']
    fps = sinfo.get('fps') or args.fps
    total = int(round(sinfo['duration'] * fps))
    if args.max_frames:
        total = min(total, args.max_frames)
    log(f'source {src}: {w}x{h} {fps:g}fps {sinfo["duration"]:.2f}s -> {total} frames')
    log(f'SR {SR_SCALE}x -> {w*SR_SCALE}x{h*SR_SCALE}, model {os.path.basename(args.model_bin)}')

    # 1. audio, copied untouched into every delivery encode
    audio = os.path.join(work, 'audio.m4a')
    if not os.path.exists(audio):
        subprocess.run([ff, '-v', 'error', '-y', '-i', src, '-map', '0:a?', '-c', 'copy', audio],
                       check=True)
    has_audio = os.path.getsize(audio) > 0

    # 2. super-resolve into lossless chunks
    net = None
    idx = 0
    while idx < total:
        out_chunk = os.path.join(work, 'chunks', f'chunk_{idx:04d}.mkv')
        os.makedirs(os.path.dirname(out_chunk), exist_ok=True)
        count = min(args.chunk_frames, total - idx)
        if os.path.exists(out_chunk):
            log(f'skip {os.path.basename(out_chunk)} (done)')
        else:
            if net is None:
                net = make_net(args.model_param, args.model_bin, args.threads)
                log(f'model loaded (threads={args.threads}, fp32)')
            log(f'chunk {idx}..{idx+count-1}')
            render_chunk(net, ff, src, w, h, idx, count, out_chunk, log, int(fps))
        idx += count

    # 3. concat list + a single lossless master (kept out of --out on purpose)
    chunks = sorted(f for f in os.listdir(os.path.join(work, 'chunks')) if f.endswith('.mkv'))
    master_list = os.path.join(work, 'master.txt')
    with open(master_list, 'w') as fh:
        for c in chunks:
            fh.write(f"file '{os.path.join(work, 'chunks', c)}'\n")
    master = os.path.join(work, 'master.mkv')
    if not os.path.exists(master):
        subprocess.run([ff, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', master_list,
                        '-map', '0:v', '-c', 'copy', master], check=True)
        log(f'lossless master {master} ({os.path.getsize(master)/1e9:.2f} GB)')

    # 4. delivery encodes
    results = {}
    for spec in args.targets.split(','):
        tw, th = (int(x) for x in spec.strip().lower().split('x'))
        base = os.path.splitext(os.path.basename(src))[0]
        suffix = f'{th}p' if th in (720, 1080, 1440, 2160) else f'{tw}x{th}'
        out = os.path.join(args.out, f'{base}-{suffix}.mp4')
        log(f'encode {tw}x{th} -> {out}')
        info = encode_target(ff, master_list, audio if has_audio else '', (tw, th),
                             out, args.crf, args.preset, log)
        info['psnr_vs_lossless_master_db'] = psnr_vs_master(ff, master_list, out, tw, th)
        results[f'{tw}x{th}'] = info
        log(f'  PSNR {info["psnr_vs_lossless_master_db"]} dB (vs the lossless 4x master)')

    if args.manifest:
        man = {
            'source': {**sinfo, 'sha256': sha256(src)},
            'sr_model': {'file': os.path.basename(args.model_bin),
                         'sha256': sha256(args.model_bin), 'scale': SR_SCALE},
            'targets': results,
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        with open(args.manifest, 'w') as fh:
            json.dump(man, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        log(f'manifest -> {args.manifest}')
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
