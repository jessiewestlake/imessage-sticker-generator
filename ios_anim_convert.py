#!/usr/bin/env python3
# ios_anim_convert.py
#
# Convert animated GIF/APNG/WebP -> (a) iOS sticker APNG (≤500 KB), (b) Live Photo pair.
# Usage examples:
#   python ios_anim_convert.py in.gif --sticker sticker.apng --size large
#   python ios_anim_convert.py in.webp --live live_out --size 618
#
# Jessie-ready: robust CLI, careful size control, and macOS-only Live Photo tagging if available.

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageSequence

try:
    from apng import APNG
except ImportError:
    print("Missing dependency: apng. Install with: pip install apng", file=sys.stderr)
    raise


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load_frames(path: Path) -> Tuple[List[Image.Image], List[int]]:
    """Return (frames, durations_ms). Supports GIF, APNG, WebP; falls back to single frame."""
    img = Image.open(path)
    frames, durs = [], []
    n = getattr(img, "n_frames", 1)
    for i in range(n):
        try:
            img.seek(i)
        except EOFError:
            break
        frame = img.convert("RGBA")
        dur = img.info.get("duration", 100)  # ms
        if dur <= 0:
            dur = 100
        frames.append(frame.copy())
        durs.append(int(dur))
    if not frames:
        frames = [img.convert("RGBA")]
        durs = [500]
    return frames, durs


def quantize_rgba(img: Image.Image, colors: int) -> Image.Image:
    # Quantize via adaptive palette, then restore alpha from original
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.getchannel("A")
    pal = img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
    pal = pal.convert("RGBA")
    pal.putalpha(alpha)
    return pal


def save_frames_to_pngs(
    frames: List[Image.Image], outdir: Path, base="frame"
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fr in enumerate(frames, 1):
        p = outdir / f"{base}_{i:04d}.png"
        fr.save(p, format="PNG", optimize=True, compress_level=9)
        paths.append(p)
    return paths


def build_apng(paths: List[Path], durations_ms: List[int], outpath: Path):
    apng = APNG()
    # APNG uses delay as (num, den). Use denominator=1000 for ms precision.
    for p, dur in zip(paths, durations_ms):
        apng.append_file(str(p), delay=(dur, 1000))
    apng.save(str(outpath))


def size_ok(path: Path, max_kb=500) -> bool:
    return path.stat().st_size <= max_kb * 1024


def make_ios_sticker(infile: Path, out_apng: Path, size_choice: str):
    # sizes per Apple sticker guidelines
    preset = {"small": 300, "medium": 408, "large": 618}
    if size_choice.isdigit():
        target_px = int(size_choice)
    else:
        target_px = preset.get(size_choice.lower(), 618)

    frames, durs = load_frames(infile)
    # Iterative compression plan: try {colors} × {size} until <=500KB
    size_steps = [target_px]
    for s in (408, 300):
        if s < target_px:
            size_steps.append(s)
    color_steps = [256, 180, 128, 96, 64, 48, 32]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for px in size_steps:
            # Resize frames to square (stickers are square); preserve aspect by padding
            sized = []
            for f in frames:
                # scale to fit within px×px, then center on transparent canvas
                ratio = min(px / f.width, px / f.height)
                # Split long tuple construction to satisfy 79-char limit
                new_w = max(1, int(f.width * ratio))
                new_h = max(1, int(f.height * ratio))
                resized = f.resize((new_w, new_h), Image.LANCZOS)
                canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
                canvas.paste(
                    resized,
                    ((px - resized.width) // 2, (px - resized.height) // 2),
                    resized,
                )
                sized.append(canvas)

            for colors in color_steps:
                qframes = [quantize_rgba(fr, colors) for fr in sized]
                pngs = save_frames_to_pngs(qframes, tmp)
                build_apng(pngs, durs, out_apng)
                if size_ok(out_apng):
                    print(
                        f"[OK] Sticker APNG created at {out_apng} ({out_apng.stat().st_size / 1024:.1f} KB, {px}px, {colors} colors)"
                    )
                    return
        # If we reach here we didn’t get under 500 KB
        print(
            f"[WARN] Could not compress < 500 KB. Output is {out_apng.stat().st_size / 1024:.1f} KB. "
            f"Try fewer frames or smaller size.",
            file=sys.stderr,
        )


def avg_fps_from_durations(durs_ms: List[int]) -> float:
    if not durs_ms:
        return 15.0
    mean = sum(durs_ms) / len(durs_ms)
    if mean <= 1:  # avoid absurd fps
        return 30.0
    return max(5.0, min(30.0, 1000.0 / mean))


def make_live_photo(infile: Path, out_base: Path, jpg_quality=92, try_tag=True):
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found on PATH. Please install it.")

    frames, durs = load_frames(infile)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    key_jpg = out_base.with_suffix(".jpg")
    video_mov = out_base.with_suffix(".mov")

    # 1) Write key photo (first frame)
    frames[0].convert("RGB").save(key_jpg, "JPEG", quality=jpg_quality, optimize=True)

    # 2) Encode .mov from frames
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pngs = save_frames_to_pngs(frames, tmp)
        fps = avg_fps_from_durations(durs)
        # Build input pattern
        pattern = str(tmp / "frame_%04d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            pattern,
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_mov),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(
        f"[OK] Live Photo components written:\n  Photo: {key_jpg}\n  Video: {video_mov}"
    )

    # 3) (macOS optional) tag both files with the same asset identifier using 'makelive'
    if try_tag:
        try:
            from makelive import make_live_photo, save_live_photo_pair_as_pvt

            asset_id = make_live_photo(str(key_jpg), str(video_mov))
            print(f"[OK] Live Photo metadata written (Asset ID: {asset_id})")

            # Optional: create a .pvt package that survives transfers
            asset_id, pvt_path = save_live_photo_pair_as_pvt(
                str(key_jpg), str(video_mov)
            )
            print(f"[OK] Also wrote shareable package: {pvt_path}")
        except Exception as e:
            print(
                "[INFO] Could not tag Live Photo automatically (likely not macOS or 'makelive' missing).",
                file=sys.stderr,
            )
            print(
                "       Files are ready. On a Mac, install 'makelive' to embed the required Asset ID.",
                file=sys.stderr,
            )
            print(f"       Reason: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description="Convert animated GIF/APNG/WebP to iOS Sticker APNG and/or Live Photo pair."
    )
    p.add_argument("input", type=Path, help="Input animated image (gif, apng, webp)")
    p.add_argument("--sticker", type=Path, help="Output APNG path for Messages sticker")
    p.add_argument(
        "--live", type=Path, help="Output base path for Live Photo (no extension)"
    )
    p.add_argument(
        "--size",
        default="large",
        help="Sticker size: small|medium|large or pixel value (default: large)",
    )
    p.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip Live Photo metadata tagging even if makelive is present",
    )
    args = p.parse_args()

    if args.sticker is None and args.live is None:
        p.error("Choose at least one output: --sticker and/or --live")

    if args.sticker:
        make_ios_sticker(args.input, args.sticker, args.size)

    if args.live:
        make_live_photo(args.input, args.live, try_tag=not args.no_tag)


if __name__ == "__main__":
    main()
