"""
make_compilation.py — 1-минутный ролик из множества видео (через OpenCV).

Не требует ffmpeg или moviepy — только OpenCV!

Берёт из каждого видео фрагмент из середины,
склеивает в одно видео с плавными переходами (fade).
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse


TARGET_DURATION = 60   # секунд
OUTPUT_FPS = 24
RESOLUTION = (1280, 720)
FADE_FRAMES = 12  # кадров на переход


def get_video_files(folder: str) -> list:
    exts = {'.mov', '.mp4', '.avi', '.mkv', '.mts'}
    return sorted([
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in exts
    ])


def get_video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": frames, "duration": duration, "w": w, "h": h}


def read_clip(path: Path, start_sec: float, duration_sec: float, target_fps: int, resolution: tuple):
    """Читает фрагмент видео, ресайзит, возвращает список кадров."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    start_frame = int(start_sec * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    target_frames = int(duration_sec * target_fps)
    frame_step = max(1, src_fps / target_fps)  # пропускать кадры для конвертации fps

    frame_idx = 0
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Выбираем нужные кадры для target_fps
        if int(frame_idx % frame_step) == 0:
            resized = cv2.resize(frame, resolution)
            frames.append(resized)

        frame_idx += 1

    cap.release()
    return frames


def apply_fade(frames: list, fade_length: int):
    """Fade in на начало, fade out на конец."""
    for i in range(min(fade_length, len(frames))):
        alpha = i / fade_length
        frames[i] = (frames[i].astype(np.float32) * alpha).astype(np.uint8)

    for i in range(min(fade_length, len(frames))):
        idx = len(frames) - 1 - i
        alpha = i / fade_length
        frames[idx] = (frames[idx].astype(np.float32) * alpha).astype(np.uint8)

    return frames


def main():
    parser = argparse.ArgumentParser(description="Create 1-min video compilation (OpenCV)")
    parser.add_argument("--input", required=True, help="Folder with source videos")
    parser.add_argument("--output", default="Jordan_2017_Highlights.mp4")
    parser.add_argument("--duration", type=int, default=TARGET_DURATION)
    args = parser.parse_args()

    videos = get_video_files(args.input)
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Target: {args.duration}s compilation")
    print()

    # Проанализировать все видео
    valid_videos = []
    for v in videos:
        info = get_video_info(v)
        if info and info["duration"] > 1:
            valid_videos.append((v, info))
            print(f"  {v.name}: {info['duration']:.1f}s ({info['w']}x{info['h']})")

    if not valid_videos:
        print("No valid videos!")
        return

    # Сколько секунд из каждого
    clip_dur = args.duration / len(valid_videos)
    clip_dur = max(1.5, min(5.0, clip_dur))

    n_clips = min(len(valid_videos), int(args.duration / clip_dur))
    clip_dur = args.duration / n_clips

    # Выбрать равномерно
    step = max(1, len(valid_videos) // n_clips)
    selected = valid_videos[::step][:n_clips]

    print(f"\nUsing {len(selected)} clips x {clip_dur:.1f}s each")
    print()

    # Создать writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, OUTPUT_FPS, RESOLUTION)

    total_written = 0
    for i, (vpath, info) in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {vpath.name}...", end=" ", flush=True)

        # Берём из середины
        start = max(0, (info["duration"] - clip_dur) / 2)

        frames = read_clip(vpath, start, clip_dur, OUTPUT_FPS, RESOLUTION)
        if not frames:
            print("skip")
            continue

        # Fade in/out
        frames = apply_fade(frames, FADE_FRAMES)

        for frame in frames:
            out.write(frame)
            total_written += 1

        print(f"{len(frames)} frames")

    out.release()
    final_dur = total_written / OUTPUT_FPS
    size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print()
    print(f"Done! {args.output}")
    print(f"  Duration: {final_dur:.1f}s")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")


if __name__ == "__main__":
    main()
