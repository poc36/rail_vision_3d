"""
make_demo_video.py — собрать демо-видео из реальных фотографий рельсов.

Настоящего видеопотока в датасете нет, поэтому для проверки видеорежима
детектора генерируется движение камеры по реальным снимкам:
плавный наезд и панорамирование (эффект Кена Бёрнса) + переходы между кадрами.
Это честная имитация движения: пиксели — из настоящих фотографий.

    python scripts/make_demo_video.py --out output/demo_rails.mp4 --seconds-per-photo 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"


def ken_burns(img: np.ndarray, n_frames: int, out_size: tuple[int, int],
              zoom_from: float = 1.0, zoom_to: float = 1.25) -> list[np.ndarray]:
    """Кадры плавного наезда с лёгким смещением по кадру."""
    W, H = out_size
    h, w = img.shape[:2]
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        z = zoom_from + (zoom_to - zoom_from) * t
        cw, ch = int(w / z), int(h / z)
        # камера едет снизу вверх по кадру — как будто движемся вдоль пути
        x0 = int((w - cw) * (0.5 + 0.10 * np.sin(np.pi * t)))
        y0 = int((h - ch) * (0.75 - 0.35 * t))
        x0 = max(0, min(w - cw, x0))
        y0 = max(0, min(h - ch, y0))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        frames.append(cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/demo_rails.mp4")
    ap.add_argument("--photos", type=int, default=6)
    ap.add_argument("--seconds-per-photo", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--source", default="holdout",
                    help="откуда брать фото: holdout|train|путь к папке")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.split("x"))
    if args.source in ("holdout", "train"):
        ann_path = DATA / "labels" / "annotations.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else {}
        # берём кадры, где рельсы точно есть (выверенная разметка)
        files = [DATA / v["path"] for v in ann.values()
                 if v.get("tracks") and f"/{args.source}/" in v.get("path", "")]
        files = sorted(files)[:args.photos]
    else:
        files = sorted(Path(args.source).glob("*.jpg"))[:args.photos]
    if not files:
        print("нет исходных фото")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    n_frames = int(args.seconds_per_photo * args.fps)
    prev_tail = None
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        frames = ken_burns(img, n_frames, (W, H))
        if prev_tail is not None:                 # плавный переход
            for i in range(args.fps // 2):
                a = i / (args.fps // 2)
                writer.write(cv2.addWeighted(frames[0], a, prev_tail, 1 - a, 0))
        for fr in frames:
            writer.write(fr)
        prev_tail = frames[-1]
    writer.release()
    print(f"видео: {out} ({len(files)} фото, {W}x{H}, {args.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
