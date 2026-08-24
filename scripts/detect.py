"""
detect.py — ГЛАВНЫЙ скрипт: найти настоящие рельсы на фото или видео.

    # одно фото
    python scripts/detect.py --input photo.jpg --out out.jpg

    # папка с фото
    python scripts/detect.py --input photos/ --out results/

    # видео (или веб-камера: --input 0)
    python scripts/detect.py --input clip.mp4 --out clip_rails.mp4

Что делает:
  1. Сеть RailNet (обучена на реальных фото) отвечает: есть ли на кадре
     НАСТОЯЩИЕ рельсы (вероятность) и где именно они проходят (маска).
  2. Маска превращается в линии рельсов, всё это рисуется поверх кадра.
  3. Если сеть не уверена, кадр помечается как "рельсы не найдены" —
     это важно, чтобы не рисовать рельсы там, где их нет.

Дополнительно (--geometry) можно включить классический геометрический
детектор: он рисует колею по перспективной модели (см. geometric_rails).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.models.railnet import RailNet, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from detection.rail_postprocess import mask_to_rails, draw_rails_overlay  # noqa: E402
from detection.geometric_rails import detect_rails, draw_detection  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class RailPredictor:
    """Обёртка над обученной сетью: кадр -> (вероятность, маска)."""

    def __init__(self, weights: str | Path, img_size: int | None = None,
                 device: str = "cpu"):
        ckpt = torch.load(str(weights), map_location=device, weights_only=False)
        self.img_size = img_size or ckpt.get("img_size", 320)
        self.model = RailNet(pretrained=False).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.device = device
        self.metrics = ckpt.get("metrics", {})

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> tuple[float, np.ndarray]:
        h, w = frame.shape[:2]
        s = self.img_size
        img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)
        img = (img[:, :, ::-1] / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))[None].to(self.device)
        p_cls, p_seg = self.model.predict(x, out_size=(h, w))
        return float(p_cls[0]), p_seg[0, 0].cpu().numpy()


def annotate(frame: np.ndarray, prob: float, mask: np.ndarray, cls_thr: float,
             seg_thr: float, geometry: bool = False) -> tuple[np.ndarray, int]:
    """Нарисовать результат на кадре. Возвращает (кадр, число найденных рельсов)."""
    h, w = frame.shape[:2]
    found = prob >= cls_thr
    rails = mask_to_rails(mask, thr=seg_thr) if found else []
    vis = draw_rails_overlay(frame, mask if found else np.zeros_like(mask), rails,
                             thr=seg_thr)
    if geometry:
        det = detect_rails(frame, min_score=0.35)
        vis = draw_detection(vis, det, draw_bed=False)

    scale = max(0.5, w / 1200.0)
    bar_h = int(38 * scale)
    cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)
    if found:
        text = f"RAILS: {prob*100:.0f}%   rails found: {len(rails)}"
        color = (60, 255, 60)
    else:
        text = f"no rails ({prob*100:.0f}%)"
        color = (60, 60, 255)
    cv2.putText(vis, text, (int(8 * scale), int(26 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, color, max(1, int(2 * scale)),
                cv2.LINE_AA)
    return vis, len(rails)


def process_image(pred: RailPredictor, path: Path, out: Path, args) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print("не читается:", path)
        return
    prob, mask = pred(img)
    vis, n = annotate(img, prob, mask, args.cls_thr, args.seg_thr, args.geometry)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"{path.name}: рельсы={prob:.2f} линий={n} -> {out}")


def process_video(pred: RailPredictor, source, out: Path, args) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("не открывается видео:", source)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ema_mask = None
    ema_prob = None
    i = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if args.stride > 1 and i % args.stride != 0 and ema_mask is not None:
            vis, _ = annotate(frame, ema_prob, ema_mask, args.cls_thr,
                              args.seg_thr, args.geometry)
            writer.write(vis)
            continue
        prob, mask = pred(frame)
        # временное сглаживание — убирает мерцание маски между кадрами
        ema_mask = mask if ema_mask is None else 0.6 * ema_mask + 0.4 * mask
        ema_prob = prob if ema_prob is None else 0.7 * ema_prob + 0.3 * prob
        vis, n = annotate(frame, ema_prob, ema_mask, args.cls_thr, args.seg_thr,
                          args.geometry)
        writer.write(vis)
        if i % 25 == 0:
            print(f"  кадр {i}, рельсы={ema_prob:.2f}, линий={n}", flush=True)
    cap.release()
    writer.release()
    print(f"видео готово: {out} ({i} кадров, {time.time()-t0:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Детекция настоящих рельсов на фото/видео")
    ap.add_argument("--input", required=True, help="файл, папка, видео или индекс камеры")
    ap.add_argument("--out", default="output/detect")
    ap.add_argument("--weights", default="runs/railnet.pt")
    ap.add_argument("--cls-thr", type=float, default=0.5, help="порог 'есть рельсы'")
    ap.add_argument("--seg-thr", type=float, default=0.5, help="порог маски")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="считать сеть раз в N кадров")
    ap.add_argument("--geometry", action="store_true",
                    help="дополнительно рисовать геометрический детектор")
    ap.add_argument("--limit", type=int, default=0, help="макс. число файлов из папки")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"нет весов {weights}. Сначала обучите: python scripts/train_railnet.py")
        return 1
    pred = RailPredictor(weights, args.img_size)
    print(f"модель: {weights} (вход {pred.img_size}px) метрики: {pred.metrics}")

    src = args.input
    out = Path(args.out)
    if src.isdigit():
        process_video(pred, int(src), out.with_suffix(".mp4"), args)
        return 0

    path = Path(src)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if args.limit:
            files = files[:args.limit]
        for f in files:
            process_image(pred, f, out / f"{f.stem}_rails.jpg", args)
    elif path.suffix.lower() in VIDEO_EXT:
        process_video(pred, str(path), out if out.suffix else out.with_suffix(".mp4"), args)
    elif path.suffix.lower() in IMAGE_EXT:
        process_image(pred, path, out if out.suffix else out / f"{path.stem}_rails.jpg", args)
    else:
        print("непонятный вход:", src)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
