"""
generate_dataset.py — Генерация синтетического датасета рельсов.

Создаёт 1000 изображений рельсов в разных положениях + бинарные маски.

Разнообразие:
 - Прямые рельсы, поворот влево, поворот вправо
 - Разные углы перспективы / положение горизонта
 - Разные условия освещения (день, сумерки, ночь, туман)
 - Разные фоны (лес, поле, город, снег)
 - Шум, дождь, блики

Структура выходных данных:
  data/synthetic/
    train/
      images/  0000.png ... 0799.png
      masks/   0000.png ... 0799.png
    val/
      images/  0800.png ... 0999.png
      masks/   0800.png ... 0999.png
"""

import os
import sys
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def rand(lo, hi):
    return np.random.uniform(lo, hi)


def rand_int(lo, hi):
    return np.random.randint(lo, hi + 1)


# ──────────────────────────────────────────────
# Background generators
# ──────────────────────────────────────────────

def make_sky(image, horizon_y, condition):
    h, w = image.shape[:2]
    if condition == "night":
        top = np.array([10, 10, 20])
        bot = np.array([30, 20, 40])
    elif condition == "dusk":
        top = np.array([30, 20, 80])
        bot = np.array([100, 80, 180])
    elif condition == "fog":
        top = np.array([200, 200, 200])
        bot = np.array([220, 220, 220])
    else:  # day
        r = rand(0, 1)
        if r < 0.33:   # blue sky
            top = np.array([220, 170, 100])
            bot = np.array([200, 210, 150])
        elif r < 0.66: # cloudy
            top = np.array([170, 170, 170])
            bot = np.array([210, 210, 205])
        else:          # sunny warm
            top = np.array([180, 160, 90])
            bot = np.array([210, 200, 140])

    for y in range(horizon_y):
        t = y / max(horizon_y, 1)
        color = (top * (1 - t) + bot * t).astype(np.uint8)
        image[y, :] = color


def make_ground(image, horizon_y, condition, ground_type):
    h, w = image.shape[:2]
    if ground_type == "snow":
        base = np.array([220, 225, 230])
    elif ground_type == "forest":
        base = np.array([30, 80, 20])
    elif ground_type == "city":
        base = np.array([80, 80, 85])
    elif ground_type == "field":
        base = np.array([50, 110, 40])
    else:
        base = np.array([70, 80, 60])

    if condition == "night":
        base = (base * 0.25).astype(np.uint8)
    elif condition == "dusk":
        base = (base * 0.6).astype(np.uint8)
    elif condition == "fog":
        base = (base * 0.5 + np.array([100, 100, 100]) * 0.5).astype(np.uint8)

    for y in range(horizon_y, h):
        t = (y - horizon_y) / max(h - horizon_y, 1)
        brightness = 1.0 + 0.3 * t
        color = np.clip(base * brightness, 0, 255).astype(np.uint8)
        image[y, :] = color

    # Add texture noise
    noise_region = image[horizon_y:, :]
    noise = np.random.randint(-15, 15, noise_region.shape, dtype=np.int16)
    image[horizon_y:, :] = np.clip(noise_region.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Rail geometry
# ──────────────────────────────────────────────

def compute_rail_points(w, h, horizon_x, horizon_y, curve, gauge_base=200):
    """
    Returns list of (left_x, right_x, y) tuples from horizon to bottom.
    curve: -1 (hard left) .. 0 (straight) .. 1 (hard right)
    """
    num_steps = h - horizon_y
    points = []
    for i in range(num_steps + 1):
        t = i / max(num_steps, 1)          # 0 at horizon, 1 at bottom
        y = horizon_y + i

        # Perspective: half-gauge grows with t
        half_gauge = t * gauge_base
        # Curve: centre shifts laterally
        cx = horizon_x + curve * (t ** 1.5) * w * 0.4
        cx = int(np.clip(cx, 0, w))

        lx = int(cx - half_gauge)
        rx = int(cx + half_gauge)
        points.append((lx, rx, y))

    return points


# ──────────────────────────────────────────────
# Draw rails + sleepers, return mask
# ──────────────────────────────────────────────

def draw_rails(image, mask, points, condition):
    h, w = image.shape[:2]

    n = len(points)
    if n < 2:
        return

    # Rail color
    if condition == "night":
        rail_color = (80, 80, 90)
    elif condition == "fog":
        rail_color = (140, 140, 145)
    elif condition == "snow":
        rail_color = (200, 200, 210)
    else:
        base_v = rand_int(140, 200)
        rail_color = (base_v, base_v, base_v + rand_int(0, 10))

    # Sleeper color
    sleeper_color = tuple(int(x) for x in (
        np.array([70, 60, 50]) * (0.3 if condition == "night" else 1.0)
    ).astype(int))

    # Draw sleepers
    sleeper_count = rand_int(15, 35)
    sleeper_indices = np.linspace(0, n - 1, sleeper_count, dtype=int)
    for idx in sleeper_indices:
        lx, rx, y = points[idx]
        if y < 0 or y >= h:
            continue
        t = idx / max(n - 1, 1)
        thickness = max(1, int(t * 5))
        cv2.line(image, (max(0, lx), y), (min(w - 1, rx), y), sleeper_color, thickness)

    # Draw rails (left & right)
    pts_l = np.array([(max(0, p[0]), p[2]) for p in points], dtype=np.int32)
    pts_r = np.array([(min(w - 1, p[1]), p[2]) for p in points], dtype=np.int32)

    for i in range(n - 1):
        t = i / max(n - 1, 1)
        thickness = max(1, int(t * 6 + 1))

        p0l = (pts_l[i][0], pts_l[i][1])
        p1l = (pts_l[i + 1][0], pts_l[i + 1][1])
        p0r = (pts_r[i][0], pts_r[i][1])
        p1r = (pts_r[i + 1][0], pts_r[i + 1][1])

        cv2.line(image, p0l, p1l, rail_color, thickness)
        cv2.line(image, p0r, p1r, rail_color, thickness)

        # Mask
        cv2.line(mask, p0l, p1l, 255, max(thickness + 2, 4))
        cv2.line(mask, p0r, p1r, 255, max(thickness + 2, 4))


# ──────────────────────────────────────────────
# Post-processing effects
# ──────────────────────────────────────────────

def apply_effects(image, condition):
    # Gaussian noise
    noise_std = {"day": 5, "dusk": 8, "night": 15, "fog": 6}.get(condition, 5)
    noise = np.random.randn(*image.shape).astype(np.float32) * noise_std
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Fog overlay
    if condition == "fog":
        fog = np.ones_like(image) * 200
        image = cv2.addWeighted(image, 0.6, fog, 0.4, 0)

    # Night — darken
    if condition == "night":
        image = (image * rand(0.2, 0.4)).astype(np.uint8)

    # Random rain streaks
    if np.random.rand() < 0.2:
        n_streaks = rand_int(50, 200)
        h, w = image.shape[:2]
        for _ in range(n_streaks):
            x = rand_int(0, w - 1)
            y = rand_int(0, h - 30)
            length = rand_int(10, 30)
            cv2.line(image, (x, y), (x + rand_int(-2, 2), y + length),
                     (200, 200, 210), 1)

    # Lens flare (rare)
    if np.random.rand() < 0.05:
        h, w = image.shape[:2]
        cx, cy = rand_int(0, w), rand_int(0, h // 2)
        cv2.circle(image, (cx, cy), rand_int(20, 60), (255, 255, 200), -1)
        image = np.clip(image.astype(np.float32), 0, 255).astype(np.uint8)

    # Blur (motion or focus)
    if np.random.rand() < 0.15:
        k = rand_int(1, 2) * 2 + 1
        image = cv2.GaussianBlur(image, (k, k), 0)

    return image


# ──────────────────────────────────────────────
# Single image generator
# ──────────────────────────────────────────────

CONDITIONS = ["day", "day", "day", "dusk", "night", "fog"]
GROUNDS    = ["default", "forest", "field", "city", "snow"]

def generate_one(width=1280, height=720):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask  = np.zeros((height, width),   dtype=np.uint8)

    condition   = CONDITIONS[rand_int(0, len(CONDITIONS) - 1)]
    ground_type = GROUNDS[rand_int(0, len(GROUNDS) - 1)]
    if ground_type == "snow" and condition == "night":
        condition = "dusk"

    # Horizon position (varied)
    horizon_y = rand_int(int(height * 0.30), int(height * 0.55))
    # Vanishing point (can be off-centre for curves)
    curve = rand(-1.0, 1.0)     # -1=hard left, 0=straight, 1=hard right
    # Vanishing x is biased opposite to curve direction
    horizon_x = int(width / 2 + curve * width * (-0.15))
    horizon_x = int(np.clip(horizon_x, width * 0.2, width * 0.8))

    # Background
    make_sky(image, horizon_y, condition)
    make_ground(image, horizon_y, condition, ground_type)

    # Rail geometry
    gauge_base = rand_int(120, 280)
    points = compute_rail_points(width, height, horizon_x, horizon_y,
                                  curve, gauge_base)
    draw_rails(image, mask, points, condition)

    # Effects
    image = apply_effects(image, condition)

    return image, mask


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic rail dataset")
    parser.add_argument("--output",  default="data/synthetic",
                        help="Root output directory")
    parser.add_argument("--total",   type=int, default=1000,
                        help="Total number of images (default: 1000)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of images for validation (default: 0.2)")
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    n_val   = int(args.total * args.val_split)
    n_train = args.total - n_val

    splits = {
        "train": n_train,
        "val":   n_val,
    }

    root = Path(args.output)
    for split in splits:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.total} images  ({n_train} train / {n_val} val)")
    logger.info(f"Output: {root.resolve()}")

    global_idx = 0
    for split, count in splits.items():
        img_dir  = root / split / "images"
        mask_dir = root / split / "masks"

        for local_idx in tqdm(range(count), desc=split):
            image, mask = generate_one(args.width, args.height)

            fname = f"{global_idx:04d}.png"
            cv2.imwrite(str(img_dir / fname), image)
            cv2.imwrite(str(mask_dir / fname), mask)
            global_idx += 1

    logger.info("✅ Done!")
    logger.info(f"  Train: {n_train} samples  →  {root / 'train'}")
    logger.info(f"  Val:   {n_val}   samples  →  {root / 'val'}")
    logger.info("")
    logger.info("To train the model run:")
    logger.info(
        f"  python scripts/train.py "
        f"--train-images {root / 'train' / 'images'} "
        f"--train-masks {root / 'train' / 'masks'} "
        f"--val-images {root / 'val' / 'images'} "
        f"--val-masks {root / 'val' / 'masks'}"
    )


if __name__ == "__main__":
    main()
