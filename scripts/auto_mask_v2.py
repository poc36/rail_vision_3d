"""
auto_mask_v2.py — Улучшенная генерация масок для реальных фото рельсов.

Вместо нейросети использует классические CV-алгоритмы:
  1. Цветовая сегментация (рельсы = серый/тёмный металл)
  2. Canny edge detection + Hough Lines
  3. Перспективная модель (рельсы сходятся к горизонту)
  4. Морфологическая обработка

Даёт ГОРАЗДО лучшие маски, чем необученная нейросеть.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_vanishing_point(edges: np.ndarray, h: int, w: int):
    """Найти точку схода (vanishing point) через линии Хафа."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=50, maxLineGap=30)
    if lines is None:
        return w // 2, h // 3

    # Найти пересечения вертикальных линий
    intersections = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 20 or angle > 160:  # горизонтальные — пропускаем
            continue
        for j in range(i + 1, len(lines)):
            x3, y3, x4, y4 = lines[j][0]
            angle2 = abs(np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi)
            if angle2 < 20 or angle2 > 160:
                continue

            # Пересечение двух линий
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)

            # Точка схода должна быть в верхней половине кадра
            if 0 < ix < w and 0 < iy < h * 0.6:
                intersections.append((ix, iy))

    if not intersections:
        return w // 2, h // 3

    # Медиана пересечений
    xs = [p[0] for p in intersections]
    ys = [p[1] for p in intersections]
    return int(np.median(xs)), int(np.median(ys))


def create_rail_mask(image: np.ndarray) -> np.ndarray:
    """Создать маску рельсов через CV-алгоритмы."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- 1. Цветовая сегментация: рельсы = низкая насыщенность, средняя яркость ---
    # Металл рельсов: серый, не яркий, не тёмный
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    metal_mask = np.zeros((h, w), dtype=np.uint8)
    metal_condition = (s_channel < 60) & (v_channel > 60) & (v_channel < 220)
    metal_mask[metal_condition] = 255

    # --- 2. Edge detection ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # --- 3. Hough Lines — найти длинные линии ---
    lines_mask = np.zeros((h, w), dtype=np.uint8)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                             minLineLength=40, maxLineGap=25)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Рельсы — линии с углом 30-90° к горизонту
            if 25 < angle < 155:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                thickness = max(3, int(length / 30))
                cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness)

    # --- 4. Точка схода для перспективной маски ---
    vx, vy = detect_vanishing_point(edges, h, w)

    # Создать трапецию ROI — область где рельсы
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    trap_top_half = int(w * 0.08)
    trap_bot_half = int(w * 0.35)
    pts = np.array([
        [vx - trap_top_half, int(vy + h * 0.05)],
        [vx + trap_top_half, int(vy + h * 0.05)],
        [min(w - 1, vx + trap_bot_half), h - 1],
        [max(0, vx - trap_bot_half), h - 1]
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [pts], 255)

    # --- 5. Объединение ---
    # Линии в ROI + металл в ROI
    combined = np.zeros((h, w), dtype=np.uint8)

    # Hough lines внутри ROI
    rail_lines = cv2.bitwise_and(lines_mask, roi_mask)

    # Металлические пиксели рядом с линиями, внутри ROI
    # Расширяем линии и пересекаем с металлом
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    lines_dilated = cv2.dilate(rail_lines, kernel_dilate)
    metal_near_lines = cv2.bitwise_and(metal_mask, lines_dilated)
    metal_near_lines = cv2.bitwise_and(metal_near_lines, roi_mask)

    combined = cv2.bitwise_or(rail_lines, metal_near_lines)

    # --- 6. Верхнюю треть убираем (редко бывают рельсы) ---
    combined[:h // 3, :] = 0

    # --- 7. Морфология ---
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_med, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_small, iterations=1)

    # --- 8. Удалить мелкие шумовые контуры ---
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.002  # минимум 0.2% изображения
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask


def main():
    parser = argparse.ArgumentParser(description="Auto-mask V2: quality rail masks via CV")
    parser.add_argument("--images", default="data/real/images")
    parser.add_argument("--masks", default="data/real/masks_v2")
    parser.add_argument("--show", action="store_true", help="Preview results")
    args = parser.parse_args()

    img_dir  = Path(args.images)
    mask_dir = Path(args.masks)
    mask_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    if not files:
        logger.error(f"No images in {img_dir}")
        return

    logger.info(f"Found {len(files)} images → generating quality masks...")

    ok = 0
    for f in tqdm(files, desc="Masking V2"):
        image = cv2.imread(str(f))
        if image is None:
            continue

        mask = create_rail_mask(image)
        cv2.imwrite(str(mask_dir / (f.stem + ".png")), mask)
        ok += 1

        if args.show:
            overlay = image.copy()
            overlay[mask > 0] = (0, 255, 0)
            preview = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            cv2.imshow("V2 Mask Preview", preview)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    logger.info(f"✅ Quality masks: {ok}/{len(files)} → {mask_dir.resolve()}")
    logger.info("")
    logger.info("Обучение:")
    logger.info(f"  python scripts/train.py \\")
    logger.info(f"    --train-images {img_dir} \\")
    logger.info(f"    --train-masks  {mask_dir} \\")
    logger.info(f"    --epochs 10 --batch-size 2")


if __name__ == "__main__":
    main()
