"""
auto_mask.py — Автоматическая генерация масок для реальных фото рельсов.

Загружает изображения из --images, прогоняет через RailSegmentor
(DeepLabV3+ с pretrained backbone) и сохраняет бинарные маски в --masks.

Затем папки можно сразу передать в train.py.

Использование:
  python scripts/auto_mask.py \
      --images data/real/images \
      --masks  data/real/masks
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def enhance_rail_mask(image: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """
    Улучшение авто-маски эвристиками для рельсов:
      - Оставляем нижнюю половину (рельсы обычно внизу)
      - Морфологическая обработка (удаление шума)
      - Детектор линий Хафа как доп. подсказка
    """
    h, w = image.shape[:2]
    mask = raw_mask.copy()

    # Рельсы в основном в нижних 2/3 кадра
    mask[:h // 3, :] = 0

    # Морфология
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Hough — находим вертикальные линии и добавляем их к маске
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=60, maxLineGap=20)
    if lines is not None:
        hough_mask = np.zeros((h, w), dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Только достаточно вертикальные линии (рельсы уходят вдаль)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 40:   # угол к горизонту > 40°
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 6)
        # Добавить Hough-линии к маске только в нижней части
        hough_mask[:h // 3, :] = 0
        mask = cv2.bitwise_or(mask, hough_mask)
        # Снова морфология после слияния
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Auto-mask real rail images")
    parser.add_argument("--images", default="data/real/images",
                        help="Input images directory")
    parser.add_argument("--masks", default="data/real/masks",
                        help="Output masks directory")
    parser.add_argument("--input-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true",
                        help="Show preview windows")
    args = parser.parse_args()

    img_dir  = Path(args.images)
    mask_dir = Path(args.masks)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not image_files:
        logger.error(f"No images found in {img_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} images in {img_dir}")

    # Загрузить модель
    logger.info("Loading segmentation model...")
    from src.detection.rail_segmentation import RailSegmentor
    segmentor = RailSegmentor(
        model_name="deeplabv3",
        device=args.device,
        input_size=tuple(args.input_size),
    )
    logger.info("Model ready. Generating masks...")

    ok = 0
    for img_path in tqdm(image_files, desc="Masking"):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Cannot read: {img_path.name}")
            continue

        # Сегментация
        result = segmentor.segment(image)
        raw_mask = result  # uint8 ndarray, 0 or 255

        # Улучшение маски эвристиками
        enhanced = enhance_rail_mask(image, raw_mask)

        # Сохранить
        out_path = mask_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), enhanced)
        ok += 1

        if args.show:
            overlay = image.copy()
            overlay[enhanced > 0] = (0, 255, 0)
            preview = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            cv2.imshow("Preview", preview)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    logger.info(f"✅ Masks saved: {ok}/{len(image_files)}  →  {mask_dir.resolve()}")
    logger.info("")
    logger.info("Теперь запусти обучение:")
    logger.info(f"  python scripts/train.py \\")
    logger.info(f"    --train-images {img_dir} \\")
    logger.info(f"    --train-masks  {mask_dir} \\")
    logger.info(f"    --epochs 30 --batch-size 2")


if __name__ == "__main__":
    main()
