"""
download_rails.py — Скачивание реальных фото рельсов из интернета.

Использует icrawler (Bing Images) — без API-ключей.

Скачивает по разным запросам для максимального разнообразия:
  - прямые рельсы, повороты, вид от машиниста
  - разное время суток, погода, страны
  - железнодорожные пути, трамвайные пути и т.д.

Результат:
  data/real/
    images/  0000.jpg ... 0999.jpg   ← готово к обучению

ВАЖНО: реальные фото не содержат масок!
После скачивания нужно:
  1. Разметить вручную (LabelMe, CVAT, Roboflow)
  2. ИЛИ использовать автоматическую pseudo-маскировку (auto_mask.py)
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Поисковые запросы — разнообразие сцен
QUERIES = [
    # Основные
    ("railway tracks perspective view", 80),
    ("railroad tracks straight ahead", 80),
    ("train tracks curved road", 70),
    # Вид от машиниста
    ("train driver view railway", 70),
    ("locomotive cab view tracks", 60),
    # Условия освещения
    ("railway tracks night lights", 60),
    ("railroad tracks sunset", 50),
    ("train tracks foggy morning", 50),
    # Погода
    ("railway tracks snow winter", 60),
    ("railroad tracks rain wet", 50),
    ("train tracks sunny day", 60),
    # Разные страны / типы
    ("metro subway tracks underground", 50),
    ("tram tracks city street", 50),
    ("high speed rail tracks", 60),
    ("freight railway rural", 50),
    # Разные ракурсы
    ("railway tracks aerial view", 40),
    ("railroad tracks side view", 40),
    ("train tracks close up detail", 40),
    # Дополнительные
    ("empty railway station platform tracks", 50),
    ("mountain railway tracks landscape", 45),
]


def download_with_icrawler(query: str, count: int, output_dir: Path) -> int:
    """Скачать изображения через Bing Image Search."""
    try:
        from icrawler.builtin import BingImageCrawler

        crawler = BingImageCrawler(
            storage={"root_dir": str(output_dir)},
            log_level=logging.WARNING,
        )
        crawler.crawl(
            keyword=query,
            max_num=count,
            min_size=(400, 300),       # минимальный размер
            file_idx_offset="auto",    # не перезаписывать
        )
        # Посчитать новые файлы
        return sum(1 for _ in output_dir.glob("*.jpg"))
    except Exception as e:
        logger.error(f"icrawler error for '{query}': {e}")
        return 0


def rename_and_clean(img_dir: Path, total_needed: int) -> int:
    """Переименовать все файлы в 0000.jpg, 0001.jpg ..."""
    files = sorted(img_dir.glob("*"))
    images = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    # Удалить лишние
    if len(images) > total_needed:
        for f in images[total_needed:]:
            f.unlink()
        images = images[:total_needed]

    # Переименовать
    kept = 0
    for i, f in enumerate(images):
        try:
            dest = img_dir / f"{i:04d}.jpg"
            if f != dest:
                f.rename(dest)
            kept += 1
        except Exception:
            pass

    return kept


def main():
    parser = argparse.ArgumentParser(description="Download real rail images")
    parser.add_argument("--output", default="data/real",
                        help="Output directory (default: data/real)")
    parser.add_argument("--total", type=int, default=1000,
                        help="Total images to download (default: 1000)")
    args = parser.parse_args()

    root = Path(args.output)
    img_dir = root / "images"
    tmp_dir = root / "_tmp_download"
    img_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target: {args.total} images  →  {img_dir.resolve()}")
    logger.info("Downloading from Bing Images (no API key required)...")
    logger.info("")

    # Масштабировать количество по запросам
    total_in_queries = sum(c for _, c in QUERIES)
    scale = args.total / total_in_queries

    downloaded = 0
    for query, base_count in QUERIES:
        count = max(5, int(base_count * scale))
        remaining = args.total - downloaded
        if remaining <= 0:
            break
        count = min(count, remaining)

        logger.info(f"[{downloaded:4d}/{args.total}] '{query}'  →  {count} images")
        q_dir = tmp_dir / query.replace(" ", "_")[:40]
        q_dir.mkdir(exist_ok=True)

        download_with_icrawler(query, count, q_dir)

        # Переместить в images/
        for f in q_dir.glob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                dest = img_dir / f"{downloaded:04d}.jpg"
                try:
                    import cv2
                    img = cv2.imread(str(f))
                    if img is not None:
                        cv2.imwrite(str(dest), img)
                        downloaded += 1
                except Exception:
                    pass
            if downloaded >= args.total:
                break

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("")
    logger.info(f"✅ Downloaded: {downloaded} images  →  {img_dir.resolve()}")

    if downloaded > 0:
        logger.info("")
        logger.info("=" * 55)
        logger.info("СЛЕДУЮЩИЙ ШАГ — разметка масок:")
        logger.info("=" * 55)
        logger.info("")
        logger.info("Вариант 1 (ручная разметка, лучшее качество):")
        logger.info("  pip install labelme")
        logger.info(f"  labelme {img_dir}")
        logger.info("")
        logger.info("Вариант 2 (авторазметка через SAM/псевдо-маски):")
        logger.info("  python scripts/auto_mask.py --images data/real/images")
        logger.info("")
        logger.info("Вариант 3 (обучение сразу на синтетике + дообучение):")
        logger.info("  python scripts/train.py \\")
        logger.info("    --train-images data/synthetic/train/images \\")
        logger.info("    --train-masks  data/synthetic/train/masks")
    else:
        logger.error("Ничего не скачалось. Проверь интернет-соединение.")


if __name__ == "__main__":
    main()
