"""
Demo — Демонстрация Rail Vision 3D на синтетических данных.

Генерирует синтетическую стерео-пару с рельсами и прогоняет
полный пайплайн обработки.
"""

import cv2
import numpy as np
import sys
import os
import logging

# Добавить путь проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_rail_image(
    width: int = 1280,
    height: int = 720,
    shift_x: int = 0,
) -> np.ndarray:
    """
    Генерация синтетического изображения с рельсами.

    Рисует перспективный вид двух рельсов, уходящих к горизонту.

    Args:
        width, height: размер изображения
        shift_x: горизонтальный сдвиг (для имитации стерео)

    Returns:
        BGR-изображение
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Небо (градиент)
    for y in range(height // 2):
        ratio = y / (height // 2)
        blue = int(200 - 80 * ratio)
        green = int(180 - 60 * ratio)
        red = int(150 - 50 * ratio)
        image[y, :] = (blue, green, red)

    # Земля (коричнево-серая)
    for y in range(height // 2, height):
        ratio = (y - height // 2) / (height // 2)
        val = int(80 + 40 * ratio)
        image[y, :] = (val, int(val * 0.8), int(val * 0.6))

    # Шпалы (горизонтальные линии между рельсами)
    vanishing_x = width // 2 + shift_x
    vanishing_y = height // 2 - 20

    num_sleepers = 30
    for i in range(num_sleepers):
        t = (i + 1) / (num_sleepers + 1)
        y = int(vanishing_y + t * (height - vanishing_y))

        # Ширина рельсов на этой глубине (перспектива)
        half_width = int(10 + t * 200)

        cx = int(vanishing_x + (width // 2 - vanishing_x) * t * 0.1)
        x1 = cx - half_width
        x2 = cx + half_width

        # Шпала
        thickness = max(2, int(t * 6))
        sleeper_color = (90, 80, 70)
        cv2.line(image, (x1, y), (x2, y), sleeper_color, thickness)

    # Левый рельс
    rail_color = (180, 180, 180)
    gauge_fraction = 0.85  # Рельсы чуть уже шпал

    pts_left = []
    pts_right = []
    num_points = 50

    for i in range(num_points):
        t = i / (num_points - 1)
        y = int(vanishing_y + t * (height - vanishing_y))
        half_width = int((10 + t * 200) * gauge_fraction)
        cx = int(vanishing_x + (width // 2 - vanishing_x) * t * 0.1)

        pts_left.append([cx - half_width, y])
        pts_right.append([cx + half_width, y])

    pts_left = np.array(pts_left, dtype=np.int32)
    pts_right = np.array(pts_right, dtype=np.int32)

    # Рисуем рельсы
    for i in range(len(pts_left) - 1):
        thickness = max(2, int((i / len(pts_left)) * 8))
        cv2.line(
            image,
            tuple(pts_left[i]),
            tuple(pts_left[i + 1]),
            rail_color,
            thickness,
        )
        cv2.line(
            image,
            tuple(pts_right[i]),
            tuple(pts_right[i + 1]),
            rail_color,
            thickness,
        )

    # Добавить шум
    noise = np.random.randn(*image.shape).astype(np.float32) * 5
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


def generate_synthetic_depth(
    width: int = 1280,
    height: int = 720,
    max_depth_mm: float = 50000.0,
) -> np.ndarray:
    """
    Генерация синтетической карты глубины.

    Глубина увеличивается от bottom (ближе, ~1м) к горизонту (дальше, ~50м).
    """
    depth = np.zeros((height, width), dtype=np.float32)

    horizon_y = height // 2 - 20

    for y in range(horizon_y, height):
        t = (y - horizon_y) / (height - horizon_y)  # 0 на горизонте, 1 внизу
        if t < 0.01:
            t = 0.01
        # Глубина обратно пропорциональна t (ближе к bottom = ближе к камере)
        row_depth = 1000.0 / t  # мм
        row_depth = min(row_depth, max_depth_mm)
        depth[y, :] = row_depth

    return depth


def main():
    """Запуск демо."""
    logger.info("=" * 60)
    logger.info("  Rail Vision 3D — DEMO")
    logger.info("=" * 60)

    # 1. Генерация синтетических данных
    logger.info("Generating synthetic stereo pair...")
    left_image = generate_synthetic_rail_image(shift_x=0)
    right_image = generate_synthetic_rail_image(shift_x=10)  # Стерео-сдвиг
    depth_map = generate_synthetic_depth()

    logger.info(f"Left image: {left_image.shape}")
    logger.info(f"Right image: {right_image.shape}")
    logger.info(f"Depth map: {depth_map.shape}, range=[{depth_map.min():.0f}, {depth_map.max():.0f}]mm")

    # 2. Показать синтетические изображения
    cv2.imshow("Left Camera", left_image)
    cv2.imshow("Right Camera", right_image)

    # Визуализация глубины
    depth_viz = depth_map.copy()
    valid = depth_viz > 0
    if np.any(valid):
        min_d = np.percentile(depth_viz[valid], 5)
        max_d = np.percentile(depth_viz[valid], 95)
        depth_viz = np.clip(depth_viz, min_d, max_d)
        depth_viz = ((depth_viz - min_d) / (max_d - min_d) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        cv2.imshow("Synthetic Depth", depth_colored)

    # 3. Инициализировать пайплайн
    logger.info("Initializing pipeline...")
    try:
        from src.pipeline.vision_pipeline import RailVisionPipeline

        pipeline = RailVisionPipeline()

        # 4. Обработать кадр
        logger.info("Processing frame...")
        result = pipeline.process_frame(left_image, right_image)

        logger.info(f"Processing time: {result.processing_time_ms:.1f}ms")

        if result.segmentation_mask is not None:
            rail_pixels = np.sum(result.segmentation_mask > 0)
            total_pixels = result.segmentation_mask.size
            logger.info(
                f"Segmentation: {rail_pixels}/{total_pixels} pixels "
                f"({100 * rail_pixels / total_pixels:.1f}%)"
            )

        if result.detection_result is not None:
            logger.info(f"Detected rails: {result.detection_result.num_rails}")

        if result.points_3d is not None:
            logger.info(f"3D points: {len(result.points_3d)}")

        if result.rail_track is not None:
            logger.info(f"Rail track: {result.rail_track.summary()}")

        # 5. Показать результат
        if result.visualization_2d is not None:
            cv2.imshow("Rail Vision 3D — Result", result.visualization_2d)

        if result.segmentation_mask is not None:
            cv2.imshow("Segmentation Mask", result.segmentation_mask)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
