"""
Visualizer2D — 2D-визуализация детекции рельсов на кадре.

Рисует:
- Полупрозрачную маску сегментации
- Контуры рельсов
- Центральные линии
- HUD с информацией (глубина, угол, кривизна)
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class Visualizer2D:
    """Визуализация детекции рельсов на 2D-изображении."""

    # Цвета (BGR)
    COLOR_RAIL_LEFT = (0, 255, 0)    # Зелёный
    COLOR_RAIL_RIGHT = (0, 200, 255)  # Оранжевый
    COLOR_CONTOUR = (255, 255, 0)     # Голубой
    COLOR_CENTER = (0, 0, 255)        # Красный
    COLOR_HOUGH = (255, 0, 255)       # Розовый
    COLOR_HUD_BG = (0, 0, 0)
    COLOR_HUD_TEXT = (255, 255, 255)

    def __init__(
        self,
        overlay_alpha: float = 0.4,
        show_depth: bool = True,
        show_hud: bool = True,
    ):
        self.alpha = overlay_alpha
        self.show_depth = show_depth
        self.show_hud = show_hud

    def draw(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        detection_result=None,
        depth_map: Optional[np.ndarray] = None,
        rail_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Полная визуализация на кадре.

        Args:
            image: BGR-кадр (H, W, 3)
            mask: маска сегментации (H, W)
            detection_result: DetectionResult из RailDetector
            depth_map: карта глубины для HUD
            rail_info: dict с доп. информацией

        Returns:
            Визуализированный кадр
        """
        canvas = image.copy()

        # 1. Наложить маску сегментации
        if mask is not None:
            canvas = self._overlay_mask(canvas, mask)

        # 2. Нарисовать контуры и линии
        if detection_result is not None:
            canvas = self._draw_detection(canvas, detection_result)

        # 3. Нарисовать карту глубины
        if depth_map is not None and self.show_depth:
            canvas = self._draw_depth_minimap(canvas, depth_map)

        # 4. HUD
        if self.show_hud:
            canvas = self._draw_hud(canvas, detection_result, rail_info)

        return canvas

    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Наложить полупрозрачную маску."""
        canvas = image.copy()

        if not np.any(mask > 0):
            return canvas

        # Создать цветное наложение
        overlay = np.zeros_like(image)
        overlay[mask > 0] = self.COLOR_RAIL_LEFT

        # Альфа-смешивание
        mask_bool = mask > 0
        mask_3d = np.stack([mask_bool] * 3, axis=-1)
        canvas = np.where(
            mask_3d,
            (canvas.astype(np.float32) * (1 - self.alpha) + overlay.astype(np.float32) * self.alpha).astype(np.uint8),
            canvas,
        )

        return canvas

    def _draw_detection(self, image: np.ndarray, detection_result) -> np.ndarray:
        """Нарисовать результаты детекции."""
        canvas = image.copy()

        for rail in detection_result.rails:
            color = (
                self.COLOR_RAIL_LEFT
                if rail.side == "left"
                else self.COLOR_RAIL_RIGHT
            )

            # Контур
            cv2.drawContours(canvas, [rail.contour], -1, self.COLOR_CONTOUR, 2)

            # Центральная линия
            if len(rail.center_line) > 1:
                for i in range(len(rail.center_line) - 1):
                    pt1 = tuple(rail.center_line[i])
                    pt2 = tuple(rail.center_line[i + 1])
                    cv2.line(canvas, pt1, pt2, self.COLOR_CENTER, 2)

            # Метка
            x, y, w, h = rail.bounding_box
            label = f"{rail.side.upper()} rail"
            cv2.putText(
                canvas,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Hough lines
        if detection_result.hough_lines is not None:
            for line in detection_result.hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (x1, y1), (x2, y2), self.COLOR_HOUGH, 1)

        return canvas

    def _draw_depth_minimap(
        self, image: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        """Нарисовать миниатюру карты глубины."""
        canvas = image.copy()
        h, w = image.shape[:2]

        # Размер миниатюры
        mini_h = h // 4
        mini_w = w // 4

        # Нормализовать глубину
        valid = depth_map[depth_map > 0]
        if len(valid) == 0:
            return canvas

        min_d = np.percentile(valid, 5)
        max_d = np.percentile(valid, 95)
        normalized = np.clip(depth_map, min_d, max_d)
        normalized = ((normalized - min_d) / (max_d - min_d + 1e-6) * 255).astype(
            np.uint8
        )

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colored[depth_map <= 0] = 0

        minimap = cv2.resize(colored, (mini_w, mini_h))

        # Поместить в правый верхний угол
        x_offset = w - mini_w - 10
        y_offset = 10
        canvas[y_offset : y_offset + mini_h, x_offset : x_offset + mini_w] = minimap

        # Рамка
        cv2.rectangle(
            canvas,
            (x_offset - 1, y_offset - 1),
            (x_offset + mini_w, y_offset + mini_h),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            "Depth",
            (x_offset, y_offset - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return canvas

    def _draw_hud(
        self,
        image: np.ndarray,
        detection_result=None,
        rail_info: Optional[dict] = None,
    ) -> np.ndarray:
        """Нарисовать HUD с информацией."""
        canvas = image.copy()
        h, w = image.shape[:2]

        lines = []
        lines.append("RAIL VISION 3D")

        if detection_result is not None:
            lines.append(f"Rails detected: {detection_result.num_rails}")
            for rail in detection_result.rails:
                lines.append(f"  {rail.side}: area={int(rail.area)}px")

        if rail_info is not None:
            if "gauge_mm" in rail_info:
                lines.append(f"Gauge: {rail_info['gauge_mm']:.0f}mm")
            if "left_rail" in rail_info:
                info = rail_info["left_rail"]
                lines.append(f"L: len={info.get('length_m', 0):.1f}m")
            if "right_rail" in rail_info:
                info = rail_info["right_rail"]
                lines.append(f"R: len={info.get('length_m', 0):.1f}m")

        # Фон HUD
        hud_h = len(lines) * 22 + 10
        hud_w = 250
        overlay = canvas.copy()
        cv2.rectangle(overlay, (5, 5), (5 + hud_w, 5 + hud_h), self.COLOR_HUD_BG, -1)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

        # Текст
        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLOR_HUD_TEXT,
                1,
            )

        return canvas
