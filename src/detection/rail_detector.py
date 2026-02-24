"""
RailDetector — Постобработка маски сегментации для извлечения рельсов.

Выполняет:
- Морфологическую обработку маски
- Извлечение контуров рельсов
- Определение левого и правого рельса
- Hough Transform для обнаружения линий
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RailLine:
    """Обнаруженный рельс в 2D."""

    contour: np.ndarray  # Контур рельса
    center_line: np.ndarray  # Центральная линия [(x, y), ...]
    side: str  # "left" или "right"
    area: float  # Площадь маски рельса
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 1.0


@dataclass
class DetectionResult:
    """Результат детекции рельсов на одном кадре."""

    mask_clean: np.ndarray  # Обработанная маска
    rails: List[RailLine] = field(default_factory=list)
    hough_lines: Optional[np.ndarray] = None  # Линии Hough (опционально)
    num_rails: int = 0


class RailDetector:
    """Постобработка масок и извлечение рельсов."""

    def __init__(
        self,
        morph_kernel_size: int = 5,
        open_iterations: int = 2,
        close_iterations: int = 3,
        min_contour_area: int = 500,
        use_hough: bool = True,
        hough_rho: int = 1,
        hough_theta_deg: float = 1.0,
        hough_threshold: int = 100,
        hough_min_line_length: int = 50,
        hough_max_line_gap: int = 20,
    ):
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.open_iter = open_iterations
        self.close_iter = close_iterations
        self.min_contour_area = min_contour_area

        self.use_hough = use_hough
        self.hough_rho = hough_rho
        self.hough_theta = np.deg2rad(hough_theta_deg)
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

    def detect(self, mask: np.ndarray) -> DetectionResult:
        """
        Обработать маску сегментации и извлечь рельсы.

        Args:
            mask: бинарная маска (H, W), uint8 {0, 255}

        Returns:
            DetectionResult с обработанной маской и найденными рельсами
        """
        # 1. Морфологическая обработка
        mask_clean = self._morphological_cleanup(mask)

        # 2. Найти контуры
        contours = self._find_contours(mask_clean)

        # 3. Фильтрация по площади и извлечение рельсов
        rails = self._extract_rails(contours, mask_clean)

        # 4. Определить left/right
        self._classify_sides(rails, mask_clean.shape[1])

        # 5. Hough Transform (опционально)
        hough_lines = None
        if self.use_hough:
            hough_lines = self._detect_hough_lines(mask_clean)

        result = DetectionResult(
            mask_clean=mask_clean,
            rails=rails,
            hough_lines=hough_lines,
            num_rails=len(rails),
        )

        logger.debug(f"Detected {len(rails)} rails")
        return result

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Морфологическая очистка маски."""
        # Opening: убрать мелкий шум
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.open_iter
        )
        # Closing: заполнить дыры
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.close_iter
        )
        return cleaned

    def _find_contours(self, mask: np.ndarray) -> list:
        """Найти контуры на маске."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _extract_rails(
        self, contours: list, mask: np.ndarray
    ) -> List[RailLine]:
        """Извлечь рельсы из контуров."""
        rails = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Вычислить центральную линию (skeleton-like)
            center_line = self._compute_center_line(contour, mask)

            rail = RailLine(
                contour=contour,
                center_line=center_line,
                side="unknown",
                area=area,
                bounding_box=(x, y, w, h),
            )
            rails.append(rail)

        # Сортировать по x позиции (слева направо)
        rails.sort(key=lambda r: r.bounding_box[0])

        return rails

    def _compute_center_line(
        self, contour: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить центральную линию контура.
        
        Использует горизонтальное сканирование: для каждой строки изображения
        находит центр масс пикселей контура.
        """
        # Создать маску только для этого контура
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(contour)
        center_points = []

        # Для каждой строки найти центр
        step = max(1, h // 50)  # Максимум 50 точек
        for row in range(y, y + h, step):
            if row >= mask.shape[0]:
                break
            row_pixels = np.where(contour_mask[row, x : x + w] > 0)[0]
            if len(row_pixels) > 0:
                cx = x + int(np.mean(row_pixels))
                center_points.append([cx, row])

        if len(center_points) == 0:
            # Fallback: центр bounding box
            center_points = [[x + w // 2, y + h // 2]]

        return np.array(center_points, dtype=np.int32)

    def _classify_sides(self, rails: List[RailLine], image_width: int):
        """Определить left/right для каждого рельса."""
        if len(rails) == 0:
            return
        if len(rails) == 1:
            center_x = rails[0].bounding_box[0] + rails[0].bounding_box[2] // 2
            rails[0].side = "left" if center_x < image_width // 2 else "right"
            return

        # Для двух и более — используем позицию
        mid = image_width // 2
        for rail in rails:
            cx = rail.bounding_box[0] + rail.bounding_box[2] // 2
            rail.side = "left" if cx < mid else "right"

    def _detect_hough_lines(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Обнаружить линии через Probabilistic Hough Transform."""
        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        return lines
