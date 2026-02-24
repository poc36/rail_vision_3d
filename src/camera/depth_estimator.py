"""
DepthEstimator — Вычисление карты глубины из стерео-пары.

Использует Semi-Global Block Matching (SGBM) с WLS-фильтрацией
для получения качественной карты диспаритета (и глубины).
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Оценка глубины на основе стерео-диспаритета."""

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        p1: int = 600,
        p2: int = 2400,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 63,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        use_wls_filter: bool = True,
        wls_lambda: float = 8000.0,
        wls_sigma: float = 1.5,
    ):
        """
        Args:
            min_disparity: минимальный диспаритет
            num_disparities: диапазон поиска диспаритета (кратно 16)
            block_size: размер блока для сопоставления
            p1, p2: параметры сглаживания SGBM
            use_wls_filter: использовать WLS-фильтр для улучшения карты
            wls_lambda: параметр lambda WLS-фильтра
            wls_sigma: параметр sigma WLS-фильтра
        """
        # Убедимся, что num_disparities кратно 16
        num_disparities = max(16, (num_disparities // 16) * 16)

        self._left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=disp12_max_diff,
            preFilterCap=pre_filter_cap,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self._use_wls = use_wls_filter
        self._wls_filter: Optional[cv2.ximgproc_DisparityWLSFilter] = None
        self._right_matcher = None

        if self._use_wls:
            try:
                self._right_matcher = cv2.ximgproc.createRightMatcher(
                    self._left_matcher
                )
                self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                    matcher_left=self._left_matcher
                )
                self._wls_filter.setLambda(wls_lambda)
                self._wls_filter.setSigmaColor(wls_sigma)
                logger.info("WLS filter enabled")
            except AttributeError:
                logger.warning(
                    "cv2.ximgproc not available. Install opencv-contrib-python for WLS filter. "
                    "Falling back to basic SGBM."
                )
                self._use_wls = False
                self._wls_filter = None
                self._right_matcher = None

        self._num_disparities = num_disparities
        self._min_disparity = min_disparity

        logger.info(
            f"DepthEstimator: SGBM, disparities={num_disparities}, block={block_size}"
        )

    def compute_disparity(
        self, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить карту диспаритета.

        Args:
            left: ректифицированный левый кадр (BGR или Gray)
            right: ректифицированный правый кадр (BGR или Gray)

        Returns:
            Карта диспаритета (float32, в пикселях)
        """
        # Конвертировать в grayscale
        if len(left.shape) == 3:
            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = left

        if len(right.shape) == 3:
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = right

        # Вычислить диспаритет
        disp_left = self._left_matcher.compute(gray_left, gray_right)

        if self._use_wls and self._wls_filter is not None and self._right_matcher is not None:
            disp_right = self._right_matcher.compute(gray_right, gray_left)
            disparity = self._wls_filter.filter(
                disp_left, gray_left, None, disp_right
            )
        else:
            disparity = disp_left

        # SGBM возвращает disparity * 16 (fixed point), конвертируем в float
        disparity = disparity.astype(np.float32) / 16.0

        # Заменить невалидные значения на 0
        disparity[disparity < 0] = 0

        return disparity

    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        focal_length_px: float,
        baseline_mm: float,
    ) -> np.ndarray:
        """
        Конвертировать диспаритет в карту глубины.

        Формула: Z = (f * B) / d

        Args:
            disparity: карта диспаритета (float32)
            focal_length_px: фокусное расстояние в пикселях
            baseline_mm: расстояние между камерами в мм

        Returns:
            Карта глубины в миллиметрах (float32)
        """
        # Избежать деления на ноль
        safe_disparity = disparity.copy()
        safe_disparity[safe_disparity < 0.1] = 0.1

        depth = (focal_length_px * baseline_mm) / safe_disparity

        # Ограничить глубину
        depth[disparity < 0.1] = 0  # Невалидные пиксели
        depth[depth > 100_000] = 0  # > 100 метров — невалидно

        return depth

    def compute_depth(
        self,
        left: np.ndarray,
        right: np.ndarray,
        focal_length_px: float,
        baseline_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычислить диспаритет и глубину за один вызов.

        Returns:
            (disparity, depth_mm)
        """
        disparity = self.compute_disparity(left, right)
        depth = self.disparity_to_depth(disparity, focal_length_px, baseline_mm)
        return disparity, depth

    def normalize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Нормализовать диспаритет для визуализации (0-255).

        Returns:
            Визуализация диспаритета (uint8)
        """
        valid = disparity[disparity > 0]
        if len(valid) == 0:
            return np.zeros(disparity.shape, dtype=np.uint8)

        min_val = np.percentile(valid, 5)
        max_val = np.percentile(valid, 95)

        normalized = np.clip(disparity, min_val, max_val)
        normalized = ((normalized - min_val) / (max_val - min_val) * 255).astype(
            np.uint8
        )

        # Цветная карта
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Чёрный цвет для невалидных
        colored[disparity <= 0] = 0

        return colored
