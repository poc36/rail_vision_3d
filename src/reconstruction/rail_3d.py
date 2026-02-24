"""
Rail3D — 3D-модель рельсов.

Классы:
- Rail3D: отдельный рельс (полилиния/сплайн в 3D)
- RailTrack3D: пара рельсов с расстоянием (колея)
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Rail3D:
    """3D-модель отдельного рельса."""

    points: np.ndarray  # (N, 3) — точки рельса, упорядоченные
    direction: np.ndarray  # (3,) — главное направление
    side: str  # "left" или "right"

    # Сплайн-аппроксимация (если доступна)
    _spline_tck: Optional[tuple] = None
    _spline_u: Optional[np.ndarray] = None

    def fit_spline(self, smoothing: float = 0.1, num_points: int = 100):
        """
        Аппроксимировать рельс сплайном.

        Args:
            smoothing: параметр сглаживания (0 = интерполяция)
            num_points: количество точек на сплайне
        """
        if len(self.points) < 4:
            logger.warning("Not enough points for spline fitting")
            return

        # Сортировать точки по главному направлению
        projections = self.points @ self.direction
        sort_idx = np.argsort(projections)
        sorted_points = self.points[sort_idx]

        try:
            tck, u = splprep(
                [sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]],
                s=smoothing * len(sorted_points),
                k=min(3, len(sorted_points) - 1),
            )
            self._spline_tck = tck
            self._spline_u = np.linspace(0, 1, num_points)
            logger.debug(f"Spline fitted with {len(sorted_points)} points")
        except Exception as e:
            logger.warning(f"Spline fitting failed: {e}")

    def get_spline_points(self, num_points: int = 100) -> Optional[np.ndarray]:
        """
        Получить сглаженные точки сплайна.

        Returns:
            (num_points, 3) или None если сплайн не построен
        """
        if self._spline_tck is None:
            return None

        u = np.linspace(0, 1, num_points)
        x, y, z = splev(u, self._spline_tck)
        return np.column_stack([x, y, z]).astype(np.float32)

    @property
    def length(self) -> float:
        """Длина рельса (сумма сегментов)."""
        if len(self.points) < 2:
            return 0.0
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def centroid(self) -> np.ndarray:
        """Центр масс рельса."""
        return np.mean(self.points, axis=0)

    def curvature_at(self, index: int) -> float:
        """
        Оценка кривизны в точке.

        Uses три последовательные точки для вычисления кривизны.
        """
        if index < 1 or index >= len(self.points) - 1:
            return 0.0

        p0, p1, p2 = self.points[index - 1], self.points[index], self.points[index + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        cross = np.linalg.norm(np.cross(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denom < 1e-10:
            return 0.0

        return float(cross / denom)

    @property
    def average_curvature(self) -> float:
        """Средняя кривизна рельса."""
        if len(self.points) < 3:
            return 0.0
        curvatures = [self.curvature_at(i) for i in range(1, len(self.points) - 1)]
        return float(np.mean(curvatures))

    def slope_angle_deg(self) -> float:
        """Угол уклона рельса (подъём/спуск) в градусах."""
        if len(self.points) < 2:
            return 0.0

        start = self.points[0]
        end = self.points[-1]
        delta = end - start
        horizontal = np.sqrt(delta[0] ** 2 + delta[2] ** 2)

        if horizontal < 1e-10:
            return 0.0

        return float(np.degrees(np.arctan2(delta[1], horizontal)))


class RailTrack3D:
    """Пара рельсов — железнодорожный путь в 3D."""

    def __init__(
        self,
        left_rail: Optional[Rail3D] = None,
        right_rail: Optional[Rail3D] = None,
    ):
        self.left_rail = left_rail
        self.right_rail = right_rail

    @property
    def is_complete(self) -> bool:
        """Оба рельса обнаружены?"""
        return self.left_rail is not None and self.right_rail is not None

    @property
    def gauge_mm(self) -> float:
        """
        Оценка ширины колеи в миллиметрах.

        Стандартная колея: 1520 мм (Россия) или 1435 мм (Европа).
        """
        if not self.is_complete:
            return 0.0

        left_center = self.left_rail.centroid
        right_center = self.right_rail.centroid

        # Расстояние между центрами рельсов
        distance_m = float(np.linalg.norm(left_center - right_center))
        return distance_m * 1000.0

    @property
    def center_line(self) -> Optional[np.ndarray]:
        """Центральная линия пути (между рельсами)."""
        if not self.is_complete:
            return None

        # Средняя точка между рельсами
        left_spline = self.left_rail.get_spline_points()
        right_spline = self.right_rail.get_spline_points()

        if left_spline is not None and right_spline is not None:
            # Интерполируем к одинаковому количеству точек
            n = min(len(left_spline), len(right_spline))
            center = (left_spline[:n] + right_spline[:n]) / 2.0
            return center

        return (self.left_rail.centroid + self.right_rail.centroid) / 2.0

    @property
    def direction(self) -> np.ndarray:
        """Общее направление пути."""
        if self.left_rail is not None:
            return self.left_rail.direction
        if self.right_rail is not None:
            return self.right_rail.direction
        return np.array([0, 0, 1], dtype=np.float32)

    def summary(self) -> dict:
        """Сводка о рельсовом пути."""
        info = {
            "is_complete": self.is_complete,
            "num_rails": sum(
                1 for r in [self.left_rail, self.right_rail] if r is not None
            ),
        }

        if self.left_rail is not None:
            info["left_rail"] = {
                "num_points": len(self.left_rail.points),
                "length_m": round(self.left_rail.length, 2),
                "curvature": round(self.left_rail.average_curvature, 4),
                "slope_deg": round(self.left_rail.slope_angle_deg(), 2),
            }

        if self.right_rail is not None:
            info["right_rail"] = {
                "num_points": len(self.right_rail.points),
                "length_m": round(self.right_rail.length, 2),
                "curvature": round(self.right_rail.average_curvature, 4),
                "slope_deg": round(self.right_rail.slope_angle_deg(), 2),
            }

        if self.is_complete:
            info["gauge_mm"] = round(self.gauge_mm, 1)

        return info
