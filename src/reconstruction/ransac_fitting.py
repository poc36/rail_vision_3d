"""
RANSACFitter — RANSAC-фитинг геометрических примитивов для рельсов.

Выполняет:
- Фитинг плоскости земли
- Фитинг линий для каждого рельса
- Изоляция рельсов от плоскости земли
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlaneModel:
    """Модель плоскости: ax + by + cz + d = 0."""

    normal: np.ndarray  # (a, b, c)
    d: float
    inlier_mask: np.ndarray  # bool mask
    inlier_ratio: float

    @property
    def equation(self) -> Tuple[float, float, float, float]:
        return (*self.normal, self.d)


@dataclass
class LineModel:
    """Модель 3D-линии: point + direction."""

    point: np.ndarray     # Точка на линии (3,)
    direction: np.ndarray  # Направление (3,), нормализовано
    inlier_mask: np.ndarray
    inlier_ratio: float

    def project_point(self, p: np.ndarray) -> np.ndarray:
        """Проекция точки на линию."""
        v = p - self.point
        t = np.dot(v, self.direction)
        return self.point + t * self.direction


class RANSACFitter:
    """RANSAC фитинг для 3D-геометрии рельсов."""

    def __init__(
        self,
        ground_distance_threshold: float = 0.05,
        ground_num_iterations: int = 1000,
        rail_distance_threshold: float = 0.03,
        rail_num_iterations: int = 500,
    ):
        """
        Args:
            ground_distance_threshold: порог расстояния до плоскости (в метрах)
            ground_num_iterations: число итераций RANSAC для плоскости
            rail_distance_threshold: порог расстояния до линии (в метрах)
            rail_num_iterations: число итераций RANSAC для линии
        """
        self.ground_dist = ground_distance_threshold
        self.ground_iters = ground_num_iterations
        self.rail_dist = rail_distance_threshold
        self.rail_iters = rail_num_iterations

    def fit_plane(self, points: np.ndarray) -> Optional[PlaneModel]:
        """
        RANSAC фитинг плоскости.

        Args:
            points: (N, 3) — 3D-точки

        Returns:
            PlaneModel или None
        """
        if len(points) < 3:
            return None

        n = len(points)
        best_inliers = None
        best_count = 0
        best_normal = None
        best_d = 0.0

        for _ in range(self.ground_iters):
            # Случайные 3 точки
            idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Нормаль к плоскости
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                continue

            normal /= norm
            d = -np.dot(normal, p1)

            # Расстояния всех точек до плоскости
            distances = np.abs(points @ normal + d)

            # Inliers
            inlier_mask = distances < self.ground_dist
            count = np.sum(inlier_mask)

            if count > best_count:
                best_count = count
                best_inliers = inlier_mask
                best_normal = normal
                best_d = d

        if best_normal is None:
            return None

        # Уточнение: пересчитать плоскость по inliers
        inlier_points = points[best_inliers]
        if len(inlier_points) >= 3:
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            _, _, Vt = np.linalg.svd(centered)
            best_normal = Vt[-1]
            best_d = -np.dot(best_normal, centroid)

            # Пересчитать inliers
            distances = np.abs(points @ best_normal + best_d)
            best_inliers = distances < self.ground_dist
            best_count = np.sum(best_inliers)

        ratio = best_count / n
        logger.debug(f"Ground plane: {ratio:.1%} inliers ({best_count}/{n})")

        return PlaneModel(
            normal=best_normal,
            d=best_d,
            inlier_mask=best_inliers,
            inlier_ratio=ratio,
        )

    def fit_line(self, points: np.ndarray) -> Optional[LineModel]:
        """
        RANSAC фитинг 3D-линии.

        Args:
            points: (N, 3) — 3D-точки

        Returns:
            LineModel или None
        """
        if len(points) < 2:
            return None

        n = len(points)
        best_inliers = None
        best_count = 0
        best_point = None
        best_direction = None

        for _ in range(self.rail_iters):
            # Случайные 2 точки
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = points[idx]

            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm

            # Расстояние от каждой точки до линии
            v = points - p1
            projections = v @ direction
            projected_points = p1 + projections[:, np.newaxis] * direction
            distances = np.linalg.norm(points - projected_points, axis=1)

            inlier_mask = distances < self.rail_dist
            count = np.sum(inlier_mask)

            if count > best_count:
                best_count = count
                best_inliers = inlier_mask
                best_point = p1
                best_direction = direction

        if best_point is None:
            return None

        # Уточнение: PCA на inliers
        inlier_points = points[best_inliers]
        if len(inlier_points) >= 2:
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            _, _, Vt = np.linalg.svd(centered)
            best_direction = Vt[0]
            best_point = centroid

            # Пересчитать inliers
            v = points - best_point
            projections = v @ best_direction
            projected_points = best_point + projections[:, np.newaxis] * best_direction
            distances = np.linalg.norm(points - projected_points, axis=1)
            best_inliers = distances < self.rail_dist
            best_count = np.sum(best_inliers)

        ratio = best_count / n
        logger.debug(f"Rail line: {ratio:.1%} inliers ({best_count}/{n})")

        return LineModel(
            point=best_point,
            direction=best_direction,
            inlier_mask=best_inliers,
            inlier_ratio=ratio,
        )

    def extract_rails(
        self,
        points: np.ndarray,
        num_rails: int = 2,
    ) -> Tuple[Optional[PlaneModel], List[LineModel]]:
        """
        Извлечить рельсы: найти плоскость земли, затем фитить линии.

        Args:
            points: (N, 3) — облако точек рельсов
            num_rails: ожидаемое количество рельсов

        Returns:
            (ground_plane, [rail_lines])
        """
        if len(points) < 10:
            logger.warning("Too few points for rail extraction")
            return None, []

        # 1. Фитинг плоскости земли (опционально)
        ground_plane = self.fit_plane(points)

        # 2. Фитинг линий рельсов
        remaining_points = points
        rail_lines = []

        for i in range(num_rails):
            if len(remaining_points) < 5:
                break

            line = self.fit_line(remaining_points)
            if line is None:
                break

            if line.inlier_ratio < 0.1:
                logger.debug(f"Rail {i}: too few inliers ({line.inlier_ratio:.1%})")
                break

            rail_lines.append(line)

            # Удалить inliers для следующей итерации
            remaining_points = remaining_points[~line.inlier_mask]
            logger.info(
                f"Rail {i}: {np.sum(line.inlier_mask)} inliers, "
                f"{len(remaining_points)} remaining"
            )

        return ground_plane, rail_lines
