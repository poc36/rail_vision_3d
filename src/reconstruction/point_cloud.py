"""
PointCloudGenerator — Генерация 3D-облака точек из маски рельсов + карты глубины.

Выполняет:
- Репроекцию 2D-пикселей в 3D с использованием intrinsic-матрицы камеры
- Фильтрацию выбросов
- Downsampling для оптимизации
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not installed. 3D visualization will be limited.")


class PointCloudGenerator:
    """Генерация 3D-облака точек из маски + глубины."""

    def __init__(
        self,
        max_depth_m: float = 50.0,
        min_depth_m: float = 0.5,
        downsample_voxel_size: float = 0.05,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
    ):
        """
        Args:
            max_depth_m: максимальная глубина в метрах
            min_depth_m: минимальная глубина в метрах
            downsample_voxel_size: размер вокселя для downsampling (0 = отключить)
            outlier_nb_neighbors: число соседей для Statistical Outlier Removal
            outlier_std_ratio: порог std для SOR
        """
        self.max_depth_mm = max_depth_m * 1000.0
        self.min_depth_mm = min_depth_m * 1000.0
        self.voxel_size = downsample_voxel_size
        self.nb_neighbors = outlier_nb_neighbors
        self.std_ratio = outlier_std_ratio

    def generate(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        camera_matrix: np.ndarray,
        color_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Генерация облака точек из глубины и маски.

        Args:
            depth_map: карта глубины (H, W), float32, в миллиметрах
            mask: бинарная маска рельсов (H, W), uint8 {0, 255}
            camera_matrix: intrinsic-матрица камеры (3x3)
            color_image: цветное изображение для раскраски точек (опционально)

        Returns:
            (points_3d, colors) — массивы (N, 3), colors может быть None
        """
        h, w = depth_map.shape

        # Параметры камеры
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Найти валидные пиксели (маска + допустимая глубина)
        valid = (
            (mask > 0)
            & (depth_map > self.min_depth_mm)
            & (depth_map < self.max_depth_mm)
        )
        ys, xs = np.where(valid)

        if len(xs) == 0:
            logger.warning("No valid points found")
            return np.zeros((0, 3), dtype=np.float32), None

        # Глубина в валидных точках
        Z = depth_map[ys, xs].astype(np.float64)

        # Репроекция: (u, v, Z) → (X, Y, Z)
        X = (xs.astype(np.float64) - cx) * Z / fx
        Y = (ys.astype(np.float64) - cy) * Z / fy

        # В метрах
        points = np.column_stack([X / 1000.0, Y / 1000.0, Z / 1000.0])

        # Цвета
        colors = None
        if color_image is not None:
            # BGR → RGB, normalize to [0, 1]
            colors = color_image[ys, xs, ::-1].astype(np.float64) / 255.0

        logger.debug(f"Generated {len(points)} 3D points from mask")

        return points.astype(np.float32), colors.astype(np.float32) if colors is not None else None

    def filter_and_downsample(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Фильтрация выбросов и downsampling через Open3D.

        Args:
            points: (N, 3)
            colors: (N, 3) опционально

        Returns:
            (filtered_points, filtered_colors)
        """
        if not HAS_OPEN3D or len(points) == 0:
            return points, colors

        # Создать Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # Voxel downsampling
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Statistical Outlier Removal
        if len(pcd.points) > self.nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors,
                std_ratio=self.std_ratio,
            )

        filtered_points = np.asarray(pcd.points, dtype=np.float32)
        filtered_colors = (
            np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        )

        logger.debug(
            f"Filtered: {len(points)} → {len(filtered_points)} points"
        )

        return filtered_points, filtered_colors

    def generate_and_filter(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        camera_matrix: np.ndarray,
        color_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Генерация + фильтрация за один вызов."""
        points, colors = self.generate(depth_map, mask, camera_matrix, color_image)
        if len(points) > 0:
            points, colors = self.filter_and_downsample(points, colors)
        return points, colors

    @staticmethod
    def to_open3d(
        points: np.ndarray, colors: Optional[np.ndarray] = None
    ):
        """Конвертировать в Open3D PointCloud."""
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for this function")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd
