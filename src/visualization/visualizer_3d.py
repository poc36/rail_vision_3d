"""
Visualizer3D — 3D-визуализация облака точек и рельсов с Open3D.

Рисует:
- Облако точек рельсов (раскрашенное)
- Линии рельсов (сплайны)
- Координатные оси
- Плоскость земли
"""

import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not installed. 3D visualization disabled.")


class Visualizer3D:
    """3D-визуализация рельсов и облака точек."""

    # Цвета (RGB, 0-1)
    COLOR_RAIL_LEFT = [0.0, 1.0, 0.0]    # Зелёный
    COLOR_RAIL_RIGHT = [1.0, 0.6, 0.0]   # Оранжевый
    COLOR_GROUND = [0.5, 0.5, 0.5]        # Серый
    COLOR_SPLINE_LEFT = [0.0, 0.8, 0.2]
    COLOR_SPLINE_RIGHT = [0.8, 0.4, 0.0]

    def __init__(
        self,
        window_width: int = 1024,
        window_height: int = 768,
        point_size: float = 2.0,
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.point_size = point_size

    def visualize(
        self,
        points: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        rail_lines: Optional[list] = None,
        rail_track=None,
        show_axes: bool = True,
        window_name: str = "Rail Vision 3D",
    ):
        """
        Показать 3D-визуализацию.

        Args:
            points: (N, 3) — облако точек
            colors: (N, 3) — цвета (RGB, [0, 1])
            rail_lines: список LineModel или Rail3D
            rail_track: RailTrack3D
            show_axes: показать координатные оси
            window_name: название окна
        """
        if not HAS_OPEN3D:
            logger.error("Open3D not available for 3D visualization")
            return

        geometries = []

        # 1. Облако точек
        if points is not None and len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            else:
                # По умолчанию — зелёный
                default_colors = np.tile(self.COLOR_RAIL_LEFT, (len(points), 1))
                pcd.colors = o3d.utility.Vector3dVector(default_colors)
            geometries.append(pcd)

        # 2. Линии рельсов (из RANSAC LineModel)
        if rail_lines is not None:
            for i, line in enumerate(rail_lines):
                line_geo = self._create_line_geometry(line, i)
                if line_geo is not None:
                    geometries.append(line_geo)

        # 3. Сплайны из RailTrack3D
        if rail_track is not None:
            spline_geos = self._create_track_geometry(rail_track)
            geometries.extend(spline_geos)

        # 4. Координатные оси
        if show_axes:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            geometries.append(axes)

        if len(geometries) == 0:
            logger.warning("Nothing to visualize")
            return

        # Показать
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=self.window_width,
            height=self.window_height,
            point_show_normal=False,
        )

    def _create_line_geometry(self, line_model, index: int):
        """Создать Open3D geometry для LineModel."""
        if not hasattr(line_model, "point") or not hasattr(line_model, "direction"):
            return None

        # Длинная линия через точку
        t = np.linspace(-20, 20, 100)
        pts = line_model.point + t[:, np.newaxis] * line_model.direction

        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        color = self.COLOR_RAIL_LEFT if index == 0 else self.COLOR_RAIL_RIGHT

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        return line_set

    def _create_track_geometry(self, rail_track) -> list:
        """Создать геометрию для RailTrack3D."""
        geometries = []

        for rail, color in [
            (rail_track.left_rail, self.COLOR_SPLINE_LEFT),
            (rail_track.right_rail, self.COLOR_SPLINE_RIGHT),
        ]:
            if rail is None:
                continue

            # Сплайн
            spline_pts = rail.get_spline_points()
            if spline_pts is not None and len(spline_pts) > 1:
                lines = [[i, i + 1] for i in range(len(spline_pts) - 1)]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(
                    spline_pts.astype(np.float64)
                )
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(
                    [color] * len(lines)
                )
                geometries.append(line_set)

            # Точки рельса (маленькие сферы)
            if len(rail.points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    rail.points.astype(np.float64)
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(rail.points), 1))
                )
                geometries.append(pcd)

        # Центральная линия пути
        center = rail_track.center_line
        if center is not None and len(center.shape) == 2 and len(center) > 1:
            lines = [[i, i + 1] for i in range(len(center) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(center.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [[1.0, 1.0, 0.0]] * len(lines)  # Жёлтая центральная линия
            )
            geometries.append(line_set)

        return geometries

    def save_screenshot(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        filepath: str,
    ):
        """Сохранить скриншот 3D-визуализации."""
        if not HAS_OPEN3D:
            logger.error("Open3D not available")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=self.window_width,
            height=self.window_height,
            visible=False,
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        vis.add_geometry(pcd)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axes)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filepath)
        vis.destroy_window()

        logger.info(f"3D screenshot saved: {filepath}")
