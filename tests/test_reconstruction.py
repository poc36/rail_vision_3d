"""Тесты для модулей 3D-реконструкции."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reconstruction.point_cloud import PointCloudGenerator
from reconstruction.ransac_fitting import RANSACFitter
from reconstruction.rail_3d import Rail3D, RailTrack3D


class TestPointCloudGenerator:
    """Тесты PointCloudGenerator."""

    def test_generate_basic(self):
        """Тест генерации облака точек."""
        gen = PointCloudGenerator(max_depth_m=50.0, min_depth_m=0.5)

        h, w = 100, 100
        depth = np.full((h, w), 5000.0, dtype=np.float32)  # 5м
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # Квадрат в центре

        camera_matrix = np.array([
            [500, 0, 50],
            [0, 500, 50],
            [0, 0, 1],
        ], dtype=np.float64)

        points, colors = gen.generate(depth, mask, camera_matrix)

        assert points.shape[1] == 3
        assert len(points) > 0  # Должны быть точки
        assert colors is None  # Не передали color_image

    def test_generate_with_colors(self):
        """Тест генерации с цветами."""
        gen = PointCloudGenerator()

        h, w = 50, 50
        depth = np.full((h, w), 3000.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        color_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        camera_matrix = np.array([
            [500, 0, 25],
            [0, 500, 25],
            [0, 0, 1],
        ], dtype=np.float64)

        points, colors = gen.generate(depth, mask, camera_matrix, color_image)

        assert points.shape[0] == colors.shape[0]
        assert colors.shape[1] == 3

    def test_empty_mask(self):
        """Пустая маска → пустое облако."""
        gen = PointCloudGenerator()

        depth = np.full((100, 100), 5000.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)

        camera_matrix = np.eye(3, dtype=np.float64)
        camera_matrix[0, 0] = camera_matrix[1, 1] = 500

        points, colors = gen.generate(depth, mask, camera_matrix)

        assert len(points) == 0


class TestRANSACFitter:
    """Тесты RANSAC фитинга."""

    def test_fit_plane(self):
        """Тест фитинга плоскости."""
        fitter = RANSACFitter(ground_distance_threshold=0.1)

        # Генерировать точки на плоскости y = 0
        n = 500
        points = np.random.randn(n, 3).astype(np.float64)
        points[:, 1] = 0  # Плоскость y=0
        points[:, 1] += np.random.randn(n) * 0.01  # Шум

        plane = fitter.fit_plane(points)

        assert plane is not None
        assert plane.inlier_ratio > 0.8

    def test_fit_line(self):
        """Тест фитинга линии."""
        fitter = RANSACFitter(rail_distance_threshold=0.05)

        # Линия вдоль оси Z
        n = 200
        t = np.linspace(0, 10, n)
        points = np.column_stack([
            np.zeros(n) + np.random.randn(n) * 0.01,
            np.zeros(n) + np.random.randn(n) * 0.01,
            t,
        ])

        line = fitter.fit_line(points)

        assert line is not None
        assert line.inlier_ratio > 0.8

        # Направление должно быть примерно вдоль Z
        assert abs(line.direction[2]) > 0.9

    def test_extract_rails(self):
        """Тест извлечения двух рельсов."""
        fitter = RANSACFitter(rail_distance_threshold=0.1)

        n = 200
        t = np.linspace(0, 10, n)

        # Левый рельс: x ≈ -1
        left = np.column_stack([
            np.full(n, -1.0) + np.random.randn(n) * 0.02,
            np.zeros(n) + np.random.randn(n) * 0.02,
            t,
        ])

        # Правый рельс: x ≈ 1
        right = np.column_stack([
            np.full(n, 1.0) + np.random.randn(n) * 0.02,
            np.zeros(n) + np.random.randn(n) * 0.02,
            t,
        ])

        all_points = np.vstack([left, right])
        _, rail_lines = fitter.extract_rails(all_points, num_rails=2)

        assert len(rail_lines) >= 1


class TestRail3D:
    """Тесты Rail3D."""

    def test_length(self):
        """Тест вычисления длины."""
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
        ], dtype=np.float32)

        rail = Rail3D(
            points=points,
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="left",
        )

        assert abs(rail.length - 3.0) < 0.01

    def test_curvature(self):
        """Тест кривизны для прямого рельса."""
        n = 50
        t = np.linspace(0, 10, n)
        points = np.column_stack([
            np.zeros(n),
            np.zeros(n),
            t,
        ]).astype(np.float32)

        rail = Rail3D(
            points=points,
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="left",
        )

        assert rail.average_curvature < 0.01  # Прямой → кривизна ≈ 0

    def test_slope(self):
        """Тест уклона для горизонтального рельса."""
        points = np.array([
            [0, 0, 0],
            [0, 0, 10],
        ], dtype=np.float32)

        rail = Rail3D(
            points=points,
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="left",
        )

        assert abs(rail.slope_angle_deg()) < 0.1  # Горизонтальный → 0°


class TestRailTrack3D:
    """Тесты RailTrack3D."""

    def test_is_complete(self):
        """Тест проверки полноты."""
        track = RailTrack3D()
        assert not track.is_complete

        left = Rail3D(
            points=np.array([[0, 0, 0]], dtype=np.float32),
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="left",
        )
        track.left_rail = left
        assert not track.is_complete

        right = Rail3D(
            points=np.array([[1.5, 0, 0]], dtype=np.float32),
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="right",
        )
        track.right_rail = right
        assert track.is_complete

    def test_gauge(self):
        """Тест ширины колеи."""
        left = Rail3D(
            points=np.array([[-0.76, 0, z] for z in range(10)], dtype=np.float32),
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="left",
        )
        right = Rail3D(
            points=np.array([[0.76, 0, z] for z in range(10)], dtype=np.float32),
            direction=np.array([0, 0, 1], dtype=np.float32),
            side="right",
        )

        track = RailTrack3D(left_rail=left, right_rail=right)

        # Расстояние = 1.52м = 1520мм (российская колея)
        assert abs(track.gauge_mm - 1520.0) < 10.0
