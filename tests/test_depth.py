"""Тесты для модуля оценки глубины."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera.depth_estimator import DepthEstimator


class TestDepthEstimator:
    """Тесты DepthEstimator."""

    def test_compute_disparity_shape(self):
        """Тест формы выходных данных."""
        estimator = DepthEstimator(
            num_disparities=64,
            block_size=5,
            use_wls_filter=False,
        )

        h, w = 240, 320
        left = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        disparity = estimator.compute_disparity(left, right)

        assert disparity.shape == (h, w)
        assert disparity.dtype == np.float32

    def test_disparity_to_depth(self):
        """Тест конвертации диспаритет → глубина."""
        estimator = DepthEstimator(use_wls_filter=False)

        # Известный диспаритет → известная глубина
        # Z = f * B / d
        f = 700.0  # px
        B = 120.0  # mm
        d = 10.0   # px

        expected_depth = f * B / d  # 8400 mm

        disparity = np.full((100, 100), d, dtype=np.float32)
        depth = estimator.disparity_to_depth(disparity, f, B)

        np.testing.assert_allclose(depth, expected_depth, rtol=1e-5)

    def test_zero_disparity_handling(self):
        """Нулевой диспаритет не должен вызывать ошибку."""
        estimator = DepthEstimator(use_wls_filter=False)

        disparity = np.zeros((100, 100), dtype=np.float32)
        depth = estimator.disparity_to_depth(disparity, 700.0, 120.0)

        # Нулевой диспаритет → глубина = 0
        assert np.all(depth == 0)

    def test_normalize_disparity(self):
        """Тест нормализации для визуализации."""
        estimator = DepthEstimator(use_wls_filter=False)

        disparity = np.random.uniform(1, 50, (100, 100)).astype(np.float32)
        colored = estimator.normalize_disparity(disparity)

        assert colored.shape == (100, 100, 3)
        assert colored.dtype == np.uint8

    def test_compute_depth(self):
        """Тест полного вычисления disparity + depth."""
        estimator = DepthEstimator(
            num_disparities=64,
            block_size=5,
            use_wls_filter=False,
        )

        h, w = 240, 320
        left = np.random.randint(50, 200, (h, w), dtype=np.uint8)
        right = np.random.randint(50, 200, (h, w), dtype=np.uint8)

        disparity, depth = estimator.compute_depth(left, right, 700.0, 120.0)

        assert disparity.shape == (h, w)
        assert depth.shape == (h, w)
