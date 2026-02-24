"""Тесты для модуля калибровки."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera.calibration import StereoCalibrator


class TestStereoCalibrator:
    """Тесты StereoCalibrator."""

    def test_default_calibration(self):
        """Тест калибровки по умолчанию."""
        cal = StereoCalibrator()
        cal.create_default_calibration(
            image_size=(640, 480),
            focal_length_px=500.0,
            baseline_mm=100.0,
        )

        assert cal.is_calibrated
        assert cal.camera_matrix_left is not None
        assert cal.camera_matrix_right is not None
        assert cal.Q is not None

        # Проверить фокусное расстояние
        assert cal.camera_matrix_left[0, 0] == 500.0
        assert cal.camera_matrix_left[1, 1] == 500.0

        # Проверить центр
        assert cal.camera_matrix_left[0, 2] == 320.0
        assert cal.camera_matrix_left[1, 2] == 240.0

    def test_rectification(self):
        """Тест ректификации (identity для дефолтной калибровки)."""
        cal = StereoCalibrator()
        cal.create_default_calibration()

        left = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        rect_l, rect_r = cal.rectify(left, right)

        assert rect_l.shape == left.shape
        assert rect_r.shape == right.shape

    def test_uncalibrated_rectify_raises(self):
        """Ректификация без калибровки должна выбросить ошибку."""
        cal = StereoCalibrator()
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError):
            cal.rectify(left, right)

    def test_save_load(self, tmp_path):
        """Тест сохранения/загрузки калибровки."""
        cal = StereoCalibrator()
        cal.create_default_calibration(
            image_size=(640, 480),
            focal_length_px=600.0,
            baseline_mm=150.0,
        )

        filepath = tmp_path / "calib"
        cal.save(str(filepath))

        cal2 = StereoCalibrator()
        cal2.load(str(filepath))

        assert cal2.is_calibrated
        np.testing.assert_array_almost_equal(
            cal.camera_matrix_left, cal2.camera_matrix_left
        )
        np.testing.assert_array_almost_equal(
            cal.Q, cal2.Q
        )
