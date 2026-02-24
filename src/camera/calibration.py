"""
StereoCalibrator — Калибровка стерео-камеры по шахматной доске.

Выполняет:
- Поиск углов шахматной доски на изображениях
- Калибровку каждой камеры отдельно
- Стерео-калибровку
- Ректификацию стерео-пары
- Сохранение/загрузку параметров
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class StereoCalibrator:
    """Калибровка и ректификация стерео-камеры."""

    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0,
    ):
        """
        Args:
            chessboard_size: (cols, rows) — количество внутренних углов
            square_size_mm: размер одной клетки в миллиметрах
        """
        self.chessboard_size = chessboard_size
        self.square_size_mm = square_size_mm

        # Результаты калибровки
        self.camera_matrix_left: Optional[np.ndarray] = None
        self.dist_coeffs_left: Optional[np.ndarray] = None
        self.camera_matrix_right: Optional[np.ndarray] = None
        self.dist_coeffs_right: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None  # Матрица поворота
        self.T: Optional[np.ndarray] = None  # Вектор трансляции
        self.E: Optional[np.ndarray] = None  # Essential matrix
        self.F: Optional[np.ndarray] = None  # Fundamental matrix

        # Ректификация
        self.R1: Optional[np.ndarray] = None
        self.R2: Optional[np.ndarray] = None
        self.P1: Optional[np.ndarray] = None
        self.P2: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None  # Disparity-to-depth mapping
        self.map_left_x: Optional[np.ndarray] = None
        self.map_left_y: Optional[np.ndarray] = None
        self.map_right_x: Optional[np.ndarray] = None
        self.map_right_y: Optional[np.ndarray] = None

        self._image_size: Optional[Tuple[int, int]] = None
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _prepare_object_points(self) -> np.ndarray:
        """Подготовка 3D-точек для шахматной доски."""
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp

    def find_chessboard_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Найти углы шахматной доски на изображении.

        Returns:
            (found, corners) - corners с субпиксельной точностью
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return found, corners

    def calibrate(
        self,
        left_images: list,
        right_images: list,
    ) -> float:
        """
        Выполнить полную стерео-калибровку.

        Args:
            left_images: список изображений с левой камеры (np.ndarray)
            right_images: список изображений с правой камеры (np.ndarray)

        Returns:
            Ошибка репроекции (RMS)
        """
        assert len(left_images) == len(right_images), "Same number of images required"
        assert len(left_images) > 0, "At least one pair required"

        objp = self._prepare_object_points()
        obj_points = []
        img_points_left = []
        img_points_right = []

        for i, (left, right) in enumerate(zip(left_images, right_images)):
            found_l, corners_l = self.find_chessboard_corners(left)
            found_r, corners_r = self.find_chessboard_corners(right)

            if found_l and found_r:
                obj_points.append(objp)
                img_points_left.append(corners_l)
                img_points_right.append(corners_r)
                logger.info(f"Frame {i}: chessboard found in both images")
            else:
                logger.warning(
                    f"Frame {i}: chessboard not found (L={found_l}, R={found_r})"
                )

        if len(obj_points) < 5:
            raise ValueError(
                f"Not enough valid pairs: {len(obj_points)}. Need at least 5."
            )

        h, w = left_images[0].shape[:2]
        self._image_size = (w, h)

        logger.info(f"Calibrating with {len(obj_points)} valid pairs...")

        # Калибровка каждой камеры
        ret_l, self.camera_matrix_left, self.dist_coeffs_left, _, _ = (
            cv2.calibrateCamera(obj_points, img_points_left, (w, h), None, None)
        )
        ret_r, self.camera_matrix_right, self.dist_coeffs_right, _, _ = (
            cv2.calibrateCamera(obj_points, img_points_right, (w, h), None, None)
        )

        logger.info(f"Left camera RMS: {ret_l:.4f}")
        logger.info(f"Right camera RMS: {ret_r:.4f}")

        # Стерео-калибровка
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        rms, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            (w, h),
            criteria=criteria,
            flags=flags,
        )

        logger.info(f"Stereo calibration RMS: {rms:.4f}")

        # Ректификация
        self._compute_rectification()

        self._is_calibrated = True
        return rms

    def _compute_rectification(self):
        """Вычислить параметры ректификации."""
        assert self._image_size is not None

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self._image_size,
            self.R,
            self.T,
            alpha=0,
        )

        # Карты для remap
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.R1,
            self.P1,
            self._image_size,
            cv2.CV_32FC1,
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.R2,
            self.P2,
            self._image_size,
            cv2.CV_32FC1,
        )

        logger.info("Rectification maps computed")

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ректификация стерео-пары.

        Args:
            left: левый кадр
            right: правый кадр

        Returns:
            (rectified_left, rectified_right)
        """
        if not self._is_calibrated:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")

        rect_left = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(
            right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR
        )

        return rect_left, rect_right

    def save(self, filepath: str):
        """Сохранить параметры калибровки."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "image_size": list(self._image_size) if self._image_size else None,
            "chessboard_size": list(self.chessboard_size),
            "square_size_mm": self.square_size_mm,
        }

        # Сохраняем матрицы в npz
        npz_path = filepath.with_suffix(".npz")
        arrays = {}
        for name in [
            "camera_matrix_left", "dist_coeffs_left",
            "camera_matrix_right", "dist_coeffs_right",
            "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q",
            "map_left_x", "map_left_y", "map_right_x", "map_right_y",
        ]:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val

        np.savez(str(npz_path), **arrays)

        # Сохраняем метаданные в yaml
        yaml_path = filepath.with_suffix(".yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Calibration saved to {filepath}")

    def load(self, filepath: str):
        """Загрузить параметры калибровки."""
        filepath = Path(filepath)

        npz_path = filepath.with_suffix(".npz")
        yaml_path = filepath.with_suffix(".yaml")

        if not npz_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {npz_path}")

        # Загрузить метаданные
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if data.get("image_size"):
                self._image_size = tuple(data["image_size"])
            if data.get("chessboard_size"):
                self.chessboard_size = tuple(data["chessboard_size"])
            if data.get("square_size_mm"):
                self.square_size_mm = data["square_size_mm"]

        # Загрузить матрицы
        arrays = np.load(str(npz_path))
        for name in arrays.files:
            setattr(self, name, arrays[name])

        self._is_calibrated = True
        logger.info(f"Calibration loaded from {filepath}")

    def create_default_calibration(
        self,
        image_size: Tuple[int, int] = (1280, 720),
        focal_length_px: float = 700.0,
        baseline_mm: float = 120.0,
    ):
        """
        Создать калибровку по умолчанию (без шахматной доски).

        Полезно для тестирования и демо.
        """
        w, h = image_size
        self._image_size = (w, h)

        cx, cy = w / 2.0, h / 2.0
        fx = fy = focal_length_px

        self.camera_matrix_left = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )
        self.camera_matrix_right = self.camera_matrix_left.copy()

        self.dist_coeffs_left = np.zeros(5, dtype=np.float64)
        self.dist_coeffs_right = np.zeros(5, dtype=np.float64)

        self.R = np.eye(3, dtype=np.float64)
        self.T = np.array([[baseline_mm], [0.0], [0.0]], dtype=np.float64)

        # Ректификация (identity для идеальных камер)
        self.R1 = np.eye(3, dtype=np.float64)
        self.R2 = np.eye(3, dtype=np.float64)
        self.P1 = np.hstack([self.camera_matrix_left, np.zeros((3, 1))])
        self.P2 = np.hstack(
            [self.camera_matrix_right, np.array([[-fx * baseline_mm], [0], [0]])]
        )

        # Q matrix: disparity to depth
        self.Q = np.array(
            [
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, fx],
                [0, 0, -1.0 / baseline_mm, 0],
            ],
            dtype=np.float64,
        )

        # Identity remap maps
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.map_left_x = map_x
        self.map_left_y = map_y
        self.map_right_x = map_x.copy()
        self.map_right_y = map_y.copy()

        self._is_calibrated = True
        logger.info(
            f"Default calibration created: {w}x{h}, f={fx}, B={baseline_mm}mm"
        )
