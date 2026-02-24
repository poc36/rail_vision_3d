"""
StereoCamera — Захват стерео-пары с двух камер или из файлов.

Поддерживает:
- Захват с физических камер (по device ID)
- Чтение из видеофайлов
- Чтение из папки с изображениями
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class StereoCamera:
    """Стерео-камера: захват левого и правого кадра."""

    def __init__(
        self,
        left_source=0,
        right_source=1,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
    ):
        """
        Args:
            left_source: ID камеры или путь к видео/директории для левого кадра
            right_source: ID камеры или путь к видео/директории для правого кадра
            resolution: (width, height) разрешение
            fps: кадры в секунду
        """
        self.resolution = resolution
        self.fps = fps
        self._left_source = left_source
        self._right_source = right_source
        self._left_cap: Optional[cv2.VideoCapture] = None
        self._right_cap: Optional[cv2.VideoCapture] = None
        self._image_mode = False
        self._image_list_left = []
        self._image_list_right = []
        self._image_index = 0

    def open(self) -> bool:
        """Открыть источники видео / камеры."""
        left_path = Path(str(self._left_source))
        right_path = Path(str(self._right_source))

        # Режим папки с изображениями
        if left_path.is_dir() and right_path.is_dir():
            self._image_mode = True
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            self._image_list_left = sorted(
                [p for p in left_path.iterdir() if p.suffix.lower() in exts]
            )
            self._image_list_right = sorted(
                [p for p in right_path.iterdir() if p.suffix.lower() in exts]
            )
            if len(self._image_list_left) == 0 or len(self._image_list_right) == 0:
                logger.error("No images found in directories")
                return False
            if len(self._image_list_left) != len(self._image_list_right):
                logger.warning(
                    f"Different number of images: left={len(self._image_list_left)}, "
                    f"right={len(self._image_list_right)}"
                )
            self._image_index = 0
            logger.info(
                f"Image mode: {len(self._image_list_left)} left, "
                f"{len(self._image_list_right)} right"
            )
            return True

        # Режим видео / камеры
        self._image_mode = False
        self._left_cap = cv2.VideoCapture(self._left_source)
        self._right_cap = cv2.VideoCapture(self._right_source)

        if not self._left_cap.isOpened():
            logger.error(f"Cannot open left source: {self._left_source}")
            return False
        if not self._right_cap.isOpened():
            logger.error(f"Cannot open right source: {self._right_source}")
            return False

        # Установить разрешение
        for cap in [self._left_cap, self._right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(
            f"Stereo camera opened: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps"
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Прочитать стерео-пару.

        Returns:
            (success, left_frame, right_frame)
        """
        if self._image_mode:
            return self._read_images()
        return self._read_video()

    def _read_video(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Чтение из видео / камеры."""
        if self._left_cap is None or self._right_cap is None:
            return False, None, None

        ret_l, frame_l = self._left_cap.read()
        ret_r, frame_r = self._right_cap.read()

        if not ret_l or not ret_r:
            return False, None, None

        # Привести к нужному разрешению
        if frame_l.shape[1] != self.resolution[0] or frame_l.shape[0] != self.resolution[1]:
            frame_l = cv2.resize(frame_l, self.resolution)
        if frame_r.shape[1] != self.resolution[0] or frame_r.shape[0] != self.resolution[1]:
            frame_r = cv2.resize(frame_r, self.resolution)

        return True, frame_l, frame_r

    def _read_images(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Чтение из папки с изображениями."""
        if self._image_index >= len(self._image_list_left):
            return False, None, None
        if self._image_index >= len(self._image_list_right):
            return False, None, None

        left_path = self._image_list_left[self._image_index]
        right_path = self._image_list_right[self._image_index]
        self._image_index += 1

        frame_l = cv2.imread(str(left_path))
        frame_r = cv2.imread(str(right_path))

        if frame_l is None or frame_r is None:
            logger.error(f"Failed to read: {left_path}, {right_path}")
            return False, None, None

        # Привести к нужному разрешению
        frame_l = cv2.resize(frame_l, self.resolution)
        frame_r = cv2.resize(frame_r, self.resolution)

        return True, frame_l, frame_r

    def release(self):
        """Освободить ресурсы."""
        if self._left_cap is not None:
            self._left_cap.release()
        if self._right_cap is not None:
            self._right_cap.release()
        logger.info("Stereo camera released")

    @property
    def frame_count(self) -> int:
        """Количество кадров (для video/image mode)."""
        if self._image_mode:
            return min(len(self._image_list_left), len(self._image_list_right))
        if self._left_cap is not None:
            return int(self._left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def reset(self):
        """Сброс на начало."""
        if self._image_mode:
            self._image_index = 0
        else:
            if self._left_cap is not None:
                self._left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self._right_cap is not None:
                self._right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
