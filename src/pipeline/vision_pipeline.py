"""
RailVisionPipeline — Главный конвейер обработки.

Оркестрирует все модули:
Захват → Ректификация → Глубина → Сегментация → Детекция →
3D-реконструкция → RANSAC → Трекинг → Визуализация
"""

import cv2
import yaml
import numpy as np
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..camera.stereo_camera import StereoCamera
from ..camera.calibration import StereoCalibrator
from ..camera.depth_estimator import DepthEstimator
from ..detection.rail_segmentation import RailSegmentor
from ..detection.rail_detector import RailDetector
from ..reconstruction.point_cloud import PointCloudGenerator
from ..reconstruction.ransac_fitting import RANSACFitter
from ..reconstruction.rail_3d import Rail3D, RailTrack3D
from ..tracking.rail_tracker import RailTracker
from ..visualization.visualizer_2d import Visualizer2D
from ..visualization.visualizer_3d import Visualizer3D

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Результат обработки одного кадра."""

    frame_id: int
    timestamp: float

    # Исходные данные
    left_image: np.ndarray
    right_image: Optional[np.ndarray] = None

    # Промежуточные результаты
    rectified_left: Optional[np.ndarray] = None
    rectified_right: Optional[np.ndarray] = None
    disparity: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    detection_result: Optional[Any] = None

    # 3D результаты
    points_3d: Optional[np.ndarray] = None
    point_colors: Optional[np.ndarray] = None
    rail_track: Optional[RailTrack3D] = None

    # Визуализация
    visualization_2d: Optional[np.ndarray] = None

    # Метрики
    processing_time_ms: float = 0.0


class RailVisionPipeline:
    """Главный конвейер системы Rail Vision 3D."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        """
        Args:
            config_path: путь к config.yaml
            config: словарь конфигурации (приоритет над файлом)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self._frame_id = 0
        self._setup_modules()

        logger.info("RailVisionPipeline initialized")

    def _default_config(self) -> dict:
        """Конфигурация по умолчанию."""
        return {
            "camera": {
                "resolution": {"width": 1280, "height": 720},
                "fps": 30,
                "left_id": 0,
                "right_id": 1,
                "baseline_mm": 120.0,
                "focal_length_px": 700.0,
            },
            "depth": {
                "sgbm": {
                    "num_disparities": 128,
                    "block_size": 5,
                    "p1": 600,
                    "p2": 2400,
                },
                "wls_filter": {"enabled": True},
            },
            "segmentation": {
                "model": "deeplabv3",
                "device": "cpu",
                "input_size": [512, 512],
                "confidence_threshold": 0.5,
            },
            "detection": {
                "morphology": {"kernel_size": 5},
                "min_contour_area": 500,
            },
            "reconstruction": {
                "point_cloud": {"max_depth_m": 50.0, "min_depth_m": 0.5},
            },
            "tracking": {
                "kalman": {"process_noise": 0.01, "measurement_noise": 0.1},
                "max_age": 10,
                "smoothing_alpha": 0.7,
            },
            "visualization": {
                "overlay_alpha": 0.4,
                "show_depth": True,
                "show_hud": True,
            },
        }

    def _setup_modules(self):
        """Инициализация всех модулей."""
        cam_cfg = self.config.get("camera", {})
        depth_cfg = self.config.get("depth", {})
        seg_cfg = self.config.get("segmentation", {})
        det_cfg = self.config.get("detection", {})
        recon_cfg = self.config.get("reconstruction", {})
        track_cfg = self.config.get("tracking", {})
        vis_cfg = self.config.get("visualization", {})

        # Калибровка (default)
        self.calibrator = StereoCalibrator()
        self.calibrator.create_default_calibration(
            image_size=(
                cam_cfg.get("resolution", {}).get("width", 1280),
                cam_cfg.get("resolution", {}).get("height", 720),
            ),
            focal_length_px=cam_cfg.get("focal_length_px", 700.0),
            baseline_mm=cam_cfg.get("baseline_mm", 120.0),
        )

        # Глубина
        sgbm = depth_cfg.get("sgbm", {})
        wls = depth_cfg.get("wls_filter", {})
        self.depth_estimator = DepthEstimator(
            num_disparities=sgbm.get("num_disparities", 128),
            block_size=sgbm.get("block_size", 5),
            p1=sgbm.get("p1", 600),
            p2=sgbm.get("p2", 2400),
            use_wls_filter=wls.get("enabled", True),
        )

        # Сегментация
        self.segmentor = RailSegmentor(
            model_name=seg_cfg.get("model", "deeplabv3"),
            device=seg_cfg.get("device", "cpu"),
            input_size=tuple(seg_cfg.get("input_size", [512, 512])),
            confidence_threshold=seg_cfg.get("confidence_threshold", 0.5),
            weights_path=seg_cfg.get("weights_path"),
        )

        # Детекция
        morph = det_cfg.get("morphology", {})
        self.detector = RailDetector(
            morph_kernel_size=morph.get("kernel_size", 5),
            min_contour_area=det_cfg.get("min_contour_area", 500),
        )

        # 3D-реконструкция
        pc_cfg = recon_cfg.get("point_cloud", {})
        self.point_cloud_gen = PointCloudGenerator(
            max_depth_m=pc_cfg.get("max_depth_m", 50.0),
            min_depth_m=pc_cfg.get("min_depth_m", 0.5),
        )

        self.ransac = RANSACFitter()

        # Трекинг
        kalman = track_cfg.get("kalman", {})
        self.tracker = RailTracker(
            process_noise=kalman.get("process_noise", 0.01),
            measurement_noise=kalman.get("measurement_noise", 0.1),
            max_age=track_cfg.get("max_age", 10),
            smoothing_alpha=track_cfg.get("smoothing_alpha", 0.7),
        )

        # Визуализация
        self.vis_2d = Visualizer2D(
            overlay_alpha=vis_cfg.get("overlay_alpha", 0.4),
            show_depth=vis_cfg.get("show_depth", True),
            show_hud=vis_cfg.get("show_hud", True),
        )
        self.vis_3d = Visualizer3D()

        self.focal_length_px = cam_cfg.get("focal_length_px", 700.0)
        self.baseline_mm = cam_cfg.get("baseline_mm", 120.0)

    def process_frame(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> FrameResult:
        """
        Обработать один стерео-кадр.

        Args:
            left_image: левый BGR-кадр
            right_image: правый BGR-кадр

        Returns:
            FrameResult со всеми результатами
        """
        start_time = time.time()
        self._frame_id += 1

        result = FrameResult(
            frame_id=self._frame_id,
            timestamp=start_time,
            left_image=left_image,
            right_image=right_image,
        )

        try:
            # 1. Ректификация
            if self.calibrator.is_calibrated:
                result.rectified_left, result.rectified_right = (
                    self.calibrator.rectify(left_image, right_image)
                )
            else:
                result.rectified_left = left_image
                result.rectified_right = right_image

            # 2. Глубина
            result.disparity, result.depth_map = self.depth_estimator.compute_depth(
                result.rectified_left,
                result.rectified_right,
                self.focal_length_px,
                self.baseline_mm,
            )

            # 3. Сегментация
            result.segmentation_mask = self.segmentor.segment(result.rectified_left)

            # 4. Постобработка (контуры, линии)
            result.detection_result = self.detector.detect(result.segmentation_mask)

            # 5. 3D-реконструкция
            if result.depth_map is not None and result.segmentation_mask is not None:
                result.points_3d, result.point_colors = (
                    self.point_cloud_gen.generate_and_filter(
                        result.depth_map,
                        result.segmentation_mask,
                        self.calibrator.camera_matrix_left,
                        result.rectified_left,
                    )
                )

            # 6. RANSAC фитинг
            rail_track = RailTrack3D()
            if result.points_3d is not None and len(result.points_3d) > 10:
                _, rail_lines = self.ransac.extract_rails(result.points_3d)

                if len(rail_lines) >= 1:
                    left_rail = Rail3D(
                        points=result.points_3d[rail_lines[0].inlier_mask],
                        direction=rail_lines[0].direction,
                        side="left",
                    )
                    left_rail.fit_spline()
                    rail_track.left_rail = left_rail

                if len(rail_lines) >= 2:
                    right_rail = Rail3D(
                        points=result.points_3d[rail_lines[1].inlier_mask],
                        direction=rail_lines[1].direction,
                        side="right",
                    )
                    right_rail.fit_spline()
                    rail_track.right_rail = right_rail

            result.rail_track = rail_track

            # 7. Трекинг
            centers = []
            directions = []
            sides = []
            for rail in [rail_track.left_rail, rail_track.right_rail]:
                if rail is not None:
                    centers.append(rail.centroid)
                    directions.append(rail.direction)
                    sides.append(rail.side)
            if centers:
                self.tracker.update(centers, directions, sides)

            # 8. 2D-визуализация
            result.visualization_2d = self.vis_2d.draw(
                result.rectified_left,
                mask=result.segmentation_mask,
                detection_result=result.detection_result,
                depth_map=result.depth_map,
                rail_info=rail_track.summary() if rail_track else None,
            )

        except Exception as e:
            logger.error(f"Frame {self._frame_id} processing error: {e}")
            import traceback
            traceback.print_exc()

        result.processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Frame {self._frame_id}: {result.processing_time_ms:.1f}ms, "
            f"rails={result.detection_result.num_rails if result.detection_result else 0}"
        )

        return result

    def process_mono(self, image: np.ndarray) -> FrameResult:
        """
        Обработка одного изображения (без стерео-глубины).

        Полезно для тестирования сегментации без второй камеры.
        """
        start_time = time.time()
        self._frame_id += 1

        result = FrameResult(
            frame_id=self._frame_id,
            timestamp=start_time,
            left_image=image,
        )

        # Только сегментация + детекция
        result.segmentation_mask = self.segmentor.segment(image)
        result.detection_result = self.detector.detect(result.segmentation_mask)

        result.visualization_2d = self.vis_2d.draw(
            image,
            mask=result.segmentation_mask,
            detection_result=result.detection_result,
        )

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def run_on_camera(self, camera: StereoCamera, max_frames: int = 0):
        """
        Запуск пайплайна на стерео-камере.

        Args:
            camera: StereoCamera (открытая)
            max_frames: максимум кадров (0 = бесконечно)
        """
        frame_count = 0

        while True:
            ret, left, right = camera.read()
            if not ret:
                logger.info("No more frames")
                break

            result = self.process_frame(left, right)

            # Показать 2D
            if result.visualization_2d is not None:
                cv2.imshow("Rail Vision 3D", result.visualization_2d)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("3"):
                # Показать 3D по клавише '3'
                if result.points_3d is not None:
                    self.vis_3d.visualize(
                        result.points_3d,
                        result.point_colors,
                        rail_track=result.rail_track,
                    )

            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                break

        cv2.destroyAllWindows()

    def run_on_images(
        self,
        left_dir: str,
        right_dir: str,
        max_frames: int = 0,
    ):
        """Запуск на папках с изображениями."""
        camera = StereoCamera(
            left_source=left_dir,
            right_source=right_dir,
        )
        camera.open()
        self.run_on_camera(camera, max_frames)
        camera.release()

    def show_3d(self, result: FrameResult):
        """Показать 3D-визуализацию результата."""
        self.vis_3d.visualize(
            result.points_3d,
            result.point_colors,
            rail_track=result.rail_track,
        )
