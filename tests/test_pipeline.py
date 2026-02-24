"""Интеграционные тесты пайплайна."""

import sys
import os
import importlib
import numpy as np
import pytest

# Add parent dir so we can import src as a package
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

# Patch the src package to allow relative imports within pipeline
import src
import src.camera
import src.detection
import src.detection.models
import src.reconstruction
import src.tracking
import src.visualization
import src.pipeline



class TestPipelineIntegration:
    """Интеграционные тесты RailVisionPipeline."""

    def test_process_mono(self):
        """Тест обработки одного изображения (без стерео)."""
        from src.pipeline.vision_pipeline import RailVisionPipeline

        pipeline = RailVisionPipeline()

        # Синтетическое изображение
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = pipeline.process_mono(image)

        assert result.frame_id == 1
        assert result.segmentation_mask is not None
        assert result.segmentation_mask.shape == (720, 1280)
        assert result.detection_result is not None
        assert result.visualization_2d is not None
        assert result.processing_time_ms > 0

    def test_process_frame(self):
        """Тест полной обработки стерео-кадра."""
        from src.pipeline.vision_pipeline import RailVisionPipeline

        pipeline = RailVisionPipeline()

        left = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        result = pipeline.process_frame(left, right)

        assert result.frame_id == 1
        assert result.rectified_left is not None
        assert result.disparity is not None
        assert result.depth_map is not None
        assert result.segmentation_mask is not None
        assert result.detection_result is not None
        assert result.visualization_2d is not None

    def test_sequential_frames(self):
        """Тест последовательной обработки кадров."""
        from src.pipeline.vision_pipeline import RailVisionPipeline

        pipeline = RailVisionPipeline()

        for i in range(3):
            left = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            right = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            result = pipeline.process_frame(left, right)
            assert result.frame_id == i + 1
