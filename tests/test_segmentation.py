"""Тесты для модулей сегментации."""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestUNet:
    """Тесты U-Net."""

    def test_forward_shape(self):
        """Тест формы выхода."""
        from detection.models.unet import UNet

        model = UNet(in_channels=3, out_channels=2, features=[32, 64, 128, 256])

        x = torch.randn(1, 3, 256, 256)
        output = model(x)

        assert output.shape == (1, 2, 256, 256)

    def test_predict(self):
        """Тест predict (softmax probabilities)."""
        from detection.models.unet import UNet

        model = UNet(in_channels=3, out_channels=2, features=[32, 64, 128, 256])

        x = torch.randn(1, 3, 128, 128)
        probs = model.predict(x)

        assert probs.shape == (1, 2, 128, 128)
        # Сумма вероятностей по классам = 1
        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_get_mask(self):
        """Тест get_mask."""
        from detection.models.unet import UNet

        model = UNet(in_channels=3, out_channels=2, features=[32, 64, 128, 256])
        x = torch.randn(1, 3, 128, 128)
        mask = model.get_mask(x, threshold=0.5)

        assert mask.shape == (1, 128, 128)
        assert mask.dtype == torch.bool


class TestDeepLabV3Plus:
    """Тесты DeepLabV3+."""

    def test_forward_shape(self):
        """Тест формы выхода."""
        from detection.models.deeplabv3 import DeepLabV3Plus

        model = DeepLabV3Plus(num_classes=2, pretrained=False)
        model.eval()  # Required for batch_size=1 (BatchNorm)

        # DeepLabV3 requires minimum ~520x520 to avoid stride mismatch
        x = torch.randn(1, 3, 520, 520)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2, 520, 520)

    def test_predict(self):
        """Тест predict."""
        from detection.models.deeplabv3 import DeepLabV3Plus

        model = DeepLabV3Plus(num_classes=2, pretrained=False)

        x = torch.randn(1, 3, 128, 128)
        probs = model.predict(x)

        assert probs.shape == (1, 2, 128, 128)


class TestRailDetector:
    """Тесты RailDetector."""

    def test_detect_empty_mask(self):
        """Тест на пустой маске."""
        from detection.rail_detector import RailDetector

        detector = RailDetector(use_hough=False)
        mask = np.zeros((480, 640), dtype=np.uint8)
        result = detector.detect(mask)

        assert result.num_rails == 0
        assert len(result.rails) == 0

    def test_detect_with_rails(self):
        """Тест с синтетическими рельсами."""
        from detection.rail_detector import RailDetector

        detector = RailDetector(min_contour_area=100, use_hough=False)

        # Создать маску с двумя вертикальными полосами (рельсы)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:400, 200:220] = 255  # Левый рельс
        mask[100:400, 420:440] = 255  # Правый рельс

        result = detector.detect(mask)

        assert result.num_rails == 2
        assert result.rails[0].side in ("left", "right")
        assert result.rails[1].side in ("left", "right")
