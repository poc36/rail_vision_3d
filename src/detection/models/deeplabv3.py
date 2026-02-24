"""
DeepLabV3+ — Модель семантической сегментации для детекции рельсов.

Использует pretrained backbone ResNet-50 из torchvision с модифицированным
выходным слоем для бинарной сегментации (фон / рельсы).
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ для бинарной сегментации рельсов.

    Основан на torchvision DeepLabV3 с backbone ResNet-50.
    Выход: тензор [B, num_classes, H, W] с логитами.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        """
        Args:
            num_classes: количество классов (2 = фон + рельсы)
            backbone: backbone сети ("resnet50" или "resnet101")
            pretrained: использовать pretrained веса на COCO
        """
        super().__init__()

        self.num_classes = num_classes

        # Загрузить pretrained DeepLabV3
        if backbone == "resnet101":
            weights = "DeepLabV3_ResNet101_Weights.DEFAULT" if pretrained else None
            self.model = seg_models.deeplabv3_resnet101(weights=weights)
        else:
            weights = "DeepLabV3_ResNet50_Weights.DEFAULT" if pretrained else None
            self.model = seg_models.deeplabv3_resnet50(weights=weights)

        # Заменить классификатор на наш (num_classes)
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(
            in_channels, num_classes, kernel_size=1
        )

        # Заменить aux классификатор
        if self.model.aux_classifier is not None:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(
                in_channels_aux, num_classes, kernel_size=1
            )

        logger.info(
            f"DeepLabV3+ initialized: backbone={backbone}, "
            f"num_classes={num_classes}, pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: входной тензор [B, 3, H, W]

        Returns:
            Логиты [B, num_classes, H, W]
        """
        result = self.model(x)
        return result["out"]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Предсказание с softmax.

        Args:
            x: входной тензор [B, 3, H, W]

        Returns:
            Вероятности [B, num_classes, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def get_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Получить бинарную маску рельсов.

        Args:
            x: входной тензор [B, 3, H, W]
            threshold: порог вероятности

        Returns:
            Бинарная маска [B, H, W] (bool)
        """
        probs = self.predict(x)
        # Класс 1 = рельсы
        rail_probs = probs[:, 1, :, :]
        mask = rail_probs > threshold
        return mask
