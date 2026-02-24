"""
RailSegmentor — Обёртка для нейросетевой сегментации рельсов.

Управляет загрузкой модели, препроцессингом изображений,
инференсом и постпроцессингом масок.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

from .models.deeplabv3 import DeepLabV3Plus
from .models.unet import UNet

logger = logging.getLogger(__name__)


class RailSegmentor:
    """Сегментация рельсов на изображениях."""

    # Нормализация ImageNet
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_name: str = "deeplabv3",
        device: str = "cpu",
        input_size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.5,
        weights_path: Optional[str] = None,
        num_classes: int = 2,
        **model_kwargs,
    ):
        """
        Args:
            model_name: "deeplabv3" или "unet"
            device: "cpu" или "cuda"
            input_size: (H, W) размер входа модели
            confidence_threshold: порог уверенности для маски
            weights_path: путь к обученным весам (None = pretrained backbone)
            num_classes: количество классов
            **model_kwargs: доп. параметры для модели
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        # Создать модель
        if model_name == "deeplabv3":
            self.model = DeepLabV3Plus(
                num_classes=num_classes,
                backbone=model_kwargs.get("backbone", "resnet50"),
                pretrained=model_kwargs.get("pretrained", True),
            )
        elif model_name == "unet":
            self.model = UNet(
                in_channels=model_kwargs.get("in_channels", 3),
                out_channels=num_classes,
                features=model_kwargs.get("features", [64, 128, 256, 512]),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Загрузить обученные веса
        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            except FileNotFoundError:
                logger.warning(
                    f"Weights not found: {weights_path}. Using pretrained backbone."
                )

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"RailSegmentor: model={model_name}, device={device}, "
            f"input_size={input_size}"
        )

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Препроцессинг изображения для модели.

        Args:
            image: BGR изображение (H, W, 3), uint8

        Returns:
            Тензор [1, 3, H, W], нормализованный
        """
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))

        # Normalize [0, 1] → ImageNet normalization
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD

        # HWC → CHW → BCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Постпроцессинг выхода модели.

        Args:
            output: логиты [1, num_classes, H, W]
            original_size: (H, W) оригинального изображения

        Returns:
            Бинарная маска рельсов (H, W), uint8 {0, 255}
        """
        # Softmax → вероятности
        probs = torch.softmax(output, dim=1)

        # Класс 1 = рельсы
        rail_prob = probs[0, 1].cpu().numpy()

        # Resize до оригинального размера
        rail_prob = cv2.resize(rail_prob, (original_size[1], original_size[0]))

        # Бинаризация
        mask = (rail_prob > self.confidence_threshold).astype(np.uint8) * 255

        return mask

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Сегментировать рельсы на изображении.

        Args:
            image: BGR изображение (H, W, 3)

        Returns:
            Бинарная маска рельсов (H, W), uint8 {0, 255}
        """
        original_size = image.shape[:2]

        tensor = self.preprocess(image)
        output = self.model(tensor)

        mask = self.postprocess(output, original_size)

        return mask

    @torch.no_grad()
    def segment_batch(self, images: list) -> list:
        """
        Batch-сегментация.

        Args:
            images: список BGR изображений

        Returns:
            Список масок
        """
        if len(images) == 0:
            return []

        original_sizes = [img.shape[:2] for img in images]

        # Батч-препроцессинг
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        # Инференс
        outputs = self.model(batch)

        # Постпроцессинг
        masks = []
        probs = torch.softmax(outputs, dim=1)
        for i in range(len(images)):
            rail_prob = probs[i, 1].cpu().numpy()
            rail_prob = cv2.resize(
                rail_prob, (original_sizes[i][1], original_sizes[i][0])
            )
            mask = (rail_prob > self.confidence_threshold).astype(np.uint8) * 255
            masks.append(mask)

        return masks

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        """
        Получить карту вероятностей (не бинарную маску).

        Returns:
            Карта вероятностей рельсов (H, W), float32 [0, 1]
        """
        original_size = image.shape[:2]
        tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(tensor)

        probs = torch.softmax(output, dim=1)
        rail_prob = probs[0, 1].cpu().numpy()
        rail_prob = cv2.resize(rail_prob, (original_size[1], original_size[0]))

        return rail_prob
