"""
U-Net — Лёгкая модель сегментации для быстрого инференса на CPU.

Классическая архитектура Encoder-Decoder с skip-connections.
Подходит для real-time обработки на CPU.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Два последовательных Conv → BN → ReLU блока."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder блок: MaxPool → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Decoder блок: Upsample → Cat skip → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Обрезать skip, если размеры отличаются
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h > 0 or diff_w > 0:
            x = nn.functional.pad(
                x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net для бинарной сегментации рельсов.

    Архитектура:
        Encoder: [64, 128, 256, 512]
        Bottleneck: 1024
        Decoder: [512, 256, 128, 64]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: list = None,
    ):
        """
        Args:
            in_channels: количество входных каналов (3 для RGB)
            out_channels: количество классов (2 = фон + рельсы)
            features: список размерностей [64, 128, 256, 512]
        """
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.num_classes = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = DownBlock(features[3], features[3] * 2)

        # Decoder
        self.up1 = UpBlock(features[3] * 2, features[3])
        self.up2 = UpBlock(features[3], features[2])
        self.up3 = UpBlock(features[2], features[1])
        self.up4 = UpBlock(features[1], features[0])

        # Output
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

        logger.info(
            f"U-Net initialized: in={in_channels}, out={out_channels}, "
            f"features={features}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: входной тензор [B, in_channels, H, W]

        Returns:
            Логиты [B, out_channels, H, W]
        """
        # Encoder
        x1 = self.inc(x)       # [B, 64, H, W]
        x2 = self.down1(x1)    # [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # [B, 512, H/8, W/8]

        # Bottleneck
        x5 = self.bottleneck(x4)  # [B, 1024, H/16, W/16]

        # Decoder
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]

        # Output
        logits = self.outc(x)  # [B, out_channels, H, W]
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание с softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def get_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Получить бинарную маску рельсов."""
        probs = self.predict(x)
        rail_probs = probs[:, 1, :, :]
        mask = rail_probs > threshold
        return mask
