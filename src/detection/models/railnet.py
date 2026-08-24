"""
railnet.py — мультизадачная сеть детекции рельсов на реальных фото.

Одна сеть решает две задачи:
  * КЛАССИФИКАЦИЯ  — есть ли на кадре настоящие рельсы (обучается на
    human-verified метках Open Images: Railway/Tram/Locomotive/... против
    проверенных негативов: дорога, улица, забор, тропа и т.д.);
  * СЕГМЕНТАЦИЯ    — где именно проходят рельсы (обучается на выверенных
    масках; см. scripts/label_rails.py и self-training).

Энкодер: ResNet-18, предобученный на ImageNet (веса тянутся через pytorchcv,
т.к. это единственный доступный источник весов в offline-окружении).
Декодер: лёгкий FPN -> маска в 1/4 разрешения входа.

Сеть намеренно небольшая: обучение и инференс идут на CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resnet18_stages(pretrained: bool = True):
    """Вернуть стадии ResNet-18 как отдельные модули (C1..C5)."""
    try:
        from pytorchcv.model_provider import get_model
        net = get_model("resnet18", pretrained=pretrained)
        feats = net.features
        stem = feats.init_block                       # -> 64, 1/4
        stages = [feats.stage1, feats.stage2, feats.stage3, feats.stage4]
        return stem, stages, [64, 64, 128, 256, 512]
    except Exception as exc:                          # офлайн — учим с нуля
        print(f"[railnet] предобученные веса недоступны ({exc}); учим с нуля")
        import torchvision
        net = torchvision.models.resnet18(weights=None)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        stages = [net.layer1, net.layer2, net.layer3, net.layer4]
        return stem, stages, [64, 64, 128, 256, 512]


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class RailNet(nn.Module):
    """Мультизадачная модель: (логит класса, логит-карта маски)."""

    def __init__(self, pretrained: bool = True, decoder_ch: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.stem, stages, chs = _resnet18_stages(pretrained)
        self.stage1, self.stage2, self.stage3, self.stage4 = stages
        c2, c3, c4, c5 = chs[1], chs[2], chs[3], chs[4]

        # --- классификация ---
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(c5, 1))

        # --- сегментация (FPN-lite) ---
        self.lat5 = nn.Conv2d(c5, decoder_ch, 1)
        self.lat4 = nn.Conv2d(c4, decoder_ch, 1)
        self.lat3 = nn.Conv2d(c3, decoder_ch, 1)
        self.lat2 = nn.Conv2d(c2, decoder_ch, 1)
        self.smooth4 = ConvBNAct(decoder_ch, decoder_ch)
        self.smooth3 = ConvBNAct(decoder_ch, decoder_ch)
        self.smooth2 = ConvBNAct(decoder_ch, decoder_ch)
        self.seg_head = nn.Sequential(
            ConvBNAct(decoder_ch, decoder_ch),
            nn.Conv2d(decoder_ch, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)             # 1/4
        c2 = self.stage1(x)          # 1/4
        c3 = self.stage2(c2)         # 1/8
        c4 = self.stage3(c3)         # 1/16
        c5 = self.stage4(c4)         # 1/32

        cls_logit = self.cls_head(c5).squeeze(1)

        p5 = self.lat5(c5)
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        seg_logit = self.seg_head(p2)          # 1/4 от входа
        return cls_logit, seg_logit

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size=None):
        """Вероятность 'есть рельсы' и карта вероятностей маски."""
        cls_logit, seg_logit = self(x)
        if out_size is not None:
            seg_logit = F.interpolate(seg_logit, size=out_size,
                                      mode="bilinear", align_corners=False)
        return torch.sigmoid(cls_logit), torch.sigmoid(seg_logit)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()
