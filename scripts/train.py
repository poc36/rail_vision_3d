"""
Train — Обучение модели сегментации рельсов.

Поддерживает:
- Обучение DeepLabV3+ и U-Net
- Аугментации (flips, color jitter, random crop)
- Метрики: IoU, Dice
- Чекпоинты
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================

class RailDataset(Dataset):
    """Датасет для тренировки сегментации рельсов."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_size: tuple = (512, 512),
        augment: bool = True,
    ):
        self.input_size = input_size
        self.augment = augment

        self.image_paths = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.mask_paths = sorted([
            os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) != masks ({len(self.mask_paths)})"

        logger.info(f"RailDataset: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, self.input_size)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Augmentations
        if self.augment:
            image, mask = self._augment(image, mask)

        # Normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - self.MEAN) / self.STD

        # Бинаризация маски
        mask = (mask > 127).astype(np.int64)

        # HWC → CHW
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()

        return image, mask

    def _augment(self, image, mask):
        """Аугментации."""
        # Горизонтальный flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Вертикальный flip (менее вероятно)
        if np.random.rand() > 0.9:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Яркость/контраст
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)

        # Гауссовский шум
        if np.random.rand() > 0.7:
            noise = np.random.randn(*image.shape).astype(np.float32) * 10
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image, mask


# ============================================================
# Metrics
# ============================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """Compute IoU per class."""
    ious = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        ious[f"iou_class_{cls}"] = iou
    ious["mean_iou"] = np.mean(list(ious.values()))
    return ious


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss для сегментации."""
    probs = torch.softmax(logits, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=logits.shape[1])
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

    intersection = (probs * target_one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


# ============================================================
# Training
# ============================================================

def train(args):
    """Основной цикл обучения."""
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Модель
    if args.model == "deeplabv3":
        from src.detection.models.deeplabv3 import DeepLabV3Plus
        model = DeepLabV3Plus(num_classes=args.num_classes, pretrained=True)
    else:
        from src.detection.models.unet import UNet
        model = UNet(in_channels=3, out_channels=args.num_classes)

    model = model.to(device)
    logger.info(f"Model: {args.model}")

    # Датасет
    train_dataset = RailDataset(
        args.train_images, args.train_masks,
        input_size=tuple(args.input_size),
        augment=True,
    )
    val_dataset = None
    if args.val_images and args.val_masks:
        val_dataset = RailDataset(
            args.val_images, args.val_masks,
            input_size=tuple(args.input_size),
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

    # Optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    ce_loss = nn.CrossEntropyLoss()

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # Validation
        if val_loader is not None:
            model.eval()
            all_ious = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    ious = compute_iou(preds, masks, args.num_classes)
                    all_ious.append(ious["mean_iou"])

            mean_iou = np.mean(all_ious)
            logger.info(
                f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_mIoU={mean_iou:.4f}"
            )

            # Save best
            if mean_iou > best_iou:
                best_iou = mean_iou
                save_path = args.save_path or "data/models/rail_segmentor.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best model saved: mIoU={best_iou:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

            # Сохранять каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                save_path = args.save_path or "data/models/rail_segmentor.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Rail Segmentation Model")
    parser.add_argument("--model", default="deeplabv3", choices=["deeplabv3", "unet"])
    parser.add_argument("--train-images", required=True, help="Path to training images")
    parser.add_argument("--train-masks", required=True, help="Path to training masks")
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-masks", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
