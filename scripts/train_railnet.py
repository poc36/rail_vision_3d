"""
train_railnet.py — обучение мультизадачной сети RailNet на РЕАЛЬНЫХ фото.

Две задачи одновременно:
  1) классификация "на кадре есть настоящие рельсы" — метки берутся из
     human-verified разметки Open Images (см. scripts/fetch_openimages.py);
  2) сегментация рельсов — обучается на:
       * выверенных масках (data/real/masks/<image_id>.png), и
       * НЕГАТИВНЫХ кадрах, где корректная маска — пустая (бесплатная
         разметка: учит сеть не рисовать рельсы на дорогах и заборах).

Запуск:
    python scripts/train_railnet.py --epochs 12 --img-size 320
    python scripts/train_railnet.py --epochs 20 --masks data/real/masks --resume runs/railnet.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.models.railnet import (RailNet, dice_loss,  # noqa: E402
                                      IMAGENET_MEAN, IMAGENET_STD)

cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
class RailDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: Path, masks_dirs: list[Path],
                 img_size: int, train: bool):
        self.rows = rows
        self.root = data_root
        self.masks_dirs = masks_dirs
        self.size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    # --- аугментации -------------------------------------------------------
    def _augment(self, img: np.ndarray, mask: np.ndarray | None):
        h, w = img.shape[:2]
        s = self.size

        if self.train:
            # случайный кроп со случайным масштабом
            scale = random.uniform(0.65, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
            if mask is not None:
                mask = mask[y0:y0 + ch, x0:x0 + cw]

        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.train:
            if random.random() < 0.5:                       # отражение
                img = img[:, ::-1].copy()
                if mask is not None:
                    mask = mask[:, ::-1].copy()
            if random.random() < 0.5:                       # небольшой поворот
                ang = random.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0)
                img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                if mask is not None:
                    mask = cv2.warpAffine(mask, M, (s, s), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img = img.astype(np.float32)
            img *= random.uniform(0.7, 1.35)                # яркость
            img = (img - img.mean()) * random.uniform(0.75, 1.3) + img.mean()  # контраст
            if random.random() < 0.25:                      # оттенок серого
                g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
            if random.random() < 0.2:                       # смаз
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            if random.random() < 0.2:                       # шум
                img = img + np.random.normal(0, random.uniform(3, 12), img.shape)
            img = np.clip(img, 0, 255)
        return img.astype(np.float32), mask

    def __getitem__(self, i: int):
        row = self.rows[i]
        img = cv2.imread(str(self.root / row["path"]), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)

        label = int(row["label"])
        mask = None
        has_mask = 0.0          # 0 нет маски, 1 выверенная маска, 2 пустая (негатив)
        if label == 0:
            # негатив: корректная маска рельсов — пустая
            mask = np.zeros(img.shape[:2], np.uint8)
            has_mask = 2.0
        else:
            for d in self.masks_dirs:
                mp = d / f"{row['image_id']}.png"
                if mp.exists():
                    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and mask.shape[:2] == img.shape[:2]:
                        has_mask = 1.0
                        break
                    mask = None

        img, mask = self._augment(img, mask)
        img = (img[:, :, ::-1] / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))

        qs = self.size // 4
        if mask is None:
            m = torch.zeros(1, qs, qs)
        else:
            mm = cv2.resize(mask, (qs, qs), interpolation=cv2.INTER_AREA)
            m = torch.from_numpy((mm > 60).astype(np.float32))[None]
        return x, torch.tensor(float(label)), m, torch.tensor(has_mask)


# ---------------------------------------------------------------------------
def load_rows(manifest: Path) -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader(manifest.open()))
    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "holdout"]
    return train, hold


def freeze_early_layers(model: nn.Module) -> int:
    """Заморозить stem и первую стадию энкодера.

    Это самые "тяжёлые" по вычислениям слои (высокое разрешение), а признаки
    там универсальные (края, текстуры) — дообучать их для рельсов не нужно.
    Даёт заметное ускорение обучения на CPU без потери качества.
    """
    n = 0
    for module in (model.stem, model.stage1):
        for p in module.parameters():
            p.requires_grad = False
            n += p.numel()
    return n


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    probs, labels = [], []
    inter = union = 0.0
    seg_n = 0
    with torch.no_grad():
        for x, y, m, hm in loader:
            x = x.to(device)
            cls_logit, seg_logit = model(x)
            probs.append(torch.sigmoid(cls_logit).cpu())
            labels.append(y)
            sel = hm > 0.5
            if sel.any() and m[sel].sum() > 0:
                p = (torch.sigmoid(seg_logit[sel]).cpu() > 0.5).float()
                t = m[sel]
                inter += float((p * t).sum())
                union += float(((p + t) > 0).float().sum())
                seg_n += int(sel.sum())
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    # ROC-AUC без sklearn
    order = np.argsort(-p)
    y_sorted = y[order]
    pos, neg = y.sum(), len(y) - y.sum()
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    auc = float(np.trapezoid(tps / max(pos, 1), fps / max(neg, 1))) if pos and neg else 0.5
    iou = float(inter / union) if union > 0 else 0.0
    return dict(acc=acc, auc=auc, iou=iou, seg_samples=seg_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--masks", default="data/real/masks")
    ap.add_argument("--out", default="runs/railnet.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--freeze-early", action="store_true",
                    help="заморозить stem+stage1 (быстрее на CPU)")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="считать метрики на первых N кадрах hold-out (0 = все)")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data_root = Path(args.data)
    masks_dirs = [Path(p) for p in args.masks.split(",") if p and Path(p).exists()]
    train_rows, hold_rows = load_rows(data_root / "manifest.csv")
    n_masks = sum(len(list(d.glob("*.png"))) for d in masks_dirs)
    print(f"train={len(train_rows)} holdout={len(hold_rows)} масок={n_masks} "
          f"({[d.name for d in masks_dirs]}) device={device}")

    train_ds = RailDataset(train_rows, data_root, masks_dirs, args.img_size, True)
    # в hold-out для метрик IoU используем только ВЫВЕРЕННЫЕ вручную маски;
    # при eval-limit оставляем все кадры с масками + часть остальных
    eval_rows = hold_rows
    if args.eval_limit and args.eval_limit < len(hold_rows):
        with_mask = [r for r in hold_rows
                     if (masks_dirs[0] / f"{r['image_id']}.png").exists()] if masks_dirs else []
        ids = {r["image_id"] for r in with_mask}
        rest = [r for r in hold_rows if r["image_id"] not in ids]
        random.Random(0).shuffle(rest)
        eval_rows = with_mask + rest[:max(0, args.eval_limit - len(with_mask))]
    hold_ds = RailDataset(eval_rows, data_root, masks_dirs[:1], args.img_size, False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    hold_dl = DataLoader(hold_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers)

    model = RailNet(pretrained=not args.no_pretrained).to(device)
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print("возобновление с", args.resume)
    if args.freeze_early:
        print(f"заморожено параметров: {freeze_early_layers(model)}")

    n_pos = sum(int(r["label"]) for r in train_rows)
    pos_weight = torch.tensor([(len(train_rows) - n_pos) / max(1, n_pos)], device=device)
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # маска рельсов — тонкая, положительных пикселей мало
    bce_seg_none = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device),
                                        reduction="none")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(train_dl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_cls = run_seg = 0.0
        for it, (x, y, m, hm) in enumerate(train_dl):
            x, y, m, hm = x.to(device), y.to(device), m.to(device), hm.to(device)
            cls_logit, seg_logit = model(x)
            loss = bce_cls(cls_logit, y)
            run_cls += float(loss.detach())

            sel = hm > 0.5
            if sel.any() and args.seg_weight > 0:
                sl, sm, kind = seg_logit[sel], m[sel], hm[sel]
                # На позитивном кадре размечен ОДИН путь, а в кадре могут быть
                # и другие рельсы. Поэтому фон на таких кадрах штрафуем слабее,
                # чтобы сеть не разучилась находить неразмеченные пути.
                # На негативных кадрах (kind=2) маска полностью верна -> вес 1.
                bg_w = torch.where(kind.view(-1, 1, 1, 1) > 1.5,
                                   torch.ones_like(sm), torch.full_like(sm, 0.3))
                wmap = torch.where(sm > 0.5, torch.ones_like(sm), bg_w)
                raw = bce_seg_none(sl, sm)
                seg_l = 0.5 * (raw * wmap).sum() / wmap.sum().clamp(min=1.0)
                pos_only = sm.sum(dim=(1, 2, 3)) > 0
                if pos_only.any():
                    seg_l = seg_l + 0.5 * dice_loss(sl[pos_only], sm[pos_only])
                loss = loss + args.seg_weight * seg_l
                run_seg += float(seg_l)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if it % 20 == 0:
                print(f"  эпоха {epoch+1} шаг {it}/{len(train_dl)} "
                      f"cls={run_cls/(it+1):.3f} seg={run_seg/(it+1):.3f}", flush=True)

        metrics = evaluate(model, hold_dl, device)
        metrics.update(epoch=epoch + 1, sec=round(time.time() - t0, 1),
                       cls_loss=round(run_cls / max(1, len(train_dl)), 4),
                       seg_loss=round(run_seg / max(1, len(train_dl)), 4))
        history.append(metrics)
        print(f"эпоха {epoch+1}/{args.epochs}: acc={metrics['acc']:.3f} "
              f"auc={metrics['auc']:.3f} iou={metrics['iou']:.3f} "
              f"({metrics['sec']}s)", flush=True)

        quality = metrics["auc"] + (metrics["iou"] if n_masks else 0.0)
        if quality > best:
            best = quality
            torch.save(dict(model=model.state_dict(), args=vars(args),
                            metrics=metrics, img_size=args.img_size), out_path)
            print("  сохранено:", out_path)

    (out_path.parent / "history.json").write_text(json.dumps(history, indent=2))
    print("лучшее качество:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
