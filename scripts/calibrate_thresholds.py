"""
calibrate_thresholds.py — подбор рабочих порогов детектора рельсов.

Два порога:
  * cls_thr — с какой вероятности считать, что рельсы в кадре есть.
    Подбирается по F1 на ПОЛОВИНЕ hold-out (чётные кадры), а итоговые
    метрики затем считаются на другой половине — иначе порог "подсматривает"
    в тестовые данные.
  * seg_thr — порог маски. Подбирается по Dice на выверенных масках
    ОБУЧАЮЩЕЙ выборки (их много, 150+), чтобы не тратить эталонные
    hold-out маски.

Результат сохраняется в runs/thresholds.json и используется detect.py.

    python scripts/calibrate_thresholds.py --weights runs/railnet_v2.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from detect import RailPredictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/railnet_v2.pt")
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--masks", default="data/real/masks")
    ap.add_argument("--out", default="runs/thresholds.json")
    ap.add_argument("--seg-images", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data = Path(args.data)
    masks_dir = Path(args.masks)
    rows = list(csv.DictReader((data / "manifest.csv").open()))
    pred = RailPredictor(args.weights)

    # --- порог классификатора: калибровочная половина hold-out ---
    hold = [r for r in rows if r["split"] == "holdout"]
    calib = hold[::2]
    ys, ps = [], []
    for i, r in enumerate(calib):
        img = cv2.imread(str(data / r["path"]))
        if img is None:
            continue
        p, _ = pred(img)
        ys.append(int(r["label"]))
        ps.append(p)
        if (i + 1) % 100 == 0:
            print(f"  cls {i+1}/{len(calib)}", flush=True)
    y = np.asarray(ys, np.float32)
    p = np.asarray(ps, np.float32)
    best_cls, best_f1 = 0.5, -1.0
    for thr in np.arange(0.20, 0.90, 0.02):
        pp = p >= thr
        tp = float(((y == 1) & pp).sum())
        fp = float(((y == 0) & pp).sum())
        fn = float(((y == 1) & ~pp).sum())
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_cls = f1, float(thr)
    print(f"cls_thr={best_cls:.2f} (F1={best_f1:.3f} на калибровочной половине)")

    # --- порог маски: выверенные маски обучающей выборки ---
    train_with_mask = [r for r in rows
                       if r["split"] == "train" and (masks_dir / f"{r['image_id']}.png").exists()]
    train_with_mask = train_with_mask[:args.seg_images]
    thrs = np.arange(0.30, 0.90, 0.05)
    num = np.zeros(len(thrs))
    den = np.zeros(len(thrs))
    for i, r in enumerate(train_with_mask):
        img = cv2.imread(str(data / r["path"]))
        gt = cv2.imread(str(masks_dir / f"{r['image_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        _, prob = pred(img)
        g = gt > 60
        for k, t in enumerate(thrs):
            pm = prob >= t
            num[k] += 2.0 * float((g & pm).sum())
            den[k] += float(g.sum() + pm.sum())
        if (i + 1) % 25 == 0:
            print(f"  seg {i+1}/{len(train_with_mask)}", flush=True)
    dice = num / np.maximum(den, 1.0)
    best_seg = float(thrs[int(np.argmax(dice))])
    print("Dice по порогам:", {round(float(t), 2): round(float(d), 3)
                               for t, d in zip(thrs, dice)})
    print(f"seg_thr={best_seg:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(weights=args.weights, cls_thr=best_cls,
                                   seg_thr=best_seg,
                                   calib_f1=round(float(best_f1), 4),
                                   seg_images=len(train_with_mask)), indent=2))
    print("сохранено:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
