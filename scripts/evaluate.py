"""
evaluate.py — честная оценка качества на hold-out.

Hold-out — это кадры из validation/test-сплитов Open Images: другие
изображения, которых не было в обучении.

Считается:
  * классификация "есть настоящие рельсы": accuracy, ROC-AUC, precision/recall,
    матрица ошибок;
  * сегментация рельсов: IoU и Dice на выверенных вручную масках;
  * скорость инференса на CPU (кадров в секунду).

    python scripts/evaluate.py --weights runs/railnet.pt --report runs/report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from detect import RailPredictor  # noqa: E402


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys = y[order]
    pos, neg = ys.sum(), len(ys) - ys.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    tps = np.cumsum(ys) / pos
    fps = np.cumsum(1 - ys) / neg
    return float(np.trapezoid(tps, fps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/railnet.pt")
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--masks", default="data/real/masks")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--cls-thr", type=float, default=0.5)
    ap.add_argument("--seg-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol-frac", type=float, default=0.006,
                    help="допуск для мягких метрик, доля ширины кадра")
    ap.add_argument("--report", default="runs/report.md")
    ap.add_argument("--qa-out", default="runs/qa_holdout.jpg")
    args = ap.parse_args()

    data = Path(args.data)
    masks_dir = Path(args.masks)
    rows = [r for r in csv.DictReader((data / "manifest.csv").open())
            if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    pred = RailPredictor(args.weights)

    ys, ps = [], []
    inter = union = dice_num = dice_den = 0.0
    tol_inter = tol_union = 0.0
    n_seg = 0
    tiles = []
    t0 = time.time()
    for i, r in enumerate(rows):
        img = cv2.imread(str(data / r["path"]))
        if img is None:
            continue
        prob, mask = pred(img)
        ys.append(int(r["label"]))
        ps.append(prob)

        mp = masks_dir / f"{r['image_id']}.png"
        if mp.exists():
            gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if gt is not None and gt.shape[:2] == mask.shape[:2]:
                g = gt > 60
                p = mask >= args.seg_thr
                inter += float((g & p).sum())
                union += float((g | p).sum())
                dice_num += 2.0 * float((g & p).sum())
                dice_den += float(g.sum() + p.sum())
                # "мягкие" метрики: рельс — тонкая линия, промах в 2-3 пикселя
                # визуально незаметен, поэтому считаем и IoU с допуском
                k = max(3, int(round(args.tol_frac * mask.shape[1])) | 1)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                g_d = cv2.dilate(g.astype(np.uint8), kern) > 0
                p_d = cv2.dilate(p.astype(np.uint8), kern) > 0
                tol_inter += float((g & p_d).sum()) + float((p & g_d).sum())
                tol_union += float(g.sum()) + float(p.sum())
                n_seg += 1
        if len(tiles) < 12 and int(r["label"]) == 1:
            from detection.rail_postprocess import draw_rails_overlay, mask_to_rails
            vis = draw_rails_overlay(img, mask, mask_to_rails(mask, args.seg_thr),
                                     thr=args.seg_thr)
            cv2.putText(vis, f"p={prob:.2f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)
            tiles.append(cv2.resize(vis, (400, 300)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    dt = time.time() - t0

    y = np.asarray(ys, np.float32)
    p = np.asarray(ps, np.float32)
    pred_pos = p >= args.cls_thr
    tp = float(((y == 1) & pred_pos).sum())
    fp = float(((y == 0) & pred_pos).sum())
    fn = float(((y == 1) & ~pred_pos).sum())
    tn = float(((y == 0) & ~pred_pos).sum())
    metrics = dict(
        n=len(y), n_pos=int(y.sum()),
        accuracy=round(float(((p >= args.cls_thr) == (y > 0.5)).mean()), 4),
        auc=round(roc_auc(y, p), 4),
        precision=round(tp / max(1.0, tp + fp), 4),
        recall=round(tp / max(1.0, tp + fn), 4),
        confusion=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)),
        seg_images=n_seg,
        seg_iou=round(float(inter / union), 4) if union else None,
        seg_dice=round(float(dice_num / dice_den), 4) if dice_den else None,
        seg_dice_tol=round(float(tol_inter / tol_union), 4) if tol_union else None,
        fps_cpu=round(len(y) / dt, 2),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if tiles:
        cols = 4
        rows_img = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)
                    if len(tiles[i:i + cols]) == cols]
        if rows_img:
            Path(args.qa_out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.qa_out, np.vstack(rows_img))
            print("QA-лист:", args.qa_out)

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(
        "# Отчёт по качеству RailNet\n\n"
        f"Веса: `{args.weights}`, сплит: `{args.split}`, кадров: {metrics['n']} "
        f"(с рельсами: {metrics['n_pos']})\n\n"
        "## Классификация «есть настоящие рельсы»\n\n"
        f"| метрика | значение |\n|---|---|\n"
        f"| accuracy | {metrics['accuracy']} |\n"
        f"| ROC-AUC | {metrics['auc']} |\n"
        f"| precision | {metrics['precision']} |\n"
        f"| recall | {metrics['recall']} |\n\n"
        f"Матрица ошибок: {metrics['confusion']}\n\n"
        "## Сегментация рельсов\n\n"
        f"| метрика | значение |\n|---|---|\n"
        f"| кадров с эталонной маской | {metrics['seg_images']} |\n"
        f"| IoU | {metrics['seg_iou']} |\n"
        f"| Dice | {metrics['seg_dice']} |\n"
        f"| Dice с допуском ±{args.tol_frac:.3f}·W | {metrics['seg_dice_tol']} |\n\n"
        f"Скорость на CPU: **{metrics['fps_cpu']} кадр/с**\n",
        encoding="utf-8")
    print("отчёт:", rep)
    (rep.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
