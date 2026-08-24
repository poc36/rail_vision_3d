"""
self_train.py — расширение обучающей выборки силами самой сети.

Ручная выверка масок дорогая, поэтому после первого обучения сеть сама
размечает остальные реальные фото, а мы оставляем только УВЕРЕННЫЕ и
ПРАВДОПОДОБНЫЕ предсказания:

  * классификатор говорит "рельсы есть" с вероятностью >= --min-prob;
  * маска не пустая, но и не залила пол-кадра (доля площади в разумных рамках);
  * в маске есть вытянутые линии (рельсы), а не круглые пятна;
  * средняя уверенность внутри маски высокая.

Полученные псевдо-маски кладутся отдельно (data/real/pseudo_masks), чтобы их
всегда можно было отличить от выверенных вручную.

    python scripts/self_train.py --weights runs/railnet.pt --limit 1200
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
from detection.rail_postprocess import mask_to_rails, clean_mask  # noqa: E402


def plausible(prob_map: np.ndarray, thr: float, min_area: float, max_area: float,
              min_rails: int, min_conf: float) -> tuple[bool, dict]:
    """Проверить, похоже ли предсказание на настоящие рельсы."""
    h, w = prob_map.shape
    m = clean_mask(prob_map, thr)
    area = float(m.sum()) / (h * w)
    if not (min_area <= area <= max_area):
        return False, dict(reason="площадь", area=round(area, 4))
    rails = mask_to_rails(prob_map, thr=thr)
    if len(rails) < min_rails:
        return False, dict(reason="мало линий", n=len(rails), area=round(area, 4))
    conf = float(prob_map[m > 0].mean()) if m.any() else 0.0
    if conf < min_conf:
        return False, dict(reason="низкая уверенность", conf=round(conf, 3))
    return True, dict(area=round(area, 4), n=len(rails), conf=round(conf, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/railnet.pt")
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--out", default="data/real/pseudo_masks")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-prob", type=float, default=0.85)
    ap.add_argument("--seg-thr", type=float, default=0.55)
    ap.add_argument("--min-area", type=float, default=0.004)
    ap.add_argument("--max-area", type=float, default=0.25)
    ap.add_argument("--min-rails", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curated = {p.stem for p in (data / "masks").glob("*.png")} \
        if (data / "masks").exists() else set()
    ann_path = data / "labels" / "annotations.json"
    ann = json.loads(ann_path.read_text()) if ann_path.exists() else {}
    rejected = {k for k, v in ann.items() if v.get("skip_seg")}

    rows = [r for r in csv.DictReader((data / "manifest.csv").open())
            if r["split"] == args.split and r["label"] == "1"
            and r["image_id"] not in curated and r["image_id"] not in rejected]
    if args.limit:
        rows = rows[:args.limit]
    print(f"кандидатов: {len(rows)} (выверенных вручную: {len(curated)})")

    pred = RailPredictor(args.weights)
    kept, stats = 0, {}
    for i, r in enumerate(rows, 1):
        img = cv2.imread(str(data / r["path"]))
        if img is None:
            continue
        p_cls, prob_map = pred(img)
        if p_cls < args.min_prob:
            stats["низкий p_cls"] = stats.get("низкий p_cls", 0) + 1
            continue
        ok, info = plausible(prob_map, args.seg_thr, args.min_area, args.max_area,
                             args.min_rails, args.min_conf)
        if not ok:
            stats[info["reason"]] = stats.get(info["reason"], 0) + 1
            continue
        mask = (prob_map >= args.seg_thr).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{r['image_id']}.png"), mask)
        kept += 1
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} принято={kept}", flush=True)
    print(f"псевдо-масок сохранено: {kept} из {len(rows)}")
    print("причины отказа:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
