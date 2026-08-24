"""
rank_proposals.py — переранжировать предложения детектора с помощью
обученного классификатора.

Геометрический детектор охотно предлагает "колею" на игрушках, машинах и
людях. Классификатор RailNet (обучен на human-verified метках) отвечает,
есть ли на кадре настоящие рельсы — умножаем на геометрический score и
получаем очередь на разметку, где почти все кадры действительно железнодорожные.

    python scripts/rank_proposals.py --weights runs/railnet_cls.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from detect import RailPredictor  # noqa: E402

DATA = ROOT / "data" / "real"
PROPOSALS = DATA / "labels" / "proposals.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/railnet_cls.pt")
    ap.add_argument("--img-size", type=int, default=192,
                    help="для ранжирования хватает меньшего разрешения")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    props = json.loads(PROPOSALS.read_text())
    pred = RailPredictor(args.weights, img_size=args.img_size)
    todo = [k for k, v in props.items()
            if v.get("cands") and "p_cls" not in v]
    print(f"кадров к оценке: {len(todo)}")
    for i, image_id in enumerate(todo, 1):
        rec = props[image_id]
        img = cv2.imread(str(DATA / rec["path"]))
        if img is None:
            continue
        p, _ = pred(img)
        rec["p_cls"] = round(float(p), 4)
        if i % 100 == 0:
            PROPOSALS.write_text(json.dumps(props))
            print(f"  {i}/{len(todo)}", flush=True)
    PROPOSALS.write_text(json.dumps(props))
    scored = [(v["p_cls"], k) for k, v in props.items() if "p_cls" in v]
    scored.sort(reverse=True)
    print(f"готово. p_cls>0.9: {sum(1 for s, _ in scored if s > 0.9)}, "
          f">0.7: {sum(1 for s, _ in scored if s > 0.7)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
