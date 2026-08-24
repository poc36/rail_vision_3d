"""
qa_grid.py — собрать контактный лист (grid) с результатами детекции.

Нужен для визуального контроля качества: скрипт рисует найденные рельсы,
подписывает каждый кадр индексом и score, склеивает в один JPEG.
Используется в цикле "прогнал -> посмотрел -> поправил алгоритм".

    python scripts/qa_grid.py --glob 'data/real/images/train/rails/*.jpg' \
        --start 0 --count 12 --out /tmp/qa/grid.jpg
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection.geometric_rails import detect_rails, draw_detection  # noqa: E402


def tile(img: np.ndarray, label: str, size=(400, 300)) -> np.ndarray:
    t = cv2.resize(img, size)
    cv2.rectangle(t, (0, 0), (size[0] - 1, 22), (0, 0, 0), -1)
    cv2.putText(t, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--out", default="/tmp/qa/grid.jpg")
    ap.add_argument("--raw", action="store_true", help="без детекции, только фото")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))[args.start:args.start + args.count]
    if not files:
        print("нет файлов")
        return 1

    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        if args.raw:
            tiles.append(tile(img, f"#{args.start + i}"))
            continue
        det = detect_rails(img, min_score=args.min_score)
        vis = draw_detection(img, det)
        tiles.append(tile(vis, f"#{args.start + i}  score={det.score:.2f}"))
        print(f"#{args.start + i} {Path(f).name} score={det.score:.3f} "
              f"{det.pairs[0].parts if det.pairs else ''}")

    cols = args.cols
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles) - cols + 1, cols)]
    if not rows:
        rows = [np.hstack(tiles)]
    grid = np.vstack(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved", args.out, grid.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
