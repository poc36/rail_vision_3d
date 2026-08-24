"""
label_rails.py — полуавтоматическая разметка рельсов (human-in-the-loop).

Зачем: у реальных фото из Open Images есть метки "есть ли ж/д", но НЕТ масок
рельсов. Чтобы обучить сегментацию, нужны маски. Полностью автоматические
маски (см. geometric_rails) шумные, поэтому разметка идёт в цикле:

    1) sheet   — отрисовать лист с фото и координатной сеткой (0..1);
                 человек (или зрячая модель) смотрит и называет ломаные,
                 по которым идут рельсы;
    2) set     — записать ломаные; они автоматически "примагничиваются"
                 к гребням яркости (головкам рельсов) -> точная маска;
    3) review  — лист с результатом для проверки; ошибки переразмечаются.

Хранение: data/real/labels/annotations.json  (нормированные координаты)
Маски:    data/real/masks/<image_id>.png     (255 = рельс)

Примеры:
    python scripts/label_rails.py sheet --start 0 --count 4
    python scripts/label_rails.py set --id 000abc... \
        --left "0.30,1.0 0.42,0.72 0.47,0.60" --right "0.62,1.0 0.52,0.72 0.50,0.60"
    python scripts/label_rails.py review --start 0 --count 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.geometric_rails import (ridge_response, _refine_curve,  # noqa: E402
                                       detect_rails)

DATA = ROOT / "data" / "real"
LABEL_DIR = DATA / "labels"
MASK_DIR = DATA / "masks"
ANN_PATH = LABEL_DIR / "annotations.json"


# ---------------------------------------------------------------------------
def load_ann() -> dict:
    if ANN_PATH.exists():
        return json.loads(ANN_PATH.read_text())
    return {}


def save_ann(ann: dict) -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    ANN_PATH.write_text(json.dumps(ann, indent=1))


def list_images(subset: str = "train", cls: str = "rails") -> list[Path]:
    return sorted((DATA / "images" / subset / cls).glob("*.jpg"))


def parse_points(text: str) -> np.ndarray:
    """'0.3,1.0 0.42,0.7' -> [[0.3,1.0],[0.42,0.7]] (нормированные координаты)."""
    pts = []
    for chunk in text.replace(";", " ").split():
        x, y = chunk.split(",")
        pts.append([float(x), float(y)])
    if len(pts) < 2:
        raise ValueError("нужно минимум 2 точки")
    return np.asarray(pts, np.float32)


# ---------------------------------------------------------------------------
def snap_polyline(img: np.ndarray, pts_norm: np.ndarray, n_samples: int = 48,
                  search_frac: float = 0.014) -> np.ndarray:
    """Уточнить ломаную по карте гребней: грубые точки -> точная линия рельса."""
    h, w = img.shape[:2]
    pts = pts_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    order = np.argsort(-pts[:, 1])                 # снизу вверх
    pts = pts[order]

    ys = np.linspace(pts[0, 1], pts[-1, 1], n_samples).astype(np.float32)
    ys = np.clip(ys, 0, h - 1)
    xs = np.interp(ys[::-1], pts[::-1, 1], pts[::-1, 0])[::-1].astype(np.float32)

    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ridge = ridge_response(gray)
    xs_ref = _refine_curve(ridge, xs, ys, search=max(3, int(w * search_frac)))
    return np.stack([xs_ref, ys], 1)


def rails_mask(shape, left: np.ndarray, right: np.ndarray,
               thickness_frac: float = 0.011) -> np.ndarray:
    """Маска двух рельсов: толщина линии убывает с удалением (перспектива)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for pts in (left, right):
        p = pts.astype(np.int32)
        n = max(1, len(p) - 1)
        for k in range(n):
            t = 1.0 - k / n
            thick = max(1, int(round(thickness_frac * w * (0.28 + 0.72 * t))))
            cv2.line(mask, tuple(p[k]), tuple(p[k + 1]), 255, thick, cv2.LINE_AA)
    return mask


def draw_rails(img: np.ndarray, tracks: list[dict], bed: bool = True) -> np.ndarray:
    vis = img.copy()
    w = vis.shape[1]
    for tr in tracks:
        left = np.asarray(tr["left"], np.float32)
        right = np.asarray(tr["right"], np.float32)
        if bed:
            ov = vis.copy()
            cv2.fillPoly(ov, [np.vstack([left, right[::-1]]).astype(np.int32)],
                         (70, 200, 70))
            vis = cv2.addWeighted(ov, 0.22, vis, 0.78, 0)
        for pts, color in ((left, (0, 230, 255)), (right, (0, 170, 255))):
            cv2.polylines(vis, [pts.astype(np.int32)], False, color,
                          max(2, int(w * 0.005)), cv2.LINE_AA)
    return vis


def grid_overlay(img: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Координатная сетка 0..1 — чтобы называть точки рельсов по картинке."""
    vis = img.copy()
    h, w = vis.shape[:2]
    for i in range(1, int(1 / step)):
        x = int(w * i * step)
        y = int(h * i * step)
        cv2.line(vis, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{i*step:.1f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{i*step:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def make_sheet(tiles: list[tuple[np.ndarray, str]], cols: int, tile_w: int,
               out: Path) -> None:
    imgs = []
    for img, label in tiles:
        th = int(tile_w * 0.75)
        t = cv2.resize(img, (tile_w, th))
        cv2.rectangle(t, (0, 0), (tile_w, 20), (0, 0, 0), -1)
        cv2.putText(t, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                       value=(30, 30, 30)))
    rows = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)
            if len(imgs[i:i + cols]) == cols]
    tail = imgs[len(rows) * cols:]
    if tail:
        pad = [np.zeros_like(tail[0])] * (cols - len(tail))
        rows.append(np.hstack(tail + pad))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
def cmd_sheet(args) -> int:
    files = list_images(args.subset, args.cls)[args.start:args.start + args.count]
    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        tiles.append((grid_overlay(img), f"#{args.start+i} {f.stem}"))
        print(f"#{args.start+i}\t{f.stem}")
    make_sheet(tiles, args.cols, args.tile, Path(args.out))
    print("лист:", args.out)
    return 0


def cmd_set(args) -> int:
    path = next((DATA / "images").rglob(f"{args.id}.jpg"), None)
    if path is None:
        print("нет такого файла:", args.id)
        return 1
    img = cv2.imread(str(path))
    ann = load_ann()

    tracks = []
    if not args.empty:
        left = snap_polyline(img, parse_points(args.left), search_frac=args.search)
        right = snap_polyline(img, parse_points(args.right), search_frac=args.search)
        tracks.append(dict(left=left.tolist(), right=right.tolist()))
        for extra in args.extra or []:
            l2, r2 = extra.split("|")
            tracks.append(dict(
                left=snap_polyline(img, parse_points(l2), search_frac=args.search).tolist(),
                right=snap_polyline(img, parse_points(r2), search_frac=args.search).tolist()))

    ann[args.id] = dict(path=str(path.relative_to(DATA)), tracks=tracks,
                        source="manual-snap")
    save_ann(ann)

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(img.shape[:2], np.uint8)
    for tr in tracks:
        mask = np.maximum(mask, rails_mask(img.shape,
                                           np.asarray(tr["left"], np.float32),
                                           np.asarray(tr["right"], np.float32)))
    cv2.imwrite(str(MASK_DIR / f"{args.id}.png"), mask)
    print(f"ok {args.id}: путей={len(tracks)}, пикселей маски={int((mask>0).sum())}")
    return 0


def cmd_review(args) -> int:
    ann = load_ann()
    ids = sorted(ann.keys())[args.start:args.start + args.count]
    tiles = []
    for i, image_id in enumerate(ids):
        rec = ann[image_id]
        img = cv2.imread(str(DATA / rec["path"]))
        if img is None:
            continue
        vis = draw_rails(img, rec["tracks"]) if rec["tracks"] else img
        if not rec["tracks"]:
            cv2.putText(vis, "EMPTY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)
        tiles.append((vis, f"#{args.start+i} {image_id[:10]} n={len(rec['tracks'])}"))
    make_sheet(tiles, args.cols, args.tile, Path(args.out))
    print("лист проверки:", args.out, "размечено всего:", len(ann))
    return 0


PROPOSALS = LABEL_DIR / "proposals.json"


def load_proposals() -> dict:
    return json.loads(PROPOSALS.read_text()) if PROPOSALS.exists() else {}


def cmd_propose(args) -> int:
    """Показать варианты колеи от геометрического детектора — для выбора глазами."""
    files = list_images(args.subset, args.cls)[args.start:args.start + args.count]
    props = load_proposals()
    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        det = detect_rails(img, max_pairs=args.cands, min_score=0.02)
        cands = []
        for p in det.pairs:
            cands.append(dict(left=p.left.tolist(), right=p.right.tolist(),
                              score=round(p.score, 3)))
        props[f.stem] = dict(path=str(f.relative_to(DATA)), cands=cands)
        tiles.append((grid_overlay(img), f"#{args.start+i} {f.stem} ORIG"))
        for k, c in enumerate(cands, 1):
            tiles.append((draw_rails(img, [c], bed=False),
                          f"#{args.start+i} cand{k} s={c['score']:.2f}"))
        for _ in range(args.cands - len(cands)):
            tiles.append((np.zeros_like(img), "-"))
        print(f"#{args.start+i}\t{f.stem}\tвариантов={len(cands)}")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS.write_text(json.dumps(props))
    make_sheet(tiles, args.cands + 1, args.tile, Path(args.out))
    print("лист вариантов:", args.out)
    return 0


def cmd_scan(args) -> int:
    """Прогнать геометрический детектор по всем кадрам и сохранить предложения."""
    files = list_images(args.subset, args.cls)
    props = load_proposals()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        if f.stem in props and not args.force:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        det = detect_rails(img, max_pairs=args.cands, min_score=0.02)
        props[f.stem] = dict(
            path=str(f.relative_to(DATA)),
            cands=[dict(left=p.left.tolist(), right=p.right.tolist(),
                        score=round(p.score, 3)) for p in det.pairs])
        if (i + 1) % 100 == 0:
            PROPOSALS.write_text(json.dumps(props))
            print(f"  {i+1}/{len(files)}", flush=True)
    PROPOSALS.write_text(json.dumps(props))
    have = sum(1 for v in props.values() if v["cands"])
    print(f"готово: {len(props)} кадров, с предложениями: {have}")
    return 0


def cmd_sheetp(args) -> int:
    """Лист лучших предложений (по убыванию score) для быстрого accept/reject."""
    props = load_proposals()
    ann = load_ann()
    def rank(v: dict) -> float:
        s = v["cands"][0]["score"]
        if args.rank == "combo" and "p_cls" in v:
            if v["p_cls"] < args.min_pcls:
                return -1.0
            return s * v["p_cls"]
        return s

    items = [(rank(v), k, v) for k, v in props.items()
             if v["cands"] and (args.include_labeled or k not in ann)
             and v["path"].split("/")[1] == args.subset]
    items = [it for it in items if it[0] > 0]
    items.sort(key=lambda t: -t[0])
    items = items[args.start:args.start + args.count]
    tiles = []
    for i, (score, image_id, rec) in enumerate(items):
        img = cv2.imread(str(DATA / rec["path"]))
        if img is None:
            continue
        vis = draw_rails(img, [rec["cands"][0]], bed=False)
        tiles.append((vis, f"#{args.start+i} {image_id[:9]} s={score:.2f}"))
        print(f"#{args.start+i}\t{image_id}\t{score:.3f}")
    make_sheet(tiles, args.cols, args.tile, Path(args.out))
    print("лист:", args.out)
    return 0


def cmd_accept(args) -> int:
    """Принять top-1 предложение для списка id (быстрый батч)."""
    props = load_proposals()
    ann = load_ann()
    n = 0
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for image_id in args.ids:
        rec = props.get(image_id)
        if not rec or not rec["cands"]:
            print("нет предложения:", image_id)
            continue
        c = rec["cands"][0]
        tracks = [dict(left=c["left"], right=c["right"])]
        ann[image_id] = dict(path=rec["path"], tracks=tracks, source="proposal-top1")
        img = cv2.imread(str(DATA / rec["path"]))
        mask = rails_mask(img.shape, np.asarray(c["left"], np.float32),
                          np.asarray(c["right"], np.float32))
        cv2.imwrite(str(MASK_DIR / f"{image_id}.png"), mask)
        n += 1
    save_ann(ann)
    print(f"принято: {n}, всего размечено: {len(ann)}")
    return 0


def cmd_reject(args) -> int:
    """Пометить кадры как 'рельсы не размечены' — они не пойдут в сегментацию."""
    ann = load_ann()
    props = load_proposals()
    for image_id in args.ids:
        rec = props.get(image_id)
        ann[image_id] = dict(path=rec["path"] if rec else "", tracks=[],
                             source="rejected", skip_seg=True)
        mp = MASK_DIR / f"{image_id}.png"
        if mp.exists():
            mp.unlink()
    save_ann(ann)
    print(f"отклонено: {len(args.ids)}, всего записей: {len(ann)}")
    return 0


def cmd_pick(args) -> int:
    """Принять один или несколько предложенных вариантов как разметку."""
    props = load_proposals()
    rec = props.get(args.id)
    if rec is None:
        print("нет предложений для", args.id)
        return 1
    ann = load_ann()
    tracks = []
    for k in args.cand:
        if 1 <= k <= len(rec["cands"]):
            c = rec["cands"][k - 1]
            tracks.append(dict(left=c["left"], right=c["right"]))
    ann[args.id] = dict(path=rec["path"], tracks=tracks, source="proposal-pick")
    save_ann(ann)
    img = cv2.imread(str(DATA / rec["path"]))
    mask = np.zeros(img.shape[:2], np.uint8)
    for tr in tracks:
        mask = np.maximum(mask, rails_mask(img.shape,
                                           np.asarray(tr["left"], np.float32),
                                           np.asarray(tr["right"], np.float32)))
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MASK_DIR / f"{args.id}.png"), mask)
    print(f"ok {args.id}: путей={len(tracks)}")
    return 0


def cmd_rebuild(args) -> int:
    """Пересобрать PNG-маски из annotations.json (маски не хранятся в git)."""
    ann = load_ann()
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for image_id, rec in ann.items():
        if not rec.get("tracks"):
            continue
        img_path = DATA / rec["path"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        mask = np.zeros(img.shape[:2], np.uint8)
        for tr in rec["tracks"]:
            mask = np.maximum(mask, rails_mask(img.shape,
                                               np.asarray(tr["left"], np.float32),
                                               np.asarray(tr["right"], np.float32)))
        cv2.imwrite(str(MASK_DIR / f"{image_id}.png"), mask)
        n += 1
    print(f"пересобрано масок: {n}")
    return 0


def cmd_stats(args) -> int:
    ann = load_ann()
    n_tracks = sum(len(v["tracks"]) for v in ann.values())
    n_empty = sum(1 for v in ann.values() if not v["tracks"])
    print(f"размечено кадров: {len(ann)}, путей: {n_tracks}, пустых: {n_empty}")
    print(f"масок на диске: {len(list(MASK_DIR.glob('*.png'))) if MASK_DIR.exists() else 0}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sheet", help="лист фото с координатной сеткой")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--cols", type=int, default=2)
    s.add_argument("--tile", type=int, default=520)
    s.add_argument("--subset", default="train")
    s.add_argument("--cls", default="rails")
    s.add_argument("--out", default="/tmp/qa/sheet.jpg")
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("set", help="записать разметку кадра")
    s.add_argument("--id", required=True)
    s.add_argument("--left", default="")
    s.add_argument("--right", default="")
    s.add_argument("--extra", action="append", help="доп. путь: 'left|right'")
    s.add_argument("--empty", action="store_true", help="рельсов нет / не видно")
    s.add_argument("--search", type=float, default=0.014, help="радиус примагничивания")
    s.set_defaults(func=cmd_set)

    s = sub.add_parser("review", help="лист проверки разметки")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=8)
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--tile", type=int, default=420)
    s.add_argument("--out", default="/tmp/qa/review.jpg")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("propose", help="лист вариантов колеи (выбор глазами)")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=2)
    s.add_argument("--cands", type=int, default=4)
    s.add_argument("--tile", type=int, default=340)
    s.add_argument("--subset", default="train")
    s.add_argument("--cls", default="rails")
    s.add_argument("--out", default="/tmp/qa/propose.jpg")
    s.set_defaults(func=cmd_propose)

    s = sub.add_parser("scan", help="прогнать детектор по всем кадрам")
    s.add_argument("--subset", default="train")
    s.add_argument("--cls", default="rails")
    s.add_argument("--cands", type=int, default=3)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("sheetp", help="лист лучших предложений для accept/reject")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=8)
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--tile", type=int, default=430)
    s.add_argument("--subset", default="train")
    s.add_argument("--include-labeled", action="store_true")
    s.add_argument("--rank", choices=["geom", "combo"], default="combo",
                   help="combo = score детектора * вероятность классификатора")
    s.add_argument("--min-pcls", type=float, default=0.6)
    s.add_argument("--out", default="/tmp/qa/sheetp.jpg")
    s.set_defaults(func=cmd_sheetp)

    s = sub.add_parser("accept", help="принять top-1 предложения (список id)")
    s.add_argument("ids", nargs="+")
    s.set_defaults(func=cmd_accept)

    s = sub.add_parser("reject", help="отклонить кадры (список id)")
    s.add_argument("ids", nargs="+")
    s.set_defaults(func=cmd_reject)

    s = sub.add_parser("pick", help="принять вариант(ы) как разметку")
    s.add_argument("--id", required=True)
    s.add_argument("--cand", type=int, action="append", required=True)
    s.set_defaults(func=cmd_pick)

    s = sub.add_parser("rebuild", help="пересобрать маски из annotations.json")
    s.set_defaults(func=cmd_rebuild)

    s = sub.add_parser("stats", help="статистика разметки")
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
