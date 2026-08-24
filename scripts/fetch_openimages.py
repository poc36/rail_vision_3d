"""
fetch_openimages.py — сбор датасета РЕАЛЬНЫХ фотографий рельсов из Open Images Dataset V7.

Почему Open Images:
  * это настоящие фотографии (Flickr, лицензия CC-BY), а не рендеры;
  * метки проверены людьми (human-verified image labels), т.е. у нас есть
    честная разметка "на фото есть железная дорога / нет";
  * зеркало картинок (CVDF S3) отдаёт JPEG напрямую, без API-ключей.

Что скачивается:
  ПОЗИТИВЫ  — Railway, Tram, Locomotive, Railroad car, High-speed rail,
              Monorail, Subway, Train  (Confidence=1)
  НЕГАТИВЫ  — снимки, где человек проверил ОТСУТСТВИЕ ж/д (Railway=0, Train=0),
              плюс "похожие" сцены с линейной перспективой:
              Road, Highway, Street, Path, Fence, Forest.

Разбиение: OID train -> наш train, OID validation+test -> наш holdout
(честный hold-out: другие изображения, другой сплит датасета).

Пример:
    python scripts/fetch_openimages.py --out data/real --pos 2200 --neg 2200
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("fetch")

META = "https://storage.googleapis.com/openimages/v7"
IMG_MIRROR = "https://open-images-dataset.s3.amazonaws.com"

# --- классы Open Images ------------------------------------------------------
RAIL_LABELS = {
    "/m/06d_3": "Railway",
    "/m/07kfm": "Tram",
    "/m/04h5c": "Locomotive",
    "/m/01g50p": "Railroad car",
    "/m/0db2f": "High-speed rail",
    "/m/056qz": "Monorail",
    "/m/0b_hy9": "Subway",
    "/m/07jdr": "Train",
}
# "сильные" ж/д метки: почти всегда в кадре видны сами рельсы
STRONG_RAIL = {"/m/06d_3", "/m/07kfm", "/m/04h5c", "/m/01g50p", "/m/0db2f", "/m/056qz"}
# Train — более шумная метка (бывают интерьеры вагонов, игрушки), берём как weak
WEAK_RAIL = {"/m/07jdr", "/m/0b_hy9"}

# сцены-негативы с похожей геометрией (уходящие в перспективу линии)
HARD_NEG_LABELS = {
    "/m/06gfj": "Road",
    "/m/0cz_0": "Highway",
    "/m/01c8br": "Street",
    "/m/02zh30": "Path",
    "/m/0blz9": "Fence",
    "/m/02zr8": "Forest",
    "/m/0hr8": "Asphalt",
    "/m/0dnhy": "Sidewalk",
    "/m/01jp76": "Parking",
    "/m/06cnp": "River",
}
ALL_LABELS = set(RAIL_LABELS) | set(HARD_NEG_LABELS)

ANNOTATION_FILES = {
    "train": f"{META}/oidv7-train-annotations-human-imagelabels.csv",
    "validation": f"{META}/oidv7-val-annotations-human-imagelabels.csv",
    "test": f"{META}/oidv7-test-annotations-human-imagelabels.csv",
}


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=64, pool_connections=64))
    s.headers["User-Agent"] = "rail-vision-3d/1.0 (dataset builder)"
    return s


# --- шаг 1: разметка ---------------------------------------------------------
def stream_annotations(subset: str, cache_dir: Path, session: requests.Session) -> Path:
    """Скачать CSV разметки потоком и оставить только интересные нам классы."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"labels_{subset}.csv"
    if out.exists() and out.stat().st_size > 0:
        logger.info("разметка %s: используем кэш %s", subset, out.name)
        return out

    url = ANNOTATION_FILES[subset]
    logger.info("разметка %s: стримим %s", subset, url.rsplit("/", 1)[-1])
    kept = 0
    tmp = out.with_suffix(".tmp")
    with session.get(url, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with tmp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ImageID", "LabelName", "Confidence"])
            first = True
            for line in r.iter_lines(chunk_size=1 << 20, decode_unicode=True):
                if first:                       # заголовок
                    first = False
                    continue
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                if parts[2] in ALL_LABELS:
                    w.writerow([parts[0], parts[2], parts[3]])
                    kept += 1
    tmp.rename(out)
    logger.info("разметка %s: сохранено %d строк", subset, kept)
    return out


def build_index(csv_path: Path) -> dict[str, dict[str, int]]:
    """image_id -> {label: confidence}"""
    idx: dict[str, dict[str, int]] = defaultdict(dict)
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            idx[row["ImageID"]][row["LabelName"]] = int(float(row["Confidence"]))
    return idx


def classify_image(labels: dict[str, int]) -> tuple[str | None, str]:
    """Вернуть ('rails'|'norails'|None, описание) по набору проверенных меток."""
    strong = [l for l in STRONG_RAIL if labels.get(l) == 1]
    weak = [l for l in WEAK_RAIL if labels.get(l) == 1]
    if strong:
        return "rails", "+".join(RAIL_LABELS[l] for l in strong + weak)
    if weak:
        return "rails_weak", "+".join(RAIL_LABELS[l] for l in weak)

    if any(labels.get(l) == 1 for l in RAIL_LABELS):
        return None, ""                      # неоднозначно — не берём

    # негатив: человек проверил, что ж/д тут нет
    rail_absent = any(labels.get(l) == 0 for l in ("/m/06d_3", "/m/07jdr"))
    scene = [HARD_NEG_LABELS[l] for l in HARD_NEG_LABELS if labels.get(l) == 1]
    if scene:
        # "трудные" негативы: дорога/улица/тропа/забор — тоже линии в перспективе
        return "norails_scene", "+".join(scene)
    if rail_absent:
        return "norails", "verified-no-railway"
    return None, ""


# --- шаг 2: загрузка изображений --------------------------------------------
def download_one(session: requests.Session, subset: str, image_id: str,
                 dest: Path, max_side: int) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    url = f"{IMG_MIRROR}/{subset}/{image_id}.jpg"
    try:
        r = session.get(url, timeout=(15, 90))
        if r.status_code != 200 or len(r.content) < 2048:
            return False
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        if min(h, w) < 200:                 # слишком мелкие — бесполезны
            return False
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать реальные фото рельсов (Open Images V7)")
    ap.add_argument("--out", default="data/real", help="каталог датасета")
    ap.add_argument("--pos", type=int, default=2200, help="сколько позитивов (train split)")
    ap.add_argument("--neg", type=int, default=2200, help="сколько негативов (train split)")
    ap.add_argument("--holdout", type=int, default=700, help="размер hold-out (val+test OID)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.out)
    session = make_session()
    meta_dir = root / "meta"

    # --- разметка ---
    plan: list[tuple[str, str, str, str, str]] = []   # (split, subset, image_id, cls, note)
    for subset in ("validation", "test", "train"):
        path = stream_annotations(subset, meta_dir, session)
        idx = build_index(path)
        split = "train" if subset == "train" else "holdout"
        buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for image_id, labels in idx.items():
            cls, note = classify_image(labels)
            if cls:
                buckets[cls].append((image_id, note))
        for k in buckets:
            random.shuffle(buckets[k])
        logger.info("%s: strong=%d weak=%d neg_scene=%d neg_generic=%d", subset,
                    len(buckets["rails"]), len(buckets["rails_weak"]),
                    len(buckets["norails_scene"]), len(buckets["norails"]))

        if split == "train":
            n_pos, n_neg = args.pos, args.neg
        else:
            n_pos, n_neg = args.holdout // 4, args.holdout // 4

        n_strong = min(len(buckets["rails"]), n_pos)
        n_weak = max(0, n_pos - n_strong)
        take = [("rails", x) for x in buckets["rails"][:n_strong]]
        take += [("rails", x) for x in buckets["rails_weak"][:n_weak]]
        # негативы: половина "трудных" (дорога/улица/забор), половина обычных
        n_scene = min(len(buckets["norails_scene"]), int(n_neg * 0.6))
        take += [("norails", x) for x in buckets["norails_scene"][:n_scene]]
        take += [("norails", x) for x in buckets["norails"][:max(0, n_neg - n_scene)]]

        for cls, (image_id, note) in take:
            plan.append((split, subset, image_id, cls, note))

    logger.info("к загрузке: %d изображений", len(plan))

    # --- загрузка ---
    manifest_rows = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for split, subset, image_id, cls, note in plan:
            dest = root / "images" / split / cls / f"{image_id}.jpg"
            futs[ex.submit(download_one, session, subset, image_id, dest, args.max_side)] = \
                (split, subset, image_id, cls, note, dest)
        for i, fut in enumerate(as_completed(futs), 1):
            split, subset, image_id, cls, note, dest = futs[fut]
            if fut.result():
                ok += 1
                manifest_rows.append(
                    dict(image_id=image_id, split=split, oid_subset=subset,
                         label=1 if cls == "rails" else 0, cls=cls,
                         oid_labels=note, path=str(dest.relative_to(root))))
            else:
                fail += 1
            if i % 250 == 0:
                logger.info("  %d/%d (ok=%d, fail=%d)", i, len(futs), ok, fail)

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "split", "oid_subset", "label",
                                           "cls", "oid_labels", "path"])
        w.writeheader()
        w.writerows(sorted(manifest_rows, key=lambda r: (r["split"], r["label"], r["image_id"])))

    n_pos = sum(r["label"] == 1 for r in manifest_rows)
    logger.info("ГОТОВО: %d фото (рельсы=%d, без рельсов=%d), ошибок=%d",
                len(manifest_rows), n_pos, len(manifest_rows) - n_pos, fail)
    logger.info("манифест: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
