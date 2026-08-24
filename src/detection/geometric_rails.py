"""
geometric_rails.py — детектор рельсов на реальных фото классическими методами CV.

Рельсы — это не просто "две линии". У настоящего пути есть жёсткая структура:

  1. Оба рельса сходятся в одной точке схода (перспектива).
  2. Колея постоянна -> после перспективной РЕКТИФИКАЦИИ (вид сверху) рельсы
     становятся двумя прямыми на фиксированных координатах.
  3. Между рельсами лежат ШПАЛЫ — квазипериодические поперечные полосы.
     В ректифицированной полосе их период постоянен, поэтому автокорреляция
     профиля поперечных градиентов даёт резкий пик. Это самый сильный признак,
     отличающий путь от дороги/забора/случайных линий.
  4. Головка рельса — узкий гребень (ridge) яркости вдоль всей линии.

Модуль работает без обучения и используется:
  * как самостоятельный детектор;
  * как генератор псевдо-разметки для обучения нейросети (self-training):
    в обучение идут только кадры с высоким score.

API:
    det = detect_rails(bgr_image)
    mask = rails_to_mask(det, image.shape)
    vis  = draw_detection(image, det)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
WORK_WIDTH = 640           # рабочее разрешение по ширине
MIN_LINE_FRAC = 0.09       # мин. длина отрезка Хафа (доля высоты кадра)
VP_TOP_FRAC = 0.02         # допустимая зона точки схода по вертикали
VP_BOTTOM_FRAC = 0.90
MAX_CANDIDATES = 16        # сколько линий-кандидатов берём в перебор пар
RECT_ROWS, RECT_COLS = 128, 64      # размер ректифицированной полосы


@dataclass
class RailPair:
    """Одна колея: два рельса, заданные полилиниями в координатах кадра."""

    left: np.ndarray                  # (N,2) снизу вверх
    right: np.ndarray                 # (N,2)
    score: float
    parts: dict = field(default_factory=dict)
    vanishing_point: tuple[float, float] | None = None
    rectified: np.ndarray | None = None    # полоса "вид сверху" (для отладки)

    @property
    def center(self) -> np.ndarray:
        return (self.left + self.right) / 2.0


@dataclass
class RailDetection:
    pairs: list[RailPair] = field(default_factory=list)
    vanishing_point: tuple[float, float] | None = None
    score: float = 0.0
    scale: float = 1.0

    @property
    def found(self) -> bool:
        return len(self.pairs) > 0


# ---------------------------------------------------------------------------
# признаки изображения
# ---------------------------------------------------------------------------
def ridge_response(gray: np.ndarray, widths=(2, 3, 5, 8, 12)) -> np.ndarray:
    """Отклик на УЗКУЮ линию (светлую или тёмную) поперёк оси X.

        r_w(x) = min( I(x) - I(x-w), I(x) - I(x+w) )     — светлый гребень
                 min( I(x-w) - I(x), I(x+w) - I(x) )     — тёмный гребень

    В отличие от лапласиана, min(...) даёт отклик ТОЛЬКО если яркость падает
    с обеих сторон, т.е. это действительно узкая линия (головка рельса), а не
    ступенька (край платформы, стена дома, граница асфальта). Это критично:
    именно ступеньки давали ложные срабатывания.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    best = np.zeros_like(g)
    for w in widths:
        left = np.roll(g, w, axis=1)
        right = np.roll(g, -w, axis=1)
        bright = np.minimum(g - left, g - right)
        dark = np.minimum(left - g, right - g)
        best = np.maximum(best, np.maximum(bright, dark))
    m = max(widths)
    best[:, :m] = 0
    best[:, -m:] = 0
    best = np.maximum(best, 0)
    hi = np.percentile(best, 99.0) + 1e-6
    return np.clip(best / hi, 0, 1)


def line_segments(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, _ = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(20, 0.66 * med)), int(min(255, 1.33 * med)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=int(h * MIN_LINE_FRAC),
                            maxLineGap=int(h * 0.035))
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4), np.float32), edges
    return np.asarray(lines, np.float32).reshape(-1, 4), edges


def estimate_vanishing_point(segs: np.ndarray, h: int, w: int) -> tuple[float, float] | None:
    """RANSAC по пересечениям длинных наклонных отрезков."""
    if len(segs) < 2:
        return None
    keep, lengths = [], []
    for s in segs:
        ang = abs(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])))
        ang = min(ang, 180 - ang)
        ln = float(np.hypot(s[2] - s[0], s[3] - s[1]))
        if 8 < ang < 89 and ln > h * MIN_LINE_FRAC:
            keep.append(s)
            lengths.append(ln)
    if len(keep) < 2:
        return None
    keep = np.asarray(keep, np.float32)
    lengths = np.asarray(lengths, np.float32)

    p1 = np.stack([keep[:, 0], keep[:, 1], np.ones(len(keep))], 1)
    p2 = np.stack([keep[:, 2], keep[:, 3], np.ones(len(keep))], 1)
    lines = np.cross(p1, p2)
    norms = np.hypot(lines[:, 0], lines[:, 1]) + 1e-9

    rng = np.random.default_rng(0)
    n = len(lines)
    pairs = rng.integers(0, n, size=(min(600, max(50, n * 4)), 2))
    best_vp, best_inl = None, -1.0
    tol = max(3.0, h * 0.012)
    for i, j in pairs:
        if i == j:
            continue
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-6:
            continue
        vx, vy = float(v[0] / v[2]), float(v[1] / v[2])
        if not (-2.5 * w < vx < 3.5 * w and h * VP_TOP_FRAC < vy < h * VP_BOTTOM_FRAC):
            continue
        d = np.abs(lines[:, 0] * vx + lines[:, 1] * vy + lines[:, 2]) / norms
        inl = float(lengths[d < tol].sum())
        if inl > best_inl:
            best_inl, best_vp = inl, (vx, vy)
    return best_vp


# ---------------------------------------------------------------------------
# ректификация колеи ("вид сверху") и признаки пути
# ---------------------------------------------------------------------------
def rectify_strip(img: np.ndarray, xs_l: np.ndarray, xs_r: np.ndarray,
                  ys: np.ndarray, margin: float = 0.35,
                  rows: int = RECT_ROWS, cols: int = RECT_COLS) -> np.ndarray:
    """Развернуть полосу пути в прямоугольник.

    Строки ys уже выбраны равномерно по 1/(y - vy), т.е. равномерно по
    расстоянию в мире -> шпалы становятся равноотстоящими полосами.
    Колонки: t от -margin до 1+margin, где t=0 — левый рельс, t=1 — правый.
    """
    k = np.linspace(0, len(ys) - 1, rows)
    xl = np.interp(k, np.arange(len(ys)), xs_l)
    xr = np.interp(k, np.arange(len(ys)), xs_r)
    yy = np.interp(k, np.arange(len(ys)), ys)

    t = np.linspace(-margin, 1.0 + margin, cols)[None, :]
    map_x = (xl[:, None] + (xr - xl)[:, None] * t).astype(np.float32)
    map_y = np.repeat(yy[:, None], cols, axis=1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _autocorr_peak(profile: np.ndarray, min_lag: int = 3, max_lag: int = 40) -> float:
    """Сила периодичности профиля (0..1) — ищем шпалы."""
    p = profile - cv2.blur(profile.reshape(-1, 1), (1, 15)).ravel()   # детренд
    if p.std() < 1e-6:
        return 0.0
    p = (p - p.mean()) / (p.std() + 1e-9)
    n = len(p)
    max_lag = min(max_lag, n // 3)
    if max_lag <= min_lag:
        return 0.0
    ac = np.array([float(np.dot(p[:-l], p[l:]) / (n - l)) for l in range(min_lag, max_lag)])
    return float(np.clip(ac.max(), 0, 1))


def track_features(img_gray: np.ndarray, ridge: np.ndarray, sat: np.ndarray,
                   xs_l: np.ndarray, xs_r: np.ndarray, ys: np.ndarray) -> dict:
    """Признаки гипотезы "между xs_l и xs_r — колея"."""
    rect_g = rectify_strip(img_gray, xs_l, xs_r, ys)
    rect_r = rectify_strip(ridge, xs_l, xs_r, ys)
    rect_s = rectify_strip(sat, xs_l, xs_r, ys)
    rows, cols = rect_g.shape
    margin = 0.35
    t_axis = np.linspace(-margin, 1 + margin, cols)
    inner = (t_axis > 0.12) & (t_axis < 0.88)
    if inner.sum() < 4:
        return {}

    # --- 1. шпалы: поперечные градиенты внутри колеи, периодичные по глубине ---
    gk = np.abs(cv2.Sobel(rect_g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    gu = np.abs(cv2.Sobel(rect_g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    prof = gk[:, inner].mean(axis=1)
    sleeper_periodicity = _autocorr_peak(prof)
    transverse_ratio = float(np.clip(
        (gk[:, inner].mean() / (gu[:, inner].mean() + 1e-6) - 0.9) / 0.8, 0, 1))

    # --- 2. рельсы: пики ridge-отклика ровно на t=0 и t=1 ---
    col_prof = rect_r.mean(axis=0)
    col_prof = col_prof / (col_prof.mean() + 1e-6)
    def peak_at(t0: float, tol: float = 0.09) -> float:
        sel = np.abs(t_axis - t0) < tol
        return float(col_prof[sel].max()) if sel.any() else 0.0
    left_peak, right_peak = peak_at(0.0), peak_at(1.0)
    rail_peaks = float(np.clip((min(left_peak, right_peak) - 1.0) / 1.1, 0, 1))

    # --- 3. непрерывность рельса: в скольких строках рядом с t=0/1 есть отклик ---
    def continuity(t0: float, tol: float = 0.09) -> float:
        sel = np.abs(t_axis - t0) < tol
        if not sel.any():
            return 0.0
        band = rect_r[:, sel].max(axis=1)
        return float((band > 0.14).mean())
    cont = min(continuity(0.0), continuity(1.0))

    # --- 4. балласт между рельсами: слабо насыщенный, но текстурный ---
    grayness = float(np.clip(1.0 - rect_s[:, inner].mean() / 95.0, 0, 1))
    texture = float(np.clip(gk[:, inner].mean() / 16.0, 0, 1))

    # --- 5. симметрия рельсов ---
    symmetry = 1.0 - abs(left_peak - right_peak) / (left_peak + right_peak + 1e-6)

    return dict(sleepers=round(sleeper_periodicity, 3),
                transverse=round(transverse_ratio, 3),
                rail_peaks=round(rail_peaks, 3),
                continuity=round(cont, 3),
                grayness=round(grayness, 3),
                texture=round(texture, 3),
                symmetry=round(float(symmetry), 3),
                _rect=rect_g)


def score_from_features(f: dict, width_frac: float) -> float:
    """Свести признаки в один score [0..1]."""
    if not f:
        return 0.0
    if not (0.05 < width_frac < 0.9):
        return 0.0
    # Ключевые признаки: рельсы видны (continuity, rail_peaks) И между ними шпалы.
    rail_term = 0.5 * f["continuity"] + 0.5 * f["rail_peaks"]
    sleeper_term = 0.65 * f["sleepers"] + 0.35 * f["transverse"]
    context = 0.5 * f["grayness"] + 0.5 * f["texture"]
    score = (0.44 * rail_term + 0.34 * sleeper_term +
             0.12 * context + 0.10 * f["symmetry"])
    # штраф, если один из рельсов почти не виден
    if f["continuity"] < 0.25 or f["rail_peaks"] < 0.05:
        score *= 0.55
    return float(np.clip(score, 0, 1))


# ---------------------------------------------------------------------------
# геометрия гипотез
# ---------------------------------------------------------------------------
def _line_x_at_y(seg: np.ndarray, y: float) -> float | None:
    x1, y1, x2, y2 = seg
    if abs(y2 - y1) < 1e-6:
        return None
    return float(x1 + (y - y1) / (y2 - y1) * (x2 - x1))


def ridge_segments(ridge: np.ndarray, h: int) -> np.ndarray:
    """Отрезки, найденные прямо по карте гребней — это кандидаты в рельсы."""
    binm = (ridge > 0.22).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(binm, 1, np.pi / 180, threshold=40,
                            minLineLength=int(h * MIN_LINE_FRAC),
                            maxLineGap=int(h * 0.05))
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4), np.float32)
    return np.asarray(lines, np.float32).reshape(-1, 4)


def _mean_along(field: np.ndarray, seg: np.ndarray, n: int = 40) -> float:
    """Среднее значение карты вдоль отрезка (с допуском ±1 px)."""
    h, w = field.shape
    xs = np.linspace(seg[0], seg[2], n)
    ys = np.linspace(seg[1], seg[3], n)
    vals = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 1 <= xi < w - 1:
            vals.append(field[yi, xi - 1:xi + 2].max())
    return float(np.mean(vals)) if vals else 0.0


def _candidate_lines(segs: np.ndarray, vp: tuple[float, float],
                     h: int, w: int, ridge: np.ndarray) -> list[np.ndarray]:
    """Отрезки, продолжение которых проходит через точку схода И которые
    лежат на гребне яркости (то есть похожи на головку рельса)."""
    vx, vy = vp
    scored = []
    tol = max(4.0, h * 0.022)
    for s in segs:
        x1, y1, x2, y2 = s
        ln = float(np.hypot(x2 - x1, y2 - y1))
        if ln < h * MIN_LINE_FRAC:
            continue
        dist = abs((y2 - y1) * vx - (x2 - x1) * vy + x2 * y1 - y2 * x1) / (ln + 1e-9)
        if dist > tol:
            continue
        if max(y1, y2) < vy + h * 0.04:          # целиком выше горизонта
            continue
        rsup = _mean_along(ridge, s)
        if rsup < 0.06:                          # на линии нет гребня — не рельс
            continue
        scored.append((ln * (0.4 + rsup), s))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored[:MAX_CANDIDATES]]


def _depth_uniform_rows(vp_y: float, y_bottom: float, y_top: float, n: int) -> np.ndarray:
    """Строки, равномерные по расстоянию в мире (равномерные по 1/(y-vy))."""
    eps = 1e-3
    u0 = 1.0 / max(eps, y_bottom - vp_y)
    u1 = 1.0 / max(eps, y_top - vp_y)
    u = np.linspace(u0, u1, n)
    return (vp_y + 1.0 / np.maximum(u, eps)).astype(np.float32)


def _profile_through_vp(vp, x_anchor: float, y_anchor: float, ys: np.ndarray) -> np.ndarray:
    vx, vy = vp
    denom = y_anchor - vy
    if abs(denom) < 1e-6:
        return np.full_like(ys, x_anchor, dtype=np.float32)
    t = (ys - vy) / denom
    return (vx + t * (x_anchor - vx)).astype(np.float32)


def _refine_curve(ridge: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  search: int = 7) -> np.ndarray:
    """Подтянуть полилинию к локальным максимумам ridge (ловит кривые пути)."""
    h, w = ridge.shape
    out = xs.astype(np.float32).copy()
    shift = 0.0
    for i in range(len(ys)):
        yi = int(round(float(ys[i])))
        if not (0 <= yi < h):
            continue
        base = out[i] + shift * 0.6
        a, b = max(0, int(base - search)), min(w, int(base + search + 1))
        if b - a < 3:
            continue
        strip = ridge[yi, a:b]
        k = int(np.argmax(strip))
        if strip[k] < 0.12:
            shift *= 0.5
            continue
        new_x = float(a + k)
        shift = new_x - out[i]
        out[i] = 0.55 * out[i] + 0.45 * new_x
    if len(out) >= 7:
        out = np.convolve(np.pad(out, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
def detect_rails(image: np.ndarray, max_pairs: int = 2, min_score: float = 0.45,
                 refine: bool = True) -> RailDetection:
    """Найти рельсы на кадре BGR."""
    if image is None or image.size == 0:
        return RailDetection()

    H, W = image.shape[:2]
    scale = WORK_WIDTH / W if W > WORK_WIDTH else 1.0
    img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else image.copy()
    h, w = img.shape[:2]

    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    ridge = ridge_response(gray)
    segs_edge, _ = line_segments(gray)
    segs_ridge = ridge_segments(ridge, h)
    segs = np.vstack([segs_edge, segs_ridge]) if len(segs_ridge) else segs_edge

    vp = estimate_vanishing_point(segs, h, w)
    det = RailDetection(scale=scale)
    if vp is None:
        return det
    det.vanishing_point = (vp[0] / scale, vp[1] / scale)

    cands = _candidate_lines(segs, vp, h, w, ridge)
    if len(cands) < 2:
        return det

    # якорная строка: чуть ниже самой нижней точки самого длинного кандидата
    y_anchor = float(min(h - 1, max(np.max(s[[1, 3]]) for s in cands)))
    y_anchor = max(y_anchor, vp[1] + h * 0.12)
    y_top = float(min(h - 5, max(0.0, vp[1]) + h * 0.05))
    if y_anchor - y_top < h * 0.12:
        return det
    ys = _depth_uniform_rows(vp[1], y_anchor, y_top, 64)

    anchors = []
    for s in cands:
        x = _line_x_at_y(s, y_anchor)
        if x is not None and -w * 0.4 < x < w * 1.4:
            anchors.append(x)
    anchors = sorted(set(round(a, 1) for a in anchors))
    if len(anchors) < 2:
        return det

    hyps = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            gauge = anchors[j] - anchors[i]
            if not (w * 0.05 < gauge < w * 0.9):
                continue
            xs_l = _profile_through_vp(vp, anchors[i], y_anchor, ys)
            xs_r = _profile_through_vp(vp, anchors[j], y_anchor, ys)
            f = track_features(gray, ridge, sat, xs_l, xs_r, ys)
            sc = score_from_features(f, gauge / w)
            if sc > min_score * 0.6:
                hyps.append((sc, f, xs_l, xs_r))
    if not hyps:
        return det
    hyps.sort(key=lambda t: -t[0])

    used: list[tuple[float, float]] = []
    for sc, f, xs_l, xs_r in hyps[:8]:
        if len(det.pairs) >= max_pairs:
            break
        cx = float((xs_l[0] + xs_r[0]) / 2)
        gauge = float(abs(xs_r[0] - xs_l[0]))
        if any(abs(cx - c) < 0.6 * max(g, gauge) for c, g in used):
            continue
        if refine:
            xs_l = _refine_curve(ridge, xs_l, ys)
            xs_r = _refine_curve(ridge, xs_r, ys)
            f = track_features(gray, ridge, sat, xs_l, xs_r, ys)
            sc = score_from_features(f, abs(xs_r[0] - xs_l[0]) / w)
        if sc < min_score:
            continue
        inv = 1.0 / scale
        rect = f.pop("_rect", None)
        det.pairs.append(RailPair(
            left=np.stack([xs_l * inv, ys * inv], 1),
            right=np.stack([xs_r * inv, ys * inv], 1),
            score=float(sc), parts={k: v for k, v in f.items() if not k.startswith("_")},
            vanishing_point=det.vanishing_point, rectified=rect))
        used.append((cx, gauge))

    det.pairs.sort(key=lambda p: -p.score)
    det.score = det.pairs[0].score if det.pairs else 0.0
    return det


# ---------------------------------------------------------------------------
def rails_to_mask(det: RailDetection, shape, thickness_frac: float = 0.010,
                  include_bed: bool = False) -> np.ndarray:
    """Растеризовать рельсы в бинарную маску (0/255)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for pair in det.pairs:
        if include_bed:
            poly = np.vstack([pair.left, pair.right[::-1]]).astype(np.int32)
            cv2.fillPoly(mask, [poly], 128)
        for pts in (pair.left, pair.right):
            pts_i = pts.astype(np.int32)
            n = max(1, len(pts_i) - 1)
            for k in range(n):
                t = 1.0 - k / n                      # ближний рельс толще
                thick = max(1, int(round(thickness_frac * w * (0.3 + 0.7 * t))))
                cv2.line(mask, tuple(pts_i[k]), tuple(pts_i[k + 1]), 255, thick)
    return mask


def draw_detection(image: np.ndarray, det: RailDetection, draw_bed: bool = True,
                   draw_vp: bool = False) -> np.ndarray:
    """Отрисовать найденные рельсы поверх кадра."""
    vis = image.copy()
    h, w = vis.shape[:2]
    for pair in det.pairs:
        if draw_bed:
            overlay = vis.copy()
            poly = np.vstack([pair.left, pair.right[::-1]]).astype(np.int32)
            cv2.fillPoly(overlay, [poly], (70, 200, 70))
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        for pts, color in ((pair.left, (0, 230, 255)), (pair.right, (0, 170, 255))):
            cv2.polylines(vis, [pts.astype(np.int32)], False, color,
                          max(2, int(w * 0.004)), cv2.LINE_AA)
        x, y = pair.left[0]
        cv2.putText(vis, f"rail {pair.score:.2f}", (int(x), int(min(h - 8, y - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1400.0),
                    (255, 255, 255), 2, cv2.LINE_AA)
    if draw_vp and det.vanishing_point:
        vx, vy = det.vanishing_point
        if 0 <= vx < w and 0 <= vy < h:
            cv2.drawMarker(vis, (int(vx), int(vy)), (255, 80, 80),
                           cv2.MARKER_CROSS, max(12, int(w * 0.02)), 2)
    return vis
