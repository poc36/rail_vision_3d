"""
rail_postprocess.py — превращение предсказанной маски в линии рельсов.

Сеть даёт вероятностную карту "это рельс". Для показа человеку удобнее
не пятна, а именно ЛИНИИ рельсов, поэтому:
  1. порог + удаление мелких компонент;
  2. каждая вытянутая компонента -> полилиния (по центрам масс строк/столбцов);
  3. сглаживание и отбраковка коротких/толстых компонент (это не рельс).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RailPolyline:
    points: np.ndarray          # (N,2) float32
    length: float               # длина в пикселях
    mean_prob: float            # средняя уверенность сети вдоль линии


def clean_mask(prob: np.ndarray, thr: float = 0.5, min_area_frac: float = 0.0006,
               close_px: int = 5) -> np.ndarray:
    """Бинаризовать карту вероятностей и убрать шум."""
    h, w = prob.shape[:2]
    m = (prob >= thr).astype(np.uint8)
    if close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    min_area = max(30, int(h * w * min_area_frac))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def component_polyline(comp: np.ndarray, prob: np.ndarray) -> RailPolyline | None:
    """Полилиния компоненты: по строкам (если она вытянута вертикально)
    или по столбцам (если горизонтально)."""
    ys, xs = np.nonzero(comp)
    if len(xs) < 30:
        return None
    hspan = xs.max() - xs.min() + 1
    vspan = ys.max() - ys.min() + 1

    pts = []
    if vspan >= hspan:                       # вертикально вытянутая — идём по строкам
        for y in range(ys.min(), ys.max() + 1):
            row = np.nonzero(comp[y])[0]
            if len(row):
                pts.append((float(row.mean()), float(y)))
    else:                                    # горизонтальная — идём по столбцам
        for x in range(xs.min(), xs.max() + 1):
            col = np.nonzero(comp[:, x])[0]
            if len(col):
                pts.append((float(x), float(col.mean())))
    if len(pts) < 10:
        return None
    p = np.asarray(pts, np.float32)

    # сглаживание
    k = max(3, (len(p) // 12) | 1)
    for axis in (0, 1):
        p[:, axis] = np.convolve(np.pad(p[:, axis], k // 2, mode="edge"),
                                 np.ones(k) / k, mode="valid")[:len(p)]
    # прореживание до ~40 точек
    if len(p) > 40:
        idx = np.linspace(0, len(p) - 1, 40).astype(int)
        p = p[idx]

    length = float(np.hypot(*(p[-1] - p[0])))
    mean_prob = float(prob[comp > 0].mean())
    return RailPolyline(points=p, length=length, mean_prob=mean_prob)


def mask_to_rails(prob: np.ndarray, thr: float = 0.5,
                  min_len_frac: float = 0.12) -> list[RailPolyline]:
    """Извлечь полилинии рельсов из карты вероятностей."""
    h, w = prob.shape[:2]
    m = clean_mask(prob, thr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    rails = []
    min_len = min_len_frac * max(h, w)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        pl = component_polyline(comp, prob)
        if pl is not None and pl.length >= min_len:
            rails.append(pl)
    rails.sort(key=lambda r: -r.length)
    return rails


def refine_rails_to_ridges(image: np.ndarray, rails: list[RailPolyline],
                           search_frac: float = 0.008) -> list[RailPolyline]:
    """Подтянуть линии сети к головкам рельсов на исходном кадре.

    Сеть даёт линию с точностью в несколько пикселей; настоящий рельс — узкий
    гребень яркости. Локальный поиск максимума гребня делает линии точными,
    не меняя их формы.
    """
    if not rails:
        return rails
    from .geometric_rails import ridge_response, _refine_curve

    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    ridge = ridge_response(gray)
    search = max(2, int(image.shape[1] * search_frac))
    out = []
    for r in rails:
        p = r.points
        vertical = (p[:, 1].max() - p[:, 1].min()) >= (p[:, 0].max() - p[:, 0].min())
        if vertical:
            xs = _refine_curve(ridge, p[:, 0].copy(), p[:, 1].copy(), search=search)
            pts = np.stack([xs, p[:, 1]], 1)
        else:                       # горизонтальная линия — ищем по столбцам
            ys = _refine_curve(ridge.T, p[:, 1].copy(), p[:, 0].copy(), search=search)
            pts = np.stack([p[:, 0], ys], 1)
        out.append(RailPolyline(points=pts.astype(np.float32), length=r.length,
                                mean_prob=r.mean_prob))
    return out


def ridge_support(image: np.ndarray, rails: list[RailPolyline]) -> list[float]:
    """Средний отклик гребня яркости вдоль каждой линии (0..1)."""
    if not rails:
        return []
    from .geometric_rails import ridge_response

    h, w = image.shape[:2]
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    ridge = ridge_response(gray)
    out = []
    for r in rails:
        vals = []
        for x, y in r.points.astype(int):
            xi, yi = int(np.clip(x, 1, w - 2)), int(np.clip(y, 0, h - 1))
            vals.append(float(ridge[yi, xi - 1:xi + 2].max()))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return out


def filter_rails_by_ridge(image: np.ndarray, rails: list[RailPolyline],
                          min_ridge: float = 0.12) -> list[RailPolyline]:
    """Оставить только линии, лежащие на реальном гребне (головке рельса).

    Сеть иногда рисует "рельсы" на посторонних вытянутых объектах. Настоящий
    рельс — это узкая яркостная линия на изображении, и эта проверка,
    независимая от сети, убирает такие ложные срабатывания.
    """
    if not rails:
        return rails
    sup = ridge_support(image, rails)
    return [r for r, s in zip(rails, sup) if s >= min_ridge]


def draw_rails_overlay(image: np.ndarray, prob: np.ndarray,
                       rails: list[RailPolyline] | None = None,
                       thr: float = 0.5, alpha: float = 0.45,
                       color=(0, 210, 255)) -> np.ndarray:
    """Наложить маску рельсов и линии на кадр."""
    vis = image.copy()
    h, w = vis.shape[:2]
    m = (prob >= thr)
    if m.any():
        tint = np.zeros_like(vis)
        tint[m] = color
        vis = cv2.addWeighted(tint, alpha, vis, 1.0, 0)
    if rails:
        for r in rails:
            cv2.polylines(vis, [r.points.astype(np.int32)], False, (30, 30, 255),
                          max(2, int(w * 0.003)), cv2.LINE_AA)
    return vis
