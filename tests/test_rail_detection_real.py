"""
Тесты детекции рельсов на реальных фото.

Проверяем то, что не зависит от наличия скачанного датасета и весов:
геометрию, постобработку маски и корректность самой сети.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.geometric_rails import (detect_rails, ridge_response,  # noqa: E402
                                       rectify_strip, rails_to_mask,
                                       estimate_vanishing_point)
from detection.rail_postprocess import (mask_to_rails, clean_mask,  # noqa: E402
                                        draw_rails_overlay)


def synthetic_track(w: int = 640, h: int = 480, vp=(320, 150), gauge: int = 220,
                    sleeper_step: int = 18) -> np.ndarray:
    """Синтетический кадр: два рельса, сходящиеся в точке схода, и шпалы."""
    img = np.full((h, w, 3), 120, np.uint8)
    vx, vy = vp
    xl_bottom, xr_bottom = w // 2 - gauge // 2, w // 2 + gauge // 2

    def x_at(y, x_bottom):
        t = (y - vy) / (h - 1 - vy)
        return int(vx + t * (x_bottom - vx))

    # шпалы (тёмные поперечины между рельсами)
    y = h - 1
    step = sleeper_step
    while y > vy + 20:
        xl, xr = x_at(y, xl_bottom), x_at(y, xr_bottom)
        cv2.line(img, (xl, y), (xr, y), (70, 65, 60), max(1, int(6 * (y - vy) / h)))
        step = max(2, int(sleeper_step * (y - vy) / (h - vy)))
        y -= step
    # рельсы (светлые узкие линии)
    for x_bottom in (xl_bottom, xr_bottom):
        pts = [(x_at(y, x_bottom), y) for y in range(h - 1, int(vy) + 15, -5)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (215, 215, 215), 3)
    return img


def test_ridge_response_prefers_lines_over_edges():
    """Фильтр гребня должен реагировать на узкую линию, а не на ступеньку."""
    line = np.full((64, 64), 60, np.uint8)
    line[:, 30:33] = 200                      # узкая светлая линия
    step = np.full((64, 64), 60, np.uint8)
    step[:, 32:] = 200                        # ступенька (край)

    r_line = ridge_response(line).max()
    r_step = ridge_response(step).max()
    assert r_line > 0.5
    assert r_line > r_step * 1.5


def test_detect_rails_on_synthetic_track():
    img = synthetic_track()
    det = detect_rails(img, min_score=0.3)
    assert det.found, "рельсы на синтетическом пути не найдены"
    pair = det.pairs[0]
    assert pair.score > 0.3
    # найденная колея должна быть примерно там, где нарисована
    bottom_left, bottom_right = pair.left[0][0], pair.right[0][0]
    assert abs(bottom_left - 210) < 60
    assert abs(bottom_right - 430) < 60
    assert bottom_left < bottom_right


def test_detect_rails_no_false_alarm_on_noise():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    det = detect_rails(noise, min_score=0.45)
    assert not det.found or det.score < 0.6


def test_vanishing_point_of_converging_lines():
    h, w = 480, 640
    segs = np.array([[100, 470, 300, 200], [540, 470, 340, 200]], np.float32)
    vp = estimate_vanishing_point(segs, h, w)
    assert vp is not None
    assert abs(vp[0] - 320) < 60


def test_rectify_strip_shape():
    img = synthetic_track()
    ys = np.linspace(479, 200, 40, dtype=np.float32)
    xs_l = np.linspace(210, 300, 40, dtype=np.float32)
    xs_r = np.linspace(430, 340, 40, dtype=np.float32)
    strip = rectify_strip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), xs_l, xs_r, ys)
    assert strip.ndim == 2 and strip.shape[0] > 0 and strip.shape[1] > 0


def test_rails_to_mask_and_back():
    img = synthetic_track()
    det = detect_rails(img, min_score=0.3)
    assert det.found
    mask = rails_to_mask(det, img.shape)
    assert mask.dtype == np.uint8 and mask.max() == 255
    rails = mask_to_rails(mask.astype(np.float32) / 255.0, thr=0.5)
    assert len(rails) >= 1


def test_clean_mask_removes_specks():
    prob = np.zeros((200, 200), np.float32)
    prob[50:150, 98:102] = 0.9          # длинная линия
    prob[10:13, 10:13] = 0.9            # мелкий шум
    m = clean_mask(prob, thr=0.5)
    assert m[100, 99] == 1
    assert m[11, 11] == 0


def test_overlay_does_not_change_shape():
    img = np.zeros((120, 160, 3), np.uint8)
    prob = np.zeros((120, 160), np.float32)
    prob[60:70, 20:140] = 0.9
    vis = draw_rails_overlay(img, prob, mask_to_rails(prob))
    assert vis.shape == img.shape


def test_railnet_forward_shapes():
    torch = pytest.importorskip("torch")
    from detection.models.railnet import RailNet

    model = RailNet(pretrained=False).eval()
    x = torch.zeros(2, 3, 128, 128)
    cls_logit, seg_logit = model(x)
    assert cls_logit.shape == (2,)
    assert seg_logit.shape == (2, 1, 32, 32)       # 1/4 от входа

    p_cls, p_seg = model.predict(x, out_size=(128, 128))
    assert p_seg.shape == (2, 1, 128, 128)
    assert float(p_cls.min()) >= 0.0 and float(p_cls.max()) <= 1.0
