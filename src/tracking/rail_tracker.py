"""
RailTracker — Трекинг рельсов между кадрами с фильтром Калмана.

Обеспечивает:
- Сглаживание положения рельсов между кадрами
- Устойчивость к пропускам детекции
- Экспоненциальное сглаживание
"""

import numpy as np
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrackedRail:
    """Отслеживаемый рельс."""

    rail_id: int
    side: str  # "left" or "right"

    # Состояние (сглаженные координаты)
    center_3d: np.ndarray  # (3,) — текущий центр
    direction_3d: np.ndarray  # (3,) — направление

    # Фильтр Калмана
    state: np.ndarray  # [x, y, z, vx, vy, vz]
    covariance: np.ndarray  # 6x6

    # Метаданные
    age: int = 0  # Сколько кадров отслеживается
    frames_without_detection: int = 0
    confidence: float = 1.0


class KalmanFilter3D:
    """Простой 3D Kalman Filter для трекинга положения."""

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.dt = 1.0  # Один кадр

        # Матрица перехода (постоянная скорость)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # Матрица наблюдений (измеряем только позицию)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Шум процесса
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.Q[3:, 3:] *= 10  # Скорость менее стабильна

        # Шум измерений
        self.R = np.eye(3, dtype=np.float64) * measurement_noise

    def predict(self, state: np.ndarray, covariance: np.ndarray):
        """Шаг предсказания."""
        state_pred = self.F @ state
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        return state_pred, cov_pred

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ):
        """Шаг обновления."""
        # Innovation
        y = measurement - self.H @ state

        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R

        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # Updated state
        state_new = state + K @ y
        cov_new = (np.eye(6) - K @ self.H) @ covariance

        return state_new, cov_new


class RailTracker:
    """Трекер рельсов по кадрам."""

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        max_age: int = 10,
        smoothing_alpha: float = 0.7,
    ):
        """
        Args:
            process_noise: шум процесса для Калмана
            measurement_noise: шум измерения
            max_age: максимум кадров без детекции до удаления
            smoothing_alpha: коэффициент экспоненциального сглаживания (0-1)
        """
        self.kf = KalmanFilter3D(process_noise, measurement_noise)
        self.max_age = max_age
        self.alpha = smoothing_alpha

        self._tracked_rails: Dict[int, TrackedRail] = {}
        self._next_id = 0

    def update(
        self,
        detected_centers: List[np.ndarray],
        detected_directions: List[np.ndarray],
        detected_sides: List[str],
    ) -> List[TrackedRail]:
        """
        Обновить трекер с новыми детекциями.

        Args:
            detected_centers: список (3,) — центры обнаруженных рельсов
            detected_directions: список (3,) — направления
            detected_sides: список "left"/"right"

        Returns:
            Список активных TrackedRail
        """
        # 1. Предсказание для всех существующих
        for rail in self._tracked_rails.values():
            rail.state, rail.covariance = self.kf.predict(rail.state, rail.covariance)
            rail.frames_without_detection += 1

        # 2. Сопоставление (по side и расстоянию)
        matched_ids = set()
        matched_detections = set()

        for i, (center, direction, side) in enumerate(
            zip(detected_centers, detected_directions, detected_sides)
        ):
            best_id = None
            best_dist = float("inf")

            for rail_id, rail in self._tracked_rails.items():
                if rail_id in matched_ids:
                    continue
                if rail.side != side:
                    continue

                predicted_center = rail.state[:3]
                dist = np.linalg.norm(center - predicted_center)

                if dist < best_dist:
                    best_dist = dist
                    best_id = rail_id

            if best_id is not None and best_dist < 5.0:  # Max 5m match distance
                # Обновить существующий
                rail = self._tracked_rails[best_id]
                rail.state, rail.covariance = self.kf.update(
                    rail.state, rail.covariance, center
                )
                rail.center_3d = self.alpha * center + (1 - self.alpha) * rail.center_3d
                rail.direction_3d = direction
                rail.frames_without_detection = 0
                rail.age += 1
                rail.confidence = min(1.0, rail.confidence + 0.1)

                matched_ids.add(best_id)
                matched_detections.add(i)
            else:
                # Новый рельс
                new_id = self._next_id
                self._next_id += 1

                state = np.zeros(6, dtype=np.float64)
                state[:3] = center
                covariance = np.eye(6, dtype=np.float64) * 1.0

                self._tracked_rails[new_id] = TrackedRail(
                    rail_id=new_id,
                    side=side,
                    center_3d=center.copy(),
                    direction_3d=direction.copy(),
                    state=state,
                    covariance=covariance,
                    age=1,
                    frames_without_detection=0,
                )

                matched_ids.add(new_id)
                matched_detections.add(i)

        # 3. Удалить старые треки
        to_remove = [
            rid
            for rid, rail in self._tracked_rails.items()
            if rail.frames_without_detection > self.max_age
        ]
        for rid in to_remove:
            del self._tracked_rails[rid]
            logger.debug(f"Removed rail {rid} (lost for {self.max_age} frames)")

        # 4. Снизить confidence для не-matched
        for rid, rail in self._tracked_rails.items():
            if rid not in matched_ids:
                rail.confidence *= 0.9

        return list(self._tracked_rails.values())

    def get_active_rails(self, min_confidence: float = 0.3) -> List[TrackedRail]:
        """Получить активные рельсы с порогом confidence."""
        return [
            r
            for r in self._tracked_rails.values()
            if r.confidence >= min_confidence
        ]

    def reset(self):
        """Сбросить все треки."""
        self._tracked_rails.clear()
        self._next_id = 0
