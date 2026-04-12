"""
drift_detector.py - PINN 정확도 모니터링
==========================================
PINN/FNO 예측 정확도를 추적하고,
정확도 저하(드리프트) 발생 시 재학습을 트리거합니다.

드리프트 감지 기준:
  1. PINN 예측 오차 > 5% → 재학습 트리거
  2. 연속 3회 오차 상승 → 경고
  3. FNO가 PINN과 5% 이상 괴리 → FNO 재증류
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """드리프트 경고."""
    level: str          # "warning", "critical"
    message: str
    metric: str         # 어떤 메트릭이 드리프트했는지
    current_value: float
    threshold: float


class DriftDetector:
    """
    PINN/FNO 정확도 드리프트 감지기.

    사용법:
        detector = DriftDetector()
        detector.record_error(0.03)  # PINN-LT 오차 기록
        alerts = detector.check()     # 드리프트 확인
    """

    ERROR_THRESHOLD = 0.05         # 5% 오차 임계값
    FNO_DIVERGENCE_THRESHOLD = 0.05  # FNO-PINN 괴리 임계값
    CONSECUTIVE_RISES = 3          # 연속 상승 경고 횟수
    HISTORY_SIZE = 50              # 오차 이력 크기

    def __init__(self):
        self._pinn_errors: deque = deque(maxlen=self.HISTORY_SIZE)
        self._fno_errors: deque = deque(maxlen=self.HISTORY_SIZE)
        self._alerts: list[DriftAlert] = []

    def record_pinn_error(self, error: float):
        """PINN vs LT 상대 오차 기록."""
        self._pinn_errors.append(error)

    def record_fno_error(self, error: float):
        """FNO vs PINN 상대 오차 기록."""
        self._fno_errors.append(error)

    def check(self) -> list[DriftAlert]:
        """드리프트 검사 실행."""
        self._alerts.clear()

        self._check_pinn_drift()
        self._check_fno_drift()
        self._check_trend()

        return self._alerts

    def _check_pinn_drift(self):
        """PINN 오차 임계값 초과 검사."""
        if not self._pinn_errors:
            return

        latest = self._pinn_errors[-1]
        if latest > self.ERROR_THRESHOLD:
            self._alerts.append(DriftAlert(
                level="critical",
                message=f"PINN error {latest:.4f} exceeds threshold {self.ERROR_THRESHOLD}. "
                        f"Retraining recommended.",
                metric="pinn_error",
                current_value=latest,
                threshold=self.ERROR_THRESHOLD,
            ))

    def _check_fno_drift(self):
        """FNO-PINN 괴리 검사."""
        if not self._fno_errors:
            return

        latest = self._fno_errors[-1]
        if latest > self.FNO_DIVERGENCE_THRESHOLD:
            self._alerts.append(DriftAlert(
                level="warning",
                message=f"FNO-PINN divergence {latest:.4f} exceeds {self.FNO_DIVERGENCE_THRESHOLD}. "
                        f"FNO re-distillation recommended.",
                metric="fno_divergence",
                current_value=latest,
                threshold=self.FNO_DIVERGENCE_THRESHOLD,
            ))

    def _check_trend(self):
        """연속 오차 상승 추세 검사."""
        if len(self._pinn_errors) < self.CONSECUTIVE_RISES:
            return

        recent = list(self._pinn_errors)[-self.CONSECUTIVE_RISES:]
        if all(recent[i] < recent[i+1] for i in range(len(recent) - 1)):
            self._alerts.append(DriftAlert(
                level="warning",
                message=f"PINN error rising for {self.CONSECUTIVE_RISES} consecutive checks: "
                        f"{[f'{e:.4f}' for e in recent]}",
                metric="pinn_trend",
                current_value=recent[-1],
                threshold=self.ERROR_THRESHOLD,
            ))

    @property
    def needs_retrain(self) -> bool:
        """재학습 필요 여부."""
        alerts = self.check()
        return any(a.level == "critical" for a in alerts)

    @property
    def needs_fno_redistill(self) -> bool:
        """FNO 재증류 필요 여부."""
        alerts = self.check()
        return any(a.metric == "fno_divergence" for a in alerts)

    def summary(self) -> dict:
        """현재 상태 요약."""
        return {
            "n_pinn_records": len(self._pinn_errors),
            "n_fno_records": len(self._fno_errors),
            "latest_pinn_error": self._pinn_errors[-1] if self._pinn_errors else None,
            "latest_fno_error": self._fno_errors[-1] if self._fno_errors else None,
            "mean_pinn_error": float(np.mean(list(self._pinn_errors))) if self._pinn_errors else None,
            "needs_retrain": self.needs_retrain,
            "needs_fno_redistill": self.needs_fno_redistill,
            "active_alerts": len(self.check()),
        }
