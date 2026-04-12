"""
active_learning.py - 능동 학습 루프
======================================
PINN ↔ LightTools 검증 루프를 자동화합니다.

능동 학습 원리:
  1. PINN/FNO로 예측
  2. MC Dropout으로 불확실도(σ) 추정
  3. σ 높은 영역 → LightTools 시뮬레이션으로 검증
  4. 검증 데이터 → PINN 재학습 데이터로 편입
  5. 반복 → 불확실한 영역이 점점 줄어듦

AGENTS.md 규칙:
  - LT 검증 결과 → 자동으로 재학습 데이터 편입
  - PINN 예측 오차 > 5% → 재학습 트리거
  - MC Dropout σ 상위 20% → LT 우선 검증
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch

from backend.api.schemas import BMDesignParams
from backend.core.uq_filter import UQFilter
from backend.data.dataset_manager import DatasetManager
from backend.harness.physical_validator import BMPhysicalValidator

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """능동 학습 설정."""
    max_lt_budget: int = 200          # LT 시뮬레이션 총 예산
    batch_size: int = 5               # 한 번에 LT 검증할 후보 수
    retrain_threshold: int = 5        # 재학습 트리거 최소 신규 데이터 수
    error_threshold: float = 0.05     # PINN 오차 재학습 임계값 (5%)
    sigma_threshold: float = 0.05     # UQ 강제 검증 임계값
    max_rounds: int = 20              # 최대 능동 학습 라운드


@dataclass
class ALRoundResult:
    """한 라운드의 결과."""
    round_num: int
    n_verified: int
    mean_error: float
    max_sigma: float
    retrained: bool


class ActiveLearningPipeline:
    """
    능동 학습 파이프라인.

    사용법:
        pipeline = ActiveLearningPipeline(
            predictor=fno_model.predict,
            lt_runner=lt_runner,
            dataset_mgr=dataset_mgr,
        )
        pipeline.run()
    """

    def __init__(
        self,
        predictor: Callable[[BMDesignParams], np.ndarray],
        lt_runner,
        dataset_mgr: DatasetManager,
        uq_filter: Optional[UQFilter] = None,
        retrain_fn: Optional[Callable] = None,
        config: Optional[ActiveLearningConfig] = None,
    ):
        """
        Args:
            predictor: 설계변수 → PSF7 예측 함수
            lt_runner: LightTools 시뮬레이션 래퍼
            dataset_mgr: 데이터셋 관리자
            uq_filter: 불확실도 추정기
            retrain_fn: PINN 재학습 함수 (호출 시 재학습 시작)
            config: 능동 학습 설정
        """
        self.predictor = predictor
        self.lt_runner = lt_runner
        self.dataset_mgr = dataset_mgr
        self.uq_filter = uq_filter
        self.retrain_fn = retrain_fn
        self.config = config or ActiveLearningConfig()
        self.validator = BMPhysicalValidator()

        self._lt_calls_used = 0
        self._new_data_since_retrain = 0
        self._round_history: list[ALRoundResult] = []

    @property
    def lt_budget_remaining(self) -> int:
        return max(0, self.config.max_lt_budget - self._lt_calls_used)

    def select_candidates(
        self,
        candidate_pool: list[BMDesignParams],
    ) -> list[BMDesignParams]:
        """
        LT 검증 대상 후보 선택.
        σ가 높은 순서로 batch_size개 선택.
        """
        if not candidate_pool:
            return []

        # 물리 제약 검증 통과한 것만
        valid = [p for p in candidate_pool if self.validator.validate(p).passed]

        if self.uq_filter is None:
            # UQ 없으면 랜덤 선택
            n = min(self.config.batch_size, len(valid), self.lt_budget_remaining)
            indices = np.random.choice(len(valid), size=n, replace=False)
            return [valid[i] for i in indices]

        # UQ 기반 선택: σ 높은 순
        sigmas = []
        for p in valid:
            psf_pred = self.predictor(p)
            # 간단한 불확실도 추정 (전체 PSF의 σ)
            sigma = np.std(psf_pred)  # placeholder
            sigmas.append(sigma)

        sigmas = np.array(sigmas)
        n = min(self.config.batch_size, len(valid), self.lt_budget_remaining)
        top_indices = np.argsort(sigmas)[-n:][::-1]

        selected = [valid[i] for i in top_indices]
        logger.info(
            f"Selected {len(selected)} candidates for LT verification "
            f"(max σ={sigmas[top_indices[0]]:.4f})"
        )
        return selected

    def verify_and_add(
        self,
        candidates: list[BMDesignParams],
    ) -> list[tuple[BMDesignParams, np.ndarray, float]]:
        """
        LT 시뮬레이션으로 검증하고 데이터 편입.

        Returns:
            list of (params, psf7_lt, error): 검증 결과
        """
        results = []

        for params in candidates:
            if self.lt_budget_remaining <= 0:
                logger.warning("LT budget exhausted")
                break

            # LT 시뮬레이션
            psf7_lt = self.lt_runner.run_single(params)
            if psf7_lt is None:
                continue

            self._lt_calls_used += 1

            # PINN 예측과 비교
            psf7_pred = self.predictor(params)
            error = np.mean((psf7_pred - psf7_lt) ** 2) / (np.mean(psf7_lt ** 2) + 1e-10)
            error = float(np.sqrt(error))  # RMSE relative

            # 데이터셋에 편입
            self.dataset_mgr.add_sample(params, psf7_lt, source="lt")
            self._new_data_since_retrain += 1

            results.append((params, psf7_lt, error))
            logger.info(
                f"LT verified: delta_bm1={params.delta_bm1:.2f}, "
                f"w1={params.w1:.2f} | error={error:.4f} | "
                f"LT budget: {self.lt_budget_remaining} remaining"
            )

        return results

    def should_retrain(self, errors: list[float]) -> bool:
        """재학습 필요 여부 판단."""
        if self._new_data_since_retrain >= self.config.retrain_threshold:
            return True
        if errors and max(errors) > self.config.error_threshold:
            return True
        return False

    def run_round(
        self,
        candidate_pool: list[BMDesignParams],
    ) -> ALRoundResult:
        """능동 학습 1라운드 실행."""
        round_num = len(self._round_history) + 1
        logger.info(f"=== Active Learning Round {round_num} ===")

        # 1. 후보 선택
        selected = self.select_candidates(candidate_pool)
        if not selected:
            logger.info("No candidates to verify")
            result = ALRoundResult(
                round_num=round_num, n_verified=0,
                mean_error=0.0, max_sigma=0.0, retrained=False,
            )
            self._round_history.append(result)
            return result

        # 2. LT 검증 & 데이터 편입
        verified = self.verify_and_add(selected)
        errors = [e for _, _, e in verified]

        # 3. 재학습 판단
        retrained = False
        if self.should_retrain(errors) and self.retrain_fn is not None:
            logger.info("Triggering PINN retraining...")
            self.retrain_fn()
            self._new_data_since_retrain = 0
            retrained = True

        result = ALRoundResult(
            round_num=round_num,
            n_verified=len(verified),
            mean_error=float(np.mean(errors)) if errors else 0.0,
            max_sigma=float(max(errors)) if errors else 0.0,
            retrained=retrained,
        )
        self._round_history.append(result)
        return result

    def run(
        self,
        candidate_pool: list[BMDesignParams],
        progress_callback: Optional[Callable] = None,
    ) -> list[ALRoundResult]:
        """전체 능동 학습 루프 실행."""
        logger.info(
            f"Starting Active Learning: "
            f"budget={self.config.max_lt_budget}, "
            f"max_rounds={self.config.max_rounds}"
        )

        for _ in range(self.config.max_rounds):
            if self.lt_budget_remaining <= 0:
                logger.info("LT budget exhausted. Stopping.")
                break

            result = self.run_round(candidate_pool)

            if progress_callback:
                progress_callback(result)

            if result.n_verified == 0:
                break

        logger.info(
            f"Active Learning complete: "
            f"{len(self._round_history)} rounds, "
            f"{self._lt_calls_used} LT calls used"
        )
        return self._round_history

    def summary(self) -> dict:
        """능동 학습 요약."""
        return {
            "rounds_completed": len(self._round_history),
            "lt_calls_used": self._lt_calls_used,
            "lt_budget_remaining": self.lt_budget_remaining,
            "total_dataset_size": self.dataset_mgr.n_samples,
            "retrain_count": sum(1 for r in self._round_history if r.retrained),
        }
