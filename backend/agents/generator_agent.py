"""
generator_agent.py - 생성 에이전트
=====================================
Planner의 계획을 받아 실제 역설계를 실행하는 에이전트.

역할:
  - BMOptimizer를 사용하여 BO 실행
  - PINN/FNO를 사용하여 PSF 예측
  - Evaluator에게 결과 전달
  - 물리 제약 위반 시 자동 재시도
"""

import logging
from typing import Optional, Callable

import numpy as np

from backend.api.schemas import BMDesignParams, BMCandidate, BMDesignSpec
from backend.harness.physical_validator import BMPhysicalValidator
from backend.agents.planner_agent import ExecutionPlan

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    역설계 실행 에이전트.

    사용법:
        generator = GeneratorAgent(optimizer=bm_optimizer)
        candidates = generator.run(plan)
    """

    def __init__(
        self,
        optimizer=None,
        pinn_predictor: Optional[Callable] = None,
        fno_predictor: Optional[Callable] = None,
    ):
        """
        Args:
            optimizer: BMOptimizer 인스턴스
            pinn_predictor: PINN 예측 함수 (BMDesignParams → np.ndarray)
            fno_predictor: FNO 예측 함수 (BMDesignParams → np.ndarray)
        """
        self.optimizer = optimizer
        self.pinn_predictor = pinn_predictor
        self.fno_predictor = fno_predictor
        self.validator = BMPhysicalValidator()

    def run(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[Callable] = None,
    ) -> list[BMCandidate]:
        """
        계획에 따라 역설계 실행.

        Args:
            plan: 실행 계획 (PlannerAgent 출력)
            progress_callback: 진행 콜백

        Returns:
            Top-5 후보 목록
        """
        logger.info(
            f"Generator starting: strategy={plan.strategy}, "
            f"iterations={plan.n_bo_iterations}, fno={plan.use_fno}"
        )

        if self.optimizer is None:
            logger.error("No optimizer configured")
            return []

        # BO 실행
        result = self.optimizer.optimize(
            n_init=plan.n_init_samples,
            n_iter=plan.n_bo_iterations,
            progress_callback=progress_callback,
        )

        candidates = result.top5
        logger.info(
            f"Generator complete: {len(candidates)} candidates, "
            f"HV={result.hypervolume:.4f}, "
            f"converged={result.converged}"
        )

        # 물리 제약 최종 검증
        valid_candidates = []
        for c in candidates:
            vr = self.validator.validate(c.params)
            if vr.passed:
                valid_candidates.append(c)
            else:
                logger.warning(
                    f"Candidate {c.label} failed validation: {vr.reason}"
                )

        return valid_candidates

    def predict_single(self, params: BMDesignParams, use_fno: bool = False) -> np.ndarray:
        """단일 설계변수 PSF 예측."""
        predictor = self.fno_predictor if (use_fno and self.fno_predictor) else self.pinn_predictor
        if predictor is None:
            raise ValueError("No predictor configured")
        return predictor(params)
