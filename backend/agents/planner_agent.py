"""
planner_agent.py - 계획 에이전트
==================================
역설계 작업의 전략을 수립하는 에이전트.

역할:
  - 목표 사양(BMDesignSpec)을 분석
  - 현재 시스템 상태(모델 준비, 데이터 양) 확인
  - Generator에게 실행 전략 지시
  - 예산(LT 호출 수) 관리
"""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.api.schemas import BMDesignSpec, ParetoWeights
from backend.harness.agents_config import AgentsConfig, load_agents_config
from backend.harness.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """실행 계획."""
    strategy: str           # "full_bo", "quick_bo", "reuse_cache"
    n_bo_iterations: int    # BO 반복 수
    n_init_samples: int     # 초기 LHS 수
    use_fno: bool           # FNO 사용 여부
    retrain_pinn: bool      # PINN 재학습 필요
    lt_budget_alloc: int    # 이번 실행에 할당할 LT 예산
    reason: str             # 전략 선택 이유


class PlannerAgent:
    """
    역설계 작업의 전략을 수립.

    사용법:
        planner = PlannerAgent()
        plan = planner.create_plan(spec, weights, system_state)
    """

    def __init__(self, config: Optional[AgentsConfig] = None):
        self.config = config or load_agents_config()

    def create_plan(
        self,
        spec: BMDesignSpec,
        weights: ParetoWeights,
        system_state: Optional[dict] = None,
    ) -> ExecutionPlan:
        """
        실행 계획 수립.

        Args:
            spec: 목표 사양
            weights: 가중치
            system_state: 현재 시스템 상태
                - fno_ready: bool
                - pinn_trained: bool
                - lt_remaining: int
                - dataset_size: int
                - drift_detected: bool
        """
        state = system_state or {}
        fno_ready = state.get("fno_ready", False)
        pinn_trained = state.get("pinn_trained", False)
        lt_remaining = state.get("lt_remaining", 200)
        dataset_size = state.get("dataset_size", 0)
        drift_detected = state.get("drift_detected", False)

        # 전략 결정
        if not pinn_trained:
            return ExecutionPlan(
                strategy="full_bo",
                n_bo_iterations=50,
                n_init_samples=20,
                use_fno=False,
                retrain_pinn=True,
                lt_budget_alloc=min(80, lt_remaining),
                reason="PINN 미학습 상태. 초기 학습 후 전체 BO 실행 필요.",
            )

        if drift_detected:
            return ExecutionPlan(
                strategy="full_bo",
                n_bo_iterations=30,
                n_init_samples=10,
                use_fno=fno_ready,
                retrain_pinn=True,
                lt_budget_alloc=min(20, lt_remaining),
                reason="드리프트 감지. PINN 재학습 후 BO 재실행.",
            )

        if fno_ready:
            return ExecutionPlan(
                strategy="quick_bo",
                n_bo_iterations=50,
                n_init_samples=20,
                use_fno=True,
                retrain_pinn=False,
                lt_budget_alloc=min(10, lt_remaining),
                reason="FNO 준비 완료. 빠른 BO 실행 (~8초).",
            )

        return ExecutionPlan(
            strategy="full_bo",
            n_bo_iterations=50,
            n_init_samples=20,
            use_fno=False,
            retrain_pinn=False,
            lt_budget_alloc=min(30, lt_remaining),
            reason="PINN 직접 추론으로 BO 실행.",
        )

    def validate_plan(self, plan: ExecutionPlan) -> tuple[bool, str]:
        """계획 유효성 검증."""
        if plan.lt_budget_alloc < 0:
            return False, "LT 예산 부족"
        if plan.n_bo_iterations < 1:
            return False, "BO 반복 수 부족"
        return True, "OK"
