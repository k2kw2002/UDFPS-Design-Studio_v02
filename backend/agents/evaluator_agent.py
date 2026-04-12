"""
evaluator_agent.py - 평가 에이전트
=====================================
Generator가 생성한 후보를 채점하는 에이전트.
Generator와 독립적으로 동작합니다 (인스턴스 공유 금지).

채점 기준 (AGENTS.md, 각 20점, 총 100점):
  1. MTF@ridge 목표 달성도
  2. skewness 목표 달성도
  3. 광량 T 목표 달성도
  4. 수렴 신뢰도 (BO acquisition value)
  5. 물리 제약 마진 (d, w 기하 조건 여유)

70점 미만: Generator 재탐색 지시
"""

import logging
from typing import Optional

from backend.api.schemas import BMDesignParams, BMCandidate, BMDesignSpec
from backend.harness.physical_validator import BMPhysicalValidator
from backend.harness.agents_config import AgentsConfig, load_agents_config

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    후보 채점 에이전트.

    사용법:
        evaluator = EvaluatorAgent(spec=spec)
        scored = evaluator.score(candidate)
        accept = evaluator.accept_or_reject(scored)
    """

    RETRY_THRESHOLD = 70  # 이 점수 미만이면 Generator 재시도

    def __init__(
        self,
        spec: Optional[BMDesignSpec] = None,
        config: Optional[AgentsConfig] = None,
    ):
        self.spec = spec or BMDesignSpec()
        self.config = config or load_agents_config()
        # 독립 validator (Generator와 공유 금지)
        self.validator = BMPhysicalValidator()

    def score(self, candidate: BMCandidate) -> BMCandidate:
        """
        후보 채점 (5개 차원, 각 20점).

        Returns:
            BMCandidate with updated evaluator_score
        """
        scores = {}

        # 1. MTF@ridge 달성도 (20점)
        scores["mtf"] = self._score_mtf(candidate.mtf_ridge)

        # 2. skewness 달성도 (20점)
        scores["skewness"] = self._score_skewness(candidate.skewness)

        # 3. 광량 T 달성도 (20점)
        scores["throughput"] = self._score_throughput(candidate.throughput)

        # 4. 수렴 신뢰도 (20점) - pareto_rank 기반
        scores["convergence"] = self._score_convergence(candidate.pareto_rank)

        # 5. 물리 제약 마진 (20점)
        scores["constraint"] = self._score_constraint_margin(candidate.params)

        total = sum(scores.values())
        candidate.evaluator_score = total

        logger.info(
            f"Candidate {candidate.label}: "
            f"total={total:.1f} "
            f"(mtf={scores['mtf']:.1f}, skew={scores['skewness']:.1f}, "
            f"T={scores['throughput']:.1f}, conv={scores['convergence']:.1f}, "
            f"constr={scores['constraint']:.1f})"
        )

        return candidate

    def _score_mtf(self, mtf: float) -> float:
        """MTF@ridge 채점 (20점 만점)."""
        target = self.spec.mtf_ridge_min
        if mtf >= target:
            return 20.0
        ratio = mtf / target
        return max(0.0, ratio * 20.0)

    def _score_skewness(self, skew: float) -> float:
        """skewness 채점 (20점 만점). 낮을수록 좋음."""
        target = self.spec.skewness_max
        if skew <= target:
            return 20.0
        if skew > target * 5:
            return 0.0
        ratio = 1.0 - (skew - target) / (target * 4)
        return max(0.0, ratio * 20.0)

    def _score_throughput(self, T: float) -> float:
        """광량 T 채점 (20점 만점)."""
        target = self.spec.throughput_min
        if T >= target:
            return 20.0
        ratio = T / target
        return max(0.0, ratio * 20.0)

    def _score_convergence(self, pareto_rank: int) -> float:
        """수렴 신뢰도 채점 (20점 만점). rank 1이 최고."""
        if pareto_rank <= 1:
            return 20.0
        if pareto_rank <= 3:
            return 15.0
        if pareto_rank <= 5:
            return 10.0
        return 5.0

    def _score_constraint_margin(self, params: BMDesignParams) -> float:
        """물리 제약 마진 채점 (20점 만점)."""
        vr = self.validator.validate(params)
        if not vr.passed:
            return 0.0

        # 마진 계산 (각 제약 조건의 여유)
        margin_scores = []

        # 오프셋 마진: |delta| / (w/2)가 1 미만이어야 함 (작을수록 여유)
        if params.w1 > 0:
            ratio1 = abs(params.delta_bm1) / (params.w1 / 2)
            margin_scores.append(max(0.0, 1.0 - ratio1))
        if params.w2 > 0:
            ratio2 = abs(params.delta_bm2) / (params.w2 / 2)
            margin_scores.append(max(0.0, 1.0 - ratio2))

        # 수용각 마진: theta_eff / theta_crit
        theta_ratio = params.theta_max_eff / 41.1
        margin_scores.append(max(0.0, 1.0 - theta_ratio))

        avg_margin = sum(margin_scores) / len(margin_scores) if margin_scores else 0.0
        return avg_margin * 20.0

    def accept_or_reject(self, candidate: BMCandidate) -> tuple[bool, str]:
        """
        후보 수락/거부 판정.

        Returns:
            (accepted, reason)
        """
        if candidate.evaluator_score >= self.RETRY_THRESHOLD:
            return True, f"Score {candidate.evaluator_score:.1f} >= {self.RETRY_THRESHOLD}"
        return False, (
            f"Score {candidate.evaluator_score:.1f} < {self.RETRY_THRESHOLD}. "
            f"Generator 재탐색 필요."
        )

    def score_batch(self, candidates: list[BMCandidate]) -> list[BMCandidate]:
        """배치 채점."""
        scored = [self.score(c) for c in candidates]
        # 점수 순 정렬
        scored.sort(key=lambda c: c.evaluator_score, reverse=True)
        return scored
