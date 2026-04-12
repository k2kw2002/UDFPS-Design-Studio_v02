"""
botorch_optimizer.py - 3목적 qNEHVI 역설계 엔진
=================================================
BoTorch를 사용한 다목적 Bayesian Optimization.

3가지 목적:
  obj1: MTF@ridge  (최대화) — 지문 인식률
  obj2: 광량 T     (최대화) — 광학 효율
  obj3: -skewness  (최대화 = skewness 최소화) — PSF 대칭성

탐색 전략:
  1. LHS 20점 초기 평가
  2. qNEHVI (qLogNEHVI fallback) acquisition
  3. 수렴 기준: 연속 5회 HV 개선 < 0.1%
  4. 결과: Pareto front + Top-5 후보

설계변수 범위 (AGENTS.md):
  δ_BM1: [-10, +10] um
  δ_BM2: [-10, +10] um
  w₁:    [5, 20] um
  w₂:    [5, 20] um
"""

import logging
import uuid
from typing import Optional, Callable

import numpy as np
import torch
from dataclasses import dataclass, field

from backend.api.schemas import BMDesignParams, BMCandidate
from backend.harness.physical_validator import BMPhysicalValidator
from backend.physics.psf_metrics import PSFMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """최적화 결과."""
    pareto_X: torch.Tensor      # Pareto 최적 설계변수 (N, 4)
    pareto_Y: torch.Tensor      # Pareto 최적 목적값 (N, 3)
    all_X: torch.Tensor         # 전체 평가 설계변수
    all_Y: torch.Tensor         # 전체 평가 목적값
    top5: list[BMCandidate]     # Top-5 후보
    hypervolume: float          # 최종 Hypervolume
    n_iterations: int           # 총 반복 수
    converged: bool             # 수렴 여부


BOUNDS_RAW = {
    "delta_bm1": (-10.0, 10.0),
    "delta_bm2": (-10.0, 10.0),
    "w1":        (5.0,   20.0),
    "w2":        (5.0,   20.0),
}
PARAM_NAMES = list(BOUNDS_RAW.keys())

# Reference point for hypervolume (worst acceptable)
REF_POINT = torch.tensor([0.0, 0.0, -1.0])


class BMOptimizer:
    """
    3목적 qNEHVI Bayesian Optimization 역설계 엔진.

    사용법:
        optimizer = BMOptimizer(fno_surrogate, validator)
        result = optimizer.optimize(n_iter=50)
        print(result.top5)  # Top-5 후보
    """

    def __init__(
        self,
        surrogate: Callable,
        validator: Optional[BMPhysicalValidator] = None,
        device: str = "cpu",
    ):
        """
        Args:
            surrogate: 설계변수 → PSF7 예측 함수.
                       (BMDesignParams) -> np.ndarray (shape: 7)
            validator: 물리 제약 검증기
            device: torch 디바이스
        """
        self.surrogate = surrogate
        self.validator = validator or BMPhysicalValidator()
        self.metrics = PSFMetrics()
        self.device = torch.device(device)

        lo = [v[0] for v in BOUNDS_RAW.values()]
        hi = [v[1] for v in BOUNDS_RAW.values()]
        self.bounds = torch.tensor([lo, hi], dtype=torch.float64, device=self.device)

    def _tensor_to_params(self, t: torch.Tensor) -> BMDesignParams:
        """(4,) 텐서 → BMDesignParams."""
        return BMDesignParams(**{
            PARAM_NAMES[i]: float(t[i]) for i in range(4)
        })

    def _evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        배치 평가: (N, 4) 설계변수 → (N, 3) 목적값.
        obj = [MTF, throughput, -skewness]
        """
        results = []
        for i in range(X.shape[0]):
            params = self._tensor_to_params(X[i])

            # 물리 제약 검증
            vr = self.validator.validate(params)
            if not vr.passed:
                results.append(torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
                continue

            # FNO/PINN surrogate로 PSF 예측
            psf7 = self.surrogate(params)
            m = self.metrics.compute(psf7)

            results.append(torch.tensor([
                m["mtf_ridge"],
                m["throughput"],
                -m["skewness"],  # 최소화 → 부호 반전
            ], dtype=torch.float64))

        return torch.stack(results).to(self.device)

    def optimize(
        self,
        n_init: int = 20,
        n_iter: int = 50,
        q: int = 4,
        convergence_window: int = 5,
        convergence_threshold: float = 0.001,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        3목적 역설계 실행.

        Args:
            n_init: 초기 LHS 샘플 수 (기본 20)
            n_iter: 최대 반복 수 (기본 50)
            q: 배치 크기 (기본 4)
            convergence_window: 수렴 판정 윈도우
            convergence_threshold: HV 개선 임계값 (0.1%)
            progress_callback: 진행 콜백

        Returns:
            OptimizationResult
        """
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

        # 초기 LHS 샘플링
        logger.info(f"Generating {n_init} initial LHS samples...")
        X = self._initial_lhs(n_init)
        Y = self._evaluate(X)

        hv_computer = Hypervolume(ref_point=REF_POINT)
        hv_history = []
        converged = False

        for iteration in range(n_iter):
            # GP 모델 피팅 (3 objectives → 3 independent GPs)
            models = []
            for j in range(3):
                gp = SingleTaskGP(
                    X.double(), Y[:, j:j+1].double()
                )
                models.append(gp)
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)

            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.warning(f"GP fitting failed at iter {iteration}: {e}")
                continue

            # Acquisition function
            try:
                from botorch.acquisition.multi_objective.logei import (
                    qLogNoisyExpectedHyperVolumeImprovement,
                )
                acqf = qLogNoisyExpectedHyperVolumeImprovement(
                    model=model,
                    ref_point=REF_POINT.tolist(),
                    X_baseline=X.double(),
                )
            except Exception:
                from botorch.acquisition.multi_objective.monte_carlo import (
                    qNoisyExpectedHypervolumeImprovement,
                )
                acqf = qNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=REF_POINT.tolist(),
                    X_baseline=X.double(),
                )

            # 후보 최적화
            from botorch.optim import optimize_acqf
            candidates, _ = optimize_acqf(
                acqf,
                bounds=self.bounds.double(),
                q=q,
                num_restarts=10,
                raw_samples=512,
            )

            # 평가 & 축적
            Y_new = self._evaluate(candidates)
            X = torch.cat([X, candidates])
            Y = torch.cat([Y, Y_new])

            # Hypervolume 계산
            pareto_mask = is_non_dominated(Y)
            pareto_Y = Y[pareto_mask]
            if pareto_Y.shape[0] > 0:
                hv = hv_computer.compute(pareto_Y)
            else:
                hv = 0.0
            hv_history.append(hv)

            logger.info(
                f"Iter {iteration+1}/{n_iter}: "
                f"HV={hv:.4f}, Pareto={pareto_mask.sum()}, Total={X.shape[0]}"
            )

            if progress_callback:
                progress_callback({
                    "iteration": iteration + 1,
                    "hypervolume": hv,
                    "n_pareto": int(pareto_mask.sum()),
                    "n_total": X.shape[0],
                })

            # 수렴 검사
            if len(hv_history) >= convergence_window:
                recent = hv_history[-convergence_window:]
                if recent[-1] > 0:
                    improvements = [
                        abs(recent[i+1] - recent[i]) / abs(recent[i] + 1e-10)
                        for i in range(len(recent) - 1)
                    ]
                    if all(imp < convergence_threshold for imp in improvements):
                        logger.info(f"Converged at iteration {iteration+1}")
                        converged = True
                        break

        # Pareto front 추출
        pareto_mask = is_non_dominated(Y)
        pareto_X = X[pareto_mask]
        pareto_Y = Y[pareto_mask]

        # Top-5 선택 (가중합 기준)
        top5 = self._select_top5(pareto_X, pareto_Y)

        final_hv = hv_history[-1] if hv_history else 0.0

        return OptimizationResult(
            pareto_X=pareto_X,
            pareto_Y=pareto_Y,
            all_X=X,
            all_Y=Y,
            top5=top5,
            hypervolume=final_hv,
            n_iterations=len(hv_history),
            converged=converged,
        )

    def _initial_lhs(self, n: int) -> torch.Tensor:
        """LHS 초기 샘플링."""
        from backend.data.lhs_sampler import generate_lhs_samples, samples_to_numpy
        samples = generate_lhs_samples(n_samples=n, validate=True)
        arr = samples_to_numpy(samples)
        return torch.tensor(arr, dtype=torch.float64, device=self.device)

    def _select_top5(
        self, pareto_X: torch.Tensor, pareto_Y: torch.Tensor
    ) -> list[BMCandidate]:
        """Pareto front에서 Top-5 후보 선택."""
        if pareto_X.shape[0] == 0:
            return []

        # 가중합 스코어 (MTF:0.4, T:0.3, -skew:0.3)
        weights = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)
        scores = (pareto_Y * weights).sum(dim=1)
        n_top = min(5, pareto_X.shape[0])
        top_indices = scores.topk(n_top).indices

        labels = ["A", "B", "C", "D", "E"]
        candidates = []

        for rank, idx in enumerate(top_indices):
            p = self._tensor_to_params(pareto_X[idx])
            y = pareto_Y[idx]

            candidates.append(BMCandidate(
                id=str(uuid.uuid4())[:8],
                label=labels[rank],
                params=p,
                mtf_ridge=float(y[0]),
                skewness=float(-y[2]),  # 부호 복원
                throughput=float(y[1]),
                crosstalk_ratio=0.0,  # 별도 계산 필요
                evaluator_score=float(scores[idx]) * 100,
                pareto_rank=rank + 1,
                uncertainty_sigma=0.0,  # UQ에서 별도 계산
                constraint_ok=True,
            ))

        return candidates
