"""
lhs_sampler.py - 4차원 Latin Hypercube Sampling
=================================================
설계변수 (δ_BM1, δ_BM2, w₁, w₂)의 초기 후보를 생성합니다.

LHS가 랜덤 샘플링보다 나은 이유:
  랜덤: 빈 영역 발생 가능 → 탐색 누락
  LHS:  전 영역을 균등 분할 → 적은 샘플로 전체 탐색

사용처:
  1. BoTorch 초기 후보 20점 생성 (Section 8, AGENTS.md)
  2. LightTools 학습 데이터 80~200점 생성
  3. 모든 샘플은 BMPhysicalValidator 통과 필수
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube

from backend.api.schemas import BMDesignParams
from backend.harness.physical_validator import BMPhysicalValidator


# 설계변수 범위 (AGENTS.md 기준)
VARIABLE_BOUNDS = {
    "delta_bm1": (-10.0, 10.0),
    "delta_bm2": (-10.0, 10.0),
    "w1":        (5.0,   20.0),
    "w2":        (5.0,   20.0),
}

VARIABLE_NAMES = list(VARIABLE_BOUNDS.keys())
N_DIMS = len(VARIABLE_NAMES)


def generate_lhs_samples(
    n_samples: int = 200,
    seed: int = 42,
    validate: bool = True,
    max_retries: int = 3,
) -> list[BMDesignParams]:
    """
    4차원 LHS로 설계변수 샘플 생성.

    Args:
        n_samples: 생성할 샘플 수 (기본 200)
        seed:      재현성을 위한 랜덤 시드
        validate:  BMPhysicalValidator 통과 여부 확인
        max_retries: 검증 실패 시 재생성 횟수

    Returns:
        list[BMDesignParams]: 검증 통과한 설계변수 목록
    """
    validator = BMPhysicalValidator() if validate else None

    valid_samples = []
    attempt = 0

    while len(valid_samples) < n_samples and attempt < max_retries:
        # 부족한 수만큼 추가 생성 (여유분 20% 포함)
        needed = int((n_samples - len(valid_samples)) * 1.2) + 10
        sampler = LatinHypercube(d=N_DIMS, seed=seed + attempt)
        raw = sampler.random(n=needed)

        # [0,1] → 실제 범위로 스케일링
        for row in raw:
            params_dict = {}
            for j, name in enumerate(VARIABLE_NAMES):
                lo, hi = VARIABLE_BOUNDS[name]
                params_dict[name] = lo + row[j] * (hi - lo)

            params = BMDesignParams(**params_dict)

            if validator:
                result = validator.validate(params)
                if not result.passed:
                    continue

            valid_samples.append(params)
            if len(valid_samples) >= n_samples:
                break

        attempt += 1

    return valid_samples[:n_samples]


def samples_to_numpy(samples: list[BMDesignParams]) -> np.ndarray:
    """BMDesignParams 리스트 → (N, 4) numpy 배열 변환."""
    return np.array([
        [s.delta_bm1, s.delta_bm2, s.w1, s.w2]
        for s in samples
    ])


def samples_to_tensor(samples: list[BMDesignParams]):
    """BMDesignParams 리스트 → (N, 4) PyTorch 텐서 변환."""
    import torch
    return torch.tensor(samples_to_numpy(samples), dtype=torch.float32)
