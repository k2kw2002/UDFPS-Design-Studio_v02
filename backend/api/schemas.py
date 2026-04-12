"""
schemas.py - 데이터 구조 정의
================================
이 파일은 "택배 송장 양식"과 같습니다.
프로젝트에서 주고받는 모든 데이터의 형태를 미리 정해놓습니다.

예시:
  BMDesignParams  = "BM 설계변수 4개를 담는 상자"
  BMDesignSpec    = "목표 스펙을 담는 상자"
  BMCandidate     = "최적화 결과 후보 1개를 담는 상자"

Pydantic이라는 도구를 사용해서,
잘못된 값이 들어오면 자동으로 에러를 냅니다.
"""

import math
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 1) BM 설계변수 (최적화의 "입력")
# ============================================================
class BMDesignParams(BaseModel):
    """
    BM(Black Matrix) 설계변수를 담는 상자.

    최적화 변수 4개:
      - delta_bm1: BM1 어퍼처(구멍)의 위치 오프셋 (um)
      - delta_bm2: BM2 어퍼처(구멍)의 위치 오프셋 (um)
      - w1:        BM1 어퍼처(구멍)의 폭 (um)
      - w2:        BM2 어퍼처(구멍)의 폭 (um)

    고정값 (바꿀 수 없는 값):
      - d:         BM1과 BM2 사이 간격 = 20um
      - 나머지:    물리적으로 정해진 값들
    """

    # --- 최적화 변수 4개 (um 단위) ---
    delta_bm1: float = Field(
        default=0.0,
        ge=-10.0, le=10.0,
        description="BM1 어퍼처 오프셋 (um). 각 OPD 픽셀 중심 기준."
    )
    delta_bm2: float = Field(
        default=0.0,
        ge=-10.0, le=10.0,
        description="BM2 어퍼처 오프셋 (um). 각 OPD 픽셀 중심 기준."
    )
    w1: float = Field(
        default=10.0,
        gt=0.0, le=20.0,
        description="BM1 어퍼처 폭 (um). 빛이 통과하는 구멍의 너비."
    )
    w2: float = Field(
        default=10.0,
        gt=0.0, le=20.0,
        description="BM2 어퍼처 폭 (um). 빛이 통과하는 구멍의 너비."
    )

    # --- 고정값 (절대 변경 불가) ---
    d: float = Field(
        default=20.0,
        description="BM1-BM2 간격 (um). 고정값."
    )
    t_bm1: float = Field(default=0.1, description="BM1 두께 (um)")
    t_bm2: float = Field(default=0.1, description="BM2 두께 (um)")
    z_encap: float = Field(default=20.0, description="Encap 두께 (um)")
    opd_pitch: float = Field(default=72.0, description="OPD 피치 (um)")
    opd_width: float = Field(default=10.0, description="OPD 픽셀 폭 (um)")
    cg_thick: float = Field(default=550.0, description="Cover Glass+OCR 합산 (um)")

    @property
    def theta_max_eff(self) -> float:
        """
        BM 기하학적 유효 수용각 (도).

        빛이 BM1 구멍을 통과해서 BM2까지 도달할 수 있는
        최대 각도를 계산합니다.

        공식: theta_eff = arctan(w1 / (2 * d))
        예시: w1=10um, d=20um -> theta_eff = arctan(10/40) = 14도
        """
        return math.degrees(math.atan(self.w1 / (2 * self.d)))


# ============================================================
# 2) 목표 스펙 (역설계의 "목표")
# ============================================================
class BMDesignSpec(BaseModel):
    """
    역설계 목표를 담는 상자.

    "이 정도 성능이 나오는 BM 구조를 찾아줘"라고 요청할 때 사용.
      - mtf_ridge_min:  지문 인식률의 최소 목표 (0~1, 높을수록 좋음)
      - skewness_max:   PSF 비대칭의 최대 허용치 (낮을수록 좋음)
      - throughput_min: 광량의 최소 목표 (0~1, 높을수록 좋음)
    """
    mtf_ridge_min: float = Field(
        default=0.60, ge=0.10, le=0.95,
        description="MTF@ridge 최소 목표. 0.60 = 60%"
    )
    skewness_max: float = Field(
        default=0.10, ge=0.01, le=0.50,
        description="skewness 최대 허용치. 0.10 이하가 목표."
    )
    throughput_min: float = Field(
        default=0.60, ge=0.10, le=0.95,
        description="광량 T 최소 목표. 0.60 = 60%"
    )


# ============================================================
# 3) Pareto 가중치 (3목적 간 중요도 비율)
# ============================================================
class ParetoWeights(BaseModel):
    """
    3가지 목표 간 중요도 비율.

    예시: mtf=0.4, throughput=0.3, skewness=0.3
    -> "지문 인식률을 가장 중요하게, 나머지는 동등하게"
    """
    mtf: float = Field(default=0.4, description="MTF 가중치")
    throughput: float = Field(default=0.3, description="광량 가중치")
    skewness: float = Field(default=0.3, description="skewness 가중치")


# ============================================================
# 4) 최적화 결과 후보 1개
# ============================================================
class BMCandidate(BaseModel):
    """
    최적화로 찾아낸 설계 후보 1개를 담는 상자.

    역설계를 돌리면 Top-5 후보가 나오는데,
    각 후보의 설계변수 + 성능 + 점수를 여기에 저장합니다.
    """
    id: str = Field(description="후보 고유 ID")
    label: str = Field(description="후보 라벨 (A, B, C, D, E)")
    params: BMDesignParams = Field(description="설계변수 4개")

    # --- 성능 지표 ---
    mtf_ridge: float = Field(description="MTF@ridge (0~1)")
    skewness: float = Field(description="PSF 비대칭도")
    throughput: float = Field(description="광량 T (0~1)")
    crosstalk_ratio: float = Field(description="크로스토크 비율 (낮을수록 좋음)")

    # --- 평가 ---
    evaluator_score: float = Field(description="Evaluator 채점 (0~100)")
    pareto_rank: int = Field(description="Pareto 순위 (1이 최고)")
    uncertainty_sigma: float = Field(description="불확실도 sigma")
    constraint_ok: bool = Field(description="물리 제약 통과 여부")
