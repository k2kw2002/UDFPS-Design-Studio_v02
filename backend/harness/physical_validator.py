"""
physical_validator.py - 물리 제약 검증기
==========================================
이 파일은 "문지기" 역할입니다.

설계변수(delta_bm1, delta_bm2, w1, w2)가 들어오면,
"이 설계가 물리적으로 가능한가?"를 체크합니다.

체크하는 것들:
  1. 각 변수가 허용 범위 안에 있는가?
  2. 어퍼처(구멍) 크기가 양수인가?
  3. 오프셋이 너무 커서 인접 픽셀을 침범하지 않는가?
  4. 유효 수용각이 임계각(41.1도)을 넘지 않는가?

통과하면: ValidationResult(passed=True)
실패하면: ValidationResult(passed=False, reason="이유", fix_hint="해결방법")
"""

import math
from dataclasses import dataclass
from typing import Optional

# BMDesignParams를 가져옵니다 (schemas.py에서 정의한 것)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.schemas import BMDesignParams


# ============================================================
# 검증 결과를 담는 상자
# ============================================================
@dataclass
class ValidationResult:
    """
    검증 결과를 담는 상자.

    passed:   True = 통과, False = 실패
    reason:   실패했을 때 "왜 실패했는지" 설명
    fix_hint: 실패했을 때 "어떻게 고치면 되는지" 힌트
    """
    passed: bool
    reason: Optional[str] = None
    fix_hint: Optional[str] = None


# ============================================================
# 물리 제약 검증기 (메인)
# ============================================================
class BMPhysicalValidator:
    """
    BM 설계변수의 물리적 타당성을 검증하는 문지기.

    사용법:
        validator = BMPhysicalValidator()
        result = validator.validate(params)
        if result.passed:
            print("통과!")
        else:
            print(f"실패: {result.reason}")
            print(f"힌트: {result.fix_hint}")
    """

    # --- 각 변수의 허용 범위 ---
    # delta_bm1, delta_bm2: -10 ~ +10 um
    # w1, w2: 5 ~ 20 um
    BOUNDS = {
        "delta_bm1": (-10.0, 10.0),
        "delta_bm2": (-10.0, 10.0),
        "w1":        (5.0,   20.0),
        "w2":        (5.0,   20.0),
    }

    # --- 고정 상수 ---
    D_FIXED = 20.0       # BM1-BM2 간격 (um), 절대 변경 불가
    THETA_CRIT = 41.1    # CG 임계각 (도), 이 이상이면 전반사
    N_CG = 1.52          # Cover Glass 굴절률

    def validate(self, p: BMDesignParams) -> ValidationResult:
        """
        설계변수를 받아서 물리 제약을 하나씩 체크합니다.

        체크 순서:
          1) 허용 범위 검사
          2) 어퍼처 양수 검사
          3) 오프셋 침범 검사
          4) 유효 수용각 검사

        하나라도 실패하면 즉시 실패 결과를 반환합니다.
        """

        # -----------------------------------------------
        # 1) 허용 범위 검사
        #    "각 변수가 정해진 범위 안에 있는가?"
        #    예: delta_bm1은 -10 ~ +10 사이여야 함
        # -----------------------------------------------
        for field_name, (lo, hi) in self.BOUNDS.items():
            value = getattr(p, field_name)  # p에서 해당 변수값을 꺼냄
            if not (lo <= value <= hi):
                return ValidationResult(
                    passed=False,
                    reason=f"{field_name}={value:.2f}um -> 범위 [{lo},{hi}] 초과",
                    fix_hint=f"{lo} <= {field_name} <= {hi} 로 조정하세요"
                )

        # -----------------------------------------------
        # 2) 어퍼처(구멍)와 간격이 양수인지 검사
        #    "구멍 크기가 0 이하이면 물리적으로 불가능"
        # -----------------------------------------------
        for field_name in ["w1", "w2", "d"]:
            if getattr(p, field_name) <= 0:
                return ValidationResult(
                    passed=False,
                    reason=f"{field_name} <= 0 -> 물리적으로 불가능",
                    fix_hint=f"{field_name}은 반드시 양수여야 합니다"
                )

        # -----------------------------------------------
        # 3) 오프셋 침범 검사
        #    "어퍼처를 너무 많이 이동시키면 옆 픽셀을 침범함"
        #
        #    조건: |delta_bm1| <= w1/2
        #    예시: w1=10um이면, delta_bm1은 -5~+5 사이여야 함
        #          delta_bm1=7이면 구멍이 옆으로 너무 이동 -> 실패
        # -----------------------------------------------
        if abs(p.delta_bm1) > p.w1 / 2:
            return ValidationResult(
                passed=False,
                reason=(
                    f"|delta_bm1|={abs(p.delta_bm1):.2f} > w1/2={p.w1/2:.2f} "
                    f"-> BM1 어퍼처가 인접 픽셀 침범"
                ),
                fix_hint="delta_bm1 감소 또는 w1 증가"
            )
        if abs(p.delta_bm2) > p.w2 / 2:
            return ValidationResult(
                passed=False,
                reason=(
                    f"|delta_bm2|={abs(p.delta_bm2):.2f} > w2/2={p.w2/2:.2f} "
                    f"-> BM2 어퍼처가 인접 픽셀 침범"
                ),
                fix_hint="delta_bm2 감소 또는 w2 증가"
            )

        # -----------------------------------------------
        # 4) 유효 수용각 검사
        #    "빛이 BM을 통과할 수 있는 최대 각도가
        #     CG 임계각(41.1도)을 넘으면 안 됨"
        #
        #    공식: theta_eff = arctan(w1 / (2*d))
        #    d=20um 고정이므로: theta_eff = arctan(w1 / 40)
        #    예시: w1=20um -> theta_eff = arctan(0.5) = 26.6도 -> OK
        # -----------------------------------------------
        if p.theta_max_eff > self.THETA_CRIT:
            return ValidationResult(
                passed=False,
                reason=(
                    f"theta_eff={p.theta_max_eff:.1f}도 > "
                    f"임계각 {self.THETA_CRIT}도"
                ),
                fix_hint="w1 감소 필요 (d=20um 고정)"
            )

        # -----------------------------------------------
        # 모든 검사 통과!
        # -----------------------------------------------
        return ValidationResult(passed=True)
