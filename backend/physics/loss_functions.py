"""
loss_functions.py - PINN 4가지 손실함수
=========================================
이 파일은 "PINN이 얼마나 잘 학습하고 있는지" 측정하는 4가지 기준입니다.

비유:
  시험을 볼 때 4개 과목이 있다고 생각하면 됩니다:
    1. 물리 시험 (L_Helmholtz): "파동 방정식을 잘 풀고 있는가?"  → 58%
    2. 코팅 시험 (L_phase):     "AR 코팅 위상을 맞추고 있는가?"  → 23%
    3. 실험 시험 (L_I):         "실제 측정값과 맞는가?"           → 11%
    4. 경계 시험 (L_BC):        "BM 벽에서 빛이 0인가?"          → 8%

  총점 = 1.0 * 물리 + 0.5 * 코팅 + 0.3 * 실험 + 0.2 * 경계

  이 총점(Loss)이 낮을수록 PINN이 잘 학습한 것입니다.
  학습 = "이 총점을 최대한 낮추는 과정"

핵심 포인트:
  - L_Helmholtz (58%): 가장 중요! 물리 법칙 자체. 데이터 0개.
  - L_phase (23%): TMM에서 계산한 위상. 데이터 0개.
  - L_I (11%): LightTools 시뮬레이션 200개. 유일한 실험 데이터.
  - L_BC (8%): BM은 불투명 → 빛=0 강제. 데이터 0개.
"""

import torch


class UDFPSPINNLosses:
    """
    UDFPS COE 스택 Helmholtz PINN의 4가지 손실함수.

    도메인: x ∈ [0, 504um] (7피치), z ∈ [0, 570um]

    총 Loss = λ1·L_Helm + λ2·L_phase + λ3·L_I + λ4·L_BC

    사용법:
        losses = UDFPSPINNLosses()
        Lh = losses.helmholtz(U, x, z, k0, n)    # 물리 법칙
        Lp = losses.phase(U_pred, dphi_tmm)        # AR 코팅
        Li = losses.intensity(U_pred, I_lt)         # 실험 데이터
        Lb = losses.boundary(U_at_BM)               # BM 경계
        total = losses.total(Lh, Lp, Li, Lb)        # 총점
    """

    def __init__(
        self,
        lam_helm: float = 1.0,
        lam_phase: float = 0.5,
        lam_I: float = 0.3,
        lam_BC: float = 0.2
    ):
        """
        각 과목의 가중치(중요도)를 설정합니다.

        lam_helm:  물리 법칙 가중치 (기본 1.0 = 가장 중요)
        lam_phase: AR 코팅 가중치 (기본 0.5)
        lam_I:     실험 데이터 가중치 (기본 0.3)
        lam_BC:    경계 조건 가중치 (기본 0.2 = 가장 덜 중요)
        """
        self.lam = dict(
            helm=lam_helm,
            phase=lam_phase,
            I=lam_I,
            BC=lam_BC
        )

    def helmholtz(
        self,
        U: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        k0: float,
        n: float
    ) -> torch.Tensor:
        """
        L_Helmholtz: 파동 방정식 잔차 (물리 시험)

        Helmholtz 방정식: ∂²U/∂x² + ∂²U/∂z² + k₀²n²U = 0

        의미:
          "빛은 파동 방정식을 만족해야 한다"
          PINN이 예측한 U(x,z)가 이 방정식을 얼마나 잘 만족하는지 측정.
          잔차(residual)가 0에 가까울수록 좋음.

        매개변수:
          U:  PINN이 예측한 파동 필드 (복소수 텐서)
          x:  x 좌표 (텐서, requires_grad=True)
          z:  z 좌표 (텐서, requires_grad=True)
          k0: 진공 파수 = 2π/λ
          n:  굴절률

        이 과목은 데이터 0개로 채점됩니다.
        순수하게 물리 법칙만으로 맞고 틀리고를 판단합니다.
        """
        # 1차 미분: ∂U/∂x (U가 x에 대해 얼마나 변하는지)
        Ux = torch.autograd.grad(
            U, x, torch.ones_like(U), create_graph=True
        )[0]

        # 2차 미분: ∂²U/∂x² (변화율의 변화율 = 곡률)
        Uxx = torch.autograd.grad(
            Ux, x, torch.ones_like(Ux), create_graph=True
        )[0]

        # z 방향도 동일하게
        Uz = torch.autograd.grad(
            U, z, torch.ones_like(U), create_graph=True
        )[0]
        Uzz = torch.autograd.grad(
            Uz, z, torch.ones_like(Uz), create_graph=True
        )[0]

        # 잔차: ∂²U/∂x² + ∂²U/∂z² + k²n²U  (이것이 0이어야 함)
        residual = Uxx + Uzz + (k0 * n) ** 2 * U

        # 잔차의 제곱 평균 → 이 값이 0에 가까울수록 물리 법칙을 잘 따름
        return torch.mean(residual ** 2)

    def phase(
        self,
        U_pred_phase: torch.Tensor,
        dphi_tmm: torch.Tensor
    ) -> torch.Tensor:
        """
        L_phase: AR 코팅 위상 경계조건 (코팅 시험)

        의미:
          AR 코팅 면(z=570um)에서 PINN이 예측한 위상이
          TMM으로 계산한 위상(dphi_tmm)과 얼마나 일치하는지 측정.

        매개변수:
          U_pred_phase: PINN이 AR 코팅 면에서 예측한 위상 (텐서)
          dphi_tmm:     TMM이 계산한 정답 위상 (텐서)

        ±θ 양방향 위상을 모두 포함합니다.
        이 과목도 데이터 0개입니다 (TMM은 물리 계산).
        """
        return torch.mean((U_pred_phase - dphi_tmm) ** 2)

    def intensity(
        self,
        U_pred: torch.Tensor,
        I_lt: torch.Tensor
    ) -> torch.Tensor:
        """
        L_I: 인텐시티 매칭 (실험 시험)

        의미:
          OPD 면(z=0)에서 PINN이 예측한 빛의 세기(|U|²)가
          LightTools 시뮬레이션 결과(I_lt)와 얼마나 일치하는지 측정.

        매개변수:
          U_pred: PINN이 OPD 면에서 예측한 파동 (복소수 텐서)
          I_lt:   LightTools가 계산한 빛의 세기 (텐서)

        7개 OPD 픽셀 모두에 적용됩니다.
        크로스토크 신호도 포함됩니다.

        이 과목만 실제 데이터(LightTools 200개)를 사용합니다.
        """
        # |U_pred|² = 빛의 세기 (복소수의 절대값의 제곱)
        I_pred = torch.abs(U_pred) ** 2
        return torch.mean((I_pred - I_lt) ** 2)

    def boundary(self, U_at_BM: torch.Tensor) -> torch.Tensor:
        """
        L_BC: BM 경계조건 (경계 시험)

        의미:
          BM(Black Matrix)은 불투명한 금속막입니다.
          BM이 있는 위치에서는 빛이 통과할 수 없으므로 U=0이어야 합니다.
          PINN이 BM 위치에서 U를 얼마나 0에 가깝게 예측하는지 측정.

        매개변수:
          U_at_BM: PINN이 BM 위치에서 예측한 파동 값들 (텐서)

        7피치 × (BM1 + BM2) = 14개 BM 영역에 적용.
        이 과목도 데이터 0개입니다 (물리적 사실).
        """
        return torch.mean(torch.abs(U_at_BM) ** 2)

    def total(
        self,
        Lh: torch.Tensor,
        Lp: torch.Tensor,
        Li: torch.Tensor,
        Lb: torch.Tensor
    ) -> dict:
        """
        4과목 종합 점수를 계산합니다.

        매개변수:
          Lh: L_Helmholtz (물리)
          Lp: L_phase (코팅)
          Li: L_I (실험)
          Lb: L_BC (경계)

        반환값:
          딕셔너리:
            total:  종합 점수 (이 값을 최소화하는 것이 학습 목표)
            helm:   물리 점수
            phase:  코팅 점수
            I:      실험 점수
            BC:     경계 점수

        종합 점수 공식:
          total = 1.0 * helm + 0.5 * phase + 0.3 * I + 0.2 * BC
        """
        t = (
            self.lam['helm'] * Lh
            + self.lam['phase'] * Lp
            + self.lam['I'] * Li
            + self.lam['BC'] * Lb
        )
        return dict(
            total=t,
            helm=Lh.item(),
            phase=Lp.item(),
            I=Li.item(),
            BC=Lb.item()
        )
