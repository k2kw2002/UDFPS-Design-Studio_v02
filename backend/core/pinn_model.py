"""
pinn_model.py - Helmholtz PINN (물리 정보 신경망)
=====================================================
이 파일은 "물리 법칙을 아는 AI"입니다.

PINN(Physics-Informed Neural Network)이란?
  일반 AI:  데이터만으로 학습 (데이터 수만개 필요)
  PINN:     데이터 + 물리 법칙으로 학습 (데이터 수백개로 충분)

비유:
  일반 AI = "기출문제만 풀어본 학생" → 새 문제에 약함
  PINN    = "교과서(물리법칙) + 기출문제" → 새 문제도 잘 풂

이 PINN이 하는 일:
  입력: (x, z) 좌표 → 출력: U(x,z) 파동 필드
  "이 위치에서 빛의 세기와 위상은 얼마인가?"

SIREN이란?
  일반 신경망: ReLU 활성화 → 고주파 패턴 표현 어려움
  SIREN:      sin() 활성화 → 파동 같은 주기적 패턴 표현에 특화

Fourier Feature Embedding이란?
  입력 좌표(x,z)를 sin/cos으로 변환해서 넣어주는 기법.
  "이 좌표가 파동의 어느 위치인지" AI가 더 잘 이해하게 해줍니다.

도메인:
  x: 0 ~ 504um (7피치)
  z: 0 ~ 570um (CG 550um + Encap 20um)
"""

import torch
import torch.nn as nn
import numpy as np
import math


# ============================================================
# Fourier Feature Embedding
# "좌표를 파동 언어로 번역해주는 번역기"
# ============================================================
class FourierFeatureEmbedding(nn.Module):
    """
    입력 좌표(x, z)를 sin/cos으로 변환합니다.

    왜 필요한가?
      신경망은 (x=100, z=200) 같은 숫자를 그냥 받으면
      "이게 파동의 어디쯤인지" 감을 못 잡습니다.

      하지만 sin(2π·x/λ), cos(2π·x/λ)로 변환하면
      "아, 이건 파장의 몇 배째 위치구나" 바로 이해합니다.

    매개변수:
      in_dim:      입력 차원 (x, z = 2차원)
      num_freqs:   사용할 주파수 개수 (많을수록 세밀한 패턴 표현)
      scale:       주파수 스케일 (파장에 맞게 조절)

    입력:  (x, z) → 2개 숫자
    출력:  sin(f1·x), cos(f1·x), sin(f2·x), ... → num_freqs×2×2개 숫자
    """

    def __init__(self, in_dim: int = 2, num_freqs: int = 64, scale: float = 10.0):
        super().__init__()
        # 랜덤 주파수 행렬 생성 (학습 중 고정)
        B = torch.randn(in_dim, num_freqs) * scale
        # register_buffer: 학습되지 않는 고정값으로 등록
        self.register_buffer('B', B)
        self.out_dim = num_freqs * 2  # sin + cos

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        좌표를 Fourier 특성으로 변환합니다.

        coords: (배치크기, 2) → (x, z) 좌표들
        반환:   (배치크기, num_freqs*2) → sin/cos 변환된 특성들
        """
        # 행렬 곱: coords @ B → 각 주파수에 대한 위상
        proj = coords @ self.B  # (배치, num_freqs)
        proj_2pi = 2 * math.pi * proj

        # sin과 cos을 나란히 붙임
        return torch.cat([torch.sin(proj_2pi), torch.cos(proj_2pi)], dim=-1)


# ============================================================
# SIREN Layer (sin 활성화 레이어)
# "파동을 잘 표현하는 특수 뉴런"
# ============================================================
class SirenLayer(nn.Module):
    """
    SIREN(Sinusoidal Representation Networks)의 한 층.

    일반 신경망:  y = ReLU(Wx + b)     → 직선적 패턴에 강함
    SIREN:       y = sin(ω₀ · (Wx + b)) → 파동 패턴에 강함

    ω₀ (omega_0):
      "얼마나 빠르게 진동하는지" 조절하는 값.
      ω₀=30: 기본값. 파장 수준의 진동을 잘 표현.

    매개변수:
      in_features:  입력 크기
      out_features: 출력 크기
      omega_0:      진동 주파수 (기본 30.0)
      is_first:     첫 번째 층인지 여부 (초기화 방법이 다름)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # 가중치 초기화 (SIREN 논문의 특수 초기화)
        # 첫 층과 나머지 층의 초기화 방법이 다릅니다
        with torch.no_grad():
            if is_first:
                # 첫 층: 균등 분포 [-1/n, 1/n]
                bound = 1.0 / in_features
            else:
                # 나머지: 균등 분포 [-sqrt(6/n)/ω₀, sqrt(6/n)/ω₀]
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """sin(ω₀ · (Wx + b))"""
        return torch.sin(self.omega_0 * self.linear(x))


# ============================================================
# Helmholtz PINN (메인 모델)
# "파동 방정식을 푸는 AI"
# ============================================================
class HelmholtzPINN(nn.Module):
    """
    Helmholtz 방정식을 푸는 PINN.

    구조:
      입력 (x, z) → Fourier Embedding → SIREN 층들 → 출력 U(x,z)

      [x, z]         (2개 숫자)
        ↓
      [Fourier]      (128개 sin/cos 특성)
        ↓
      [SIREN 층 1]   (256개 뉴런, sin 활성화)
        ↓
      [SIREN 층 2]   (256개 뉴런, sin 활성화)
        ↓
      [SIREN 층 3]   (256개 뉴런, sin 활성화)
        ↓
      [SIREN 층 4]   (256개 뉴런, sin 활성화)
        ↓
      [출력 층]       (2개: 실수부 Re(U), 허수부 Im(U))

    매개변수:
      hidden_dim:    은닉층 뉴런 수 (기본 256)
      num_layers:    은닉층 수 (기본 4)
      num_freqs:     Fourier 주파수 수 (기본 64)
      omega_0:       SIREN 진동 주파수 (기본 30.0)

    사용법:
        model = HelmholtzPINN()
        x = torch.tensor([[100.0, 300.0]])  # (x=100um, z=300um)
        U = model(x)                         # 복소수 파동 값
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_freqs: int = 64,
        omega_0: float = 30.0
    ):
        super().__init__()

        # 1. Fourier Feature Embedding (좌표 → 파동 특성)
        self.embedding = FourierFeatureEmbedding(
            in_dim=2, num_freqs=num_freqs, scale=10.0
        )

        # 2. SIREN 네트워크 구축
        layers = []

        # 첫 번째 층: Fourier 출력(num_freqs*2) → hidden_dim
        layers.append(SirenLayer(
            self.embedding.out_dim, hidden_dim,
            omega_0=omega_0, is_first=True
        ))

        # 중간 층들: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(
                hidden_dim, hidden_dim, omega_0=omega_0
            ))

        self.network = nn.Sequential(*layers)

        # 3. 출력 층: hidden_dim → 2 (실수부, 허수부)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        좌표를 받아서 파동 필드 U(x,z)를 예측합니다.

        매개변수:
            coords: (배치크기, 2) 텐서. 각 행 = [x, z] (um 단위)

        반환값:
            U: (배치크기, 2) 텐서. 각 행 = [Re(U), Im(U)]
               Re(U) = 실수부, Im(U) = 허수부
               |U|² = Re² + Im² = 빛의 세기
        """
        # 1단계: 좌표를 Fourier 특성으로 변환
        features = self.embedding(coords)

        # 2단계: SIREN 네트워크 통과
        hidden = self.network(features)

        # 3단계: 실수부/허수부 출력
        output = self.output_layer(hidden)

        return output

    def predict_complex(self, coords: torch.Tensor) -> torch.Tensor:
        """
        복소수 형태로 파동 필드를 반환합니다.

        반환값:
            U: (배치크기,) 복소수 텐서
               U = Re(U) + i·Im(U)
        """
        out = self.forward(coords)
        return torch.complex(out[:, 0], out[:, 1])

    def predict_intensity(self, coords: torch.Tensor) -> torch.Tensor:
        """
        빛의 세기(인텐시티)를 반환합니다.

        반환값:
            I: (배치크기,) 텐서
               I = |U|² = Re(U)² + Im(U)²
        """
        out = self.forward(coords)
        return out[:, 0] ** 2 + out[:, 1] ** 2
