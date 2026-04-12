"""
fno_model.py - FNO Surrogate (빠른 대리 모델)
================================================
이 파일은 "PINN의 요약본"입니다.

PINN vs FNO:
  PINN: 정확하지만 느림 (수 분 ~ 수십 분)
  FNO:  PINN만큼 정확하면서 빠름 (~0.8ms)

비유:
  PINN = "수학 시험을 한 문제씩 꼼꼼히 푸는 것"   → 정확하지만 느림
  FNO  = "PINN이 만든 답안지를 암기한 것"           → 빠르게 답을 냄

FNO(Fourier Neural Operator)란?
  주파수 영역에서 학습하는 특수 신경망.
  일반 신경망이 "점 → 점" 매핑을 배우는 반면,
  FNO는 "함수 → 함수" 매핑을 배웁니다.

이 FNO가 하는 일:
  입력: 설계변수 p = (delta_bm1, delta_bm2, w1, w2)  (4개 숫자)
  출력: PSF 7개 OPD 값                                 (7개 숫자)

증류(Distillation):
  PINN으로 10,000개 케이스를 미리 계산해서
  FNO에게 "이 입력이면 이 출력이다"를 가르칩니다.
  학생(FNO)이 선생(PINN)의 지식을 물려받는 것.
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# Fourier Layer (FNO의 핵심 블록)
# "주파수 영역에서 계산하는 특수 층"
# ============================================================
class FourierLayer(nn.Module):
    """
    FNO의 핵심 구성요소: Fourier Layer.

    동작 원리:
      1. 입력을 FFT로 주파수 분해
      2. 주파수 영역에서 가중치 곱셈 (학습되는 부분)
      3. IFFT로 복원
      4. 일반 선형 변환과 더함

    비유:
      일반 층:   "각 점을 하나씩 처리"
      Fourier층: "전체 패턴(주파수)을 한번에 처리"

      음악으로 비유하면:
      일반 층 = "한 음씩 듣고 판단"
      Fourier층 = "전체 화음(주파수 스펙트럼)을 한번에 듣고 판단"

    매개변수:
      width:     채널 수 (은닉 차원)
      modes:     사용할 Fourier 모드 수 (주파수 성분 개수)
    """

    def __init__(self, width: int, modes: int = 8):
        super().__init__()
        self.width = width
        self.modes = modes

        # 주파수 영역 가중치 (복소수, 학습됨)
        # shape: (modes, width_in, width_out) — einsum 'bmi,mio->bmo'
        scale = 1.0 / (width * width)
        self.weights = nn.Parameter(
            scale * torch.randn(modes, width, width, dtype=torch.cfloat)
        )

        # 일반 선형 변환 (바이패스 경로)
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력: (배치, 길이, 채널)
        출력: (배치, 길이, 채널)
        """
        batch_size, seq_len, _ = x.shape

        # --- 경로 1: 주파수 영역 처리 ---

        # 1. FFT: 시간 → 주파수
        x_ft = torch.fft.rfft(x, dim=1)

        # 2. 주파수 가중치 곱셈 (낮은 주파수 modes개만 사용)
        #    높은 주파수는 노이즈이므로 무시
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x_ft.shape[1])
        out_ft[:, :modes, :] = torch.einsum(
            'bmi,mio->bmo',
            x_ft[:, :modes, :].to(torch.cfloat),
            self.weights[:modes, :, :]
        )

        # 3. IFFT: 주파수 → 시간
        x_fourier = torch.fft.irfft(out_ft, n=seq_len, dim=1)

        # --- 경로 2: 일반 선형 변환 (바이패스) ---
        x_linear = self.linear(x)

        # --- 두 경로를 더하고 활성화 ---
        return torch.nn.functional.gelu(x_fourier + x_linear)


# ============================================================
# FNO Surrogate (메인 모델)
# "설계변수 → PSF 7개"를 빠르게 예측하는 모델
# ============================================================
class FNOSurrogate(nn.Module):
    """
    FNO 기반 Surrogate 모델.

    구조:
      설계변수 p (4개)
        ↓
      [Lifting]       4 → 64 (차원 확장)
        ↓
      [Fourier 층 1]  64 → 64
        ↓
      [Fourier 층 2]  64 → 64
        ↓
      [Fourier 층 3]  64 → 64
        ↓
      [Fourier 층 4]  64 → 64
        ↓
      [Projection]    64 → 7 (PSF 7개 OPD)

    매개변수:
      in_dim:      입력 차원 (설계변수 4개)
      out_dim:     출력 차원 (PSF 7개 OPD)
      width:       채널 수 (기본 64)
      num_layers:  Fourier 층 수 (기본 4)
      modes:       Fourier 모드 수 (기본 8)

    사용법:
        model = FNOSurrogate()
        p = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # 설계변수
        psf7 = model(p)                               # PSF 7개
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 7,
        width: int = 64,
        num_layers: int = 4,
        modes: int = 8
    ):
        super().__init__()

        self.out_dim = out_dim

        # 1. Lifting: 입력 차원 확장 (4 → width)
        #    "설계변수 4개를 더 풍부한 표현으로 변환"
        self.lifting = nn.Linear(in_dim, width * out_dim)

        # 2. Fourier 층들
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes) for _ in range(num_layers)
        ])

        # 3. Projection: 출력 차원 축소 (width → 1, 각 OPD당)
        #    "풍부한 표현을 하나의 인텐시티 값으로 요약"
        self.projection = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1)
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        설계변수로부터 PSF 7개 값을 예측합니다.

        매개변수:
            p: (배치크기, 4) 텐서
               각 행 = [delta_bm1, delta_bm2, w1, w2]

        반환값:
            psf7: (배치크기, 7) 텐서
                  각 행 = 7개 OPD 픽셀의 인텐시티
        """
        batch_size = p.shape[0]

        # 1. Lifting: (배치, 4) → (배치, 7, width)
        h = self.lifting(p)
        h = h.view(batch_size, self.out_dim, -1)

        # 2. Fourier 층들 통과
        for layer in self.fourier_layers:
            h = layer(h)

        # 3. Projection: (배치, 7, width) → (배치, 7)
        psf7 = self.projection(h).squeeze(-1)

        # 4. 인텐시티는 항상 양수 → softplus 적용
        #    softplus(x) = log(1 + exp(x)), 항상 > 0
        psf7 = torch.nn.functional.softplus(psf7)

        return psf7
