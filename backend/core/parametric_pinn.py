"""
parametric_pinn.py - Parametric Helmholtz PINN
================================================
설계변수 (δ_BM1, δ_BM2, w₁, w₂)를 추가 입력으로 받아
모든 BM 설계에 대해 단일 모델로 PSF를 예측.

기존 PINN:  입력 (x, z) → U(x, z)         설계마다 별도 학습
Parametric: 입력 (x, z, δ₁, δ₂, w₁, w₂) → U(x, z; p)  한 번 학습

장점:
  - 학습 1회 → 모든 설계변수에서 PSF 추론 가능
  - FNO 증류 불필요 (직접 BoTorch에 연결 가능)
  - 단, 학습 시간 더 걸림 (입력 차원 증가)
"""

import math
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """좌표 + 설계변수를 Fourier 특성으로 변환."""

    def __init__(self, in_dim: int = 6, num_freqs: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_freqs) * scale
        self.register_buffer('B', B)
        self.out_dim = num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ParametricHelmholtzPINN(nn.Module):
    """
    Parametric PINN: (x, z, δ₁, δ₂, w₁, w₂) → U(x, z; p)

    입력 6차원:
      x, z:        좌표 (um)
      δ₁, δ₂:     BM 오프셋 (um), [-10, +10]으로 정규화
      w₁, w₂:     BM 아퍼처 폭 (um), [5, 20]으로 정규화

    출력 2차원: [Re(U), Im(U)]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_freqs: int = 64,
        omega_0: float = 30.0,
    ):
        super().__init__()

        # 입력: (x, z, δ₁, δ₂, w₁, w₂) = 6차원
        self.embedding = FourierFeatureEmbedding(in_dim=6, num_freqs=num_freqs)

        layers = []
        layers.append(SirenLayer(self.embedding.out_dim, hidden_dim, omega_0, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
        self.network = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dim, 2)

        # 정규화 상수 (40um BM 도메인)
        # x: [0, 504], z: [0, 40], d1: [-10,10], d2: [-10,10], w1: [5,20], w2: [5,20]
        self.register_buffer('x_scale', torch.tensor([504.0, 40.0, 20.0, 20.0, 15.0, 15.0]))
        self.register_buffer('x_offset', torch.tensor([252.0, 20.0, 0.0, 0.0, 12.5, 12.5]))

    def forward(self, coords_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_params: (batch, 6) = [x, z, δ₁, δ₂, w₁, w₂]

        Returns:
            (batch, 2) = [Re(U), Im(U)]
        """
        # 정규화: [0,504]→[-1,1], [-10,10]→[-1,1], [5,20]→[-1,1]
        normalized = (coords_params - self.x_offset) / (self.x_scale / 2)
        features = self.embedding(normalized)
        hidden = self.network(features)
        return self.output_layer(hidden)

    def predict_intensity(self, coords_params: torch.Tensor) -> torch.Tensor:
        out = self.forward(coords_params)
        return out[:, 0] ** 2 + out[:, 1] ** 2

    def predict_psf7(
        self, delta_bm1, delta_bm2, w1, w2,
        device='cpu'
    ):
        """설계변수 → 7-OPD PSF."""
        import numpy as np
        self.eval()
        psf = np.zeros(7)
        with torch.no_grad():
            for i in range(7):
                x_opd = (i - 3) * 72.0 + 36.0
                inp = torch.tensor(
                    [[x_opd, 0.0, delta_bm1, delta_bm2, w1, w2]],
                    dtype=torch.float32, device=device
                )
                I = self.predict_intensity(inp)
                psf[i] = I.item()
        return psf
