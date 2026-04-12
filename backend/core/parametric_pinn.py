"""
parametric_pinn.py - Carrier-Envelope Parametric PINN
======================================================
u(x,z; theta, params) = exp(i*kx*sin(theta)*x) * A(x,z; theta, params)

캐리어(고주파 평면파)는 해석적으로 분리.
PINN은 느린 포락선 A(x,z)만 학습.

포락선 A: BM 회절이 만드는 부드러운 변화 (~10um 스케일)
캐리어:   파장 0.52um의 고주파 진동 → 해석적 처리

L_phase 타겟: A(x,z_top) = t(theta) → 상수! (고주파 문제 해소)
"""

import math
import numpy as np
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_dim=8, num_freqs=64, scale=10.0):
        super().__init__()
        # 차원별 스케일: 공간(고주파), theta(저주파)
        if in_dim == 8:
            dim_scales = torch.tensor([scale, scale, scale*0.5, scale*0.5,
                                       scale*0.5, scale*0.5, 2.0, 2.0])
            B = torch.randn(in_dim, num_freqs) * dim_scales.unsqueeze(1)
        else:
            B = torch.randn(in_dim, num_freqs) * scale
        self.register_buffer('B', B)
        self.out_dim = num_freqs * 2

    def forward(self, x):
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
    Carrier-Envelope PINN.

    출력: A(x,z; theta, params) = 포락선 (Re, Im)
    전체 해: u = exp(i*kx*sin(theta)*x) * A

    입력 8D: (x, z, d1, d2, w1, w2, sin_theta, cos_theta)
    """

    def __init__(self, hidden_dim=256, num_layers=5, num_freqs=64, omega_0=30.0):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(in_dim=8, num_freqs=num_freqs)

        layers = []
        layers.append(SirenLayer(self.embedding.out_dim, hidden_dim, omega_0, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 2)

        # 정규화 (sin/cos는 그대로)
        self.register_buffer('x_scale', torch.tensor([504.0, 40.0, 20.0, 20.0, 15.0, 15.0, 1.0, 1.0]))
        self.register_buffer('x_offset', torch.tensor([252.0, 20.0, 0.0, 0.0, 12.5, 12.5, 0.0, 0.0]))

        # 파수
        self.k = 2 * math.pi / 0.520 * 1.52  # k0 * n_CG

    def forward_envelope(self, coords_params):
        """포락선 A(x,z;theta,params) 출력."""
        normalized = (coords_params - self.x_offset) / (self.x_scale / 2)
        features = self.embedding(normalized)
        hidden = self.network(features)
        return self.output_layer(hidden)

    def forward(self, coords_params):
        """전체 해 u = carrier * envelope. 양방향 캐리어 분리."""
        A = self.forward_envelope(coords_params)
        x = coords_params[:, 0]
        z = coords_params[:, 1]
        sin_th = coords_params[:, 6]
        cos_th = coords_params[:, 7]
        kx = self.k * sin_th
        kz = self.k * cos_th
        # Carrier reference at z=40 (top boundary)
        # → A(z=40) = t(θ) (no exp(-i·kz·40) phase, easily representable)
        carrier_phase = kx * x + kz * (z - 40.0)
        carrier_re = torch.cos(carrier_phase)
        carrier_im = torch.sin(carrier_phase)
        u_re = carrier_re * A[:, 0] - carrier_im * A[:, 1]
        u_im = carrier_re * A[:, 1] + carrier_im * A[:, 0]
        return torch.stack([u_re, u_im], dim=1)

    def predict_intensity(self, coords_params):
        """|u|^2 = |A|^2 (carrier는 진폭 1이므로)."""
        A = self.forward_envelope(coords_params)
        return A[:, 0] ** 2 + A[:, 1] ** 2

    def predict_psf7(self, d1, d2, w1, w2, device='cpu', n_angles=7):
        """설계변수 → 7-OPD PSF (다각도 비간섭 합)."""
        angles = [0, 15, -15, 30, -30, 41, -41][:n_angles]
        pitch = 72.0
        n_pitches = 7
        self.eval()
        total_psf = np.zeros(n_pitches)

        with torch.no_grad():
            for theta in angles:
                th_rad = math.radians(theta)
                sin_th = math.sin(th_rad)
                cos_th = math.cos(th_rad)
                for i in range(n_pitches):
                    x_opd = i * pitch + pitch / 2
                    inp = torch.tensor(
                        [[x_opd, 0.0, d1, d2, w1, w2, sin_th, cos_th]],
                        dtype=torch.float32, device=torch.device(device)
                    )
                    I = self.predict_intensity(inp)
                    total_psf[i] += I.item()
        total_psf /= len(angles)
        return total_psf
