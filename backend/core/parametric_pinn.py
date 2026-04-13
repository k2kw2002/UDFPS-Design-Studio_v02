"""
parametric_pinn.py - Carrier-Envelope Parametric PINN
======================================================
u(x,z; theta, params) = exp(i*kx*sin(theta)*x) * A(x,z; theta, params)

캐리어(고주파 평면파)는 해석적으로 분리.
PINN은 느린 포락선 A(x,z)만 학습.

입력 9D: (x, z, d1, d2, w1, w2, sin_theta, cos_theta, slit_dist)
slit_dist: 부호 있는 슬릿 거리 (양=슬릿 안, 음=BM 영역)
"""

import math
import numpy as np
import torch
import torch.nn as nn


OPD_PITCH = 72.0
N_PITCHES = 7
Z_BM1 = 40.0
Z_BM2 = 20.0


def compute_slit_dist(x, z, d1, d2, w1, w2):
    """
    부호 있는 슬릿 거리: +면 슬릿 안, -면 BM 불투명.
    |값|은 가장 가까운 슬릿 경계까지의 거리.
    z=40 근처에선 (d1, w1), z=20 근처에선 (d2, w2) 사용.
    BM 평면에서 먼 z에선 큰 양수 반환 (투명 영역).
    """
    pitch_idx = torch.floor(x / OPD_PITCH).clamp(0, N_PITCHES - 1)
    center_x = pitch_idx * OPD_PITCH + OPD_PITCH / 2

    # BM1 (z~40): slit edge distance
    edge_dist1 = w1 / 2 - torch.abs(x - center_x - d1)
    bm1_z = torch.exp(-((z - Z_BM1) / 2.0) ** 2)

    # BM2 (z~20): slit edge distance
    edge_dist2 = w2 / 2 - torch.abs(x - center_x - d2)
    bm2_z = torch.exp(-((z - Z_BM2) / 2.0) ** 2)

    # BM 평면에서 멀면 투명 (큰 양수)
    far_from_bm = 20.0 * (1.0 - bm1_z - bm2_z).clamp(min=0)

    return bm1_z * edge_dist1 + bm2_z * edge_dist2 + far_from_bm


def compute_bm_mask(slit_dist, sharpness=2.0):
    """
    Hard BM mask: sigmoid(slit_dist).
    슬릿 안(slit_dist>0) → ~1, BM(slit_dist<0) → ~0.
    """
    return torch.sigmoid(sharpness * slit_dist)


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_dim=9, num_freqs=64, scale=10.0):
        super().__init__()
        if in_dim == 9:
            # 9D: x, z, d1, d2, w1, w2, sin_th, cos_th, slit_dist
            dim_scales = torch.tensor([scale, scale, scale*0.5, scale*0.5,
                                       scale*0.5, scale*0.5, 2.0, 2.0, 5.0])
        elif in_dim == 8:
            dim_scales = torch.tensor([scale, scale, scale*0.5, scale*0.5,
                                       scale*0.5, scale*0.5, 2.0, 2.0])
        else:
            dim_scales = torch.ones(in_dim) * scale
        B = torch.randn(in_dim, num_freqs) * dim_scales.unsqueeze(1)
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
    Carrier-Envelope PINN with BM slit indicator.

    출력: A(x,z; theta, params) = 포락선 (Re, Im)
    전체 해: u = exp(i*kx*sin(theta)*x) * A

    입력 9D: (x, z, d1, d2, w1, w2, sin_theta, cos_theta, slit_dist)
    """

    def __init__(self, hidden_dim=256, num_layers=5, num_freqs=64, omega_0=30.0,
                 in_dim=9, use_bm_mask=True, mask_sharpness=2.0):
        super().__init__()
        self.in_dim = in_dim
        self.use_bm_mask = use_bm_mask
        self.mask_sharpness = mask_sharpness

        self.embedding = FourierFeatureEmbedding(in_dim=in_dim, num_freqs=num_freqs)

        layers = []
        layers.append(SirenLayer(self.embedding.out_dim, hidden_dim, omega_0, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 2)

        # 정규화 (9D)
        if in_dim == 9:
            self.register_buffer('x_scale', torch.tensor(
                [504.0, 40.0, 20.0, 20.0, 15.0, 15.0, 1.0, 1.0, 40.0]))
            self.register_buffer('x_offset', torch.tensor(
                [252.0, 20.0, 0.0, 0.0, 12.5, 12.5, 0.0, 0.0, 0.0]))
        else:
            self.register_buffer('x_scale', torch.tensor(
                [504.0, 40.0, 20.0, 20.0, 15.0, 15.0, 1.0, 1.0]))
            self.register_buffer('x_offset', torch.tensor(
                [252.0, 20.0, 0.0, 0.0, 12.5, 12.5, 0.0, 0.0]))

        # 파수
        self.k = 2 * math.pi / 0.520 * 1.52  # k0 * n_CG

    def forward_envelope(self, coords_params):
        """포락선 A(x,z;theta,params) 출력. BM mask 적용."""
        normalized = (coords_params - self.x_offset) / (self.x_scale / 2)
        features = self.embedding(normalized)
        hidden = self.network(features)
        A = self.output_layer(hidden)

        if self.use_bm_mask and self.in_dim == 9:
            slit_dist = coords_params[:, 8]
            mask = compute_bm_mask(slit_dist, self.mask_sharpness)
            A = A * mask.unsqueeze(1)  # (N,2) * (N,1)

        return A

    def forward(self, coords_params):
        """전체 해 u = carrier * envelope."""
        A = self.forward_envelope(coords_params)
        x = coords_params[:, 0]
        z = coords_params[:, 1]
        sin_th = coords_params[:, 6]
        cos_th = coords_params[:, 7]
        kx = self.k * sin_th
        kz = self.k * cos_th
        carrier_phase = kx * x + kz * (z - 40.0)
        carrier_re = torch.cos(carrier_phase)
        carrier_im = torch.sin(carrier_phase)
        u_re = carrier_re * A[:, 0] - carrier_im * A[:, 1]
        u_im = carrier_re * A[:, 1] + carrier_im * A[:, 0]
        return torch.stack([u_re, u_im], dim=1)

    def predict_intensity(self, coords_params):
        """|u|^2 = |A|^2."""
        A = self.forward_envelope(coords_params)
        return A[:, 0] ** 2 + A[:, 1] ** 2

    def predict_psf7(self, d1, d2, w1, w2, device='cpu', n_angles=7):
        """
        설계변수 → 7-OPD PSF (다각도 비간섭 합).

        하이브리드 파이프라인:
          ASM(지문+AR+CG 550um) → PINN BM1 soft mask → ASM(ILD 20um)
          → BM2 binary mask → ASM(Encap 20um) → OPD 적분

        PINN 기여: BM1에서 slit/BM 분화 (soft mask, 회절 효과 포함).
        ASM 기여: 지문 패턴, CG blur, 자유공간 전파.
        """
        from backend.physics.asm_propagator import ASMPropagator
        from backend.physics.ar_coating.ar_boundary import ARLutInterpolator

        angles = [0, 15, -15, 30, -30, 41, -41][:n_angles]
        pitch = OPD_PITCH
        n_pitches = N_PITCHES
        self.eval()
        asm = ASMPropagator(wl_um=0.520, n_medium=1.52)
        ar_lut = ARLutInterpolator()

        # x 그리드 (dx=1um, 504 points — ASM과 동일)
        N_pts = 504
        dx = 1.0
        x_arr = np.arange(N_pts) * dx

        # 지문 패턴 (ridge=1.0, valley=0.1, period=288um)
        finger = np.ones(N_pts) * 0.1
        for i in range(N_pts):
            if (x_arr[i] % 288) < 144:
                finger[i] = 1.0

        # BM1 soft mask from PINN (slit indicator at z=40)
        x_t = torch.tensor(x_arr, dtype=torch.float32)
        z40_t = torch.full((N_pts,), Z_BM1)
        d1_t = torch.full((N_pts,), float(d1))
        d2_t = torch.full((N_pts,), float(d2))
        w1_t = torch.full((N_pts,), float(w1))
        w2_t = torch.full((N_pts,), float(w2))
        sd40 = compute_slit_dist(x_t, z40_t, d1_t, d2_t, w1_t, w2_t)
        bm1_mask = compute_bm_mask(sd40, self.mask_sharpness).numpy()

        # BM2 binary mask
        bm2_mask = np.zeros(N_pts)
        for p in range(n_pitches):
            center = p * pitch + pitch / 2 + d2
            i_left = max(0, int(center - w2 / 2))
            i_right = min(N_pts, int(center + w2 / 2))
            bm2_mask[i_left:i_right] = 1.0

        total_psf = np.zeros(n_pitches)

        for theta in angles:
            # Step 1: ASM — 지문 → AR → CG 550um → z=40 (BM1 면)
            t_amp, dphi = ar_lut.get_complex_t(520, theta, unpolarized=True)
            U0 = asm.make_incident_field(x_arr, theta, dphi) * finger * t_amp
            U_bm1 = asm.propagate_1d(U0, dx, 550.0)

            # Step 2: PINN BM1 soft mask (slit=~1, BM=~0)
            U_after_bm1 = U_bm1 * bm1_mask

            # Step 3: ASM z=40 → z=20 (ILD 20um)
            U_at_bm2 = asm.propagate_1d(U_after_bm1, dx, 20.0)

            # Step 4: BM2 binary mask
            U_after_bm2 = U_at_bm2 * bm2_mask

            # Step 5: ASM z=20 → z=0 (Encap 20um)
            U_z0 = asm.propagate_1d(U_after_bm2, dx, 20.0)
            intensity = np.abs(U_z0) ** 2

            # Step 6: OPD 픽셀 적분 (10um 폭)
            for i in range(n_pitches):
                opd_center = i * pitch + pitch / 2
                i_left = max(0, int(opd_center - 5))
                i_right = min(N_pts, int(opd_center + 5))
                total_psf[i] += intensity[i_left:i_right].sum() * dx

        total_psf /= len(angles)
        return total_psf
