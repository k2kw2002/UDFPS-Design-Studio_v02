"""
train_phase_b.py - 3-Stage Envelope PINN Training
==================================================
Lh/Lp 스케일 불균형 해결 (Lh~5000 vs Lp~1):

  Stage 1 (0~S1):       L_phase only → |A|=0.77 확립
  Stage 2 (S1~S2):      L_PDE λ=1e-4 도입 → baseline 유지
  Stage 3 (S2~end):     L_PDE 점진 1e-4→1e-2 + L_BC → BM 회절

모든 손실은 envelope A에 대해 계산 (forward_envelope 사용).
Envelope PDE: ∇²A + 2i(kx·∂A/∂x + kz·∂A/∂z) + source·A = 0

Usage:
  python train_phase_b.py --epochs 3000 --device cpu
  python train_phase_b.py --epochs 10000 --device cuda
"""

import sys, math, time, logging, argparse
import numpy as np
import torch
import torch.optim as optim

from backend.core.parametric_pinn import ParametricHelmholtzPINN
from backend.physics.ar_coating.ar_boundary import ARLutInterpolator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--n_phase', type=int, default=500)
parser.add_argument('--n_colloc', type=int, default=5000)
parser.add_argument('--n_bm', type=int, default=300)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--lr', type=float, default=1e-3)
# Stage boundaries
parser.add_argument('--stage1_end', type=int, default=500)
parser.add_argument('--stage2_end', type=int, default=1500)
# Loss weights
parser.add_argument('--lam_phase', type=float, default=5.0)
parser.add_argument('--lam_pde_min', type=float, default=1e-4,
                    help='Stage 2 PDE weight (Lh~5000이므로 1e-4 → 실효 ~0.5)')
parser.add_argument('--lam_pde_max', type=float, default=1e-2,
                    help='Stage 3 최대 PDE weight')
parser.add_argument('--lam_bc', type=float, default=1.0)
parser.add_argument('--normalize', action='store_true', default=True,
                    help='Dynamic loss normalization (L/L.detach())')
parser.add_argument('--norm_floor', type=float, default=1e-4,
                    help='Floor for normalization denominator')
# BM source
parser.add_argument('--bm_strength', type=float, default=-1.0,
                    help='BM absorption coefficient (음수)')
parser.add_argument('--bm_sharpness', type=float, default=2.0,
                    help='BM indicator sigmoid sharpness (um^-1)')
args = parser.parse_args()

# ============================================================
# Constants
# ============================================================
DOMAIN_X = (0.0, 504.0)
DOMAIN_Z = (0.0, 40.0)
Z_BM1 = 40.0
Z_BM2 = 20.0
Z_OPD = 0.0
OPD_PITCH = 72.0
N_PITCHES = 7
WL_UM = 0.520
K0 = 2 * math.pi / WL_UM
N_CG = 1.52
K = K0 * N_CG  # 매질 내 파수

# Device
if args.device == 'cuda' and not torch.cuda.is_available():
    log.warning("CUDA not available, falling back to CPU")
    device = torch.device('cpu')
else:
    device = torch.device(args.device)
log.info(f"Device: {device}")


# ============================================================
# Helpers
# ============================================================
def rand_range(n, lo, hi, dev):
    return torch.rand(n, device=dev) * (hi - lo) + lo


def sample_theta(n):
    """sin(θ) 균일 샘플링."""
    sin_max = math.sin(math.radians(41.1))
    sin_th = rand_range(n, -sin_max, sin_max, device)
    cos_th = torch.sqrt(1 - sin_th ** 2)
    return sin_th, cos_th


# ============================================================
# Envelope Phase BC: A(x, z=40) = t(θ) * exp(-i·kz·z_top)
# ============================================================
def get_envelope_phase_bc(n, slit_only=False):
    """
    Envelope L_phase: z=40에서 A = t(θ).
    carrier reference가 z=40이므로 carrier(x,40) = exp(i·kx·x).
    u(x,40) = carrier·A = exp(i·kx·x)·t(θ) = t(θ)·exp(i·kx·x). ✓
    |A| = |t(θ)| ≈ 0.77. Target은 θ에만 느리게 의존.

    slit_only=True: BM1 슬릿 내부에서만 샘플 (불투명 영역 제외).
    물리적으로 빛은 슬릿을 통해서만 진입하므로 이것이 정확.
    """
    ar_lut = ARLutInterpolator()

    if slit_only:
        # Rejection sampling: BM1 슬릿 내부만
        all_coords, all_re, all_im = [], [], []
        collected = 0
        while collected < n:
            batch = max(n * 5, 500)
            x = rand_range(batch, 0, 504, device)
            d1 = rand_range(batch, -10, 10, device)
            d2 = rand_range(batch, -10, 10, device)
            w1 = rand_range(batch, 5, 20, device)
            w2 = rand_range(batch, 5, 20, device)
            sin_th, cos_th = sample_theta(batch)

            pitch_idx = (x / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
            center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2 + d1
            in_slit = (x >= center - w1 / 2) & (x <= center + w1 / 2)

            if in_slit.sum() == 0:
                continue

            valid = in_slit.nonzero(as_tuple=True)[0]
            z_vals = torch.full((len(valid),), Z_BM1, device=device)

            coords = torch.stack([x[valid], z_vals, d1[valid], d2[valid],
                                  w1[valid], w2[valid],
                                  sin_th[valid], cos_th[valid]], dim=1)

            re = torch.zeros(len(valid), device=device)
            im = torch.zeros(len(valid), device=device)
            for j, idx in enumerate(valid):
                theta_rad = math.asin(sin_th[idx].item())
                theta_deg = math.degrees(theta_rad)
                t_amp, dphi_deg = ar_lut.get_complex_t(WL_UM * 1000, theta_deg, unpolarized=True)
                envelope_phase = math.radians(dphi_deg)
                re[j] = t_amp * math.cos(envelope_phase)
                im[j] = t_amp * math.sin(envelope_phase)

            all_coords.append(coords)
            all_re.append(re)
            all_im.append(im)
            collected += len(valid)

        return torch.cat(all_coords)[:n], torch.cat(all_re)[:n], torch.cat(all_im)[:n]

    else:
        # z=40 전체 x에서 샘플 (Stage 1, 2 용)
        x_vals = rand_range(n, 0, 504, device)
        z_vals = torch.full((n,), Z_BM1, device=device)
        d1 = rand_range(n, -10, 10, device)
        d2 = rand_range(n, -10, 10, device)
        w1 = rand_range(n, 5, 20, device)
        w2 = rand_range(n, 5, 20, device)
        sin_th, cos_th = sample_theta(n)

        coords = torch.stack([x_vals, z_vals, d1, d2, w1, w2, sin_th, cos_th], dim=1)
        target_re = torch.zeros(n, device=device)
        target_im = torch.zeros(n, device=device)

        for i in range(n):
            theta_rad = math.asin(sin_th[i].item())
            theta_deg = math.degrees(theta_rad)
            t_amp, dphi_deg = ar_lut.get_complex_t(WL_UM * 1000, theta_deg, unpolarized=True)
            envelope_phase = math.radians(dphi_deg)
            target_re[i] = t_amp * math.cos(envelope_phase)
            target_im[i] = t_amp * math.sin(envelope_phase)

        return coords, target_re, target_im


# ============================================================
# Envelope Phase Loss
# ============================================================
def envelope_phase_loss(model, coords, target_re, target_im):
    """A at z=40 vs t(θ)·exp(-i·kz·40)."""
    A = model.forward_envelope(coords)
    return torch.mean((A[:, 0] - target_re) ** 2 + (A[:, 1] - target_im) ** 2)


# ============================================================
# BM Smooth Indicator
# ============================================================
def compute_bm_indicator(x, z, d1, d2, w1, w2, sharpness=2.0):
    """
    BM 불투명 영역 indicator [0,1].
    1 = opaque (BM), 0 = transparent (slit).
    BM1(z≈40), BM2(z≈20) 근처에서만 활성.
    """
    # z 방향: BM 평면 근처에서만 활성 (Gaussian, σ=2um)
    bm1_z = torch.exp(-((z - Z_BM1) / 2.0) ** 2)
    bm2_z = torch.exp(-((z - Z_BM2) / 2.0) ** 2)

    # 어느 BM인지에 따라 d, w 선택
    bm_z_total = bm1_z + bm2_z + 1e-8
    d = (d1 * bm1_z + d2 * bm2_z) / bm_z_total
    w = (w1 * bm1_z + w2 * bm2_z) / bm_z_total

    # x 방향: 슬릿 바깥이면 opaque
    pitch_idx = torch.floor(x / OPD_PITCH).clamp(0, N_PITCHES - 1)
    center = pitch_idx * OPD_PITCH + OPD_PITCH / 2 + d
    dist = torch.abs(x - center)
    slit_open = torch.sigmoid(sharpness * (w / 2 - dist))  # 1 inside slit
    bm_opaque = 1.0 - slit_open  # 1 outside slit

    # BM 평면 근처에서만 활성
    bm_zone = (bm1_z + bm2_z).clamp(0, 1)
    return bm_opaque * bm_zone


# ============================================================
# Envelope PDE Loss (with optional BM source)
# ============================================================
def envelope_pde_loss(model, n_colloc, bm_strength=0.0, bm_sharpness=2.0):
    """
    Envelope equation:
      ∇²A + 2i(kx·∂A/∂x + kz·∂A/∂z) + source·A = 0

    source = bm_strength · bm_indicator (BM 영역에서만 활성)

    Real: ∇²A_re - 2(kx·∂A_im/∂x + kz·∂A_im/∂z) + source·A_re = 0
    Imag: ∇²A_im + 2(kx·∂A_re/∂x + kz·∂A_re/∂z) + source·A_im = 0
    """
    x = rand_range(n_colloc, 0, 504, device)
    z = rand_range(n_colloc, 0, 40, device)
    x.requires_grad_(True)
    z.requires_grad_(True)

    d1 = rand_range(n_colloc, -10, 10, device)
    d2 = rand_range(n_colloc, -10, 10, device)
    w1 = rand_range(n_colloc, 5, 20, device)
    w2 = rand_range(n_colloc, 5, 20, device)
    sin_th, cos_th = sample_theta(n_colloc)

    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)

    A = model.forward_envelope(coords)
    A_re = A[:, 0]
    A_im = A[:, 1]

    kx = K * sin_th
    kz = K * cos_th

    # ---- First derivatives ----
    dAr_dx = torch.autograd.grad(A_re.sum(), x, create_graph=True)[0]
    dAr_dz = torch.autograd.grad(A_re.sum(), z, create_graph=True)[0]
    dAi_dx = torch.autograd.grad(A_im.sum(), x, create_graph=True)[0]
    dAi_dz = torch.autograd.grad(A_im.sum(), z, create_graph=True)[0]

    # ---- Second derivatives ----
    d2Ar_dx2 = torch.autograd.grad(dAr_dx.sum(), x, create_graph=True)[0]
    d2Ar_dz2 = torch.autograd.grad(dAr_dz.sum(), z, create_graph=True)[0]
    d2Ai_dx2 = torch.autograd.grad(dAi_dx.sum(), x, create_graph=True)[0]
    d2Ai_dz2 = torch.autograd.grad(dAi_dz.sum(), z, create_graph=True)[0]

    # ---- Envelope PDE residual ----
    res_re = d2Ar_dx2 + d2Ar_dz2 - 2 * (kx * dAi_dx + kz * dAi_dz)
    res_im = d2Ai_dx2 + d2Ai_dz2 + 2 * (kx * dAr_dx + kz * dAr_dz)

    # ---- BM source (optional) ----
    if bm_strength != 0.0:
        bm = compute_bm_indicator(x, z, d1, d2, w1, w2, sharpness=bm_sharpness)
        source = bm_strength * bm
        res_re = res_re + source * A_re
        res_im = res_im + source * A_im

    return torch.mean(res_re ** 2 + res_im ** 2)


# ============================================================
# BM BC Loss: A=0 at opaque regions (rejection sampling)
# ============================================================
def get_bm_bc(n):
    """BM 불투명 영역 점 생성 (rejection sampling)."""
    points = []
    generated = 0
    while generated < n:
        k = max(n * 3, 500)
        d1 = rand_range(k, -10, 10, device)
        d2 = rand_range(k, -10, 10, device)
        w1 = rand_range(k, 5, 20, device)
        w2 = rand_range(k, 5, 20, device)

        is_bm1 = torch.randint(0, 2, (k,), device=device) == 0
        z_curr = torch.where(is_bm1,
                             torch.tensor(Z_BM1, device=device),
                             torch.tensor(Z_BM2, device=device))
        w_curr = torch.where(is_bm1, w1, w2)
        d_curr = torch.where(is_bm1, d1, d2)
        x_curr = rand_range(k, 0, 504, device)

        pitch_idx = (x_curr / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
        center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2 + d_curr
        slit_left = center - w_curr / 2
        slit_right = center + w_curr / 2
        is_opaque = (x_curr < slit_left) | (x_curr > slit_right)

        if is_opaque.sum() > 0:
            valid = is_opaque.nonzero(as_tuple=True)[0]
            sin_th, cos_th = sample_theta(len(valid))
            batch = torch.stack([x_curr[valid], z_curr[valid],
                                 d1[valid], d2[valid], w1[valid], w2[valid],
                                 sin_th, cos_th], dim=1)
            points.append(batch)
            generated += len(valid)

    return torch.cat(points, dim=0)[:n]


def bm_bc_loss(model, bm_coords):
    """L_BC: BM 불투명 → A = 0."""
    A = model.forward_envelope(bm_coords)
    return torch.mean(A[:, 0] ** 2 + A[:, 1] ** 2)


# ============================================================
# Monitor |A| amplitude
# ============================================================
@torch.no_grad()
def monitor_amplitude(model, n=504):
    """z=40에서 θ=0 기준 |A| 확인. std 포함 (공간 변동 감지)."""
    x = torch.linspace(0, 504, n, device=device)
    z = torch.full((n,), Z_BM1, device=device)
    d1 = torch.zeros(n, device=device)
    d2 = torch.zeros(n, device=device)
    w1 = torch.full((n,), 10.0, device=device)
    w2 = torch.full((n,), 10.0, device=device)
    sin_th = torch.zeros(n, device=device)
    cos_th = torch.ones(n, device=device)

    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
    A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)
    return amp.mean().item(), amp.min().item(), amp.max().item(), amp.std().item()


@torch.no_grad()
def monitor_amplitude_slit_vs_bm(model, w=10.0, n=200):
    """z=40에서 θ=0, w1=10 기준: slit 내부 vs BM 불투명 |A| 비교."""
    x = torch.linspace(0, 504, n, device=device)
    z = torch.full((n,), Z_BM1, device=device)
    d1 = torch.zeros(n, device=device)
    d2 = torch.zeros(n, device=device)
    w1 = torch.full((n,), w, device=device)
    w2 = torch.full((n,), w, device=device)
    sin_th = torch.zeros(n, device=device)
    cos_th = torch.ones(n, device=device)

    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
    A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)

    # Slit vs BM classification (d=0, w=10)
    pitch_idx = (x / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
    center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2
    in_slit = (torch.abs(x - center) < w / 2)

    slit_amp = amp[in_slit].mean().item() if in_slit.sum() > 0 else 0.0
    bm_amp = amp[~in_slit].mean().item() if (~in_slit).sum() > 0 else 0.0
    return slit_amp, bm_amp


@torch.no_grad()
def monitor_amplitude_bottom(model, n=200):
    """z=0 (OPD)에서 |A| 확인."""
    x = torch.linspace(0, 504, n, device=device)
    z = torch.full((n,), Z_OPD, device=device)
    d1 = torch.zeros(n, device=device)
    d2 = torch.zeros(n, device=device)
    w1 = torch.full((n,), 10.0, device=device)
    w2 = torch.full((n,), 10.0, device=device)
    sin_th = torch.zeros(n, device=device)
    cos_th = torch.ones(n, device=device)

    coords = torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th], dim=1)
    A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)
    return amp.mean().item()


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("Phase B: 3-Stage Envelope PINN Training")
    log.info("=" * 60)
    log.info(f"  Stage 1: epoch 0~{args.stage1_end}  | L_phase only")
    log.info(f"  Stage 2: epoch {args.stage1_end}~{args.stage2_end} | +L_PDE λ={args.lam_pde_min:.0e}")
    log.info(f"  Stage 3: epoch {args.stage2_end}~{args.epochs} | L_PDE ramp → {args.lam_pde_max:.0e} + L_BC")
    log.info(f"  lambda_phase={args.lam_phase}, lambda_bc={args.lam_bc}")
    log.info(f"  normalize={args.normalize}, norm_floor={args.norm_floor}")
    log.info(f"  BM source={args.bm_strength}, sharpness={args.bm_sharpness}")
    log.info(f"  n_phase={args.n_phase}, n_colloc={args.n_colloc}, n_bm={args.n_bm}")

    # Verify target amplitude
    ar_lut = ARLutInterpolator()
    t_amp_0, dphi_0 = ar_lut.get_complex_t(WL_UM * 1000, 0.0, unpolarized=True)
    log.info(f"\n  AR LUT θ=0: |t|={t_amp_0:.4f}, dphi={dphi_0:.2f} deg")
    log.info(f"  Target |A| at z=40: {t_amp_0:.4f} (carrier ref z=40, no kz*40 phase)")

    # Model
    model = ParametricHelmholtzPINN(hidden_dim=128, num_layers=4, num_freqs=48)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model: {n_params:,} parameters\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t0 = time.time()

    # Fixed normalization scales (measured once at stage transitions)
    Lh_fixed_scale = None  # Set at Stage 2 start
    Lp_fixed_scale = None  # Set at Stage 3 start

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # ---- Stage & λ_PDE schedule ----
        if epoch < args.stage1_end:
            lam_pde = 0.0
            lam_bc = 0.0
            stage = 1
        elif epoch < args.stage2_end:
            lam_pde = args.lam_pde_min
            lam_bc = 0.0
            stage = 2
        else:
            lam_pde = args.lam_pde_max
            lam_bc = args.lam_bc
            stage = 3

        # ---- L_phase (slit-only in Stage 3 to avoid conflict with L_BC) ----
        slit_only = (stage >= 3)
        ph_coords, ph_re, ph_im = get_envelope_phase_bc(args.n_phase, slit_only=slit_only)
        Lp = envelope_phase_loss(model, ph_coords, ph_re, ph_im)

        # ---- L_PDE (Stage 2, 3) ----
        Lh_val = 0.0
        Lh = None
        if lam_pde > 0:
            bm_src = args.bm_strength if stage == 3 else 0.0
            Lh = envelope_pde_loss(model, args.n_colloc,
                                   bm_strength=bm_src,
                                   bm_sharpness=args.bm_sharpness)
            Lh_val = Lh.item()

        # ---- L_BC (Stage 3 only) ----
        Lb_val = 0.0
        Lb = None
        if lam_bc > 0 and stage >= 3:
            bm_coords = get_bm_bc(args.n_bm)
            Lb = bm_bc_loss(model, bm_coords)
            Lb_val = Lb.item()

        # ---- Loss assembly (different strategy per stage) ----
        if stage <= 2:
            # Stage 1-2: Dynamic normalization (proven to keep |A|=0.77)
            loss = args.lam_phase * Lp / Lp.detach().clamp(min=args.norm_floor)
            if Lh is not None:
                # Capture Lh scale at Stage 2 entry for later use
                if Lh_fixed_scale is None:
                    Lh_fixed_scale = Lh.detach().clamp(min=1.0).item()
                    log.info(f"  [Stage 2] Lh_fixed_scale = {Lh_fixed_scale:.1f}")
                loss = loss + lam_pde * Lh / Lh.detach().clamp(min=args.norm_floor)
        else:
            # Stage 3: RAW losses — no Lp normalization
            # slit-only L_phase (0.06) << L_BC (1.74) → BM에서 A→0 학습 가능
            # L_PDE는 Lh_fixed_scale로만 정규화 (초기값 ~100M → 실효 ~0)
            if Lp_fixed_scale is None:
                Lp_fixed_scale = Lp.detach().item()
                log.info(f"  [Stage 3] Raw Lp = {Lp_fixed_scale:.4f} (no normalization)")
                log.info(f"  [Stage 3] Lh_fixed_scale = {Lh_fixed_scale:.1f}")

            loss = args.lam_phase * Lp  # raw: ~5*0.012 = 0.06
            if Lh is not None:
                loss = loss + lam_pde * Lh / max(Lh_fixed_scale, 1.0)
            if Lb is not None:
                loss = loss + lam_bc * Lb  # raw: ~3*0.58 = 1.74

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        log_interval = 50
        if epoch % log_interval == 0 or epoch == args.epochs - 1:
            amp_mean, amp_min, amp_max, amp_std = monitor_amplitude(model)
            slit_amp, bm_amp = monitor_amplitude_slit_vs_bm(model)

            log.info(
                f"  [{epoch:5d}] S{stage} λpde={lam_pde:.1e} "
                f"| Lp={Lp.item():.4f} Lh={Lh_val:.1f} Lb={Lb_val:.4f} "
                f"| slit={slit_amp:.3f} bm={bm_amp:.3f} "
                f"std={amp_std:.4f} [{amp_min:.2f},{amp_max:.2f}]"
            )

        # ---- Stage transition alerts ----
        if epoch == args.stage1_end:
            amp_mean, _, _, amp_std = monitor_amplitude(model)
            log.info(f"\n  >>> Stage 1->2: |A|={amp_mean:.3f} std={amp_std:.4f} (target~{t_amp_0:.3f})")
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, 'phase_b_stage1.pt')
            log.info("")
        if epoch == args.stage2_end:
            amp_mean, _, _, amp_std = monitor_amplitude(model)
            log.info(f"\n  >>> Stage 2->3: |A|={amp_mean:.3f} std={amp_std:.4f}")
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, 'phase_b_stage2.pt')
            log.info("")

    # ---- Final ----
    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Training complete: {elapsed:.0f}s")

    amp_mean, amp_min, amp_max, amp_std = monitor_amplitude(model)
    slit_amp, bm_amp = monitor_amplitude_slit_vs_bm(model)
    log.info(f"|A(top)| mean={amp_mean:.4f} std={amp_std:.4f} [{amp_min:.4f}, {amp_max:.4f}]")
    log.info(f"slit={slit_amp:.4f}  bm={bm_amp:.4f}  diff={slit_amp-bm_amp:.4f}")

    # Save FINAL model (not best — best_loss tracks across stages incorrectly)
    ckpt_path = "phase_b_stage3.pt"
    torch.save({
        'model_state': model.state_dict(),
        'config': vars(args),
        'epoch': args.epochs,
    }, ckpt_path)
    log.info(f"Saved: {ckpt_path}")


if __name__ == '__main__':
    main()
