"""
train_phase_b.py - 9D Envelope PINN with BM slit indicator
===========================================================
핵심 발견: SIREN+Fourier가 (x,d,w)에서 슬릿 indicator 학습 불가.
해결: slit_dist를 9번째 입력 + hard mask로 BM 자동 0.

  Stage 1 (0~S1):       L_phase(slit) only → 슬릿 |A|=0.77
  Stage 2 (S1~S2):      + L_PDE (BM source) → 물리 학습
  (L_BC 불필요 — hard mask가 BM에서 A=0 보장)

입력 9D: (x, z, d1, d2, w1, w2, sin_th, cos_th, slit_dist)
출력: A_raw * sigmoid(slit_dist) → 슬릿=A, BM=0

Usage:
  python train_phase_b.py --epochs 3000 --device cpu
  python train_phase_b.py --epochs 10000 --device cuda
"""

import sys, math, time, logging, argparse
import numpy as np
import torch
import torch.optim as optim

from backend.core.parametric_pinn import (
    ParametricHelmholtzPINN, compute_slit_dist
)
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
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--lr', type=float, default=1e-3)
# Stage boundaries
parser.add_argument('--stage1_end', type=int, default=500)
parser.add_argument('--stage2_end', type=int, default=1500)
# Loss weights
parser.add_argument('--lam_phase', type=float, default=5.0)
parser.add_argument('--lam_pde', type=float, default=0.01,
                    help='PDE weight in Stage 2 (small — physics fine-tuning only)')
# BM
parser.add_argument('--bm_strength', type=float, default=-1.0)
parser.add_argument('--bm_sharpness', type=float, default=2.0)
parser.add_argument('--mask_sharpness', type=float, default=2.0,
                    help='Hard mask sigmoid sharpness')
args = parser.parse_args()

# ============================================================
# Constants
# ============================================================
DOMAIN_X = (0.0, 504.0)
DOMAIN_Z = (0.0, 40.0)
Z_BM1 = 40.0
Z_BM2 = 20.0
OPD_PITCH = 72.0
N_PITCHES = 7
WL_UM = 0.520
K0 = 2 * math.pi / WL_UM
N_CG = 1.52
K = K0 * N_CG

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
    sin_max = math.sin(math.radians(41.1))
    sin_th = rand_range(n, -sin_max, sin_max, device)
    cos_th = torch.sqrt(1 - sin_th ** 2)
    return sin_th, cos_th


def make_9d_coords(x, z, d1, d2, w1, w2, sin_th, cos_th):
    """8D + slit_dist → 9D coords."""
    sd = compute_slit_dist(x, z, d1, d2, w1, w2)
    return torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th, sd], dim=1)


# ============================================================
# Phase BC: slit positions only (rejection sampling)
# ============================================================
def get_phase_bc_slit(n):
    """
    z=40 슬릿 내부에서 A = t(θ).
    Hard mask가 BM=0 보장하므로 L_BC 불필요.
    """
    ar_lut = ARLutInterpolator()
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

        # BM1 슬릿 판정
        pitch_idx = (x / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
        center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2 + d1
        in_slit = (x >= center - w1 / 2) & (x <= center + w1 / 2)

        if in_slit.sum() == 0:
            continue

        v = in_slit.nonzero(as_tuple=True)[0]
        z_vals = torch.full((len(v),), Z_BM1, device=device)

        coords = make_9d_coords(x[v], z_vals, d1[v], d2[v],
                                w1[v], w2[v], sin_th[v], cos_th[v])

        re = torch.zeros(len(v), device=device)
        im = torch.zeros(len(v), device=device)
        for j, idx in enumerate(v):
            theta_deg = math.degrees(math.asin(sin_th[idx].item()))
            t_amp, dphi_deg = ar_lut.get_complex_t(WL_UM * 1000, theta_deg, unpolarized=True)
            ph = math.radians(dphi_deg)
            re[j] = t_amp * math.cos(ph)
            im[j] = t_amp * math.sin(ph)

        all_coords.append(coords)
        all_re.append(re)
        all_im.append(im)
        collected += len(v)

    return torch.cat(all_coords)[:n], torch.cat(all_re)[:n], torch.cat(all_im)[:n]


# ============================================================
# Phase Loss
# ============================================================
def phase_loss(model, coords, target_re, target_im):
    A = model.forward_envelope(coords)
    return torch.mean((A[:, 0] - target_re) ** 2 + (A[:, 1] - target_im) ** 2)


# ============================================================
# Envelope PDE Loss (with BM source)
# ============================================================
def envelope_pde_loss(model, n_colloc, bm_strength=0.0, bm_sharpness=2.0):
    """
    ∇²A + 2i(kx·∂A/∂x + kz·∂A/∂z) + source·A = 0
    A는 이미 hard mask 적용됨 (forward_envelope에서).
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

    coords = make_9d_coords(x, z, d1, d2, w1, w2, sin_th, cos_th)

    A = model.forward_envelope(coords)
    A_re = A[:, 0]
    A_im = A[:, 1]

    kx = K * sin_th
    kz = K * cos_th

    dAr_dx = torch.autograd.grad(A_re.sum(), x, create_graph=True)[0]
    dAr_dz = torch.autograd.grad(A_re.sum(), z, create_graph=True)[0]
    dAi_dx = torch.autograd.grad(A_im.sum(), x, create_graph=True)[0]
    dAi_dz = torch.autograd.grad(A_im.sum(), z, create_graph=True)[0]

    d2Ar_dx2 = torch.autograd.grad(dAr_dx.sum(), x, create_graph=True)[0]
    d2Ar_dz2 = torch.autograd.grad(dAr_dz.sum(), z, create_graph=True)[0]
    d2Ai_dx2 = torch.autograd.grad(dAi_dx.sum(), x, create_graph=True)[0]
    d2Ai_dz2 = torch.autograd.grad(dAi_dz.sum(), z, create_graph=True)[0]

    res_re = d2Ar_dx2 + d2Ar_dz2 - 2 * (kx * dAi_dx + kz * dAi_dz)
    res_im = d2Ai_dx2 + d2Ai_dz2 + 2 * (kx * dAr_dx + kz * dAr_dz)

    return torch.mean(res_re ** 2 + res_im ** 2)


# ============================================================
# Monitor
# ============================================================
@torch.no_grad()
def monitor(model, n=504):
    """z=40 슬라이스: slit vs BM |A| + std."""
    x = torch.linspace(0, 504, n, device=device)
    z = torch.full((n,), Z_BM1, device=device)
    d1 = torch.zeros(n, device=device)
    d2 = torch.zeros(n, device=device)
    w1 = torch.full((n,), 10.0, device=device)
    w2 = torch.full((n,), 10.0, device=device)
    sin_th = torch.zeros(n, device=device)
    cos_th = torch.ones(n, device=device)

    coords = make_9d_coords(x, z, d1, d2, w1, w2, sin_th, cos_th)
    A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)

    # Slit vs BM (d=0, w=10)
    pitch_idx = (x / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
    center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2
    in_slit = torch.abs(x - center) < 5.0  # w/2=5

    slit_a = amp[in_slit].mean().item() if in_slit.sum() > 0 else 0.0
    bm_a = amp[~in_slit].mean().item() if (~in_slit).sum() > 0 else 0.0
    return slit_a, bm_a, amp.std().item()


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("Phase B: 9D Envelope PINN with BM slit indicator")
    log.info("=" * 60)
    log.info(f"  Stage 1: epoch 0~{args.stage1_end}  | L_phase(slit) only")
    log.info(f"  Stage 2: epoch {args.stage1_end}~{args.epochs} | +L_PDE λ={args.lam_pde}")
    log.info(f"  Hard mask sharpness={args.mask_sharpness} (L_BC 불필요)")
    log.info(f"  lam_phase={args.lam_phase}, lam_pde={args.lam_pde}")
    log.info(f"  BM source={args.bm_strength}")
    log.info(f"  n_phase={args.n_phase}, n_colloc={args.n_colloc}")

    ar_lut = ARLutInterpolator()
    t_amp_0, _ = ar_lut.get_complex_t(WL_UM * 1000, 0.0, unpolarized=True)
    log.info(f"\n  Target |A| slit: {t_amp_0:.4f}")

    # 9D model with hard mask
    model = ParametricHelmholtzPINN(
        hidden_dim=128, num_layers=4, num_freqs=48,
        in_dim=9, use_bm_mask=True, mask_sharpness=args.mask_sharpness
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model: {n_params:,} params (9D input, hard BM mask)\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t0 = time.time()
    Lh_scale = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        stage = 1 if epoch < args.stage1_end else 2

        # ---- L_phase (always) ----
        ph_coords, ph_re, ph_im = get_phase_bc_slit(args.n_phase)
        Lp = phase_loss(model, ph_coords, ph_re, ph_im)
        loss = args.lam_phase * Lp

        # ---- L_PDE (Stage 2) ----
        Lh_val = 0.0
        if stage >= 2:
            Lh = envelope_pde_loss(model, args.n_colloc,
                                   bm_strength=args.bm_strength,
                                   bm_sharpness=args.bm_sharpness)
            Lh_val = Lh.item()
            # Dynamic normalization for PDE (proven to maintain |A|)
            loss = loss + args.lam_pde * Lh / Lh.detach().clamp(min=0.01)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            slit_a, bm_a, std = monitor(model)
            log.info(
                f"  [{epoch:5d}] S{stage} "
                f"| Lp={Lp.item():.4f} Lh={Lh_val:.1f} "
                f"| slit={slit_a:.3f} bm={bm_a:.3f} diff={slit_a-bm_a:.3f} "
                f"std={std:.4f}"
            )

        if epoch == args.stage1_end:
            s, b, std = monitor(model)
            log.info(f"\n  >>> Stage 1->2: slit={s:.3f} bm={b:.3f} diff={s-b:.3f} std={std:.4f}")
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, 'phase_b_stage1.pt')
            log.info("")

    # ---- Final ----
    elapsed = time.time() - t0
    slit_a, bm_a, std = monitor(model)
    log.info(f"\n{'=' * 60}")
    log.info(f"Training complete: {elapsed:.0f}s")
    log.info(f"slit={slit_a:.4f}  bm={bm_a:.4f}  diff={slit_a-bm_a:.4f}  std={std:.4f}")

    torch.save({
        'model_state': model.state_dict(),
        'config': vars(args),
    }, 'phase_b_stage2.pt')
    log.info(f"Saved: phase_b_stage2.pt")


if __name__ == '__main__':
    main()
