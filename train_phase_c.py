"""
train_phase_c.py - Phase C: PINN이 진짜 BM 물리 학습
=====================================================
Phase B: A_raw = 0.799 상수 (sigmoid mask만 기여)
Phase C: A_raw가 (x, z, d, w, θ)에 따라 변하는 물리적 필드

핵심 추가: L_data = |PSF_pinn(d,w) - PSF_asm(d,w)|²
  ASM 정답 데이터(100개 설계)로 PINN이 정확한 PSF를 출력하도록 학습.
  이 과정에서 A_raw가 상수에서 벗어나 진짜 BM 전파 물리를 학습.

학습 순서:
  Stage 1 (0~S1):     L_phase(slit) + L_data → 경계 + PSF 매칭
  Stage 2 (S1~end):   + L_PDE → 물리 일관성 추가

Usage:
  python train_phase_c.py --epochs 3000 --device cpu
  python train_phase_c.py --epochs 10000 --device cuda
"""

import sys, math, time, logging, argparse
import numpy as np
import torch
import torch.optim as optim

from backend.core.parametric_pinn import (
    ParametricHelmholtzPINN, compute_slit_dist, compute_bm_mask,
    OPD_PITCH, N_PITCHES, Z_BM1, Z_BM2
)
from backend.physics.ar_coating.ar_boundary import ARLutInterpolator
from backend.physics.asm_propagator import ASMPropagator
from backend.physics.psf_metrics import compute_psf_mtf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--n_phase', type=int, default=300)
parser.add_argument('--n_colloc', type=int, default=3000)
parser.add_argument('--batch_data', type=int, default=16,
                    help='L_data batch size (from ASM training data)')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--stage1_end', type=int, default=1000)
parser.add_argument('--lam_phase', type=float, default=5.0)
parser.add_argument('--lam_data', type=float, default=10.0,
                    help='L_data weight (PSF matching)')
parser.add_argument('--lam_pde', type=float, default=0.01)
parser.add_argument('--mask_sharpness', type=float, default=2.0)
parser.add_argument('--resume', type=str, default='phase_b_stage2.pt',
                    help='Resume from Phase B checkpoint')
args = parser.parse_args()

WL_UM = 0.520
K0 = 2 * math.pi / WL_UM
N_CG = 1.52
K = K0 * N_CG

device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


# ============================================================
# Helpers
# ============================================================
def rand_range(n, lo, hi):
    return torch.rand(n, device=device) * (hi - lo) + lo


def sample_theta(n):
    sin_max = math.sin(math.radians(41.1))
    sin_th = rand_range(n, -sin_max, sin_max)
    cos_th = torch.sqrt(1 - sin_th ** 2)
    return sin_th, cos_th


def make_9d(x, z, d1, d2, w1, w2, sin_th, cos_th):
    sd = compute_slit_dist(x, z, d1, d2, w1, w2)
    return torch.stack([x, z, d1, d2, w1, w2, sin_th, cos_th, sd], dim=1)


# ============================================================
# L_phase: slit-only phase BC at z=40
# ============================================================
def get_phase_bc(n):
    ar_lut = ARLutInterpolator()
    all_c, all_r, all_i = [], [], []
    collected = 0
    while collected < n:
        batch = max(n * 5, 300)
        x = rand_range(batch, 0, 504)
        d1 = rand_range(batch, -10, 10); d2 = rand_range(batch, -10, 10)
        w1 = rand_range(batch, 5, 20); w2 = rand_range(batch, 5, 20)
        sin_th, cos_th = sample_theta(batch)
        pi = (x / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
        cen = pi.float() * OPD_PITCH + OPD_PITCH / 2 + d1
        ok = (x >= cen - w1 / 2) & (x <= cen + w1 / 2)
        if ok.sum() == 0:
            continue
        v = ok.nonzero(as_tuple=True)[0]
        zv = torch.full((len(v),), Z_BM1, device=device)
        c = make_9d(x[v], zv, d1[v], d2[v], w1[v], w2[v], sin_th[v], cos_th[v])
        re = torch.zeros(len(v), device=device)
        im = torch.zeros(len(v), device=device)
        for j, idx in enumerate(v):
            td = math.degrees(math.asin(sin_th[idx].item()))
            ta, dp = ar_lut.get_complex_t(WL_UM * 1000, td, unpolarized=True)
            ph = math.radians(dp)
            re[j] = ta * math.cos(ph)
            im[j] = ta * math.sin(ph)
        all_c.append(c); all_r.append(re); all_i.append(im)
        collected += len(v)
    return torch.cat(all_c)[:n], torch.cat(all_r)[:n], torch.cat(all_i)[:n]


# ============================================================
# L_data: PSF matching (PINN hybrid pipeline vs ASM)
# ============================================================
def torch_asm_propagate(u, dx, dz, k):
    """Differentiable ASM in PyTorch (gradient flows through)."""
    N = u.shape[0]
    fx = torch.fft.fftfreq(N, d=dx, device=u.device)
    kx = 2 * math.pi * fx
    kz_sq = k ** 2 - kx ** 2
    propagating = kz_sq > 0
    kz = torch.where(propagating, torch.sqrt(torch.abs(kz_sq)), torch.zeros_like(kz_sq))
    H = torch.where(propagating, torch.exp(1j * kz * dz), torch.zeros_like(kz, dtype=torch.complex64))
    U = torch.fft.fft(u)
    return torch.fft.ifft(U * H)


def compute_pinn_psf(model, d1, d2, w1, w2, asm, ar_lut, n_angles=7):
    """
    Fully differentiable PINN hybrid PSF.
    ASM(finger+CG, numpy) → PINN BM1 mask → torch ASM(ILD+BM2+Encap) → OPD.
    Gradient flows: PSF → torch_asm → PINN mask → model parameters.
    """
    angles = [0, 15, -15, 30, -30, 41, -41][:n_angles]
    pitch = OPD_PITCH
    n_pitches = N_PITCHES
    dx = 1.0
    N_pts = 504
    x_arr = np.arange(N_pts) * dx
    k = K0 * N_CG

    # Finger pattern (fixed)
    finger = np.ones(N_pts) * 0.1
    for i in range(N_pts):
        if (x_arr[i] % 288) < 144:
            finger[i] = 1.0

    # BM2 mask (torch, for differentiability)
    bm2_mask = torch.zeros(N_pts, device=device)
    for p in range(n_pitches):
        center = p * pitch + pitch / 2 + d2
        il = max(0, int(center - w2 / 2))
        ir = min(N_pts, int(center + w2 / 2))
        bm2_mask[il:ir] = 1.0

    total_psf = torch.zeros(n_pitches, device=device)

    for theta in angles:
        t_amp, dphi = ar_lut.get_complex_t(520, theta, unpolarized=True)
        U0 = asm.make_incident_field(x_arr, theta, dphi) * finger * t_amp
        U_bm1_np = asm.propagate_1d(U0, dx, 550.0)

        # Convert to torch (CG part is fixed, no grad needed)
        U_bm1 = torch.tensor(U_bm1_np, dtype=torch.complex64, device=device)

        # PINN BM1 mask (DIFFERENTIABLE — gradient flows here!)
        x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
        z_t = torch.full((N_pts,), Z_BM1, device=device)
        d1_t = torch.full((N_pts,), float(d1), device=device)
        d2_t = torch.full((N_pts,), float(d2), device=device)
        w1_t = torch.full((N_pts,), float(w1), device=device)
        w2_t = torch.full((N_pts,), float(w2), device=device)
        sin_t = torch.full((N_pts,), math.sin(math.radians(theta)), device=device)
        cos_t = torch.full((N_pts,), math.cos(math.radians(theta)), device=device)
        coords = make_9d(x_t, z_t, d1_t, d2_t, w1_t, w2_t, sin_t, cos_t)

        A = model.forward_envelope(coords)
        amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2 + 1e-12)
        amax = amp.max().detach()
        pinn_mask = amp / (amax + 1e-8)

        # Apply mask
        U_after_bm1 = U_bm1 * pinn_mask.to(torch.complex64)

        # Differentiable ASM: ILD 20um
        U_at_bm2 = torch_asm_propagate(U_after_bm1, dx, 20.0, k)

        # BM2 mask
        U_after_bm2 = U_at_bm2 * bm2_mask.to(torch.complex64)

        # Differentiable ASM: Encap 20um
        U_z0 = torch_asm_propagate(U_after_bm2, dx, 20.0, k)

        # Intensity (differentiable)
        intensity = U_z0.real ** 2 + U_z0.imag ** 2

        # OPD pixel integration
        for i in range(n_pitches):
            opd_cen = i * pitch + pitch / 2
            il = max(0, int(opd_cen - 5))
            ir = min(N_pts, int(opd_cen + 5))
            total_psf[i] = total_psf[i] + intensity[il:ir].mean()

    total_psf = total_psf / len(angles)
    return total_psf


def data_loss(model, data_batch, asm, ar_lut):
    """L_data = MSE(PSF_pinn, PSF_asm) over a batch of designs."""
    loss = torch.tensor(0.0, device=device)
    for sample in data_batch:
        d1, d2, w1, w2 = sample['d1'], sample['d2'], sample['w1'], sample['w2']
        psf_target = torch.tensor(sample['psf7'], dtype=torch.float32, device=device)
        psf_pred = compute_pinn_psf(model, d1, d2, w1, w2, asm, ar_lut, n_angles=3)
        loss += torch.mean((psf_pred - psf_target) ** 2)
    return loss / len(data_batch)


# ============================================================
# L_PDE: envelope equation
# ============================================================
def envelope_pde_loss(model, n_colloc):
    x = rand_range(n_colloc, 0, 504)
    z = rand_range(n_colloc, 0, 40)
    x.requires_grad_(True)
    z.requires_grad_(True)
    d1 = rand_range(n_colloc, -10, 10); d2 = rand_range(n_colloc, -10, 10)
    w1 = rand_range(n_colloc, 5, 20); w2 = rand_range(n_colloc, 5, 20)
    sin_th, cos_th = sample_theta(n_colloc)
    coords = make_9d(x, z, d1, d2, w1, w2, sin_th, cos_th)

    A = model.forward_envelope(coords)
    A_re, A_im = A[:, 0], A[:, 1]
    kx = K * sin_th; kz = K * cos_th

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
def monitor(model, asm, ar_lut):
    # Baseline PSF
    psf = compute_pinn_psf(model, 0, 0, 10, 10, asm, ar_lut, n_angles=3)
    mtf = compute_psf_mtf(psf.tolist())

    # Check A_raw variation (is it still constant?)
    n = 504
    x = torch.linspace(0, 504, n, device=device)
    z = torch.full((n,), 30.0, device=device)  # z=30 (between BMs, no mask)
    d1 = torch.zeros(n, device=device); d2 = torch.zeros(n, device=device)
    w1 = torch.full((n,), 10.0, device=device); w2 = torch.full((n,), 10.0, device=device)
    sin_th = torch.zeros(n, device=device); cos_th = torch.ones(n, device=device)
    coords = make_9d(x, z, d1, d2, w1, w2, sin_th, cos_th)
    A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)

    return {
        'mtf': mtf,
        'psf_sum': float(psf.sum()),
        'A_z30_mean': amp.mean().item(),
        'A_z30_std': amp.std().item(),  # 0이면 아직 상수, >0이면 물리 학습 중
    }


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("Phase C: PINN learns real BM physics (L_data)")
    log.info("=" * 60)
    log.info(f"  Stage 1: epoch 0~{args.stage1_end} | L_phase + L_data")
    log.info(f"  Stage 2: epoch {args.stage1_end}~{args.epochs} | + L_PDE")
    log.info(f"  lam_phase={args.lam_phase}, lam_data={args.lam_data}, lam_pde={args.lam_pde}")

    # Load ASM training data
    data = torch.load('asm_training_data.pt', map_location='cpu', weights_only=False)
    log.info(f"  ASM training data: {len(data)} samples")

    # Model (resume from Phase B)
    model = ParametricHelmholtzPINN(
        hidden_dim=128, num_layers=4, num_freqs=48,
        in_dim=9, use_bm_mask=True, mask_sharpness=args.mask_sharpness
    )
    ckpt_path = args.resume
    if ckpt_path and torch.load(ckpt_path, map_location='cpu', weights_only=False):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        log.info(f"  Resumed from {ckpt_path}")
    model.to(device)

    asm = ASMPropagator(wl_um=WL_UM, n_medium=N_CG)
    ar_lut = ARLutInterpolator()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t0 = time.time()
    log.info("")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        stage = 1 if epoch < args.stage1_end else 2

        # L_phase
        ph_c, ph_r, ph_i = get_phase_bc(args.n_phase)
        A_ph = model.forward_envelope(ph_c)
        Lp = torch.mean((A_ph[:, 0] - ph_r) ** 2 + (A_ph[:, 1] - ph_i) ** 2)

        # L_data (batch from ASM data)
        batch_idx = np.random.choice(len(data), args.batch_data, replace=False)
        batch = [data[i] for i in batch_idx]
        Ld = data_loss(model, batch, asm, ar_lut)

        loss = args.lam_phase * Lp + args.lam_data * Ld

        # L_PDE (Stage 2)
        Lh_val = 0.0
        if stage >= 2:
            Lh = envelope_pde_loss(model, args.n_colloc)
            Lh_val = Lh.item()
            loss = loss + args.lam_pde * Lh / (Lh.detach().clamp(min=0.01))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            m = monitor(model, asm, ar_lut)
            log.info(
                f"  [{epoch:5d}] S{stage} "
                f"| Lp={Lp.item():.4f} Ld={Ld.item():.6f} Lh={Lh_val:.1f} "
                f"| MTF={m['mtf']:.1%} sum={m['psf_sum']:.3f} "
                f"A_z30={m['A_z30_mean']:.3f}+/-{m['A_z30_std']:.4f}"
            )

        if epoch == args.stage1_end:
            log.info(f"\n  >>> Stage 1->2")
            torch.save({'model_state': model.state_dict()}, 'phase_c_stage1.pt')

    # Final
    elapsed = time.time() - t0
    m = monitor(model, asm, ar_lut)
    log.info(f"\n{'=' * 60}")
    log.info(f"Phase C complete: {elapsed:.0f}s")
    log.info(f"MTF={m['mtf']:.1%} sum={m['psf_sum']:.3f}")
    log.info(f"A_z30 mean={m['A_z30_mean']:.3f} std={m['A_z30_std']:.4f}")
    log.info(f"  (std>0 = PINN learned real physics, std~0 = still constant)")

    torch.save({'model_state': model.state_dict(), 'config': vars(args)},
               'phase_c_final.pt')
    log.info("Saved: phase_c_final.pt")


if __name__ == '__main__':
    main()
