"""
train_parametric_pinn.py - Parametric PINN 학습
================================================
ASM + PINN 하이브리드 아키텍처:
  ASM: 지문 -> AR(LUT) -> CG 550um 전파 -> BM1 면 복소 필드
  PINN: BM 40um 도메인 Helmholtz (x, z, d1, d2, w1, w2) -> U(x,z;p)

한 번 학습으로 모든 BM 설계를 커버하는 Parametric PINN.
학습 후 임의의 (d1,d2,w1,w2)에 대해 ~1ms 추론.

Usage:
  GPU: python train_parametric_pinn.py --epochs 10000 --n_colloc 20000 --device cuda
  CPU: python train_parametric_pinn.py --epochs 1000 --n_colloc 2000 --device cpu
"""

import sys, math, time, logging, argparse
import numpy as np
import torch
import torch.optim as optim

from backend.core.parametric_pinn import ParametricHelmholtzPINN
from backend.physics.asm_propagator import ASMPropagator
from backend.physics.ar_coating.ar_boundary import ARLutInterpolator
from backend.physics.psf_metrics import PSFMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--n_colloc', type=int, default=10000)
parser.add_argument('--n_phase', type=int, default=500, help='L_phase points (AR coating)')
parser.add_argument('--n_I', type=int, default=100, help='L_I points (OPD intensity)')
parser.add_argument('--n_bm', type=int, default=500, help='L_BC points (BM opacity)')
parser.add_argument('--n_angles', type=int, default=7, help='ASM angles for L_I')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# ============================================================
# Constants (검증됨)
# ============================================================
DOMAIN_X = (0.0, 504.0)
DOMAIN_Z = (0.0, 40.0)   # BM 영역만
Z_BM1 = 40.0             # PINN 도메인 상단 = ASM 출력
Z_BM2 = 20.0             #
Z_OPD = 0.0
OPD_PITCH = 72.0
N_PITCHES = 7
WL_UM = 0.520
K0 = 2 * math.pi / WL_UM
N_CG = 1.52

# Device
if args.device == 'cuda' and not torch.cuda.is_available():
    log.warning("CUDA not available, falling back to CPU")
    device = torch.device('cpu')
else:
    device = torch.device(args.device)

log.info(f"Device: {device}")
if device.type == 'cuda':
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# ASM: CG 전파 -> BM1 면 복소 필드 (각 각도별)
# ============================================================
def compute_asm_fields():
    """
    여러 각도에서 ASM으로 CG 전파한 결과를 미리 계산.
    학습 중 Top BC로 사용.
    """
    asm = ASMPropagator(wl_um=WL_UM, n_medium=N_CG)
    ar_lut = ARLutInterpolator()

    N = 504
    dx = 1.0
    x = np.arange(N) * dx

    # 지문 패턴
    finger = np.ones(N) * 0.1
    for i in range(N):
        if (x[i] % 288) < 144:
            finger[i] = 1.0

    # 각도
    angles = [0, 15, -15, 30, -30, 41, -41][:args.n_angles]

    fields = {}
    for theta in angles:
        t_amp, dphi = ar_lut.get_complex_t(WL_UM * 1000, theta, unpolarized=True)
        U0 = asm.make_incident_field(x, theta, dphi) * finger * t_amp
        U_bm1 = asm.propagate_1d(U0, dx, 550.0)
        fields[theta] = {
            'x': x,
            're': U_bm1.real.astype(np.float32),
            'im': U_bm1.imag.astype(np.float32),
        }
        log.info(f"  ASM theta={theta:+3d}: |U| range [{np.abs(U_bm1).min():.4f}, {np.abs(U_bm1).max():.4f}]")

    return fields, angles


# ============================================================
# Collocation 생성
# ============================================================
def rand_range(n, lo, hi, dev):
    return torch.rand(n, device=dev) * (hi - lo) + lo


def get_collocation(n):
    """도메인 내부 랜덤 점 + 랜덤 설계변수."""
    x = rand_range(n, 0, 504, device)
    z = rand_range(n, 0, 40, device)
    x.requires_grad_(True)
    z.requires_grad_(True)
    d1 = rand_range(n, -10, 10, device)
    d2 = rand_range(n, -10, 10, device)
    w1 = rand_range(n, 5, 20, device)
    w2 = rand_range(n, 5, 20, device)
    return torch.stack([x, z, d1, d2, w1, w2], dim=1), x, z


def get_phase_bc(n):
    """
    L_phase: z=40 (AR 코팅면)에서 U = t(θ)·E_inc.
    t(θ) = AR LUT 복소 투과계수 (진폭 + 위상).
    랜덤 x, 랜덤 θ, 랜덤 설계변수로 샘플.
    """
    ar_lut = ARLutInterpolator()

    x_vals = rand_range(n, 0, 504, device)
    z_vals = torch.full((n,), Z_BM1, device=device)
    d1 = rand_range(n, -10, 10, device)
    d2 = rand_range(n, -10, 10, device)
    w1 = rand_range(n, 5, 20, device)
    w2 = rand_range(n, 5, 20, device)

    coords = torch.stack([x_vals, z_vals, d1, d2, w1, w2], dim=1)

    target_re = torch.zeros(n, device=device)
    target_im = torch.zeros(n, device=device)

    for i in range(n):
        xi = x_vals[i].item()
        # x 위치 → 유효 입사각
        xc = xi - 252.0
        theta_deg = max(-41, min(41, math.degrees(math.atan2(xc, 590.0))))

        # AR LUT: 복소 투과계수 t(θ) = |t|·e^(j·dphi)
        t_amp, dphi_deg = ar_lut.get_complex_t(WL_UM * 1000, theta_deg, unpolarized=True)

        # 입사파: E_inc = exp(j·kx·x)
        kx = K0 * N_CG * math.sin(math.radians(theta_deg))
        inc_phase = kx * xi

        # 타겟: t(θ) · E_inc
        total_phase = math.radians(dphi_deg) + inc_phase
        target_re[i] = t_amp * math.cos(total_phase)
        target_im[i] = t_amp * math.sin(total_phase)

    return coords, target_re, target_im


def get_intensity_bc(n, asm_fields, angles):
    """
    L_I: z=0 (OPD면)에서 |U|² = ASM 타겟 세기.
    ASM이 계산한 CG blur + 크로스토크를 PINN에 전달.
    """
    # 7 OPD 위치에서 ASM 세기 타겟
    asm_psf = np.zeros(N_PITCHES)
    for theta in angles:
        f = asm_fields[theta]
        for i in range(N_PITCHES):
            xi = int(i * OPD_PITCH + OPD_PITCH / 2)
            xi = max(0, min(len(f['re']) - 1, xi))
            asm_psf[i] += f['re'][xi]**2 + f['im'][xi]**2
    asm_psf /= len(angles)

    # 랜덤 설계변수로 n개 샘플
    d1 = rand_range(n, -10, 10, device)
    d2 = rand_range(n, -10, 10, device)
    w1 = rand_range(n, 5, 20, device)
    w2 = rand_range(n, 5, 20, device)

    # OPD 위치 중 랜덤 선택
    opd_idx = np.random.randint(0, N_PITCHES, size=n)
    x_vals = torch.tensor([opd_idx[i] * OPD_PITCH + OPD_PITCH / 2 for i in range(n)],
                          dtype=torch.float32, device=device)
    z_vals = torch.full((n,), Z_OPD, device=device)

    coords = torch.stack([x_vals, z_vals, d1, d2, w1, w2], dim=1)
    targets = torch.tensor([asm_psf[opd_idx[i]] for i in range(n)],
                           dtype=torch.float32, device=device)

    return coords, targets


def get_bm_bc(n):
    """BM 불투명 영역 (rejection sampling)."""
    points = []
    generated = 0

    while generated < n:
        k = max(n * 3, 1000)
        d1 = rand_range(k, -10, 10, device)
        d2 = rand_range(k, -10, 10, device)
        w1 = rand_range(k, 5, 20, device)
        w2 = rand_range(k, 5, 20, device)

        # BM1(z=40) or BM2(z=20) 랜덤 선택
        is_bm1 = torch.randint(0, 2, (k,), device=device) == 0
        z_curr = torch.where(is_bm1, torch.tensor(Z_BM1, device=device), torch.tensor(Z_BM2, device=device))
        w_curr = torch.where(is_bm1, w1, w2)
        d_curr = torch.where(is_bm1, d1, d2)

        x_curr = rand_range(k, 0, 504, device)

        # 어떤 피치에 속하는지
        pitch_idx = (x_curr / OPD_PITCH).long().clamp(0, N_PITCHES - 1)
        center = pitch_idx.float() * OPD_PITCH + OPD_PITCH / 2 + d_curr
        slit_left = center - w_curr / 2
        slit_right = center + w_curr / 2

        # 불투명 영역 (슬릿 바깥)
        is_opaque = (x_curr < slit_left) | (x_curr > slit_right)

        if is_opaque.sum() > 0:
            valid = is_opaque.nonzero(as_tuple=True)[0]
            batch = torch.stack([x_curr[valid], z_curr[valid],
                                 d1[valid], d2[valid], w1[valid], w2[valid]], dim=1)
            points.append(batch)
            generated += len(valid)

    return torch.cat(points, dim=0)[:n]


# ============================================================
# Loss 함수
# ============================================================
def helmholtz_loss(model, coords, x, z):
    """∇²U + k²n²U = 0 (Re, Im 각각)."""
    U = model(coords)
    # Re part
    Ux = torch.autograd.grad(U[:, 0], x, torch.ones_like(U[:, 0]), create_graph=True)[0]
    Uxx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
    Uz = torch.autograd.grad(U[:, 0], z, torch.ones_like(U[:, 0]), create_graph=True)[0]
    Uzz = torch.autograd.grad(Uz, z, torch.ones_like(Uz), create_graph=True)[0]
    # Im part
    Ix = torch.autograd.grad(U[:, 1], x, torch.ones_like(U[:, 1]), create_graph=True)[0]
    Ixx = torch.autograd.grad(Ix, x, torch.ones_like(Ix), create_graph=True)[0]
    Iz = torch.autograd.grad(U[:, 1], z, torch.ones_like(U[:, 1]), create_graph=True)[0]
    Izz = torch.autograd.grad(Iz, z, torch.ones_like(Iz), create_graph=True)[0]

    helm_re = Uxx + Uzz + (K0 * N_CG) ** 2 * U[:, 0]
    helm_im = Ixx + Izz + (K0 * N_CG) ** 2 * U[:, 1]
    return torch.mean(helm_re ** 2 + helm_im ** 2)


def phase_loss(model, coords, target_re, target_im):
    """L_phase: AR 코팅면에서 U = t(θ)·E_inc (위상 왜곡 반영)."""
    out = model(coords)
    return torch.mean((out[:, 0] - target_re) ** 2 + (out[:, 1] - target_im) ** 2)


def intensity_loss(model, coords, targets):
    """L_I: OPD면에서 |U|² = ASM 타겟 (CG blur 반영)."""
    out = model(coords)
    I_pred = out[:, 0] ** 2 + out[:, 1] ** 2
    return torch.mean((I_pred - targets) ** 2)


def bm_bc_loss(model, bm_coords):
    """L_BC: BM 불투명 U = 0."""
    out = model(bm_coords)
    return torch.mean(out[:, 0] ** 2 + out[:, 1] ** 2)


# ============================================================
# PSF 추론
# ============================================================
def predict_psf7(model, d1, d2, w1, w2):
    """학습된 모델로 PSF 추론 (~1ms)."""
    model.eval()
    psf = np.zeros(N_PITCHES)
    with torch.no_grad():
        for i in range(N_PITCHES):
            x_opd = i * OPD_PITCH + OPD_PITCH / 2
            inp = torch.tensor([[x_opd, Z_OPD, d1, d2, w1, w2]],
                               dtype=torch.float32, device=device)
            out = model(inp)
            psf[i] = (out[0, 0] ** 2 + out[0, 1] ** 2).item()
    return psf


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("Parametric PINN Training (Hybrid ASM + PINN)")
    log.info("=" * 60)
    log.info(f"  Domain: {DOMAIN_X[1]}um x {DOMAIN_Z[1]}um (BM region)")
    log.info(f"  BM1=z={Z_BM1}, BM2=z={Z_BM2}, OPD=z={Z_OPD}")
    log.info(f"  Epochs: {args.epochs}, Colloc: {args.n_colloc}")
    log.info(f"  L_phase: {args.n_phase}, L_I: {args.n_I}, L_BC: {args.n_bm}")
    log.info(f"  Angles: {args.n_angles}")

    # ASM 필드 사전 계산
    log.info("\nComputing ASM fields at BM1...")
    asm_fields, angles = compute_asm_fields()
    log.info(f"  {len(angles)} angles computed")

    # 모델
    model = ParametricHelmholtzPINN(hidden_dim=128, num_layers=4, num_freqs=48)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel: {n_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t0 = time.time()
    best_loss = float('inf')
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # 가이드 원래 구조: L_H + L_phase + L_I + L_BC
        # 1. L_Helmholtz (58%): 파동방정식
        coords, x_col, z_col = get_collocation(args.n_colloc)
        Lh = helmholtz_loss(model, coords, x_col, z_col)

        # 2. L_phase (23%): AR 코팅 위상 (LUT)
        ph_coords, ph_re, ph_im = get_phase_bc(args.n_phase)
        Lp = phase_loss(model, ph_coords, ph_re, ph_im)

        # 3. L_I (11%): OPD 세기 (ASM 타겟)
        li_coords, li_targets = get_intensity_bc(args.n_I, asm_fields, angles)
        Li = intensity_loss(model, li_coords, li_targets)

        # 4. L_BC (8%): BM 불투명
        bm_coords = get_bm_bc(args.n_bm)
        Lb = bm_bc_loss(model, bm_coords)

        # L_total = 1.0·L_H + 0.5·L_phase + 0.3·L_I + 0.2·L_BC
        loss = 1.0 * Lh + 0.5 * Lp + 0.3 * Li + 0.2 * Lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        if lv < best_loss and not math.isnan(lv):
            best_loss = lv
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            # 테스트: baseline 설계 PSF
            psf = predict_psf7(model, 0, 0, 10, 10)
            pmax, pmin = psf.max(), psf.min()
            c = (pmax - pmin) / (pmax + pmin + 1e-10)
            log.info(
                f"  [{epoch:5d}/{args.epochs}] loss={lv:.2f} "
                f"Lh={Lh.item():.1f} Lp={Lp.item():.4f} Li={Li.item():.4f} Lb={Lb.item():.4f} "
                f"contrast={c * 100:.1f}% | {time.time() - t0:.0f}s"
            )

    # 최종 결과
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(f"Training complete: {elapsed:.0f}s, best loss={best_loss:.2f}")

    # 여러 설계에서 PSF 추론
    log.info(f"\nPSF predictions:")
    designs = [
        ("Baseline", 0, 0, 10, 10),
        ("Offset -3", -3, 0, 10, 10),
        ("Offset +3", 3, 0, 10, 10),
        ("Narrow w=6", 0, 0, 6, 6),
        ("Wide w=18", 0, 0, 18, 18),
    ]
    metrics = PSFMetrics()
    for name, d1, d2, w1, w2 in designs:
        t1 = time.time()
        psf = predict_psf7(model, d1, d2, w1, w2)
        dt = (time.time() - t1) * 1000
        m = metrics.compute(psf)
        pmax, pmin = psf.max(), psf.min()
        c = (pmax - pmin) / (pmax + pmin + 1e-10)
        log.info(f"  {name:15s}: contrast={c * 100:.1f}% skew={m['skewness']:.3f} ({dt:.1f}ms)")

    # 저장
    ckpt_path = "parametric_pinn_ckpt.pt"
    torch.save({
        'model_state': model.state_dict() if not best_state else best_state,
        'config': {
            'domain_z': DOMAIN_Z, 'z_bm1': Z_BM1, 'z_bm2': Z_BM2,
            'epochs': args.epochs, 'n_colloc': args.n_colloc,
            'n_angles': args.n_angles, 'hidden_dim': 128, 'num_layers': 4,
        },
        'best_loss': best_loss,
    }, ckpt_path)
    log.info(f"\nSaved: {ckpt_path}")
    log.info(f"Run on GPU: python train_parametric_pinn.py --epochs 10000 --n_colloc 20000 --device cuda")


if __name__ == '__main__':
    main()
