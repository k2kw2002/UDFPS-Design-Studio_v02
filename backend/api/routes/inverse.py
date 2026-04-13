"""
inverse.py - Mock Inverse Design (랜덤 search)
================================================
실제 Phase D에서 BoTorch qNEHVI로 교체 예정.
현재: 랜덤 search N회 → Pareto frontier 추출.
"""

from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import torch
from pathlib import Path

from backend.core.parametric_pinn import ParametricHelmholtzPINN
from backend.physics.psf_metrics import (
    compute_psf_skewness, compute_psf_mtf, compute_throughput
)

router = APIRouter()

# ============================================================
# Global model cache
# ============================================================
_model_cache = None


def get_9d_model():
    global _model_cache
    if _model_cache is None:
        root = Path(__file__).parent.parent.parent.parent
        # Phase B stage2 checkpoint (9D)
        ckpt_path = root / "phase_b_stage2.pt"
        if not ckpt_path.exists():
            ckpt_path = root / "parametric_pinn_ckpt.pt"

        model = ParametricHelmholtzPINN(
            hidden_dim=128, num_layers=4, num_freqs=48,
            in_dim=9, use_bm_mask=True, mask_sharpness=2.0
        )
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache = model
    return _model_cache


# ============================================================
# Request / Response models
# ============================================================
class DesignPredictRequest(BaseModel):
    d1: float = 0.0
    d2: float = 0.0
    w1: float = 10.0
    w2: float = 10.0
    theta_deg: float = 0.0


class DesignPredictResponse(BaseModel):
    psf_7: list[float]
    skewness: float
    mtf: float
    throughput: float
    amplitude_profile_z40: list[float]
    intensity_profile_z0: list[float]
    x_coords: list[float]


class InverseRequest(BaseModel):
    target_skewness: float = 0.0
    target_mtf: float = 0.8
    target_throughput: float = 1.0
    n_trials: int = 50
    w_skewness: float = 1.0
    w_mtf: float = 1.0
    w_throughput: float = 1.0


class Candidate(BaseModel):
    d1: float
    d2: float
    w1: float
    w2: float
    psf_7: list[float]
    skewness: float
    mtf: float
    throughput: float
    score: float


class InverseResponse(BaseModel):
    best: Candidate
    pareto: list[Candidate]
    all_trials: list[Candidate]


# ============================================================
# Predict endpoint (9D PINN)
# ============================================================
@router.post("/predict", response_model=DesignPredictResponse)
def predict_design(req: DesignPredictRequest):
    """설계변수 → PSF + amplitude profiles."""
    import math
    from backend.core.parametric_pinn import compute_slit_dist

    model = get_9d_model()

    # PSF 7-OPD
    psf = model.predict_psf7(req.d1, req.d2, req.w1, req.w2)
    psf_list = psf.tolist()

    # z=40 amplitude profile
    n = 504
    x = torch.linspace(0, 504, n)
    theta_rad = math.radians(req.theta_deg)
    sin_th = torch.full((n,), math.sin(theta_rad))
    cos_th = torch.full((n,), math.cos(theta_rad))
    d1_t = torch.full((n,), req.d1)
    d2_t = torch.full((n,), req.d2)
    w1_t = torch.full((n,), req.w1)
    w2_t = torch.full((n,), req.w2)

    z40 = torch.full((n,), 40.0)
    sd40 = compute_slit_dist(x, z40, d1_t, d2_t, w1_t, w2_t)
    coords40 = torch.stack([x, z40, d1_t, d2_t, w1_t, w2_t, sin_th, cos_th, sd40], dim=1)
    with torch.no_grad():
        A40 = model.forward_envelope(coords40)
    amp40 = torch.sqrt(A40[:, 0] ** 2 + A40[:, 1] ** 2).tolist()

    # z=0 intensity profile
    z0 = torch.full((n,), 0.0)
    sd0 = compute_slit_dist(x, z0, d1_t, d2_t, w1_t, w2_t)
    coords0 = torch.stack([x, z0, d1_t, d2_t, w1_t, w2_t, sin_th, cos_th, sd0], dim=1)
    with torch.no_grad():
        A0 = model.forward_envelope(coords0)
    int0 = (A0[:, 0] ** 2 + A0[:, 1] ** 2).tolist()

    return DesignPredictResponse(
        psf_7=psf_list,
        skewness=compute_psf_skewness(psf_list),
        mtf=compute_psf_mtf(psf_list),
        throughput=compute_throughput(psf_list),
        amplitude_profile_z40=amp40,
        intensity_profile_z0=int0,
        x_coords=x.tolist(),
    )


@router.post("/amplitude_map")
def amplitude_map(req: DesignPredictRequest):
    """2D |A| 히트맵 (x x z)."""
    import math
    from backend.core.parametric_pinn import compute_slit_dist

    model = get_9d_model()

    nx, nz = 200, 80
    x_grid = torch.linspace(0, 504, nx)
    z_grid = torch.linspace(0, 40, nz)
    xx, zz = torch.meshgrid(x_grid, z_grid, indexing='xy')
    x_flat = xx.flatten()
    z_flat = zz.flatten()
    n = len(x_flat)

    theta_rad = math.radians(req.theta_deg)
    sin_th = torch.full((n,), math.sin(theta_rad))
    cos_th = torch.full((n,), math.cos(theta_rad))
    d1_t = torch.full((n,), req.d1)
    d2_t = torch.full((n,), req.d2)
    w1_t = torch.full((n,), req.w1)
    w2_t = torch.full((n,), req.w2)
    sd = compute_slit_dist(x_flat, z_flat, d1_t, d2_t, w1_t, w2_t)

    coords = torch.stack([x_flat, z_flat, d1_t, d2_t, w1_t, w2_t, sin_th, cos_th, sd], dim=1)
    with torch.no_grad():
        A = model.forward_envelope(coords)
    amp = torch.sqrt(A[:, 0] ** 2 + A[:, 1] ** 2)
    amp_2d = amp.reshape(nz, nx).tolist()

    return {"amplitude": amp_2d, "nx": nx, "nz": nz}


# ============================================================
# Inverse search (mock BoTorch)
# ============================================================
def _objective(skew, mtf, thru, req):
    return (req.w_skewness * abs(skew - req.target_skewness)
            + req.w_mtf * max(0, req.target_mtf - mtf)
            + req.w_throughput * max(0, req.target_throughput - thru))


def _find_pareto(candidates, top_k=10):
    pareto = []
    for c in candidates:
        dominated = False
        for o in candidates:
            if o is c:
                continue
            if (o.skewness <= c.skewness and o.mtf >= c.mtf
                    and o.throughput >= c.throughput
                    and (o.skewness < c.skewness or o.mtf > c.mtf
                         or o.throughput > c.throughput)):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    pareto.sort(key=lambda x: x.score)
    return pareto[:top_k]


@router.post("/search", response_model=InverseResponse)
def inverse_search(req: InverseRequest):
    """랜덤 search로 설계 공간 탐색 (mock BoTorch)."""
    model = get_9d_model()

    candidates = []
    for _ in range(req.n_trials):
        d1 = float(np.random.uniform(-10, 10))
        d2 = float(np.random.uniform(-10, 10))
        w1 = float(np.random.uniform(5, 20))
        w2 = float(np.random.uniform(5, 20))

        psf = model.predict_psf7(d1, d2, w1, w2)
        psf_list = psf.tolist()

        skew = compute_psf_skewness(psf_list)
        mtf = compute_psf_mtf(psf_list)
        thru = compute_throughput(psf_list)

        candidates.append(Candidate(
            d1=round(d1, 2), d2=round(d2, 2),
            w1=round(w1, 2), w2=round(w2, 2),
            psf_7=psf_list, skewness=round(skew, 4),
            mtf=round(mtf, 4), throughput=round(thru, 4),
            score=round(_objective(skew, mtf, thru, req), 4),
        ))

    best = min(candidates, key=lambda c: c.score)
    pareto = _find_pareto(candidates)

    return InverseResponse(best=best, pareto=pareto, all_trials=candidates)
