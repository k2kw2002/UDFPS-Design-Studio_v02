"""
design.py - 역설계 실행/상태 라우트
====================================
POST /api/design/run       역설계 작업 시작
GET  /api/design/status/{job_id}  작업 상태 조회
POST /api/inference/psf    실시간 PSF 추론 (FNO)
GET  /api/inference/pinn   PINN 학습 결과 PSF
"""

import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.schemas import BMDesignParams, BMDesignSpec, ParetoWeights
from backend.harness.physical_validator import BMPhysicalValidator

router = APIRouter()

# --- 인메모리 작업 저장소 (추후 Redis/Celery 교체) ---
_jobs: Dict[str, Dict[str, Any]] = {}

validator = BMPhysicalValidator()


# --- 요청/응답 모델 ---
class DesignRunRequest(BaseModel):
    spec: BMDesignSpec = BMDesignSpec()
    weights: ParetoWeights = ParetoWeights()


class DesignRunResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    iteration: int
    pinn_loss: Dict[str, float] | None = None


class PsfInferenceRequest(BaseModel):
    params: BMDesignParams


class PsfInferenceResponse(BaseModel):
    psf7: list[float]
    metrics: Dict[str, float]


# --- 엔드포인트 ---
@router.post("/run", response_model=DesignRunResponse)
def run_design(req: DesignRunRequest):
    """
    역설계 작업 시작.
    Phase 3 이후: Celery 태스크로 BoTorch qNEHVI 실행.
    현재: 작업 ID만 생성하고 pending 상태 반환.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "iteration": 0,
        "spec": req.spec.model_dump(),
        "weights": req.weights.model_dump(),
        "pinn_loss": None,
        "candidates": [],
    }
    return DesignRunResponse(job_id=job_id, status="pending")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    """역설계 작업 상태 조회."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    j = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        iteration=j["iteration"],
        pinn_loss=j["pinn_loss"],
    )


@router.post("/psf", response_model=PsfInferenceResponse)
def infer_psf(req: PsfInferenceRequest):
    """
    설계변수 → PSF 실시간 추론.
    Phase 1: 물리 근사 모델 (ASM 기반).
    Phase 3+: FNO 서로게이트 (0.8ms).
    """
    # 물리 제약 검증
    vr = validator.validate(req.params)
    if not vr.passed:
        raise HTTPException(status_code=422, detail=vr.reason)

    # Phase 1: ASM 기반 근사 PSF 계산
    psf7, metrics = _compute_approx_psf(req.params)
    return PsfInferenceResponse(psf7=psf7, metrics=metrics)


@router.get("/pinn-result")
def get_pinn_result():
    """
    PINN 학습 결과 반환.
    체크포인트 파일에서 PSF, MTF, loss 정보를 로드.
    """
    import torch
    import numpy as np
    from backend.physics.psf_metrics import PSFMetrics

    # 배치 결과 (7개 설정) 우선 로드
    batch_path = Path(__file__).parent.parent.parent.parent / "pinn_results_all.pt"
    single_path = Path(__file__).parent.parent.parent.parent / "pinn_checkpoint_no_lt.pt"

    if batch_path.exists():
        all_results = torch.load(str(batch_path), map_location="cpu", weights_only=False)
        return {
            "source": "pinn_batch",
            "results": all_results,
            "note": "Batch PINN: CG only + AR baseline + 5 optimized candidates",
        }

    if single_path.exists():
        ckpt = torch.load(str(single_path), map_location="cpu", weights_only=False)
        psf7 = ckpt["psf7"]
        mtf = ckpt["mtf"]
        psf_arr = np.array(psf7)
        metrics = PSFMetrics().compute(psf_arr)
        return {
            "source": "pinn",
            "results": {
                "AR_baseline": {
                    "name": "AR_baseline",
                    "label": "With AR + Baseline",
                    "psf7": psf7,
                    "mtf": mtf,
                    "metrics": metrics,
                    "params": ckpt["params"],
                    "best_loss": ckpt["best_loss"],
                }
            },
            "note": "Single PINN baseline only",
        }

    raise HTTPException(status_code=404, detail="No PINN checkpoint. Run train_pinn_batch.py")


def _compute_approx_psf(params: BMDesignParams) -> tuple[list[float], dict]:
    """
    Phase 1 근사 PSF 계산 (ASM + 기하광학).
    FNO 학습 전까지의 임시 구현.
    """
    import numpy as np

    # 기하광학 기반 단순 PSF 근사
    n_opd = 7
    pitch = params.opd_pitch
    center_idx = 3  # 중심 OPD

    psf = np.zeros(n_opd)
    for i in range(n_opd):
        x_opd = (i - center_idx) * pitch
        # BM1 아퍼처 통과 계산
        bm1_center = x_opd + params.delta_bm1
        bm1_half = params.w1 / 2.0
        # BM2 아퍼처 통과 계산
        bm2_center = x_opd + params.delta_bm2
        bm2_half = params.w2 / 2.0

        # 유효 아퍼처 오버랩 (BM1 & BM2 교차)
        overlap_lo = max(bm1_center - bm1_half, bm2_center - bm2_half)
        overlap_hi = min(bm1_center + bm1_half, bm2_center + bm2_half)
        eff_width = max(0.0, overlap_hi - overlap_lo)

        # 직진 성분 (중심 OPD에서 가장 강함)
        geom_factor = eff_width / pitch
        # 거리 감쇠 (크로스토크 반영)
        dist = abs(i - center_idx)
        decay = np.exp(-0.5 * dist)

        psf[i] = geom_factor * (1.0 + 0.3 * decay)

    # 정규화
    total = psf.sum()
    if total > 0:
        psf = psf / total

    # 메트릭 계산
    from backend.physics.psf_metrics import PSFMetrics
    metrics = PSFMetrics().compute(psf)

    return psf.tolist(), metrics
