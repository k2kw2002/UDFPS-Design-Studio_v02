"""
training.py - 학습 데이터 관리 라우트
======================================
POST /api/training/add_gt     LightTools 검증 데이터 추가
GET  /api/training/stats      학습 데이터 통계
"""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.schemas import BMDesignParams

router = APIRouter()

# --- 인메모리 학습 데이터 저장소 (추후 DB 교체) ---
_training_data: list[dict] = []


class AddGroundTruthRequest(BaseModel):
    params: BMDesignParams
    psf7_measured: List[float] = Field(
        ..., min_length=7, max_length=7,
        description="LightTools에서 측정한 7개 OPD PSF 값",
    )


class AddGroundTruthResponse(BaseModel):
    status: str
    total_samples: int


class TrainingStatsResponse(BaseModel):
    total_samples: int
    needs_retraining: bool
    last_pinn_error: float | None = None


@router.post("/add_gt", response_model=AddGroundTruthResponse)
def add_ground_truth(req: AddGroundTruthRequest):
    """
    LightTools 측정 데이터 추가.
    Active Learning 루프에서 자동 호출됨.
    PINN 재학습 트리거 조건: 누적 5개 이상 새 데이터.
    """
    _training_data.append({
        "params": req.params.model_dump(),
        "psf7": req.psf7_measured,
    })
    return AddGroundTruthResponse(
        status="added",
        total_samples=len(_training_data),
    )


@router.get("/stats", response_model=TrainingStatsResponse)
def get_training_stats():
    """학습 데이터 현황."""
    return TrainingStatsResponse(
        total_samples=len(_training_data),
        needs_retraining=len(_training_data) >= 5,
        last_pinn_error=None,
    )
