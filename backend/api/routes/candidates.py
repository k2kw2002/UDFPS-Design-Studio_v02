"""
candidates.py - 최적화 후보 조회/비교 라우트
=============================================
GET  /api/design/candidates/{job_id}   후보 목록
GET  /api/design/compare               후보 비교
"""

from typing import List

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import BMCandidate
from backend.api.routes.design import _jobs

router = APIRouter()


@router.get("/candidates/{job_id}", response_model=List[BMCandidate])
def get_candidates(job_id: str):
    """역설계 결과 후보 목록 조회."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is {job['status']}, not completed yet",
        )
    return job["candidates"]


@router.get("/compare")
def compare_candidates(candidate_ids: List[str] = Query(...)):
    """
    후보 간 비교.
    candidate_ids로 전달된 후보들의 설계변수/성능을 나란히 비교.
    """
    results = []
    for job in _jobs.values():
        for c in job.get("candidates", []):
            if isinstance(c, dict) and c.get("id") in candidate_ids:
                results.append(c)
            elif hasattr(c, "id") and c.id in candidate_ids:
                results.append(c.model_dump() if hasattr(c, "model_dump") else c)

    if not results:
        raise HTTPException(status_code=404, detail="No matching candidates found")

    return {"candidates": results, "count": len(results)}
