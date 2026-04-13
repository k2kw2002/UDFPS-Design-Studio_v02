"""
main.py - FastAPI 서버 진입점
================================
uvicorn backend.api.main:app --reload 로 실행.

라우트:
  /api/design/*      역설계 실행/상태/후보 조회
  /api/inference/*    FNO 실시간 PSF 추론
  /api/training/*     학습 데이터 추가
  /api/export/*       LightTools 내보내기
  /                   Design Studio UI (정적 파일)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes.design import router as design_router
from backend.api.routes.candidates import router as candidates_router
from backend.api.routes.training import router as training_router
from backend.api.routes.export import router as export_router
from backend.api.routes.inverse import router as inverse_router

app = FastAPI(
    title="UDFPS BM Design Studio",
    description="PINN 기반 UDFPS BM 광학 역설계 플랫폼 API",
    version="0.1.0",
)

# CORS - 프론트엔드(Vite dev server) 연결 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(design_router, prefix="/api/design", tags=["design"])
app.include_router(candidates_router, prefix="/api/design", tags=["candidates"])
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(export_router, prefix="/api/design", tags=["export"])
app.include_router(inverse_router, prefix="/api/inverse", tags=["inverse"])

# PSF inference는 /api/inference 경로로 별도 등록
from backend.api.routes.design import router as _design_router
from fastapi import APIRouter as _AR
_inference_router = _AR()
_inference_router.add_api_route(
    "/psf",
    endpoint=__import__("backend.api.routes.design", fromlist=["infer_psf"]).infer_psf,
    methods=["POST"],
    tags=["inference"],
)
_inference_router.add_api_route(
    "/pinn-result",
    endpoint=__import__("backend.api.routes.design", fromlist=["get_pinn_result"]).get_pinn_result,
    methods=["GET"],
    tags=["inference"],
)
app.include_router(_inference_router, prefix="/api/inference")


@app.get("/api/health")
def health_check():
    """서버 상태 + 모델 상태."""
    from pathlib import Path
    root = Path(__file__).parent.parent.parent

    models = {
        "parametric_pinn": (root / "parametric_pinn_ckpt.pt").exists(),
        "fno": (root / "fno_surrogate_ckpt.pt").exists(),
        "botorch": (root / "botorch_top5.pt").exists(),
        "pinn_results": (root / "pinn_results_all.pt").exists(),
        "ar_lut": (root / "backend" / "physics" / "ar_coating" / "data" / "ar_lut.npz").exists(),
    }

    # 활성 엔진: 가장 상위 모델 자동 선택
    if models["fno"]:
        active = "fno"
    elif models["parametric_pinn"]:
        active = "parametric_pinn"
    else:
        active = "asm"

    return {
        "status": "ok",
        "platform": "UDFPS BM Design Studio",
        "models": models,
        "active_engine": active,
        "pipeline": {
            "training": "PINN (Helmholtz + AR phase + ASM L_I + BM BC)",
            "distillation": "PINN -> FNO (10k teacher samples)" if models["fno"] else "not done",
            "optimization": "FNO + BoTorch qNEHVI 3-objective" if models["botorch"] else "not done",
            "inference": active,
        }
    }


@app.get("/api/inference/predict")
def unified_predict(
    delta_bm1: float = 0, delta_bm2: float = 0,
    w1: float = 10, w2: float = 10,
    use_ar: bool = True,
):
    """
    통합 추론 API — 가용한 최상위 모델로 PSF 예측.
    우선순위: FNO(0.1ms) > Parametric PINN(1ms) > ASM(1s)
    """
    from pathlib import Path
    import torch
    import numpy as np
    from backend.physics.psf_metrics import PSFMetrics

    root = Path(__file__).parent.parent.parent
    metrics_calc = PSFMetrics()

    def make_response(psf7, source, latency):
        m = metrics_calc.compute(psf7)
        pmax, pmin = float(psf7.max()), float(psf7.min())
        return {
            "psf7": psf7.tolist(), "metrics": m,
            "contrast": (pmax - pmin) / (pmax + pmin + 1e-10),
            "source": source, "latency_ms": latency,
        }

    # 1순위: FNO Surrogate (~0.1ms)
    fno_path = root / "fno_surrogate_ckpt.pt"
    if fno_path.exists():
        try:
            from backend.core.fno_model import FNOSurrogate
            ckpt = torch.load(str(fno_path), map_location="cpu", weights_only=False)
            fno = FNOSurrogate(in_dim=4, out_dim=7, width=32, modes=4, num_layers=3)
            fno.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
            fno.eval()
            with torch.no_grad():
                x = torch.tensor([[delta_bm1, delta_bm2, w1, w2]], dtype=torch.float32)
                psf7 = fno(x).squeeze().numpy()
            return make_response(psf7, "fno_surrogate", "<0.1")
        except Exception:
            pass

    # 2순위: Parametric PINN (~1ms)
    pinn_path = root / "parametric_pinn_ckpt.pt"
    if pinn_path.exists():
        try:
            from backend.core.parametric_pinn import ParametricHelmholtzPINN
            ckpt = torch.load(str(pinn_path), map_location="cpu", weights_only=False)
            model = ParametricHelmholtzPINN(hidden_dim=128, num_layers=4, num_freqs=48)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            psf7 = model.predict_psf7(delta_bm1, delta_bm2, w1, w2)
            return make_response(psf7, "parametric_pinn", "~1")
        except Exception:
            pass

    # 3순위: ASM (~1s)
    from backend.physics.optical_pipeline import OpticalPipeline
    pipe = OpticalPipeline(dx=0.5, n_angles=21)
    result = pipe.compute_with_metrics(
        delta_bm1=delta_bm1, delta_bm2=delta_bm2,
        w1=w1, w2=w2, use_ar=use_ar,
    )
    return make_response(result["psf7"], "asm_wave_optics", "~1000")


@app.get("/api/inference/reverse-design")
def reverse_design_endpoint(
    n_candidates: int = 10,
    mtf_min: float = 0.3,
):
    """
    역설계 API — BoTorch 결과 또는 그리드 서치.
    BoTorch Top-5 있으면 반환, 없으면 실시간 탐색.
    """
    from pathlib import Path
    import torch
    import numpy as np
    from backend.physics.psf_metrics import PSFMetrics

    root = Path(__file__).parent.parent.parent
    metrics_calc = PSFMetrics()

    # 1순위: BoTorch 사전 계산 결과
    bo_path = root / "botorch_top5.pt"
    if bo_path.exists():
        try:
            data = torch.load(str(bo_path), map_location="cpu", weights_only=False)
            return {"candidates": data["top5"], "source": "botorch_qnehvi"}
        except Exception:
            pass

    # 2순위: FNO/PINN 기반 그리드 서치
    candidates = []
    for d1 in np.arange(-8, 9, 2):
        for w1 in np.arange(6, 20, 2):
            for w2 in np.arange(6, 20, 2):
                if abs(d1) > w1 / 2:
                    continue
                resp = unified_predict(delta_bm1=d1, w1=w1, w2=w2)
                if resp["contrast"] >= mtf_min:
                    candidates.append({
                        "params": {"delta_bm1": d1, "delta_bm2": 0, "w1": w1, "w2": w2},
                        "contrast": resp["contrast"],
                        "metrics": resp["metrics"],
                        "source": resp["source"],
                    })

    candidates.sort(key=lambda c: c["contrast"], reverse=True)
    return {
        "candidates": candidates[:n_candidates],
        "total_searched": len(candidates),
        "source": "grid_search_" + (candidates[0]["source"] if candidates else "none"),
    }


@app.get("/api/inference/pinn-result")
def pinn_result_endpoint():
    """PINN 학습 결과 반환."""
    from backend.api.routes.design import get_pinn_result
    return get_pinn_result()


@app.get("/api/inference/asm-psf")
def asm_psf_endpoint(
    delta_bm1: float = 0, delta_bm2: float = 0,
    w1: float = 10, w2: float = 10,
    use_ar: bool = True, cg_thick: float = 550,
):
    """ASM 파동광학 기반 PSF 계산."""
    from backend.physics.optical_pipeline import OpticalPipeline
    pipe = OpticalPipeline(dx=0.5, n_angles=21)
    result = pipe.compute_with_metrics(
        delta_bm1=delta_bm1, delta_bm2=delta_bm2,
        w1=w1, w2=w2, use_ar=use_ar, cg_override=cg_thick,
    )
    psf7 = result["psf7"]
    pmax, pmin = float(psf7.max()), float(psf7.min())
    contrast = (pmax - pmin) / (pmax + pmin + 1e-10)
    return {
        "psf7": psf7.tolist(),
        "metrics": result["metrics"],
        "contrast": contrast,
        "cg_thick": cg_thick,
        "use_ar": use_ar,
        "source": "asm_wave_optics",
    }


@app.get("/api/inference/asm-comparison")
def asm_comparison_endpoint(w1: float = 10, w2: float = 10):
    """CG only vs AR baseline vs AR optimized 비교 (ASM 기반)."""
    from backend.physics.optical_pipeline import OpticalPipeline
    pipe = OpticalPipeline(dx=0.5, n_angles=21)

    results = {}
    configs = [
        ("CG_only", {"w1": w1, "w2": w2, "use_ar": False}),
        ("AR_baseline", {"w1": 10, "w2": 10, "use_ar": True}),
        ("AR_optimized", {"w1": w1, "w2": w2, "use_ar": True}),
    ]
    for name, kw in configs:
        r = pipe.compute_with_metrics(**kw)
        psf7 = r["psf7"]
        pmax, pmin = float(psf7.max()), float(psf7.min())
        results[name] = {
            "psf7": psf7.tolist(),
            "metrics": r["metrics"],
            "contrast": (pmax - pmin) / (pmax + pmin + 1e-10),
        }

    # CG 두께별 contrast
    cg_sweep = []
    for cg in [50, 100, 200, 300, 400, 550]:
        psf7 = pipe.compute_psf7(w1=10, w2=10, use_ar=False, cg_override=cg)
        pmax, pmin = float(psf7.max()), float(psf7.min())
        cg_sweep.append({"cg_um": cg, "contrast": (pmax-pmin)/(pmax+pmin+1e-10)})

    return {"results": results, "cg_sweep": cg_sweep, "source": "asm_wave_optics"}


@app.get("/api/inference/botorch-result")
def botorch_result_endpoint():
    """BoTorch qNEHVI 최적화 결과."""
    import torch
    from pathlib import Path
    ckpt = Path(__file__).parent.parent.parent / "botorch_results.pt"
    if not ckpt.exists():
        return {"status": "not_found", "message": "Run run_fno_botorch.py first"}
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return {
        "status": "ok",
        "top5": data["top5"],
        "hv_history": [float(h) for h in data.get("hv_history", [])],
    }


# 프론트엔드 정적 파일 서빙
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/src", StaticFiles(directory=str(_frontend_dir / "src")), name="frontend-src")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(_frontend_dir / "index.html"))
