"""
test_e2e.py - End-to-End 통합 테스트
=====================================
API → PINN → PSF → Metrics → UI 전체 흐름 검증.

실행: python -m pytest tests/test_e2e.py -v
"""

import pytest
import numpy as np


# ============================================================
# 1. 스키마 & 검증기
# ============================================================
class TestSchemas:
    def test_design_params_default(self):
        from backend.api.schemas import BMDesignParams
        p = BMDesignParams()
        assert p.w1 == 10.0
        assert p.d == 20.0
        assert abs(p.theta_max_eff - 14.036) < 0.1

    def test_design_params_custom(self):
        from backend.api.schemas import BMDesignParams
        p = BMDesignParams(delta_bm1=3.0, delta_bm2=-2.0, w1=15.0, w2=12.0)
        assert p.delta_bm1 == 3.0
        assert p.w1 == 15.0

    def test_validator_pass(self):
        from backend.api.schemas import BMDesignParams
        from backend.harness.physical_validator import BMPhysicalValidator
        v = BMPhysicalValidator()
        p = BMDesignParams(delta_bm1=2.0, w1=12.0, w2=10.0)
        r = v.validate(p)
        assert r.passed

    def test_validator_fail_offset(self):
        from backend.api.schemas import BMDesignParams
        from backend.harness.physical_validator import BMPhysicalValidator
        v = BMPhysicalValidator()
        p = BMDesignParams(delta_bm1=8.0, w1=10.0, w2=10.0)
        r = v.validate(p)
        assert not r.passed


# ============================================================
# 2. 물리 모듈
# ============================================================
class TestPhysics:
    def test_tmm_table(self):
        from backend.physics.tmm_calculator import GorillaDXTMM
        tmm = GorillaDXTMM()
        table = tmm.compute_table()
        assert len(table) == 83
        assert table[0.0] == 0.0
        assert table[30.0] != 0.0
        # 대칭: |Δφ(+30)| == |Δφ(-30)|
        assert abs(abs(table[30.0]) - abs(table[-30.0])) < 0.01

    def test_psf_metrics(self):
        from backend.physics.psf_metrics import PSFMetrics
        m = PSFMetrics()
        # Ridge > Valley → MTF > 0
        psf = np.array([0.2, 0.05, 0.2, 0.05, 0.2, 0.05, 0.2])
        result = m.compute(psf)
        assert result['mtf_ridge'] > 0.5
        assert 'skewness' in result
        assert 'throughput' in result

    def test_psf_metrics_symmetric(self):
        from backend.physics.psf_metrics import PSFMetrics
        m = PSFMetrics()
        psf = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = m.compute(psf)
        assert abs(result['mtf_ridge']) < 0.01  # 대칭이면 MTF ≈ 0


# ============================================================
# 3. PINN 모델
# ============================================================
class TestPINN:
    def test_model_forward(self):
        import torch
        from backend.core.pinn_model import HelmholtzPINN
        model = HelmholtzPINN(hidden_dim=32, num_layers=2, num_freqs=16)
        x = torch.rand(10, 2) * 504
        out = model(x)
        assert out.shape == (10, 2)  # Re, Im

    def test_model_intensity(self):
        import torch
        from backend.core.pinn_model import HelmholtzPINN
        model = HelmholtzPINN(hidden_dim=32, num_layers=2, num_freqs=16)
        x = torch.rand(5, 2) * 504
        I = model.predict_intensity(x)
        assert I.shape == (5,)
        assert (I >= 0).all()  # 세기는 항상 ≥ 0

    def test_checkpoint_load(self):
        import torch
        from pathlib import Path
        ckpt_path = Path("pinn_checkpoint_no_lt.pt")
        if not ckpt_path.exists():
            pytest.skip("No PINN checkpoint")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert 'psf7' in ckpt
        assert len(ckpt['psf7']) == 7
        assert 'mtf' in ckpt


# ============================================================
# 4. FNO 모델
# ============================================================
class TestFNO:
    def test_fno_forward(self):
        import torch
        from backend.core.fno_model import FNOSurrogate
        # modes를 out_dim(7)의 rfft 크기 이하로 설정
        fno = FNOSurrogate(in_dim=4, out_dim=7, width=32, modes=4, num_layers=2)
        x = torch.rand(3, 4)  # 4 design params
        out = fno(x)
        assert out.shape == (3, 7)  # 7 OPD PSF


# ============================================================
# 5. 데이터
# ============================================================
class TestData:
    def test_lhs_sampler(self):
        from backend.data.lhs_sampler import generate_lhs_samples, samples_to_numpy
        samples = generate_lhs_samples(n_samples=20, seed=42)
        assert len(samples) == 20
        arr = samples_to_numpy(samples)
        assert arr.shape == (20, 4)
        # 범위 검증
        assert arr[:, 2].min() >= 5.0   # w1 >= 5
        assert arr[:, 2].max() <= 20.0  # w1 <= 20

    def test_dataset_manager(self):
        from backend.data.dataset_manager import DatasetManager
        from backend.api.schemas import BMDesignParams
        mgr = DatasetManager()
        p = BMDesignParams(delta_bm1=1.0, w1=12.0)
        mgr.add_sample(p, np.random.rand(7), source="test")
        assert mgr.n_samples == 1
        t_params, t_psf = mgr.get_tensors()
        assert t_params.shape == (1, 4)
        assert t_psf.shape == (1, 7)


# ============================================================
# 6. 에이전트
# ============================================================
class TestAgents:
    def test_planner(self):
        from backend.agents.planner_agent import PlannerAgent
        from backend.api.schemas import BMDesignSpec, ParetoWeights
        planner = PlannerAgent()
        plan = planner.create_plan(
            BMDesignSpec(), ParetoWeights(),
            {'fno_ready': True, 'pinn_trained': True}
        )
        assert plan.strategy == 'quick_bo'
        assert plan.use_fno is True

    def test_evaluator(self):
        from backend.agents.evaluator_agent import EvaluatorAgent
        from backend.api.schemas import BMDesignSpec, BMCandidate, BMDesignParams
        ev = EvaluatorAgent(spec=BMDesignSpec())
        c = BMCandidate(
            id='t', label='A',
            params=BMDesignParams(delta_bm1=1, w1=12, w2=10),
            mtf_ridge=0.65, skewness=0.05, throughput=0.70,
            crosstalk_ratio=0.08, evaluator_score=0,
            pareto_rank=1, uncertainty_sigma=0.02, constraint_ok=True,
        )
        scored = ev.score(c)
        assert scored.evaluator_score > 70
        accepted, _ = ev.accept_or_reject(scored)
        assert accepted

    def test_drift_detector(self):
        from backend.harness.drift_detector import DriftDetector
        dd = DriftDetector()
        dd.record_pinn_error(0.03)
        dd.record_pinn_error(0.04)
        dd.record_pinn_error(0.06)  # > 0.05 threshold
        assert dd.needs_retrain


# ============================================================
# 7. FastAPI
# ============================================================
class TestAPI:
    @pytest.fixture
    def client(self):
        from backend.api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health(self, client):
        r = client.get('/api/health')
        assert r.status_code == 200
        assert r.json()['status'] == 'ok'

    def test_psf_inference(self, client):
        r = client.post('/api/inference/psf', json={
            'params': {'delta_bm1': 1.0, 'w1': 12.0, 'w2': 10.0}
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data['psf7']) == 7
        assert 'mtf_ridge' in data['metrics']

    def test_psf_invalid(self, client):
        r = client.post('/api/inference/psf', json={
            'params': {'delta_bm1': 15.0, 'w1': 10.0, 'w2': 10.0}
        })
        assert r.status_code == 422

    def test_design_run(self, client):
        r = client.post('/api/design/run', json={})
        assert r.status_code == 200
        assert 'job_id' in r.json()

    def test_pinn_result(self, client):
        r = client.get('/api/inference/pinn-result')
        if r.status_code == 200:
            data = r.json()
            assert 'results' in data

    def test_training_stats(self, client):
        r = client.get('/api/training/stats')
        assert r.status_code == 200

    def test_frontend_served(self, client):
        r = client.get('/')
        assert r.status_code == 200
        assert 'Design Studio' in r.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
