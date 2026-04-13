# Project Map - UDFPS BM Design Studio

## 한눈에 보는 파이프라인

```
[지문 패턴] → [AR 코팅(TMM)] → [CG 전파(ASM)] → [BM 회절(PINN)] → [PSF 추출] → [역설계(BoTorch)]
                                                         ↓
                                                   [FNO 증류] → 실시간 추론
                                                         ↓
                                                   [Design Studio UI]
```

---

## 읽는 순서 (권장)

### 1단계: 물리 엔진 (입력 → 출력 흐름)

| 순서 | 파일 | 역할 | 핵심 |
|------|------|------|------|
| 1 | `backend/physics/tmm_calculator.py` | AR 코팅 위상/투과 계산 | SiO2/TiO2 4층, dphi(30deg)=-10.3deg |
| 2 | `backend/physics/ar_coating/ar_boundary.py` | AR LUT 조회 (O(1)) | ar_lut.npz에서 T(theta) 즉시 반환 |
| 3 | `backend/physics/asm_propagator.py` | CG 550um 파동 전파 | FFT 기반 Angular Spectrum Method |
| 4 | `backend/physics/optical_pipeline.py` | 전체 광학 시뮬레이션 | 지문→AR→CG→BM1→ILD→BM2→OPD |
| 5 | `backend/physics/psf_metrics.py` | PSF 성능 지표 | MTF, skewness, throughput |
| 6 | `backend/physics/loss_functions.py` | PINN 손실 함수 | L_Helmholtz + L_phase + L_I + L_BC |

### 2단계: PINN 모델 (핵심)

| 순서 | 파일 | 역할 | 핵심 |
|------|------|------|------|
| 7 | `backend/core/parametric_pinn.py` | **9D Carrier-Envelope PINN** | 현재 주력 모델. slit_dist 입력 + hard mask |
| 8 | `train_phase_b.py` | **학습 스크립트** | Stage1(L_phase) → Stage2(+PDE). 현재 작업 중 |
| 9 | `train_parametric_pinn.py` | 이전 학습 스크립트 (8D) | Phase B 이전 버전. 참고용 |
| 10 | `backend/core/pinn_model.py` | 기본 PINN (2D, 단일 설계) | 초기 프로토타입. 사용 안 함 |

### 3단계: 서로게이트 + 최적화

| 순서 | 파일 | 역할 | 핵심 |
|------|------|------|------|
| 11 | `backend/core/fno_model.py` | FNO 서로게이트 | PINN→FNO 증류. 4D→PSF7, ~0.1ms |
| 12 | `backend/core/botorch_optimizer.py` | BoTorch 3목적 최적화 | qNEHVI. MTF↑ skewness↓ throughput↑ |
| 13 | `run_distill_optimize.py` | 증류+최적화 파이프라인 | PINN학습→FNO증류→BoTorch→Top5 |

### 4단계: API 서버 + UI

| 순서 | 파일 | 역할 |
|------|------|------|
| 14 | `backend/api/main.py` | FastAPI 서버 |
| 15 | `backend/api/schemas.py` | API 데이터 모델 |
| 16 | `frontend/index.html` | UI 진입점 |
| 17 | `frontend/src/app.js` | SPA 메인 컨트롤러 |
| 18 | `frontend/src/api/client.js` | API 클라이언트 |

---

## 디렉토리 구조

```
AI_PINN_PROJECT_v02/
│
├── backend/
│   ├── api/                    # 웹 서버
│   │   ├── main.py             # FastAPI 진입점
│   │   ├── schemas.py          # 데이터 모델 (BMDesignParams 등)
│   │   └── routes/
│   │       ├── design.py       # POST /api/design/run
│   │       ├── candidates.py   # GET /api/candidates
│   │       ├── training.py     # POST /api/training/add
│   │       └── export.py       # GET /api/export/{id}
│   │
│   ├── core/                   # 딥러닝 모델
│   │   ├── parametric_pinn.py  # ★ 9D Carrier-Envelope PINN (현재 주력)
│   │   ├── pinn_model.py       # 기본 PINN (2D, 레거시)
│   │   ├── pinn_trainer.py     # 학습 루프 (레거시)
│   │   ├── fno_model.py        # FNO 서로게이트
│   │   ├── botorch_optimizer.py # BoTorch 최적화
│   │   ├── active_learning.py  # 능동 학습
│   │   └── uq_filter.py       # MC Dropout 불확실성
│   │
│   ├── physics/                # 광학 물리
│   │   ├── tmm_calculator.py   # AR 코팅 TMM
│   │   ├── asm_propagator.py   # CG 전파 ASM
│   │   ├── optical_pipeline.py # 전체 시뮬레이션
│   │   ├── psf_metrics.py      # MTF/skewness/throughput
│   │   ├── loss_functions.py   # PINN 손실 함수
│   │   └── ar_coating/
│   │       ├── ar_boundary.py  # AR LUT 조회
│   │       ├── data/ar_lut.npz # AR LUT 데이터
│   │       └── generate_lut.py # LUT 생성
│   │
│   ├── data/                   # 데이터 관리
│   │   ├── dataset_manager.py  # 통합 데이터 인터페이스
│   │   ├── lighttools_runner.py # LightTools 연동
│   │   └── lhs_sampler.py      # Latin Hypercube 샘플링
│   │
│   ├── harness/                # 오케스트레이션
│   │   ├── physical_validator.py # 물리 제약 검증
│   │   ├── drift_detector.py    # 모델 정확도 모니터링
│   │   └── agents_config.py     # 도메인 규칙 파서
│   │
│   └── agents/                 # AI 에이전트
│       ├── planner_agent.py    # 전략 에이전트
│       ├── generator_agent.py  # 실행 에이전트
│       └── evaluator_agent.py  # 평가 에이전트
│
├── frontend/
│   ├── index.html              # SPA 진입점
│   └── src/
│       ├── app.js              # 메인 앱 (상태 관리)
│       ├── api/client.js       # API 호출
│       └── components/
│           ├── FingerprintView.js  # 지문+PSF 시각화
│           ├── SensorSimView.js    # OPD 센서 응답
│           ├── CandidateCard.js    # 설계 후보 카드
│           ├── drawPareto.js       # Pareto front 차트
│           ├── drawPsf.js          # PSF 프로파일
│           └── drawStack.js        # 레이어 구조도
│
├── train_phase_b.py            # ★ 현재 학습 스크립트 (9D PINN)
├── train_parametric_pinn.py    # 이전 학습 (8D, 참고용)
├── run_distill_optimize.py     # FNO증류 + BoTorch 파이프라인
├── start.py                    # 서버 시작
│
├── experiments/                # 실험 아카이브 (사용 안 함)
├── tests/                      # 테스트
│   ├── test_e2e.py
│   └── test_physics_verification.py
│
├── Dockerfile                  # Docker 빌드
├── docker-compose.yml          # Docker 오케스트레이션
└── requirements.txt            # Python 의존성
```

---

## 현재 상태 (Phase B 완료)

### 해결된 것
- Carrier-Envelope 분리 → L_phase 0.619 문제 해결
- Carrier reference z=40 → target 위상 문제 해결
- 동적 정규화 → Lh/Lp 스케일 불균형 해결
- 9D 입력(slit_dist) + hard mask → BM 공간 분화 달성

### 현재 결과
```
slit |A| = 0.743  (target 0.793)
BM   |A| = 0.009  (≈ 0)
PDE  Lh  = 378    (수렴)
```

### 남은 단계
1. **Phase C**: L_data (ASM PSF 타겟) 추가 → z=0 PSF 정확도
2. **Phase D**: FNO 증류 + BoTorch 최적화 연결
3. **GPU 학습**: epochs/collocation 증가 → slit 0.743 → 0.793
4. **UI 연결**: 9D PINN 추론 API 업데이트

---

## 핵심 물리 상수

| 상수 | 값 | 의미 |
|------|-----|------|
| 파장 | 0.520 um | 녹색 LED |
| n_CG | 1.52 | Cover Glass 굴절률 |
| CG 두께 | 550 um | Gorilla DX |
| k0 | 12.08 um^-1 | 자유공간 파수 |
| k | 18.37 um^-1 | 매질 내 파수 (k0*n_CG) |
| OPD pitch | 72 um | OPD 픽셀 간격 |
| N_pitches | 7 | 시뮬레이션 픽셀 수 |
| BM1 | z=40 um | 상단 BM (CG 쪽) |
| BM2 | z=20 um | 하단 BM (OPD 쪽) |
| 설계 변수 | d1,d2 ∈ [-10,10], w1,w2 ∈ [5,20] um | BM 오프셋/폭 |
| 입사각 | ±41.1° (7각도) | 지문 산란 범위 |
