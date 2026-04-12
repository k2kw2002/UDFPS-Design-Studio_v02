# UDFPS BM 광학 역설계 플랫폼 — Claude Code 구현 가이드 v2

> **목적**: UDFPS COE 스택 BM 광학 구조 역설계 자동화를 위한
> PINN 기반 플랫폼 구현 지침서.
> Claude Code가 처음부터 구현할 수 있도록 아키텍처, 컴포넌트,
> 구현 순서를 정리합니다.

---

## 1. 프로젝트 개요

### 1.1 플랫폼이 하는 일

```
목표 사양 입력 (MTF@ridge, PSF skewness, 광량 T)
        ↓
LHS 샘플링 → LightTools(L_I) + TMM(L_phase) + ASM(L_BC)
        ↓
Helmholtz PINN 학습
L_total = L_Helmholtz + L_phase + L_I + L_BC
        ↓
FNO Surrogate 증류 (PINN → FNO, ~0.8ms/PSF)
        ↓
BoTorch qNEHVI 3목적 역설계 (8초)
        ↓
Pareto front → 최적 BM 설계변수 p* Top-5
        ↓
Design Studio UI — PSF 시각화 / 후보 비교 / 설계변수 확인
```

### 1.2 핵심 가치 (LightTools 단독 대비)

- **역설계 자동화**: 목표 MTF/skewness → 최적 BM 구조 자동 도출
- **위상 왜곡 모델링**: LightTools 불가 → PINN+TMM으로 PSF 비대칭 최초 정량화
- **BO 연결**: LightTools 833시간 → FNO+BoTorch 8초
- **Sim-to-Real**: L_I → I_measured 교체만으로 fine-tuning 완료
- **크로스토크 정량화**: 7피치 도메인으로 2.65피치 크로스토크 직접 포착

---

## 2. 확정 COE 스택 구조

```
┌─────────────────────────────────┐
│  지문 (난반사 — ±41° 전방향)     │
├─────────────────────────────────┤
│  AR Coating (Gorilla DX)        │  Δφ(θ) 위상 왜곡 발생
├─────────────────────────────────┤
│  Cover Glass + OCR  550μm      │  n=1.52  (CG+OCR 합산)
│                                 │  크로스토크 굴절 발생
│                                 │  θ_crit = 41.1°
├─[w₁][][][][w₁][][][][w₁]───────┤  BM1  t=0.1μm [고정]
│  ← δ_BM1 →  아퍼처 피치마다 반복 │  아퍼처 사이: 불투명 BM
├─────────────────────────────────┤
│  ILD (간격 d)                   │  d: 변수
├─[w₂][][][][w₂][][][][w₂]───────┤  BM2  t=0.1μm [고정]
│  ← δ_BM2 →                     │  아퍼처 사이: 불투명 BM
├─────────────────────────────────┤
│  Encap  20μm                    │  [고정]
├──┤R├──────┤V├──────┤R├──────────┤  OPD  10μm폭 / 72μm피치
└─────────────────────────────────┘

R = Ridge (지문 융선), V = Valley (지문 골짜기)
OPD 픽셀: 10μm 폭 (감지 영역) + 62μm 불감지 영역 = 72μm 피치
```

### 2.1 레이어별 두께 요약

| 레이어 | 두께 | 상태 |
|---|---|---|
| AR Coating (Gorilla DX) | ~ 수백nm | 고정 |
| Cover Glass + OCR | 550μm (합산) | 고정 |
| BM1 | 0.1μm | 고정 |
| ILD (d) | 5 ~ 50μm | **변수** |
| BM2 | 0.1μm | 고정 |
| Encap | 20μm | 고정 |
| OPD | 10μm 폭 / 72μm 피치 | 고정 |

---

## 3. 확정 설계변수

### 3.1 최적화 변수 (5개)

```python
class BMDesignParams(BaseModel):
    # ── 최적화 변수 4개 (μm) ──
    delta_bm1: float   # BM1 아퍼처 오프셋 (각 OPD 픽셀 중심 기준)
    delta_bm2: float   # BM2 아퍼처 오프셋 (각 OPD 픽셀 중심 기준)
    w1:        float   # BM1 아퍼처 폭
    w2:        float   # BM2 아퍼처 폭

    # ── 고정값 ──
    d:         float = 20.0   # BM1-BM2 간격 (μm) [고정]
    t_bm1:     float = 0.1    # BM1 두께 (μm)
    t_bm2:     float = 0.1    # BM2 두께 (μm)
    z_encap:   float = 20.0   # Encap 두께 (μm, BM2~OPD 사이)
    opd_pitch: float = 72.0   # OPD 피치 (μm)
    opd_width: float = 10.0   # OPD 픽셀 폭 (μm)
    cg_thick:  float = 550.0  # Cover Glass + OCR 합산 (μm)

    @property
    def theta_max_eff(self) -> float:
        """BM 기하학적 유효 수용각 (도) — d 고정(20μm), w1에 종속"""
        return math.degrees(math.atan(self.w1 / (2 * self.d)))
```

### 3.2 구조 이해 핵심

```
OPD 위치:   피치(72μm) 기준 고정 → δ_OPD 없음
BM 아퍼처:  OPD 픽셀마다 1개씩 → 피치마다 반복
오프셋 정의: δ_BM1, δ_BM2 각각 해당 OPD 픽셀 중심 기준

예) OPD #0 중심 = 0μm
    BM1 아퍼처 중심 = 0 + δ_BM1
    OPD #1 중심 = 72μm
    BM1 아퍼처 중심 = 72 + δ_BM1   ← 동일한 δ 반복
```

---

## 4. 입사각 범위

```
물리적 상한:  θ_crit = arcsin(1/n_CG) = arcsin(1/1.52) ≈ 41.1°
              CG 임계각 — 이 이상은 전반사로 OPD 도달 불가

방향:         ±방향 모두 — 지문 난반사 → 좌우 모든 방향 입사
              단방향(+만)이 아닌 ±41° 양방향

BM 수용각:    θ_eff = ±arctan(w₁ / (2×d))  ← d=20μm 고정, w₁에만 종속
              w₁=10μm, d=20μm → θ_eff ≈ ±14°
              w₁=20μm, d=20μm → θ_eff ≈ ±27°

TMM 계산 범위: -41° ~ +41° 전체
              Δφ(θ) = TMM(|θ|)  (크기 동일, 방향 반전)
```

---

## 5. PINN 도메인 설정

### 5.1 왜 단일 유닛셀(72μm)로 부족한가

```
크로스토크 이동 거리:
  θ ≈ 22.5° 입사 → CG(550μm) 통과 후 Δx ≈ 191μm ≈ 2.65 피치

단일 유닛셀(72μm) PINN:
  직진 신호: 볼 수 있음 ✓
  크로스토크(191μm 이동): 도메인 밖으로 나감 ✗
  → 핵심 문제를 놓침
```

### 5.2 확정 PINN 도메인

```
가로 x:  504μm  = 72μm × 7피치  (중심 ±3 피치)
깊이 z:  570μm  = CG(550) + Encap(20)  [BM 두께 무시 가능]

이유:
  크로스토크 최대 범위 2.65피치 → ±3피치 포함 시 완전 포착
  30mm 대비 1.7% — GPU 1장으로 계산 가능
  중심 OPD의 직진 신호 + 인접 OPD 크로스토크 동시 포착

경계조건:
  x 좌/우: Periodic BC (주기 구조)
  z 상단:  입사파 소스 (AR 코팅 면)
  z 하단:  OPD 면 세기 측정
  BM 위치: U = 0 강제 (불투명)

콜로케이션 포인트: 30,000 ~ 50,000개 (GPU 1장 권장)
```

### 5.3 전체 30mm×30mm와의 관계

```
PINN 출력 (504μm × 7 OPD PSF)
      ↓ 동일 패턴 타일링
전체 30mm × 30mm 지문 이미지
(417 × 417 = 174,000개 OPD 픽셀)

MTF, skewness, FRR/FAR:
  7개 OPD 블록에서 계산 → 전체에 적용
```

---

## 6. 기술 스택

### 6.1 백엔드

```
Python 3.11+
├── 물리 모델링
│   ├── PyTorch 2.x            # PINN / FNO 학습
│   ├── NVIDIA PhysicsNeMo     # Physics-Informed 프레임워크
│   ├── numpy / scipy          # TMM 행렬, ASM 전파
│   └── tmm                    # Transfer Matrix Method
│
├── 광학 시뮬레이션
│   ├── LightTools Python API  # L_I 정답지 (200개)
│   ├── tmm_calculator.py      # Δφ(θ,λ) — L_phase 소스
│   └── asm_propagator.py      # Angular Spectrum Method
│
├── 최적화
│   ├── BoTorch                # qNEHVI 3목적 BO
│   ├── GPyTorch               # GP + UQ
│   └── pyDOE2                 # LHS 샘플링
│
└── API 서버
    ├── FastAPI
    ├── celery                 # 비동기 학습 태스크
    └── uvicorn
```

---

## 7. 프로젝트 구조

```
udfps-pinn-platform/
│
├── backend/
│   ├── core/
│   │   ├── pinn_model.py          # Helmholtz PINN (SIREN)
│   │   ├── fno_model.py           # FNO Surrogate (p→PSF)
│   │   ├── botorch_optimizer.py   # qNEHVI 3목적 역설계
│   │   ├── uq_filter.py           # MC Dropout UQ
│   │   └── active_learning.py     # 능동 학습 루프
│   │
│   ├── physics/
│   │   ├── tmm_calculator.py      # TMM: Δφ(±θ, λ)
│   │   ├── asm_propagator.py      # ASM: CG 전파
│   │   ├── loss_functions.py      # L_Helm + L_phase + L_I + L_BC
│   │   └── psf_metrics.py         # skewness, MTF, T, 크로스토크
│   │
│   ├── harness/
│   │   ├── physical_validator.py  # BM 물리 제약 게이트
│   │   ├── agents_config.py       # AGENTS.md 파싱
│   │   └── drift_detector.py      # PINN 정확도 모니터링
│   │
│   ├── agents/
│   │   ├── planner_agent.py
│   │   ├── generator_agent.py
│   │   └── evaluator_agent.py
│   │
│   ├── data/
│   │   ├── lhs_sampler.py         # 4차원 LHS (δ_BM1,δ_BM2,w₁,w₂)
│   │   ├── lighttools_runner.py   # LT API 래퍼
│   │   ├── dataset_manager.py     # (p, PSF×7) 쌍 관리
│   │   └── flywheel.py            # 재학습 파이프라인
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── design.py
│   │   │   ├── candidates.py
│   │   │   ├── training.py
│   │   │   └── export.py
│   │   └── schemas.py
│   │
│   ├── AGENTS.md                  # BM 도메인 규칙 (최우선 작성)
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── SpecInput/
│       │   ├── ParetoPlot/
│       │   ├── PsfChart/          # PSF 비대칭 + 크로스토크 시각화
│       │   ├── MtfChart/
│       │   ├── FingerprintView/
│       │   ├── DesignVarTable/
│       │   ├── CandidateCard/
│       │   └── KpiBanner/
│       ├── store/designStore.js
│       ├── api/client.js
│       └── App.jsx
│
└── docker-compose.yml
```

---

## 8. AGENTS.md — BM 도메인 규칙

```markdown
# UDFPS BM 광학 설계 도메인 규칙 (AGENTS.md)

## 시스템 고정값 (절대 변경 불가)

### COE 스택
- Cover Glass + OCR: 550μm (합산), n=1.52
- AR 코팅: Gorilla DX
- BM1 두께: 0.1μm
- BM2 두께: 0.1μm
- Encap (BM2~OPD): 20μm
- OPD 픽셀: 폭 10μm, 피치 72μm

### 광학 상수
- 평가 파장: 520nm
- CG 임계각: 41.1° (= arcsin(1/1.52))
- 크로스토크 각도: 22.5° → 이동거리 191μm (2.65 피치)
- Δφ(30°, 520nm): -10.6° (TMM 기준)

### PINN 도메인
- 가로 x: 504μm (7피치 = 72×7) — 크로스토크 포착을 위해 필수
- 깊이 z: 570μm (CG 550 + Encap 20)
- 입사각 범위: -41.1° ~ +41.1° (양방향 필수)

## 최적화 설계변수 (4개, μm)

| 변수 | 범위 | 설명 |
|---|---|---|
| δ_BM1 | -10 ~ +10 | BM1 아퍼처 오프셋 (각 OPD 중심 기준) |
| δ_BM2 | -10 ~ +10 | BM2 아퍼처 오프셋 (각 OPD 중심 기준) |
| w₁    |  5 ~ 20   | BM1 아퍼처 폭 |
| w₂    |  5 ~ 20   | BM2 아퍼처 폭 |

## 물리 하드 제약 (위반 시 즉시 거부)

- w₁ > 0, w₂ > 0 (아퍼처 양수)
- d = 20μm (고정, 변경 불가)
- |δ_BM1| ≤ w₁ / 2  (아퍼처가 인접 픽셀 침범 방지)
- |δ_BM2| ≤ w₂ / 2
- θ_eff = arctan(w₁/40) ≤ 41.1° (d=20μm 고정 기준)
- PINN 훈련 도메인 외삽 금지

## 최적화 목표 (3목적 qNEHVI)

- MTF@ridge  ≥ 60%   (최대화)
- skewness   ≤ 0.10  (최소화 → 부호 반전 후 최대화)
- 광량 T     ≥ 60%   (최대화)

## BoTorch 탐색 규칙

- 초기 후보: LHS 20점 (4차원)
- Acquisition: qNEHVI (qLogNEHVI fallback)
- 수렴 기준: 연속 5 iteration HV 개선 < 0.1%
- UQ 임계값: σ > 0.05 → LightTools 강제 검증

## Evaluator 채점 기준 (각 20점)

1. MTF@ridge 목표 달성도
2. skewness 목표 달성도
3. 광량 T 목표 달성도
4. 수렴 신뢰도 (BO acquisition value)
5. 물리 제약 마진 (d, w 기하 조건 여유)
- 70점 미만: Generator 재탐색 지시

## Active Learning 규칙

- LT 검증 결과 → 자동으로 재학습 데이터 편입
- PINN 예측 오차 > 5% → 재학습 트리거
- MC Dropout σ 상위 20% → LT 우선 검증

## 출력 형식

- 설계변수: μm 단위 명시
- PSF 결과: skewness, FWHM, 피크 세기, 크로스토크 비율 포함
- 불확실도 σ 항상 출력
```

---

## 9. 핵심 모듈 구현 명세

### 9.1 Physical Validator

```python
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class BMDesignParams:
    delta_bm1: float
    delta_bm2: float
    w1:        float
    w2:        float
    d:         float
    t_bm1:     float = 0.1
    t_bm2:     float = 0.1
    z_encap:   float = 20.0
    opd_pitch: float = 72.0
    opd_width: float = 10.0
    cg_thick:  float = 550.0

    @property
    def theta_max_eff(self) -> float:
        return math.degrees(math.atan(self.w1 / (2 * self.d)))

@dataclass
class ValidationResult:
    passed:   bool
    reason:   Optional[str] = None
    fix_hint: Optional[str] = None

class BMPhysicalValidator:
    BOUNDS = {
        "delta_bm1": (-10.0, 10.0),
        "delta_bm2": (-10.0, 10.0),
        "w1":        (5.0,   20.0),
        "w2":        (5.0,   20.0),
    }
    D_FIXED    = 20.0   # BM1-BM2 간격 고정값 (μm)
    THETA_CRIT = 41.1
    N_CG       = 1.52

    def validate(self, p: BMDesignParams) -> ValidationResult:
        # 1) 탐색 범위 검사
        for f, (lo, hi) in self.BOUNDS.items():
            v = getattr(p, f)
            if not (lo <= v <= hi):
                return ValidationResult(False,
                    f"{f}={v:.2f}μm — 범위 [{lo},{hi}] 초과",
                    f"{lo} ≤ {f} ≤ {hi}")

        # 2) 아퍼처/간격 양수
        for f in ["w1", "w2", "d"]:
            if getattr(p, f) <= 0:
                return ValidationResult(False, f"{f} ≤ 0 불가")

        # 3) 오프셋이 아퍼처 절반 초과 방지
        if abs(p.delta_bm1) > p.w1 / 2:
            return ValidationResult(False,
                f"|δ_BM1|={abs(p.delta_bm1):.2f} > w₁/2={p.w1/2:.2f}",
                "δ_BM1 감소 또는 w₁ 증가")
        if abs(p.delta_bm2) > p.w2 / 2:
            return ValidationResult(False,
                f"|δ_BM2|={abs(p.delta_bm2):.2f} > w₂/2={p.w2/2:.2f}",
                "δ_BM2 감소 또는 w₂ 증가")

        # 4) 수용각 — d 고정(20μm) 기준
        if p.theta_max_eff > self.THETA_CRIT:
            return ValidationResult(False,
                f"θ_eff={p.theta_max_eff:.1f}° > 임계각 {self.THETA_CRIT}°",
                "w₁ 감소 필요 (d=20μm 고정)")

        return ValidationResult(True)
```

---

### 9.2 TMM 계산기

```python
import numpy as np
from tmm import coh_tmm

class GorillaDXTMM:
    """
    Gorilla DX AR 코팅 TMM 계산.
    양방향 (-41° ~ +41°) Δφ(θ, λ) 계산.
    PINN L_phase 경계조건 소스.
    데이터 0개 — 물리 방정식에서 자동 계산.
    """
    # Gorilla DX 4층 (SiO2/TiO2 반복, 실제 코팅 스펙 기준 조정 필요)
    N_LIST = [1.0, 1.46, 2.35, 1.46, 2.35, 1.52]
    D_LIST = [np.inf, 100, 70, 120, 50, np.inf]   # nm

    def compute_phase(self, theta_deg: float, wl_nm: float = 520.0) -> float:
        """단일 (θ, λ) 투과 위상 Δφ (도). θ=0° 기준 정규화."""
        th_rad = np.radians(abs(theta_deg))   # TMM은 |θ| 사용
        res  = coh_tmm('p', self.N_LIST, self.D_LIST, th_rad, wl_nm)
        ref  = coh_tmm('p', self.N_LIST, self.D_LIST, 0.0,    wl_nm)
        dphi = np.angle(res['t'], deg=True) - np.angle(ref['t'], deg=True)
        # 방향 부호: +θ와 -θ는 Δφ 크기 동일, 위상 방향 반전
        return float(dphi * np.sign(theta_deg) if theta_deg != 0 else 0.0)

    def compute_table(
        self,
        theta_range=np.arange(-41, 42, 1),   # ±41° 양방향
        wl_nm: float = 520.0
    ) -> dict:
        """L_phase 경계조건 생성용 전체 테이블."""
        return {float(th): self.compute_phase(th, wl_nm) for th in theta_range}
```

---

### 9.3 PINN 손실함수

```python
import torch

class UDFPSPINNLosses:
    """
    UDFPS COE 스택 Helmholtz PINN 4가지 손실함수.

    도메인: x ∈ [0, 504μm] (7피치), z ∈ [0, 570μm]
    L_total = λ1·L_Helm + λ2·L_phase + λ3·L_I + λ4·L_BC

    비중:
      L_Helmholtz ~58% — 파동 방정식 (데이터 0개)
      L_phase     ~23% — AR 코팅 위상 경계조건 (TMM 소스, 데이터 0개)
      L_I         ~11% — LightTools 세기 (200개, 유일한 실험 데이터)
      L_BC        ~ 8% — BM 불투명 경계 (형상 정의, 데이터 0개)
    """

    def __init__(self, lam_helm=1.0, lam_phase=0.5,
                 lam_I=0.3, lam_BC=0.2):
        self.lam = dict(helm=lam_helm, phase=lam_phase,
                        I=lam_I, BC=lam_BC)

    def helmholtz(self, U, x, z, k0, n):
        """∇²U + k₀²n²U = 0"""
        Ux  = torch.autograd.grad(U, x, torch.ones_like(U), create_graph=True)[0]
        Uxx = torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True)[0]
        Uz  = torch.autograd.grad(U, z, torch.ones_like(U), create_graph=True)[0]
        Uzz = torch.autograd.grad(Uz, z, torch.ones_like(Uz), create_graph=True)[0]
        return torch.mean((Uxx + Uzz + (k0*n)**2 * U)**2)

    def phase(self, U_pred_phase, dphi_tmm):
        """
        L_phase = ||∠U_pred − Δφ_TMM||²
        AR 코팅 면(z=570μm)에서 적용.
        ±θ 양방향 위상 포함.
        """
        return torch.mean((U_pred_phase - dphi_tmm)**2)

    def intensity(self, U_pred, I_lt):
        """
        L_I = |||U_pred|² − I_LT||²
        OPD 면(z=0)에서 7개 OPD 픽셀 모두 적용.
        크로스토크 신호도 포함.
        """
        return torch.mean((torch.abs(U_pred)**2 - I_lt)**2)

    def boundary(self, U_at_BM):
        """
        L_BC = ||U||²_BM
        BM 불투명 영역 U=0 강제.
        7피치 × BM1 + BM2 = 14개 BM 영역.
        """
        return torch.mean(torch.abs(U_at_BM)**2)

    def total(self, Lh, Lp, Li, Lb):
        t = (self.lam['helm']*Lh + self.lam['phase']*Lp
           + self.lam['I']*Li    + self.lam['BC']*Lb)
        return dict(total=t, helm=Lh.item(), phase=Lp.item(),
                    I=Li.item(), BC=Lb.item())
```

---

### 9.4 PSF 지표 계산

```python
import numpy as np

class PSFMetrics:
    """
    7개 OPD 픽셀 PSF에서 최적화 지표 계산.
    크로스토크도 정량화.
    """
    PITCH    = 72.0
    OPD_W    = 10.0
    N_PIXELS = 7

    def compute(self, psf_7: np.ndarray) -> dict:
        """
        Args:
            psf_7: 7개 OPD 픽셀의 세기 배열 (shape: 7)
                   인덱스 3 = 중심 OPD (Ridge)

        Returns:
            mtf_ridge, skewness, throughput, crosstalk_ratio
        """
        # Ridge(짝수)와 Valley(홀수) 분리
        ridge_vals  = psf_7[[0, 2, 4, 6]]
        valley_vals = psf_7[[1, 3, 5]]

        # 1) MTF@ridge
        r_mean = ridge_vals.mean()
        v_mean = valley_vals.mean()
        mtf = (r_mean - v_mean) / (r_mean + v_mean + 1e-8)

        # 2) Skewness (중심 OPD 기준)
        xs     = np.arange(self.N_PIXELS) * self.PITCH
        norm   = psf_7 / (psf_7.sum() + 1e-8)
        mu     = np.sum(xs * norm)
        sigma  = np.sqrt(np.sum((xs - mu)**2 * norm) + 1e-8)
        skew   = np.sum(((xs - mu)/sigma)**3 * norm)

        # 3) 광량 T (전체 세기 합, 정규화)
        T = float(psf_7.sum())

        # 4) 크로스토크 비율
        # 중심 OPD(인덱스3) 대비 인접 Valley(인덱스2,4) 세기
        center_val = psf_7[3]
        xtalk_vals = (psf_7[2] + psf_7[4]) / 2.0
        xtalk_ratio = xtalk_vals / (center_val + 1e-8)

        return dict(
            mtf_ridge     = float(np.clip(mtf, 0, 1)),
            skewness      = float(skew),
            throughput    = T,
            crosstalk_ratio = float(xtalk_ratio),
        )
```

---

### 9.5 BoTorch 역설계 엔진

```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHyperVolumeImprovement
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf

class BMOptimizer:
    """
    3목적 qNEHVI 역설계.
    obj1: MTF@ridge (최대화)
    obj2: 광량 T    (최대화)
    obj3: -skewness (최대화 = skewness 최소화)
    """
    BOUNDS_RAW = {
        "delta_bm1": (-10.0, 10.0),
        "delta_bm2": (-10.0, 10.0),
        "w1":        (5.0,   20.0),
        "w2":        (5.0,   20.0),
    }
    D_FIXED   = 20.0              # BM1-BM2 간격 고정
    REF_POINT = torch.tensor([0.0, 0.0, -1.0])

    def __init__(self, fno_surrogate, validator):
        self.fno = fno_surrogate
        self.val = validator
        lo = [v[0] for v in self.BOUNDS_RAW.values()]
        hi = [v[1] for v in self.BOUNDS_RAW.values()]
        self.bounds = torch.tensor([lo, hi], dtype=torch.float64)

    def _eval(self, p_batch):
        results = []
        for p in p_batch:
            params = self._to_params(p)
            v = self.val.validate(params)
            if not v.passed:
                results.append(torch.tensor([0.0, 0.0, -1.0]))
                continue
            psf7 = self.fno(p.unsqueeze(0)).squeeze().numpy()
            m    = PSFMetrics().compute(psf7)
            results.append(torch.tensor([
                m["mtf_ridge"],
                m["throughput"],
                -m["skewness"],
            ]))
        return torch.stack(results)

    def optimize(self, n_iter=50):
        from botorch.utils.sampling import draw_sobol_samples
        X = draw_sobol_samples(bounds=self.bounds, n=1, q=20).squeeze(0)
        Y = self._eval(X)

        for _ in range(n_iter):
            models = [SingleTaskGP(X, Y[:, j:j+1]) for j in range(3)]
            model  = ModelListGP(*models)
            acqf   = qLogNoisyExpectedHyperVolumeImprovement(
                model=model, ref_point=self.REF_POINT, X_baseline=X)
            cands, _ = optimize_acqf(
                acqf, bounds=self.bounds, q=4,
                num_restarts=10, raw_samples=512)
            Y_new = self._eval(cands)
            X = torch.cat([X, cands])
            Y = torch.cat([Y, Y_new])

        mask   = is_non_dominated(Y)
        top5   = Y[mask].sum(dim=1).topk(min(5, mask.sum())).indices
        return dict(pareto_X=X[mask], pareto_Y=Y[mask],
                    top5=[{"p": X[mask][i], "Y": Y[mask][i]} for i in top5])

    def _to_params(self, t):
        k = list(self.BOUNDS_RAW.keys())
        return BMDesignParams(**{k[i]: float(t[i]) for i in range(4)})
```

---

## 10. API 명세

```
POST /api/design/run
     body: { spec: BMDesignSpec, weights: ParetoWeights }
     response: { job_id }

GET  /api/design/status/{job_id}
     response: { status, progress, iteration, pareto_front,
                 pinn_loss: {helm, phase, I, BC} }

GET  /api/design/candidates/{job_id}
     response: { candidates: [BMCandidate] }
     — 각 후보: 설계변수 + MTF/skewness/T/crosstalk + σ + 점수

GET  /api/design/compare?candidate_ids[]=A&candidate_ids[]=B
POST /api/design/export/{candidate_id}

POST /api/inference/psf
     body: { params: BMDesignParams }
     response: { psf7: list[float], metrics: PSFMetrics }
     — Design Explorer 실시간 탐색용 (FNO 0.8ms)
     — Phase 1: 물리 근사 모델 / Phase 3+: FNO

POST /api/training/add_gt
     body: { p: BMDesignParams, psf7_measured: list[float] }
```

### 스키마

```python
class BMDesignSpec(BaseModel):
    mtf_ridge_min:  float = Field(0.60, ge=0.10, le=0.95)
    skewness_max:   float = Field(0.10, ge=0.01, le=0.50)
    throughput_min: float = Field(0.60, ge=0.10, le=0.95)

class ParetoWeights(BaseModel):
    mtf:        float = 0.4
    throughput: float = 0.3
    skewness:   float = 0.3

class BMCandidate(BaseModel):
    id:               str
    label:            str          # "A"~"E"
    params:           BMDesignParams
    mtf_ridge:        float
    skewness:         float
    throughput:       float
    crosstalk_ratio:  float        # 크로스토크 비율
    evaluator_score:  float
    pareto_rank:      int
    uncertainty_sigma: float
    constraint_ok:    bool
```

---

## 11. 구현 순서

### Phase 1 — 기반 (1~2일)
```
1. 프로젝트 구조 생성
2. AGENTS.md 작성 (BM 도메인 규칙)
3. BMPhysicalValidator 구현 + 단위 테스트
4. Pydantic 스키마 정의
5. FastAPI 기본 구조
6. TMM 계산기 — ±41° 양방향 Δφ 테이블 생성 및 검증
```

### Phase 2 — 물리 코어 (3~5일)
```
7.  LHS 샘플러 (4차원, δ_BM1·δ_BM2·w₁·w₂)
8.  LightTools API 래퍼 (7피치 도메인 PSF 7개 수집)
9.  ASM 전파기 (CG 570μm 도메인)
10. PINN 손실함수 4개 + 단위 테스트
11. Helmholtz PINN (SIREN + Fourier Feature Embedding)
12. PINN 학습 파이프라인 (Adam → L-BFGS)
    — 도메인: 504μm × 570μm, 콜로케이션 40,000개
```

### Phase 3 — FNO + 최적화 (3~4일)
```
13. FNO Surrogate (Lifting + Fourier Layer)
    — 입력: p (5개), 출력: PSF×7 OPD
14. PINN → FNO 증류 (10,000쌍)
15. PSFMetrics (skewness, MTF, T, 크로스토크)
16. BoTorch qNEHVI 단일 목적 테스트
17. 3목적 BMOptimizer 구현
18. UQ Filter (MC Dropout)
```

### Phase 4 — 하네스 + 에이전트 (2~3일)
```
19. Active Learning 파이프라인
20. Drift Detector
21. 3-에이전트 오케스트레이터
22. API 엔드포인트 완성
```

### Phase 5 — Design Studio UI (3~4일)
```
23. Vite + React + Zustand 세팅
24. Summary 탭 — Executive Banner + Pareto front
25. Detail 탭  — 역설계 실행 + 후보 선택 (Summary와 공유 Workspace)
26. Explore 탭 — 5개 슬라이더 + 실시간 PSF/KPI
    주의: 슬라이더 addEventListener는 DOMContentLoaded 이후 바인딩
27. drawStack() — 탭 공용 COE 단면 렌더러 (cvEl 파라미터로 캔버스 교체)
28. CandidateCard — 현재 대비 개선량 (+pp) + LT 내보내기 버튼
29. 전체 레이아웃 통합
```

### Phase 6 — 검증 (1~2일)
```
31. 백엔드-프론트엔드 E2E 연결
32. 2D PoC: LT 80개 + PINN coarse 학습
33. 크로스토크 재현 확인 (7피치 도메인)
34. Docker Compose
```

---

## 12. Design Studio UI 구현 명세

### 12.1 전체 레이아웃 (Design Studio v3 확정)

```
플랫폼명: Design Studio

Nav
├── 탭: Summary | Detail | Explore
├── 백엔드 모드 chip (approx / FNO)
├── Iter / HV 상태 chip
└── 역설계 실행 버튼

─── 탭별 화면 구성 ─────────────────────────────────────

[Summary 탭] — 임원·경영진 뷰
  Executive Banner (5개 KPI):
    FRR 개선율 | skewness 개선율 | 광량 개선율 | 설계시간 | 달성 후보 수
  Workspace:
    좌: 목표 사양 슬라이더 + Pareto 가중치 + 진행 상태
    중: Pareto front (현재 설계 노란점 + 최적안 화살표)
        하단: PSF 7개 OPD + 설계변수 비교표 + COE 단면
    우: Top-5 후보 카드 (Hypervolume + 상태 메시지)

[Detail 탭] — 엔지니어 뷰 (역설계 메인)
  Summary와 동일 Workspace, Executive Banner 숨김
  기술 용어 그대로 노출 (MTF, skewness, L_Helm 등)

[Explore 탭] — 수동 탐색 뷰 (설계변수 직접 조정)
  좌: 5개 슬라이더 + 물리 제약 상태 + 수용각 θ_eff + 역설계 결과와 비교
  우: KPI 4개 카드 + PSF 7개 OPD 바 차트 + COE 스택 단면
```

### 12.2 탭 역할 분리

| 탭 | 대상 | 핵심 기능 |
|---|---|---|
| Summary | 임원·경영진 | FRR/FAR 개선율, 현재 대비 수치, 설계 시간 단축 |
| Detail | 엔지니어 | 역설계 실행, Pareto 탐색, 후보 선택, LT 내보내기 |
| Explore | 엔지니어 | 5개 슬라이더 → PSF/MTF/skewness 실시간 확인 |

### 12.3 Executive Banner — KPI 5개

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ FRR 개선      │ PSF 비대칭 개선│ 광학 효율 개선 │ 설계 최적화 시간│ Pareto 후보  │
│ (지문 인식률) │ (선명도)      │ (광량 T)      │              │             │
│ 7.1%          │ 0.047         │ 74%           │ 8초          │ 3/5         │
│ ↓-42%         │ ↓-83%         │ ↑+26pp        │ vs 833h      │ 목표 달성   │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

현재 양산 기준값 (비교 기준):
  MTF_baseline  = 0.42
  skew_baseline = 0.28
  T_baseline    = 0.48
```

### 12.4 Pareto front 시각화 규칙

```
x축: MTF@ridge (지문 인식률)
y축: 광량 T (광학 효율)
원 크기: 1/skewness (클수록 선명도 좋음)

포인트:
  노란 점 = 현재 양산 설계 (기준점)
  파란 점 = Pareto 후보 A~E
  초록 점 = 선택된 최적안 (강조)
  화살표  = 현재 → 최적안 개선 방향

목표 영역:
  초록 점선 = MTF 목표 / T 목표 교차점
  오른쪽 위 영역 = 목표 달성 구간
```

### 12.5 KPI 색상 규칙

| 지표 | 녹색 (달성) | 노란 (근접) | 빨간 (미달) |
|---|---|---|---|
| MTF@ridge | ≥ 60% | 45~60% | < 45% |
| Skewness | ≤ 0.10 | 0.10~0.20 | > 0.20 |
| 광량 T | ≥ 60% | 45~60% | < 45% |
| 크로스토크 | ≤ 10% | 10~20% | > 20% |

### 12.6 Explore 탭 — 실시간 탐색기

```
목적: 설계자가 4개 변수를 직접 조작하며 PSF/성능을 즉시 확인
      역설계 결과 후보를 수동으로 미세조정하는 보조 도구

백엔드 모드:
  approx 모드 (Phase 1~2): 물리 근사 모델, 즉시 계산
  FNO 모드    (Phase 3+):   /api/inference/psf 호출, 0.8ms

슬라이더 → 실시간 업데이트 항목:
  □ KPI 4개 카드 (MTF, skewness, T, 크로스토크)
  □ PSF 7개 OPD 바 차트
  □ COE 스택 단면 + 수용각 광선
  □ θ_eff = ±arctan(w₁/2d) 자동 계산
  □ 물리 제약 위반 여부 (⚠ / △ / ✓)
  □ 역설계 결과 최적안 대비 MTF/skewness 차이

중요 구현 주의사항:
  Explore 탭 슬라이더 addEventListener는
  DOMContentLoaded 이후에 바인딩해야 함
  (슬라이더가 <script> 이후 HTML에 정의되어 있기 때문)
```

```javascript
// ✓ 올바른 바인딩 방법
document.addEventListener('DOMContentLoaded', () => {
  ['esl0','esl1','esl2','esl3','esl4'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', exploreUpdate);
  });
});

// ✗ 잘못된 방법 (null 에러 발생)
['esl0','esl1','esl2','esl3','esl4'].forEach(id => {
  document.getElementById(id).addEventListener('input', exploreUpdate); // TypeError
});
```

### 12.7 후보 카드 구성

```
[안 A]                    Pareto #1
┌────────────────────────────────┐
│ 인식률 MTF  │ 선명도 skew       │
│ 68%  +26pp  │ 0.047  ▼         │
│ 광량 T      │ 크로스토크        │
│ 74%  +26pp  │ 8%               │
├────────────────────────────────┤
│ δ₁+2.5  δ₂-1.0  w₁12.5  w₂10.0  d18 │
├────────────────────────────────┤
│  LightTools 검증 내보내기 →    │
└────────────────────────────────┘

현재 설계 대비 개선량 (+pp) 표시
LightTools 내보내기 버튼 — 선택 후보를 .lt 형식으로 export
```

### 12.8 Zustand 전역 상태

```javascript
const useDesignStore = create((set, get) => ({
  // 역설계 목표
  spec: { mtf_min: 0.60, skew_max: 0.10, T_min: 0.60 },
  weights: { mtf: 0.4, T: 0.3, skew: 0.3 },

  // 현재 양산 기준 (비교용)
  baseline: { mtf: 0.42, skew: 0.28, T: 0.48 },

  // Explore 탭 설계변수
  explore_params: { delta_bm1:0, delta_bm2:0, w1:10, w2:10, d:20 },

  // 역설계 결과
  candidates: [],
  pareto_points: [],
  selected_id: null,
  hypervolume: null,

  // 학습/최적화 상태
  pinn_loss: { helm:null, phase:null, I:null, BC:null },
  iter_current: 0,
  iter_total: 50,
  fno_ready: false,
  backend_mode: 'approx',   // 'approx' | 'fno'
  lt_count: 0,              // LT 사용 횟수 (200개 한도)

  // 액션
  setSpec:          (s)   => set({ spec: {...get().spec, ...s} }),
  setExploreParams: (p)   => set({ explore_params: {...get().explore_params, ...p} }),
  setBackendMode:   (m)   => set({ backend_mode: m }),
  selectCandidate:  (id)  => set({ selected_id: id }),
}));
```

### 12.9 FNO 연동 — 백엔드 모드 전환

```javascript
// api/inference_client.js
const getMetrics = async (params, mode = 'approx') => {
  if (mode === 'approx') {
    return physicsApproxModel(params);   // 브라우저 내 즉시 계산
  }
  // FNO API (Phase 3+)
  const res = await fetch('/api/inference/psf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const { psf7, metrics } = await res.json();
  return metrics;
};
```

---

## 14. Phase 1 PoC 최소 목표 (2D 단순 구조)

```
착수 기준:

1. TMM ±41° 테이블 생성 → LightTools와 5% 이내 검증
2. LHS 4차원 80개 생성 (δ_BM1·δ_BM2·w₁·w₂)
3. LightTools 80개 수집 (7피치 도메인, PSF 7개/케이스)
4. Helmholtz PINN 2D coarse 학습
5. 크로스토크 신호 재현 가능 여부 확인
6. PSF 비대칭 (skewness ≠ 0) 재현 가능 여부 확인

완료 기준:
  □ skewness 정량화 성공
  □ 크로스토크 7피치 도메인에서 포착
  □ PINN ↔ LightTools MSE < 10%
  □ Phase 2 (실제 COE 스택) 착수 판단
```

---

## 15. 주요 주의사항

### 물리 일관성
- TMM 계산: θ=0° 기준 정규화 필수 (Δφ(0°) = 0° 강제)
- 입사각: ±방향 모두 — 단방향만 처리하면 PSF 비대칭 과소평가
- 도메인: 크로스토크 포착 위해 7피치(504μm) 필수
- 단위: 모든 길이 μm, 위상 도(°), 파장 nm

### 하네스 원칙
- BMPhysicalValidator: 절대 우회 불가
- AGENTS.md: 반드시 에이전트 코드보다 먼저 작성
- Evaluator: Generator와 독립 (인스턴스 공유 금지)

### Active Learning
- LT/실측 데이터: `add_ground_truth()` 경유 필수
- 재학습: 비동기 백그라운드 (API 블로킹 금지)
- LT 사용 카운터: 200개 한도 모니터링

### UI
- 플랫폼명: Design Studio (추후 변경 예정)
- 탭 3개: Summary (임원) / Detail (엔지니어) / Explore (수동 탐색)
- Explore 슬라이더: DOMContentLoaded 후 바인딩 필수
- drawStack(): cvEl 파라미터로 Detail/Explore 탭 캔버스 공용 사용
- 후보 카드: 현재 양산 설계 대비 개선량(+pp) 항상 표시
- Pareto: 노란 점 = 현재 설계, 초록 점 = 선택 최적안, 화살표 = 개선 방향
- skewness 배지: ≤0.05 녹색 / ≤0.10 노란 / >0.10 빨간
