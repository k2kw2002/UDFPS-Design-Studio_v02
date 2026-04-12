# UDFPS BM 광학 설계 도메인 규칙 (AGENTS.md)
#
# 이 파일은 "물리 법칙 사전"입니다.
# 모든 코드는 이 규칙을 반드시 따라야 합니다.
# AI 에이전트(Planner, Generator, Evaluator)도 이 규칙을 읽고 판단합니다.

## 시스템 고정값 (절대 변경 불가)

### COE 스택 (층 구조)
- Cover Glass + OCR: 550um (합산), 굴절률 n=1.52
- AR 코팅: Gorilla DX (4층 SiO2/TiO2)
- BM1 두께: 0.1um
- BM2 두께: 0.1um
- Encap (BM2~OPD 사이): 20um
- OPD 픽셀: 폭 10um, 피치 72um

### 광학 상수
- 평가 파장: 520nm
- CG 임계각: 41.1도 (= arcsin(1/1.52))
- 크로스토크 각도: 22.5도 -> 이동거리 191um (2.65피치)
- DeltaPhi(30도, 520nm): -10.6도 (TMM 기준)

### PINN 도메인
- 가로 x: 504um (7피치 = 72 x 7) -> 크로스토크 포착 위해 필수
- 깊이 z: 570um (CG 550 + Encap 20)
- 입사각 범위: -41.1도 ~ +41.1도 (양방향 필수)

## 최적화 설계변수 (4개, um 단위)

| 변수 | 범위 | 설명 |
|---|---|---|
| delta_bm1 | -10 ~ +10 | BM1 어퍼처 오프셋 (각 OPD 중심 기준) |
| delta_bm2 | -10 ~ +10 | BM2 어퍼처 오프셋 (각 OPD 중심 기준) |
| w1        |  5 ~ 20   | BM1 어퍼처 폭 |
| w2        |  5 ~ 20   | BM2 어퍼처 폭 |

## 물리 하드 제약 (위반 시 즉시 거부)

- w1 > 0, w2 > 0 (어퍼처 양수)
- d = 20um (고정, 변경 불가)
- |delta_bm1| <= w1 / 2  (어퍼처가 인접 픽셀 침범 방지)
- |delta_bm2| <= w2 / 2
- theta_eff = arctan(w1/40) <= 41.1도 (d=20um 고정 기준)
- PINN 훈련 도메인 인코딩 금지

## 최적화 목표 (3목적 qNEHVI)

- MTF@ridge  >= 60%   (최대화)
- skewness   <= 0.10  (최소화 -> 부호 반전 후 최대화)
- 광량 T     >= 60%   (최대화)

## BoTorch 필수 규칙

- 초기 후보: LHS 20점 (4차원)
- Acquisition: qNEHVI (qLogNEHVI fallback)
- 수렴 기준: 연속 5 iteration HV 개선 < 0.1%
- UQ 임계값: sigma > 0.05 -> LightTools 감별 검증

## Evaluator 채점 기준 (각 20점, 총 100점)

1. MTF@ridge 목표 달성도
2. skewness 목표 달성도
3. 광량 T 목표 달성도
4. 수렴 신뢰도 (BO acquisition value)
5. 물리 제약 마진 (d, w 기하 조건 여유)
- 70점 미만: Generator 재시도 지시

## Active Learning 규칙

- LT 검증 결과 -> 자동으로 재학습 데이터 편입
- PINN 예측 오차 > 5% -> 재학습 트리거
- MC Dropout sigma 상위 20% -> LT 우선 검증

## 출력 형식

- 설계변수: um 단위 명시
- PSF 결과: skewness, FWHM, 피크 인텐시티, 크로스토크 비율 포함
- 불확실도 sigma 항상 출력
