"""
psf_metrics.py - PSF 성능 지표 계산기
=======================================
이 파일은 "성적표"를 만드는 도구입니다.

PINN이나 FNO가 계산한 PSF(Point Spread Function, 점퍼짐함수)에서
4가지 핵심 지표를 뽑아냅니다:

  1. MTF@ridge (지문 인식률)
     "지문의 ridge(산)와 valley(골)를 얼마나 잘 구분하는가?"
     높을수록 좋음. 목표: 60% 이상.

  2. Skewness (비대칭도)
     "PSF가 한쪽으로 치우쳤는가?"
     0에 가까울수록 좋음 (대칭). 목표: 0.10 이하.

  3. Throughput (광량)
     "빛이 얼마나 많이 통과하는가?"
     높을수록 좋음. 목표: 60% 이상.

  4. Crosstalk Ratio (크로스토크 비율)
     "옆 픽셀로 새는 빛이 얼마나 되는가?"
     낮을수록 좋음.

PSF 7개란?
  7피치(504um) 도메인에서 7개 OPD 픽셀 각각의
  빛의 세기를 나타냅니다.

  인덱스:  [0]  [1]  [2]  [3]  [4]  [5]  [6]
  의미:     R    V    R   중심   R    V    R
           (R=Ridge, V=Valley)

  인덱스 3 = 중심 OPD (빛이 가장 강해야 하는 곳)
"""

import numpy as np


class PSFMetrics:
    """
    7개 OPD 픽셀 PSF에서 최적화 지표를 계산합니다.

    사용법:
        metrics = PSFMetrics()
        result = metrics.compute(psf_7)
        print(f"MTF: {result['mtf_ridge']:.1%}")
        print(f"Skewness: {result['skewness']:.3f}")
    """

    PITCH = 72.0       # OPD 피치 (um)
    OPD_W = 10.0       # OPD 픽셀 폭 (um)
    N_PIXELS = 7        # OPD 픽셀 수

    def compute(self, psf_7: np.ndarray) -> dict:
        """
        PSF 7개 값에서 4가지 성능 지표를 계산합니다.

        매개변수:
            psf_7: 7개 OPD 픽셀의 인텐시티(빛의 세기) 배열
                   shape: (7,)
                   인덱스 3 = 중심 OPD (Ridge)

        반환값:
            딕셔너리:
              mtf_ridge:      MTF@ridge (0~1, 높을수록 좋음)
              skewness:       비대칭도 (0에 가까울수록 좋음)
              throughput:     광량 (총 빛의 세기)
              crosstalk_ratio: 크로스토크 비율 (낮을수록 좋음)
        """

        # -----------------------------------------------
        # 1) MTF@ridge (지문 인식률)
        #    "지문의 산과 골을 구분하는 능력"
        #
        #    Ridge(산) 위치: 인덱스 0, 2, 4, 6
        #    Valley(골) 위치: 인덱스 1, 3, 5
        #
        #    공식: MTF = (평균_산 - 평균_골) / (평균_산 + 평균_골)
        #    범위: 0(전혀 구분 못함) ~ 1(완벽히 구분)
        #
        #    비유: 흑백 줄무늬의 명암 대비
        #          MTF=1: 검정/흰색 뚜렷 → 지문 완벽 인식
        #          MTF=0: 전체 회색 → 지문 인식 불가
        # -----------------------------------------------
        ridge_vals = psf_7[[0, 2, 4, 6]]   # Ridge 위치 값들
        valley_vals = psf_7[[1, 3, 5]]     # Valley 위치 값들

        r_mean = ridge_vals.mean()   # Ridge 평균
        v_mean = valley_vals.mean()  # Valley 평균

        # MTF 계산 (1e-8은 0으로 나누기 방지용 아주 작은 수)
        mtf = (r_mean - v_mean) / (r_mean + v_mean + 1e-8)

        # -----------------------------------------------
        # 2) Skewness (비대칭도)
        #    "PSF가 한쪽으로 치우친 정도"
        #
        #    통계학의 왜도(skewness) 공식 사용:
        #      1. 무게중심(mu) 계산
        #      2. 표준편차(sigma) 계산
        #      3. 3차 모멘트 계산
        #
        #    skewness = 0: 완벽하게 대칭
        #    skewness > 0: 오른쪽으로 치우침
        #    skewness < 0: 왼쪽으로 치우침
        #
        #    비유: 시소의 균형
        #          0 = 시소가 수평 (대칭)
        #          ≠0 = 시소가 기울어짐 (비대칭)
        # -----------------------------------------------
        # 각 OPD 픽셀의 x 위치 (um)
        xs = np.arange(self.N_PIXELS) * self.PITCH
        # 예: [0, 72, 144, 216, 288, 360, 432]

        # 정규화 (전체 합이 1이 되도록)
        norm = psf_7 / (psf_7.sum() + 1e-8)

        # 무게중심 mu (어디에 빛이 집중되는지)
        mu = np.sum(xs * norm)

        # 표준편차 sigma (빛이 얼마나 퍼져있는지)
        sigma = np.sqrt(np.sum((xs - mu) ** 2 * norm) + 1e-8)

        # 왜도(skewness) 계산 (3차 모멘트)
        skew = np.sum(((xs - mu) / sigma) ** 3 * norm)

        # -----------------------------------------------
        # 3) Throughput (광량)
        #    "OPD 센서에 도달한 총 빛의 양"
        #    단순히 7개 값의 합입니다.
        #    높을수록 빛이 많이 통과한 것 → 좋음.
        # -----------------------------------------------
        T = float(psf_7.sum())

        # -----------------------------------------------
        # 4) Crosstalk Ratio (크로스토크 비율)
        #    "옆 픽셀로 새는 빛의 비율"
        #
        #    중심 OPD(인덱스3)에 비해
        #    바로 옆 Valley(인덱스2, 4)에 얼마나 빛이 새는지.
        #
        #    공식: xtalk = 옆칸 평균 / 중심값
        #    예시: 중심=1.0, 옆칸=0.1 → xtalk=0.1 (10%)
        #
        #    10% 이하가 목표입니다.
        # -----------------------------------------------
        center_val = psf_7[3]  # 중심 OPD 인텐시티
        xtalk_vals = (psf_7[2] + psf_7[4]) / 2.0  # 인접 Valley 평균
        xtalk_ratio = xtalk_vals / (center_val + 1e-8)

        return dict(
            mtf_ridge=float(np.clip(mtf, 0, 1)),
            skewness=float(skew),
            throughput=T,
            crosstalk_ratio=float(xtalk_ratio),
        )


# ============================================================
# Standalone helper functions (API 용)
# ============================================================

def compute_psf_skewness(psf_7):
    """PSF 비대칭성. 대칭이면 0, 비대칭이면 큼."""
    psf = np.array(psf_7)
    center = 3
    left = psf[:center].sum()
    right = psf[center + 1:].sum()
    total = psf.sum() + 1e-8
    return float(abs(left - right) / total)


def compute_psf_mtf(psf_7):
    """간이 MTF: Peak / (Peak + Secondary)."""
    psf = np.array(psf_7)
    sorted_psf = np.sort(psf)[::-1]
    return float(sorted_psf[0] / (sorted_psf[0] + sorted_psf[1] + 1e-8))


def compute_throughput(psf_7):
    """전체 에너지 (정규화 전)."""
    return float(np.sum(psf_7))
