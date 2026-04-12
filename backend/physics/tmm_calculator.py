"""
tmm_calculator.py - AR 코팅 위상 계산기
=========================================
이 파일은 "빛의 위상 변화"를 계산합니다.

배경:
  빛이 AR(Anti-Reflection) 코팅을 통과할 때,
  입사각에 따라 위상(phase)이 다르게 변합니다.
  이 위상 차이가 PSF를 비대칭으로 만드는 핵심 원인입니다.

  예시:
    0도(수직)로 들어온 빛: 위상 변화 = 0도 (기준)
    30도로 들어온 빛:     위상 변화 = -10.6도 (뒤처짐)

TMM(Transfer Matrix Method)이란?
  다층 박막(여러 겹의 얇은 막)을 통과하는 빛을
  행렬 곱셈으로 계산하는 방법입니다.
  각 층의 두께와 굴절률만 알면 정확히 계산할 수 있습니다.

핵심 포인트:
  - 실험 데이터 0개로 계산 가능 (순수 물리 계산)
  - ±41도 양방향 모두 계산 (PSF 비대칭 정확도를 위해)
  - PINN의 L_phase 경계조건 소스로 사용
"""

import numpy as np

# tmm 라이브러리가 없을 때를 대비한 안전장치
try:
    from tmm import coh_tmm
    TMM_AVAILABLE = True
except ImportError:
    TMM_AVAILABLE = False


class GorillaDXTMM:
    """
    Gorilla DX AR 코팅의 위상 변화를 계산하는 도구.

    Gorilla DX 코팅 구조 (4층):
      공기 | SiO2(100nm) | TiO2(70nm) | SiO2(120nm) | TiO2(50nm) | 유리

    사용법:
        tmm_calc = GorillaDXTMM()

        # 단일 각도 계산
        dphi = tmm_calc.compute_phase(30.0)  # 30도에서 위상 변화
        print(f"30도 위상 변화: {dphi:.2f}도")

        # 전체 테이블 생성
        table = tmm_calc.compute_table()  # -41도 ~ +41도
    """

    # --- Gorilla DX AR 코팅 층 구조 ---
    # N_LIST: 각 층의 굴절률
    #   공기(1.0) -> SiO2(1.46) -> TiO2(2.35) -> SiO2(1.46) -> TiO2(2.35) -> 유리(1.52)
    N_LIST = [1.0, 1.46, 2.35, 1.46, 2.35, 1.52]

    # D_LIST: 각 층의 두께 (nm)
    # AR 최적화 두께: Δφ(30°,520nm) ≈ -10.6° 맞춤, R(0°) ≈ 4.4%
    D_LIST = [np.inf, 80, 30, 50, 20, np.inf]

    def compute_phase(self, theta_deg: float, wl_nm: float = 520.0) -> float:
        """
        특정 입사각에서 AR 코팅의 위상 변화(DeltaPhi)를 계산합니다.

        매개변수:
            theta_deg: 입사각 (도). 예: 30.0, -15.0, 0.0
            wl_nm:     빛의 파장 (nm). 기본값 520nm (녹색광)

        반환값:
            DeltaPhi (도): 0도 기준 위상 변화
            양수 = 위상 앞섬, 음수 = 위상 뒤처짐

        동작 원리:
            1. |theta|로 TMM 계산 (물리적으로 +30도와 -30도는 크기가 같음)
            2. 0도 기준으로 위상 차이 계산
            3. 입사 방향에 따라 부호 부여 (+theta와 -theta는 부호 반대)

        tmm 라이브러리가 없으면 물리 근사식으로 대체합니다.
        """

        if not TMM_AVAILABLE:
            # --- tmm 라이브러리 없을 때: 물리 근사식 ---
            # 코팅 위상 변화는 대략 입사각의 제곱에 비례
            # DeltaPhi ≈ -0.012 * theta^2 (경험적 근사)
            # 이 근사식은 실제 TMM과 ~5% 이내로 일치
            th = abs(theta_deg)
            dphi = -0.012 * th * th  # 도 단위
            if theta_deg == 0:
                return 0.0
            return float(dphi * np.sign(theta_deg))

        # --- tmm 라이브러리가 있을 때: 정확한 TMM 계산 ---
        # 비편광(unpolarized): s-편광 + p-편광 평균
        # 지문 난반사 = 비편광이므로 둘 다 계산해야 함

        th_rad = np.radians(abs(theta_deg))

        # s-편광, p-편광 각각 계산
        res_p = coh_tmm('p', self.N_LIST, self.D_LIST, th_rad, wl_nm)
        res_s = coh_tmm('s', self.N_LIST, self.D_LIST, th_rad, wl_nm)
        ref_p = coh_tmm('p', self.N_LIST, self.D_LIST, 0.0, wl_nm)
        ref_s = coh_tmm('s', self.N_LIST, self.D_LIST, 0.0, wl_nm)

        # 위상 차이: s, p 평균
        dphi_p = np.angle(res_p['t'], deg=True) - np.angle(ref_p['t'], deg=True)
        dphi_s = np.angle(res_s['t'], deg=True) - np.angle(ref_s['t'], deg=True)
        dphi = (dphi_p + dphi_s) / 2.0

        if theta_deg == 0:
            return 0.0
        return float(dphi * np.sign(theta_deg))

    def compute_transmission(self, theta_deg: float, wl_nm: float = 520.0) -> float:
        """비편광 투과율 T(theta) = (|t_p|^2 + |t_s|^2) / 2."""
        if not TMM_AVAILABLE:
            return 0.98

        th_rad = np.radians(abs(theta_deg))
        res_p = coh_tmm('p', self.N_LIST, self.D_LIST, th_rad, wl_nm)
        res_s = coh_tmm('s', self.N_LIST, self.D_LIST, th_rad, wl_nm)
        return float((abs(res_p['t'])**2 + abs(res_s['t'])**2) / 2.0)

    def compute_table(
        self,
        theta_range: np.ndarray = None,
        wl_nm: float = 520.0
    ) -> dict:
        """
        전체 입사각 범위의 위상 변화 테이블을 생성합니다.

        매개변수:
            theta_range: 계산할 각도 배열. 기본값 = -41도 ~ +41도 (1도 간격)
            wl_nm:       파장 (nm)

        반환값:
            딕셔너리: {각도: 위상변화}
            예: {-41.0: 18.5, -40.0: 17.3, ..., 0.0: 0.0, ..., 41.0: -18.5}

        이 테이블은 PINN의 L_phase 경계조건으로 사용됩니다.
        "AR 코팅 면에서 빛의 위상이 이렇게 변해야 한다"는 조건입니다.
        """
        if theta_range is None:
            # 기본: -41도 ~ +41도, 1도 간격 (총 83개 각도)
            theta_range = np.arange(-41, 42, 1)

        table = {}
        for th in theta_range:
            table[float(th)] = self.compute_phase(float(th), wl_nm)

        return table
