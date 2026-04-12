"""
asm_propagator.py - Angular Spectrum Method (빛의 전파 계산기)
================================================================
이 파일은 "빛이 유리(Cover Glass)를 통과하면서 어떻게 퍼지는지" 계산합니다.

배경:
  지문에서 반사된 빛이 Cover Glass(550um)를 통과하면,
  각도에 따라 옆으로 이동합니다.

  예시:
    0도(수직): 옆으로 이동 안 함
    22.5도:    옆으로 191um 이동 (2.65피치분!)
    → 이것이 "크로스토크"의 원인

ASM(Angular Spectrum Method)이란?
  빛의 파동을 "주파수 성분"으로 분해해서 전파를 계산하는 방법입니다.
  물 위의 파도를 여러 파도의 합으로 분해하는 것과 비슷합니다.

  1. 입력 파동을 FFT로 주파수 분해
  2. 각 주파수 성분에 "전파 거리만큼의 위상 변화" 적용
  3. IFFT로 다시 합성 → 전파된 파동

핵심 포인트:
  - 실험 데이터 0개로 계산 가능 (순수 물리 계산)
  - PINN의 입력 경계조건(상단면) 생성에 사용
  - 크로스토크 거리를 정확히 반영
"""

import numpy as np


class ASMPropagator:
    """
    Angular Spectrum Method로 빛의 전파를 계산합니다.

    빛이 유리판(Cover Glass)을 통과하면서
    회절(퍼짐)과 굴절(방향 변화)을 겪는 과정을 시뮬레이션합니다.

    사용법:
        asm = ASMPropagator(wl_um=0.52, n_medium=1.52)

        # 1D 전파 (x방향만)
        U_out = asm.propagate_1d(U_in, dx=0.1, dz=550.0)

    매개변수 설명:
        wl_um:    빛의 파장 (um 단위). 520nm = 0.52um
        n_medium: 매질의 굴절률. Cover Glass = 1.52
    """

    def __init__(self, wl_um: float = 0.52, n_medium: float = 1.52):
        """
        초기 설정.

        wl_um:    파장 (um). 기본값 0.52um = 520nm (녹색광)
        n_medium: 굴절률. 기본값 1.52 (Cover Glass)
        """
        self.wl = wl_um                      # 파장 (um)
        self.n = n_medium                     # 굴절률
        self.k0 = 2 * np.pi / wl_um          # 진공 파수 (2pi/lambda)
        self.k = self.k0 * n_medium           # 매질 내 파수 (k0 * n)

    def propagate_1d(
        self,
        U_in: np.ndarray,
        dx: float,
        dz: float
    ) -> np.ndarray:
        """
        1D 파동을 z 방향으로 dz만큼 전파합니다.

        매개변수:
            U_in: 입력 파동 (복소수 배열, 1D)
                  각 점의 빛의 세기와 위상 정보를 담고 있음
            dx:   x 방향 격자 간격 (um)
                  예: dx=0.1이면 0.1um마다 하나의 점
            dz:   전파 거리 (um)
                  예: dz=550이면 Cover Glass 550um 통과

        반환값:
            U_out: 전파 후 파동 (복소수 배열, 1D)

        동작 순서:
            1. FFT: 파동을 주파수(공간 주파수) 성분으로 분해
            2. Transfer Function: 각 주파수에 전파 위상 적용
            3. IFFT: 다시 합성 → 전파된 파동
        """
        N = len(U_in)  # 격자점 개수

        # -----------------------------------------------
        # 1단계: 공간 주파수 배열 생성
        #   "빛의 파동을 여러 각도의 평면파로 분해"
        #   fx: 공간 주파수 (1/um 단위)
        #   fftfreq: FFT에서 사용하는 주파수 배열 생성 함수
        # -----------------------------------------------
        fx = np.fft.fftfreq(N, d=dx)

        # -----------------------------------------------
        # 2단계: 전파 전달함수(Transfer Function) 계산
        #   H(fx) = exp(i * kz * dz)
        #
        #   kz = sqrt(k^2 - (2*pi*fx)^2)
        #     = "z 방향 파수"
        #
        #   의미: 각 주파수 성분이 dz만큼 이동할 때
        #         위상이 얼마나 변하는지
        #
        #   주의: kx^2 > k^2이면 에바네센트파 (감쇠)
        #         → kz를 0으로 처리 (전파 불가)
        # -----------------------------------------------
        kx = 2 * np.pi * fx
        kz_sq = self.k**2 - kx**2

        # 전파 가능한 성분만 선택 (kz^2 > 0인 것만)
        propagating = kz_sq > 0
        kz = np.zeros_like(kz_sq)
        kz[propagating] = np.sqrt(kz_sq[propagating])

        # 전달함수: 각 주파수 성분에 적용할 위상 변화
        H = np.exp(1j * kz * dz)

        # 에바네센트파(전파 불가 성분)는 제거
        H[~propagating] = 0.0

        # -----------------------------------------------
        # 3단계: FFT → 전달함수 적용 → IFFT
        #   "분해 → 위상 변화 적용 → 재합성"
        # -----------------------------------------------
        U_freq = np.fft.fft(U_in)        # 1. 주파수 분해
        U_prop = U_freq * H              # 2. 전파 위상 적용
        U_out = np.fft.ifft(U_prop)      # 3. 재합성

        return U_out

    def make_incident_field(
        self,
        x_grid: np.ndarray,
        theta_deg: float,
        dphi_deg: float = 0.0
    ) -> np.ndarray:
        """
        특정 각도의 입사 평면파를 생성합니다.

        매개변수:
            x_grid:    x 좌표 배열 (um)
            theta_deg: 입사각 (도). 예: 22.5
            dphi_deg:  AR 코팅 위상 변화 (도). TMM에서 계산한 값.

        반환값:
            U_in: 입사 파동 (복소수 배열)

        입사 평면파:
            U(x) = exp(i * kx * x + i * dphi)
            kx = k * sin(theta)

        이 평면파는 ASM 전파의 시작점(입력)이 됩니다.
        """
        theta_rad = np.radians(theta_deg)
        dphi_rad = np.radians(dphi_deg)

        # kx: x 방향 파수 성분
        #   빛이 비스듬히 입사하면 x 방향으로도 진행 성분이 있음
        kx = self.k * np.sin(theta_rad)

        # 평면파 생성
        U_in = np.exp(1j * (kx * x_grid + dphi_rad))

        return U_in

    def compute_crosstalk_distance(self, theta_deg: float, dz: float) -> float:
        """
        특정 입사각에서 크로스토크 이동 거리를 계산합니다.

        매개변수:
            theta_deg: 입사각 (도)
            dz:        전파 거리 (um)

        반환값:
            이동 거리 (um)

        공식: delta_x = dz * tan(theta)
        예시: theta=22.5도, dz=550um
              → delta_x = 550 * tan(22.5) = 228um ≈ 3.2피치

        이 값이 크면 크로스토크가 심합니다.
        7피치 도메인(504um)이 필요한 이유입니다.
        """
        theta_rad = np.radians(abs(theta_deg))
        return abs(dz * np.tan(theta_rad))
