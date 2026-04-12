"""
ar_boundary.py
학습 중 매번 무거운 TMM을 계산하지 않고, ar_lut.npz 를 로드해 O(1) 수준으로 투과 계수 반환.
"""
import numpy as np
import os
import math

class ARLutInterpolator:
    def __init__(self, data_dir=None):
        # 모듈이 어디에서 호출되건 상대 경로를 매칭
        if data_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_path, 'data')
            
        lut_path = os.path.join(data_dir, 'ar_lut.npz')
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT not found at {lut_path}. Run generate_lut.py first.")
            
        data = np.load(lut_path)
        self.t_amp = data['t_amp']
        # 0도 기준 상대 위상으로 정규화 (라디안)
        raw_phase = data['t_phase']  # shape: (wl, angle, pol)
        self.t_phase = raw_phase - raw_phase[:, 0:1, :]  # 각 파장/편광에서 0도 위상 빼기
        self.wls = data['wavelengths']
        self.angles = data['angles']
        
    def get_complex_t(self, wavelength_nm, theta_deg, unpolarized=True):
        """
        주어진 파장과 입사각에 대해 t(complex)의 amplitude와 phase 값을 반환합니다.
        가장 가까운 값을 찾는 Nearest-neighbor 보간 기반입니다.
        
        파라미터:
            wavelength_nm: 쿼리할 빛 파장(nm 단위, 예: 520.0)
            theta_deg: 입사각 (-45도 ~ 45도 여유)
            unpolarized: 무편광 평균치 (지문 산란 기반 반사 가정시 True 권장)
        """
        # 1. 파장 인덱스
        wl_idx = (np.abs(self.wls - wavelength_nm)).argmin()
        
        # 2. 각도 인덱스 (대칭 구조이므로 절댓값)
        ang = abs(theta_deg)
        # 만약 설정 각도 (45)를 초과한다면 경계값
        if ang >= self.angles[-1]:
            ang_idx = -1
        else:
            ang_idx = (np.abs(self.angles - ang)).argmin()
        
        if unpolarized:
            # S파와 P파의 Incoherent / Average 특성 반영
            amp_s = self.t_amp[wl_idx, ang_idx, 0]
            phase_s = self.t_phase[wl_idx, ang_idx, 0]
            t_s = amp_s * complex(math.cos(phase_s), math.sin(phase_s))
            
            amp_p = self.t_amp[wl_idx, ang_idx, 1]
            phase_p = self.t_phase[wl_idx, ang_idx, 1]
            t_p = amp_p * complex(math.cos(phase_p), math.sin(phase_p))
            
            # Complex amplitude 평균치
            t_avg = (t_s + t_p) / 2.0
            amp = abs(t_avg)
            phase_rad = math.atan2(t_avg.imag, t_avg.real)
        else:
            amp = self.t_amp[wl_idx, ang_idx, 1]
            phase_rad = self.t_phase[wl_idx, ang_idx, 1]

        # degree로 반환 (부호: +theta와 -theta는 위상 방향 반전)
        phase_deg = math.degrees(phase_rad)
        if theta_deg < 0:
            phase_deg = -phase_deg

        return float(amp), float(phase_deg)
