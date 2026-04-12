"""
fit_ar_stack.py
Corning DX 공개 스펙 (420~680nm 에서 T≈98%, R≈1% 유지)에 등가하는 
4층 (Nb2O5 / SiO2) 스택 두께를 SciPy로 최적화(피팅)합니다.
"""
import numpy as np
from scipy.optimize import minimize
import json
import os
from tmm import coh_tmm

# 파장 범위: 420 ~ 680 nm (5nm 간격)
wavelengths_nm = np.arange(420, 685, 5)

# 레이어 정보 (4-layer AR + Glass substrate)
# 매질: 공기(1.0) | Nb2O5 | SiO2 | Nb2O5 | SiO2 | CoverGlass(1.52)
n_air = 1.0
n_nb2o5 = 2.30  # High Index
n_sio2 = 1.46   # Low Index
n_glass = 1.52  # Glass

N_LIST = [n_air, n_nb2o5, n_sio2, n_nb2o5, n_sio2, n_glass]

# 타겟 반사율: R = 1.0% (0.01) across all wavelengths
TARGET_R = 0.01

def objective_function(d):
    """
    주어진 두께 배열 d = [d1, d2, d3, d4] (nm) 에 대해
    목표 반사율 스펙트럼과의 매칭 오차(SSE)를 계산합니다.
    """
    total_error = 0.0
    D_LIST = [np.inf, d[0], d[1], d[2], d[3], np.inf]
    
    for wl in wavelengths_nm:
        # 광대역 수직 입사 최적화
        res_s = coh_tmm('s', N_LIST, D_LIST, 0.0, wl)
        total_error += (res_s['R'] - TARGET_R)**2
    
    return total_error

def main():
    print(f"Fitting 4-layer AR stack to Target R={TARGET_R*100}% over {wavelengths_nm[0]}~{wavelengths_nm[-1]}nm")
    
    # 초기 두께 구상 (대략 광학적 두께 기준 랜덤값)
    initial_guess = [15.0, 30.0, 115.0, 90.0]
    
    # 두께 제한 조건 (범위: 5nm ~ 200nm 사이)
    bounds = [(5, 200) for _ in range(4)]
    
    result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        print("Fitting Successful!")
    else:
        print("Fitting Failed!", result.message)
        
    optimal_d = result.x.tolist()
    d_list_full = [float('inf'), optimal_d[0], optimal_d[1], optimal_d[2], optimal_d[3], float('inf')]
    
    print(f"Optimal Thicknesses (nm):\n  L1 (Nb2O5): {optimal_d[0]:.1f}\n  L2 (SiO2): {optimal_d[1]:.1f}\n  L3 (Nb2O5): {optimal_d[2]:.1f}\n  L4 (SiO2): {optimal_d[3]:.1f}")
    
    # 520nm 검증
    r_check = coh_tmm('s', N_LIST, d_list_full, 0.0, 520.0)['R']
    print(f"Check Reflection at 520nm: {r_check*100:.2f}% (Target: {TARGET_R*100}%)")
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/fitted_stack.json'
    with open(out_path, 'w') as f:
        json.dump({
            "n_list": N_LIST,
            "d_list": d_list_full,
            "wavelengths": wavelengths_nm.tolist(),
            "target_R": TARGET_R,
            "mse": result.fun / len(wavelengths_nm)
        }, f, indent=4)
    print(f"Saved fitted stack parameters to {out_path}")

if __name__ == "__main__":
    main()
