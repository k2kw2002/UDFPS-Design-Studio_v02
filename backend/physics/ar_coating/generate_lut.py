"""
generate_lut.py
피팅된 스택 정보를 바탕으로 3D LUT (angle, wavelength, pol) 생성
"""
import numpy as np
import json
import os
from tmm import coh_tmm

def main():
    stack_file = 'data/fitted_stack.json'
    if not os.path.exists(stack_file):
        raise FileNotFoundError(f"Run fit_ar_stack.py first! Cannot find {stack_file}")
        
    with open(stack_file, 'r') as f:
        stack = json.load(f)
        
    n_list = stack['n_list']
    d_list = stack['d_list']
    
    # 0도 제한, 최대 45도까지 1도 간격 탐색
    angles_deg = np.arange(0, 46, 1)  
    wavelengths_nm = np.arange(420, 685, 5) 
    
    # LUT 텐서 초기화
    lut_t_amp = np.zeros((len(wavelengths_nm), len(angles_deg), 2))
    lut_t_phase = np.zeros((len(wavelengths_nm), len(angles_deg), 2))
    
    print(f"Generating LUT...")
    print(f" - Wavelength points: {len(wavelengths_nm)}")
    print(f" - Angle points:      {len(angles_deg)}")
    print(f" - Polarizations:     2 (s, p)")
    
    for i, wl in enumerate(wavelengths_nm):
        for j, ang in enumerate(angles_deg):
            th_rad = np.radians(ang)
            
            # S파 전파
            res_s = coh_tmm('s', n_list, d_list, th_rad, wl)
            lut_t_amp[i, j, 0] = np.abs(res_s['t'])
            lut_t_phase[i, j, 0] = np.angle(res_s['t'])
            
            # P파 전파
            res_p = coh_tmm('p', n_list, d_list, th_rad, wl)
            lut_t_amp[i, j, 1] = np.abs(res_p['t'])
            lut_t_phase[i, j, 1] = np.angle(res_p['t'])
            
    out_path = 'data/ar_lut.npz'
    np.savez_compressed(
        out_path, 
        t_amp=lut_t_amp, 
        t_phase=lut_t_phase, 
        wavelengths=wavelengths_nm, 
        angles=angles_deg,
        polarizations=np.array(['s', 'p'])
    )
    print(f"Saved optimized LUT matrix to {out_path}")
    print(f"Shape: {lut_t_amp.shape} = (wl, angles, pol)")

if __name__ == "__main__":
    main()
