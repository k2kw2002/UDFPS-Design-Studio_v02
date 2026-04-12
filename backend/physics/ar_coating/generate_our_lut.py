"""
generate_our_lut.py - 우리 TMM(SiO2/TiO2) 기반 AR LUT 생성
============================================================
비편광(s+p avg), broadband(420~680nm), 0~45도
"""
import numpy as np
import json
from pathlib import Path
from tmm import coh_tmm

# 우리 코팅 구조 (dphi(30)=-10.3 deg, R=4.4%)
N_LIST = [1.0, 1.46, 2.35, 1.46, 2.35, 1.52]
D_LIST = [np.inf, 80, 30, 50, 20, np.inf]

# 그리드
wavelengths = np.arange(420, 681, 5)  # 420~680nm, 5nm 간격 (53점)
angles = np.arange(0, 46, 1)          # 0~45도, 1도 간격 (46점)
pols = ['s', 'p']

print(f"Generating LUT: {len(wavelengths)} wavelengths x {len(angles)} angles x {len(pols)} pols")

# LUT 배열
t_amp = np.zeros((len(wavelengths), len(angles), len(pols)))
t_phase = np.zeros((len(wavelengths), len(angles), len(pols)))
r_amp = np.zeros((len(wavelengths), len(angles), len(pols)))
r_phase = np.zeros((len(wavelengths), len(angles), len(pols)))

for wi, wl in enumerate(wavelengths):
    for ai, ang in enumerate(angles):
        for pi, pol in enumerate(pols):
            res = coh_tmm(pol, N_LIST, D_LIST, np.radians(ang), wl)
            ref = coh_tmm(pol, N_LIST, D_LIST, 0, wl)

            t_amp[wi, ai, pi] = abs(res['t'])
            t_phase[wi, ai, pi] = np.angle(res['t'], deg=True) - np.angle(ref['t'], deg=True)
            r_amp[wi, ai, pi] = abs(res['r'])
            r_phase[wi, ai, pi] = np.angle(res['r'], deg=True)

# 저장
out_path = Path(__file__).parent / "data" / "ar_lut.npz"
np.savez(out_path,
    wavelengths=wavelengths,
    angles=angles,
    polarizations=np.array(pols),
    t_amp=t_amp, t_phase=t_phase,
    r_amp=r_amp, r_phase=r_phase,
)

# stack 정보도 저장
stack_path = Path(__file__).parent / "data" / "fitted_stack.json"
with open(stack_path, 'w') as f:
    json.dump({
        "n_list": N_LIST,
        "d_list": [None if np.isinf(d) else d for d in D_LIST],
        "materials": ["air", "SiO2", "TiO2", "SiO2", "TiO2", "glass"],
        "target_dphi_30": -10.6,
        "actual_dphi_30": float(t_phase[wavelengths.tolist().index(520), angles.tolist().index(30), :].mean()),
        "R_0deg": float(r_amp[wavelengths.tolist().index(520), 0, :].mean()**2),
        "polarization": "unpolarized (s+p average)",
    }, f, indent=2)

print(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")
print(f"Saved: {stack_path}")

# 검증
idx_520 = wavelengths.tolist().index(520)
idx_30 = angles.tolist().index(30)
dp_p = t_phase[idx_520, idx_30, 1]  # p-pol
dp_s = t_phase[idx_520, idx_30, 0]  # s-pol
dp_avg = (dp_p + dp_s) / 2
print(f"\nVerification at 520nm, 30deg:")
print(f"  dphi_p = {dp_p:.2f} deg")
print(f"  dphi_s = {dp_s:.2f} deg")
print(f"  dphi_avg = {dp_avg:.2f} deg (guide: -10.6)")
