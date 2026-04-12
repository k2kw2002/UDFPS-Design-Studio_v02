# UDFPS PINN - GPU Cluster Execution Guide

## Architecture (Verified)

```
ASM: finger -> AR(LUT, unpolarized) -> CG 550um propagation -> field at BM1
PINN: BM region 40um only (504um x 40um) Helmholtz equation
      Top BC = ASM output complex field
      BM BC = U=0 at opaque regions
Incoherent sum: 7 angles (0, +/-15, +/-30, +/-41)
```

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r backend/requirements.txt
pip install tmm botorch gpytorch

# 2. Run hybrid ASM+PINN training (auto-detects GPU)
python run_hybrid_4angle.py

# 3. Check results
python -c "import torch; d=torch.load('pinn_results_all.pt'); print(d)"
```

## Verified Physics

| Component | Status | Value |
|-----------|--------|-------|
| TMM (AR coating) | Verified | dphi(30)=-10.3 deg (guide -10.6) |
| TMM polarization | Unpolarized (s+p avg) | Corrected from p-only |
| AR LUT | Generated | 53 wl x 46 angles x 2 pol, O(1) lookup |
| ASM CG propagation | Verified | CG50:98% -> CG550:29% contrast |
| ASM crosstalk | Verified | 228um = 3.2 pitches at 22.5 deg |
| BM positions | Corrected | BM1=z=40, BM2=z=20 (was swapped!) |
| Domain | Corrected | 590um total (was 570) |
| PINN domain | 40um only | BM region (not full 590um CG) |

## Expected Results

```
CG Only (no AR):     contrast ~29% (matches ASM)
AR + Baseline:       contrast < 29% (AR degrades)
Difference = AR phase distortion effect
```

## GPU Settings (in script)

```
GPU: N_COLLOC=50000, N_EPOCHS=5000, ~2min/angle, ~30min total
CPU: N_COLLOC=20000, N_EPOCHS=3000, ~26min/angle, ~6hr total
```
