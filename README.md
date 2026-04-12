---
title: UDFPS BM Design Studio
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
---

# UDFPS BM Design Studio

Physics-Informed Neural Network based reverse design platform for Under-Display Fingerprint Sensor BM (Black Matrix) optical structure.

## Features
- **ASM Wave Optics**: CG 550um propagation + crosstalk (verified)
- **AR Coating LUT**: Unpolarized TMM, dphi(30°)=-10.3°
- **Parametric PINN**: Helmholtz + AR phase + BM diffraction
- **FNO Surrogate**: 0.1ms inference
- **BoTorch qNEHVI**: 3-objective Pareto optimization

## API
- `GET /api/health` — Model status
- `GET /api/inference/predict?delta_bm1=0&w1=10&w2=10` — Unified inference (FNO > PINN > ASM)
- `GET /api/inference/reverse-design` — Top-5 optimal designs
