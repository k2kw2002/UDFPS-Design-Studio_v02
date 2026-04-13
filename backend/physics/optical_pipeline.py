"""
optical_pipeline.py - ASM 기반 전체 광학 시뮬레이션 파이프라인
=============================================================
지문 → CG(550um) ASM 전파 → BM1 마스킹 → ILD(20um) → BM2 → OPD 세기

LightTools 없이 파동광학 기반으로 PSF를 계산합니다.
CG 두께에 따른 blur, BM 설계에 따른 MTF, AR 위상 왜곡 모두 반영.

사용법:
    pipe = OpticalPipeline()
    psf7 = pipe.compute_psf7(delta_bm1=0, delta_bm2=0, w1=10, w2=10)
    psf7_no_ar = pipe.compute_psf7(..., use_ar=False)
"""

import numpy as np
import math

from backend.physics.asm_propagator import ASMPropagator
from backend.physics.tmm_calculator import GorillaDXTMM
from backend.physics.psf_metrics import PSFMetrics


class OpticalPipeline:
    """
    전체 광학 파이프라인.

    구조 (위에서 아래로):
      z=570um  지문 면 (입사파 소스)
      z=570~20um  Cover Glass 550um (ASM 전파)
      z=40um   BM1 (아퍼처 마스킹) — PINN soft mask 또는 binary
      z=40~20um ILD 20um (ASM 전파)
      z=20um   BM2 (아퍼처 마스킹) — PINN soft mask 또는 binary
      z=20~0um Encap 20um (ASM 전파)
      z=0um    OPD (세기 측정)

    use_pinn=True: BM 마스크를 PINN soft sigmoid로 교체
                   (회절 edge 효과 반영, Phase B 학습 결과)
    """

    def __init__(
        self,
        wl_um: float = 0.520,
        n_cg: float = 1.52,
        cg_thick: float = 550.0,
        ild_thick: float = 20.0,
        encap_thick: float = 20.0,
        opd_pitch: float = 72.0,
        opd_width: float = 10.0,
        n_pitches: int = 7,
        dx: float = 0.5,
        n_angles: int = 21,
        theta_max: float = 41.0,
        use_pinn: bool = False,
        pinn_model=None,
    ):
        self.wl = wl_um
        self.n_cg = n_cg
        self.cg_thick = cg_thick
        self.ild_thick = ild_thick
        self.encap_thick = encap_thick
        self.opd_pitch = opd_pitch
        self.opd_width = opd_width
        self.n_pitches = n_pitches
        self.dx = dx
        self.n_angles = n_angles
        self.theta_max = theta_max

        # ASM 전파기
        self.asm = ASMPropagator(wl_um=wl_um, n_medium=n_cg)
        # TMM (AR 코팅 위상)
        self.tmm = GorillaDXTMM()
        self.tmm_table = self.tmm.compute_table()
        # PSF 메트릭
        self.psf_metrics = PSFMetrics()

        # 도메인 격자
        self.domain_width = opd_pitch * n_pitches  # 504um
        self.N = int(self.domain_width / dx)
        self.x_grid = np.linspace(0, self.domain_width, self.N, endpoint=False)

        # 입사각 배열 (±theta_max, 양방향)
        self.thetas = np.linspace(-theta_max, theta_max, n_angles)

        # PINN hybrid (Phase B soft BM mask)
        self.use_pinn = use_pinn
        self.pinn_model = pinn_model

    def _compute_system_psf(self, bm1_mask, bm2_mask, use_ar, cg_thick):
        """
        시스템 PSF 계산 (점 광원 응답).

        지문의 한 점에서 산란된 빛이 여러 각도로 퍼짐.
        같은 점에서 나온 빛이므로 간섭(coherent sum).
        AR 위상이 각도마다 다르게 적용 → PSF 왜곡.

        Returns:
            psf_1d: (N,) — 시스템 PSF (세기)
        """
        # 점 광원 → 여러 각도로 산란 → 간섭 합 (같은 광원)
        total_field = np.zeros(self.N, dtype=complex)

        for theta in self.thetas:
            # AR 위상
            dphi = 0.0
            if use_ar:
                theta_round = max(-41, min(41, round(theta)))
                dphi = self.tmm_table.get(float(theta_round), 0.0)

            # 평면파 (AR 위상 포함)
            U = self.asm.make_incident_field(self.x_grid, theta, dphi)

            # CG 전파
            U = self.asm.propagate_1d(U, self.dx, cg_thick)

            # BM1
            U = U * bm1_mask

            # ILD 전파
            U = self.asm.propagate_1d(U, self.dx, self.ild_thick)

            # BM2
            U = U * bm2_mask

            # Encap 전파
            U = self.asm.propagate_1d(U, self.dx, self.encap_thick)

            # 간섭 합 (같은 점 광원이므로 coherent)
            total_field += U

        # PSF = |합산 필드|²
        psf = np.abs(total_field) ** 2
        # 정규화
        s = psf.sum()
        if s > 0:
            psf /= s
        return psf

    def compute_psf7(
        self,
        delta_bm1: float = 0.0,
        delta_bm2: float = 0.0,
        w1: float = 10.0,
        w2: float = 10.0,
        use_ar: bool = True,
        cg_override: float = None,
    ) -> np.ndarray:
        """
        BM 설계변수 -> 7-OPD PSF 계산.

        비간섭 직접 전파 (CG 효과 + 크로스토크 정확):
          각 입사각에 대해:
            1) 지문 패턴 x 입사 평면파
            2) AR 투과 진폭 T(theta) 가중 (각도별 투과율 차이)
            3) CG ASM 전파 -> 빛 퍼짐 (크로스토크)
            4) BM 마스킹 -> OPD
          비간섭 합 (각 각도 독립)

        AR 효과:
          - 각도별 투과 진폭 T(theta) 변화 -> PSF 형태 변화
          - 큰 각도에서 투과 감소 -> 유효 NA 축소 -> PSF 변화
        """
        cg_thick = cg_override if cg_override is not None else self.cg_thick

        bm1_mask = self._make_bm_mask(delta_bm1, w1)
        bm2_mask = self._make_bm_mask(delta_bm2, w2)
        finger = self._make_finger_pattern()

        # AR 투과 진폭 테이블 (비편광 LUT 우선, fallback TMM)
        ar_T_table = {}
        if use_ar:
            try:
                from backend.physics.ar_coating.ar_boundary import ARLutInterpolator
                lut = ARLutInterpolator()
                for th in self.thetas:
                    t_amp, _ = lut.get_complex_t(self.wl * 1000, th, unpolarized=True)
                    ar_T_table[th] = t_amp
            except Exception:
                # fallback: TMM 비편광
                for th in self.thetas:
                    ar_T_table[th] = self.tmm.compute_transmission(th, self.wl * 1000) ** 0.5

        total_intensity = np.zeros(self.N)

        for theta in self.thetas:
            # AR 투과 진폭 가중
            if use_ar and theta in ar_T_table:
                T_weight = ar_T_table[theta]
            else:
                T_weight = 1.0  # AR 없으면 가중치 1

            # 1) 지문 x 입사파
            U = self.asm.make_incident_field(self.x_grid, theta, 0.0)
            U = U * finger * T_weight

            # 2) CG 전파 (크로스토크 발생)
            U = self.asm.propagate_1d(U, self.dx, cg_thick)

            # 3) BM1
            if self.use_pinn and self.pinn_model is not None:
                U = U * self._pinn_bm_mask(delta_bm1, w1, delta_bm2, w2, z=40.0)
            else:
                U = U * bm1_mask

            # 4) ILD
            U = self.asm.propagate_1d(U, self.dx, self.ild_thick)

            # 5) BM2
            if self.use_pinn and self.pinn_model is not None:
                U = U * self._pinn_bm_mask(delta_bm1, w1, delta_bm2, w2, z=20.0)
            else:
                U = U * bm2_mask

            # 6) Encap
            U = self.asm.propagate_1d(U, self.dx, self.encap_thick)

            # 비간섭 합
            total_intensity += np.abs(U) ** 2

        total_intensity /= len(self.thetas)

        psf7 = self._extract_opd_intensity(total_intensity)
        return psf7

    def _tmm_available(self):
        try:
            from tmm import coh_tmm
            return True
        except ImportError:
            return False

    def _make_finger_pattern(self) -> np.ndarray:
        """
        지문 Ridge/Valley 패턴 생성.
        Ridge(짝수 OPD 피치) = 1.0 (강한 난반사)
        Valley(홀수 OPD 피치) = 0.1 (약한 반사)
        실제 지문: Ridge 폭 ~150um, Valley 폭 ~150um
        OPD 피치 72um → 약 2 피치가 1 Ridge
        """
        pattern = np.ones(self.N) * 0.1  # 기본: Valley (약함)
        # 지문 Ridge: 약 144um 폭 (2 피치), 144um 간격
        ridge_period = self.opd_pitch * 2  # 144um
        for x_idx in range(self.N):
            x_um = x_idx * self.dx
            phase = (x_um % (ridge_period * 2)) / (ridge_period * 2)
            if phase < 0.5:
                pattern[x_idx] = 1.0  # Ridge
        return pattern

    def _make_bm_mask(self, delta: float, w: float) -> np.ndarray:
        """BM 아퍼처 마스크 (투과=1, 불투명=0)."""
        mask = np.zeros(self.N)
        for pitch_idx in range(self.n_pitches):
            opd_center = (pitch_idx - 3) * self.opd_pitch + self.opd_pitch / 2 + self.domain_width / 2
            # 도메인 중심 보정
            opd_center = pitch_idx * self.opd_pitch + self.opd_pitch / 2

            aper_center = opd_center + delta
            aper_left = aper_center - w / 2
            aper_right = aper_center + w / 2

            # 격자 인덱스
            i_left = max(0, int(aper_left / self.dx))
            i_right = min(self.N, int(aper_right / self.dx))
            mask[i_left:i_right] = 1.0

        return mask

    def _extract_opd_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """도메인 세기에서 7개 OPD 픽셀 값 추출."""
        psf7 = np.zeros(self.n_pitches)
        for i in range(self.n_pitches):
            opd_center = i * self.opd_pitch + self.opd_pitch / 2
            # OPD 감지 영역 (10um 폭)
            opd_left = opd_center - self.opd_width / 2
            opd_right = opd_center + self.opd_width / 2

            i_left = max(0, int(opd_left / self.dx))
            i_right = min(self.N, int(opd_right / self.dx))

            if i_right > i_left:
                psf7[i] = np.mean(intensity[i_left:i_right])

        return psf7

    def compute_with_metrics(self, **kwargs) -> dict:
        """PSF + 메트릭 한번에 계산."""
        psf7 = self.compute_psf7(**kwargs)
        metrics = self.psf_metrics.compute(psf7)
        return {"psf7": psf7, "metrics": metrics}

    def _pinn_bm_mask(self, d1, w1, d2, w2, z):
        """
        PINN soft BM mask at given z plane.
        sigmoid(slit_dist) → smooth transition at slit edges.
        """
        import torch
        from backend.core.parametric_pinn import compute_slit_dist, compute_bm_mask

        x_t = torch.tensor(self.x_grid, dtype=torch.float32)
        z_t = torch.full((self.N,), float(z))
        d1_t = torch.full((self.N,), float(d1))
        d2_t = torch.full((self.N,), float(d2))
        w1_t = torch.full((self.N,), float(w1))
        w2_t = torch.full((self.N,), float(w2))

        sd = compute_slit_dist(x_t, z_t, d1_t, d2_t, w1_t, w2_t)
        mask = compute_bm_mask(sd, self.pinn_model.mask_sharpness)
        return mask.numpy()

    def compare_cg_thickness(
        self, thicknesses: list[float], **bm_kwargs
    ) -> list[dict]:
        """CG 두께별 PSF 비교."""
        results = []
        for t in thicknesses:
            r = self.compute_with_metrics(cg_override=t, **bm_kwargs)
            r["cg_thick"] = t
            results.append(r)
        return results
