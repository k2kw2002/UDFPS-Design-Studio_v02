"""
pinn_trainer.py - PINN 학습 파이프라인
========================================
Helmholtz PINN의 전체 학습 과정을 관리합니다.

학습 전략 (2단계):
  1. Adam (20,000 iterations): 빠르게 대략적인 해 탐색
  2. L-BFGS (5,000 iterations): 정밀하게 수렴

콜로케이션 포인트:
  도메인 내부:  30,000 ~ 50,000개 (Helmholtz 잔차 평가)
  AR 코팅 면:  1,000개 (z=570um, 위상 경계조건)
  OPD 면:      7개 × 배치 (z=0, 세기 매칭)
  BM 영역:     14개 영역 × 200개 = 2,800개 (U=0 경계)

도메인:
  x: [0, 504] um (7피치)
  z: [0, 570] um (CG 550 + Encap 20)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch
import torch.optim as optim

from backend.core.pinn_model import HelmholtzPINN
from backend.physics.loss_functions import UDFPSPINNLosses
from backend.physics.tmm_calculator import GorillaDXTMM
from backend.api.schemas import BMDesignParams

logger = logging.getLogger(__name__)


# ============================================================
# 도메인 상수 (AGENTS.md 기준)
# ============================================================
DOMAIN_X = (0.0, 504.0)     # 7피치 = 72 × 7
DOMAIN_Z = (0.0, 590.0)     # Encap(20) + ILD(20) + CG(550) = 590
WAVELENGTH_UM = 0.520        # 520nm = 0.520um
K0 = 2 * math.pi / WAVELENGTH_UM  # 진공 파수
N_CG = 1.52                 # Cover Glass 굴절률
OPD_PITCH = 72.0
N_PITCHES = 7
Z_AR_COATING = 590.0        # AR 코팅 면 (도메인 상단, 지문 쪽)
Z_OPD = 0.0                 # OPD 면 (도메인 하단)
Z_BM2 = 20.0                # BM2 위치 (Encap 위, OPD 가까운 쪽)
Z_BM1 = 40.0                # BM1 위치 (ILD 위, CG 가까운 쪽)


@dataclass
class TrainConfig:
    """학습 설정."""
    # 콜로케이션 포인트
    n_collocation: int = 40_000
    n_boundary_per_bm: int = 200
    n_phase_points: int = 1000

    # Adam 단계
    adam_epochs: int = 20_000
    adam_lr: float = 1e-4

    # L-BFGS 단계
    lbfgs_epochs: int = 5_000
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 5

    # 로깅
    log_every: int = 500

    # 디바이스
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainState:
    """학습 상태 추적."""
    epoch: int = 0
    total_epochs: int = 0
    phase: str = "idle"  # "idle", "adam", "lbfgs", "done"
    loss_history: list = field(default_factory=list)
    best_loss: float = float("inf")
    best_model_state: Optional[dict] = None


class PINNTrainer:
    """
    Helmholtz PINN 학습 파이프라인.

    사용법:
        trainer = PINNTrainer(config=TrainConfig())
        trainer.set_design_params(params)       # BM 설계변수 설정
        trainer.set_lt_data(params_list, psf7s)  # LightTools 데이터 설정
        state = trainer.train()                  # 학습 실행
    """

    def __init__(
        self,
        model: Optional[HelmholtzPINN] = None,
        config: Optional[TrainConfig] = None,
    ):
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device)

        # 모델
        self.model = model or HelmholtzPINN()
        self.model.to(self.device)

        # 손실함수
        self.losses = UDFPSPINNLosses()

        # TMM (위상 경계조건)
        self.tmm = GorillaDXTMM()

        # 학습 상태
        self.state = TrainState()

        # 데이터 (나중에 설정)
        self._design_params: Optional[BMDesignParams] = None
        self._lt_coords: Optional[torch.Tensor] = None  # (N, 2) OPD 좌표
        self._lt_intensity: Optional[torch.Tensor] = None  # (N,) 세기

    def set_design_params(self, params: BMDesignParams):
        """BM 설계변수 설정 → BM 경계 좌표 생성."""
        self._design_params = params
        logger.info(
            f"Design params: delta_bm1={params.delta_bm1:.2f}, "
            f"delta_bm2={params.delta_bm2:.2f}, "
            f"w1={params.w1:.2f}, w2={params.w2:.2f}"
        )

    def set_lt_data(
        self,
        opd_intensities: list[np.ndarray],
    ):
        """
        LightTools 세기 데이터 설정.

        Args:
            opd_intensities: 각 LT 케이스의 7-OPD 세기 리스트
        """
        if not opd_intensities:
            logger.warning("No LT data provided, L_I will be zero")
            return

        # OPD 중심 좌표 (7피치)
        opd_centers = []
        for i in range(N_PITCHES):
            x = (i - 3) * OPD_PITCH + OPD_PITCH / 2  # 중심 좌표
            opd_centers.append(x)

        # 평균 세기 (여러 LT 케이스의 평균)
        avg_psf = np.mean(opd_intensities, axis=0)

        coords = []
        intensities = []
        for i, x_c in enumerate(opd_centers):
            coords.append([x_c, Z_OPD])
            intensities.append(avg_psf[i])

        self._lt_coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        self._lt_intensity = torch.tensor(intensities, dtype=torch.float32, device=self.device)
        logger.info(f"LT data set: {len(opd_intensities)} cases, avg peak={avg_psf.max():.4f}")

    # ============================================================
    # 콜로케이션 포인트 생성
    # ============================================================
    def _generate_collocation_points(self) -> torch.Tensor:
        """
        도메인 내부 콜로케이션 포인트 생성.
        Helmholtz 잔차를 이 포인트들에서 평가합니다.
        """
        n = self.config.n_collocation
        x = torch.rand(n, 1) * (DOMAIN_X[1] - DOMAIN_X[0]) + DOMAIN_X[0]
        z = torch.rand(n, 1) * (DOMAIN_Z[1] - DOMAIN_Z[0]) + DOMAIN_Z[0]
        coords = torch.cat([x, z], dim=1).to(self.device)
        coords.requires_grad_(True)
        return coords

    def _generate_phase_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        AR 코팅 면 (z=570um) 위상 경계조건 포인트 생성.
        TMM에서 계산한 Δφ(θ)를 타겟으로 사용.
        """
        n = self.config.n_phase_points
        x = torch.linspace(DOMAIN_X[0], DOMAIN_X[1], n)
        z = torch.full((n,), Z_AR_COATING)
        coords = torch.stack([x, z], dim=1).to(self.device)
        coords.requires_grad_(True)

        # TMM 위상 테이블 (x 좌표 → 입사각 → 위상)
        dphi_values = []
        tmm_table = self.tmm.compute_table()
        for xi in x.numpy():
            # x 좌표 → 대략적 입사각 추정 (중심으로부터의 각도)
            x_from_center = xi - DOMAIN_X[1] / 2
            theta_approx = math.degrees(math.atan2(x_from_center, DOMAIN_Z[1]))
            theta_clamped = max(-41, min(41, round(theta_approx)))
            dphi = tmm_table.get(float(theta_clamped), 0.0)
            dphi_values.append(math.radians(dphi))  # 라디안으로 변환

        dphi_tensor = torch.tensor(dphi_values, dtype=torch.float32, device=self.device)
        return coords, dphi_tensor

    def _generate_bm_points(self) -> torch.Tensor:
        """
        BM(Black Matrix) 불투명 영역 경계조건 포인트 생성.
        BM 영역에서 U=0을 강제.

        BM 구조 (7피치 반복):
          각 피치마다 아퍼처(투명) + BM(불투명)
          BM 포인트 = 아퍼처 밖의 영역
        """
        if self._design_params is None:
            raise ValueError("Design params not set. Call set_design_params() first.")

        p = self._design_params
        n_per = self.config.n_boundary_per_bm
        bm_points = []

        for pitch_idx in range(N_PITCHES):
            opd_center = (pitch_idx - 3) * OPD_PITCH + OPD_PITCH / 2

            # --- BM1 불투명 영역 ---
            bm1_center = opd_center + p.delta_bm1
            bm1_left = bm1_center - p.w1 / 2
            bm1_right = bm1_center + p.w1 / 2

            # 왼쪽 BM1 불투명 (피치 시작 ~ 아퍼처 왼쪽)
            pitch_start = opd_center - OPD_PITCH / 2
            if bm1_left > pitch_start:
                x_bm = torch.linspace(pitch_start, bm1_left, n_per // 2)
                z_bm = torch.full_like(x_bm, Z_BM1)
                bm_points.append(torch.stack([x_bm, z_bm], dim=1))

            # 오른쪽 BM1 불투명 (아퍼처 오른쪽 ~ 피치 끝)
            pitch_end = opd_center + OPD_PITCH / 2
            if bm1_right < pitch_end:
                x_bm = torch.linspace(bm1_right, pitch_end, n_per // 2)
                z_bm = torch.full_like(x_bm, Z_BM1)
                bm_points.append(torch.stack([x_bm, z_bm], dim=1))

            # --- BM2 불투명 영역 ---
            bm2_center = opd_center + p.delta_bm2
            bm2_left = bm2_center - p.w2 / 2
            bm2_right = bm2_center + p.w2 / 2

            if bm2_left > pitch_start:
                x_bm = torch.linspace(pitch_start, bm2_left, n_per // 2)
                z_bm = torch.full_like(x_bm, Z_BM2)
                bm_points.append(torch.stack([x_bm, z_bm], dim=1))

            if bm2_right < pitch_end:
                x_bm = torch.linspace(bm2_right, pitch_end, n_per // 2)
                z_bm = torch.full_like(x_bm, Z_BM2)
                bm_points.append(torch.stack([x_bm, z_bm], dim=1))

        if not bm_points:
            return torch.zeros(0, 2, device=self.device)

        return torch.cat(bm_points, dim=0).to(self.device)

    # ============================================================
    # 학습 1스텝
    # ============================================================
    def _compute_loss(self) -> dict:
        """4가지 손실 합산."""
        # 1) Helmholtz 잔차 (콜로케이션 포인트)
        colloc = self._generate_collocation_points()
        x_col = colloc[:, 0:1]
        z_col = colloc[:, 1:2]
        x_col.requires_grad_(True)
        z_col.requires_grad_(True)

        coords_col = torch.cat([x_col, z_col], dim=1)
        U_col = self.model.predict_complex(coords_col)
        L_helm = self.losses.helmholtz(U_col, x_col, z_col, K0, N_CG)

        # 2) Phase 경계조건 (AR 코팅 면)
        phase_coords, dphi_tmm = self._generate_phase_points()
        U_phase = self.model.predict_complex(phase_coords)
        U_pred_angle = torch.angle(U_phase)
        L_phase = self.losses.phase(U_pred_angle, dphi_tmm)

        # 3) Intensity 매칭 (LT 데이터)
        if self._lt_coords is not None and self._lt_intensity is not None:
            U_opd = self.model.predict_complex(self._lt_coords)
            L_I = self.losses.intensity(U_opd, self._lt_intensity)
        else:
            L_I = torch.tensor(0.0, device=self.device)

        # 4) BM 경계조건
        bm_coords = self._generate_bm_points()
        if bm_coords.shape[0] > 0:
            U_bm = self.model.predict_complex(bm_coords)
            L_BC = self.losses.boundary(U_bm)
        else:
            L_BC = torch.tensor(0.0, device=self.device)

        return self.losses.total(L_helm, L_phase, L_I, L_BC)

    # ============================================================
    # 학습 루프
    # ============================================================
    def train(
        self,
        progress_callback: Optional[Callable[[TrainState], None]] = None,
    ) -> TrainState:
        """
        2단계 학습 실행: Adam → L-BFGS.

        Args:
            progress_callback: 진행 상황 콜백 (선택)

        Returns:
            TrainState: 학습 결과
        """
        if self._design_params is None:
            raise ValueError("Call set_design_params() before training")

        total = self.config.adam_epochs + self.config.lbfgs_epochs
        self.state.total_epochs = total
        self.state.phase = "adam"

        logger.info(f"Starting PINN training on {self.device}")
        logger.info(f"  Adam: {self.config.adam_epochs} epochs, lr={self.config.adam_lr}")
        logger.info(f"  L-BFGS: {self.config.lbfgs_epochs} epochs")

        # ---- Phase 1: Adam ----
        self._train_adam(progress_callback)

        # ---- Phase 2: L-BFGS ----
        self._train_lbfgs(progress_callback)

        # 최고 모델 복원
        if self.state.best_model_state is not None:
            self.model.load_state_dict(self.state.best_model_state)

        self.state.phase = "done"
        logger.info(
            f"Training complete. Best loss: {self.state.best_loss:.6f}"
        )
        return self.state

    def _train_adam(self, callback: Optional[Callable] = None):
        """Adam 옵티마이저로 1단계 학습."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.adam_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.adam_epochs
        )

        for epoch in range(self.config.adam_epochs):
            self.model.train()
            optimizer.zero_grad()

            loss_dict = self._compute_loss()
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            self._update_state(epoch, loss_dict, callback)

    def _train_lbfgs(self, callback: Optional[Callable] = None):
        """L-BFGS 옵티마이저로 2단계 정밀 학습."""
        self.state.phase = "lbfgs"
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.config.lbfgs_lr,
            max_iter=self.config.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        for epoch in range(self.config.lbfgs_epochs):
            loss_dict_ref = [None]

            def closure():
                optimizer.zero_grad()
                ld = self._compute_loss()
                loss = ld["total"]
                loss.backward()
                loss_dict_ref[0] = ld
                return loss

            optimizer.step(closure)

            global_epoch = self.config.adam_epochs + epoch
            if loss_dict_ref[0] is not None:
                self._update_state(global_epoch, loss_dict_ref[0], callback)

    def _update_state(
        self,
        epoch: int,
        loss_dict: dict,
        callback: Optional[Callable],
    ):
        """학습 상태 업데이트 + 로깅."""
        loss_val = loss_dict["total"].item() if torch.is_tensor(loss_dict["total"]) else loss_dict["total"]

        self.state.epoch = epoch
        self.state.loss_history.append({
            "epoch": epoch,
            "total": loss_val,
            "helm": loss_dict["helm"],
            "phase": loss_dict["phase"],
            "I": loss_dict["I"],
            "BC": loss_dict["BC"],
        })

        # 최고 모델 갱신
        if loss_val < self.state.best_loss:
            self.state.best_loss = loss_val
            self.state.best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

        # 로깅
        if epoch % self.config.log_every == 0:
            logger.info(
                f"[{self.state.phase}] epoch={epoch:5d} | "
                f"total={loss_val:.6f} | "
                f"helm={loss_dict['helm']:.4f} phase={loss_dict['phase']:.4f} "
                f"I={loss_dict['I']:.4f} BC={loss_dict['BC']:.4f}"
            )

        if callback:
            callback(self.state)

    # ============================================================
    # 유틸리티
    # ============================================================
    def predict_psf7(self, params: BMDesignParams) -> np.ndarray:
        """
        학습된 PINN으로 7개 OPD PSF 예측.

        Args:
            params: BM 설계변수

        Returns:
            np.ndarray: 7개 OPD 세기 (shape: 7)
        """
        self.model.eval()
        psf = np.zeros(N_PITCHES)

        with torch.no_grad():
            for i in range(N_PITCHES):
                x_opd = (i - 3) * OPD_PITCH + OPD_PITCH / 2
                coord = torch.tensor(
                    [[x_opd, Z_OPD]], dtype=torch.float32, device=self.device
                )
                intensity = self.model.predict_intensity(coord)
                psf[i] = intensity.cpu().item()

        return psf

    def save_checkpoint(self, path: str):
        """모델 체크포인트 저장."""
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "state": {
                "epoch": self.state.epoch,
                "best_loss": self.state.best_loss,
                "phase": self.state.phase,
            },
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """모델 체크포인트 로드."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.state.epoch = ckpt["state"]["epoch"]
        self.state.best_loss = ckpt["state"]["best_loss"]
        logger.info(f"Checkpoint loaded from {path}")
