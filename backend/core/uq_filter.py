"""
uq_filter.py - MC Dropout 불확실도 추정
==========================================
PINN/FNO 예측의 불확실도(σ)를 MC Dropout으로 추정합니다.

MC Dropout 원리:
  1. Dropout을 활성화한 채로 같은 입력에 대해 N번 추론
  2. N개 출력의 분산(σ²) = 모델의 불확실도
  3. σ가 크면 → 모델이 자신 없는 영역 → LightTools 검증 필요

AGENTS.md 규칙:
  - σ > 0.05 → LightTools 강제 검증
  - σ 상위 20% → LT 우선 검증 (Active Learning)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from backend.api.schemas import BMDesignParams

logger = logging.getLogger(__name__)


class MCDropoutWrapper(nn.Module):
    """
    기존 모델에 MC Dropout을 추가하는 래퍼.
    모델의 Linear 층 사이에 Dropout을 삽입합니다.
    """

    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 원래 모델의 forward에 dropout 적용
        features = self.model.embedding(x)
        for layer in self.model.network:
            features = layer(features)
            features = self.dropout(features)  # 각 층 후 dropout
        return self.model.output_layer(features)

    def enable_mc_dropout(self):
        """추론 시에도 Dropout 활성화."""
        self.dropout.train()

    def disable_mc_dropout(self):
        """Dropout 비활성화."""
        self.dropout.eval()


class UQFilter:
    """
    불확실도 추정 및 필터링.

    사용법:
        uq = UQFilter(model, n_samples=50)
        sigma = uq.estimate_uncertainty(coords)
        needs_lt = uq.needs_lt_verification(sigma)
    """

    SIGMA_THRESHOLD = 0.05  # LT 강제 검증 임계값 (AGENTS.md)
    TOP_PERCENT = 0.20      # LT 우선 검증 비율

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        dropout_rate: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.n_samples = n_samples

        # MC Dropout 래퍼
        self.mc_model = MCDropoutWrapper(model, dropout_rate)
        self.mc_model.to(self.device)

    def estimate_uncertainty(
        self,
        coords: torch.Tensor,
    ) -> dict:
        """
        MC Dropout으로 불확실도 추정.

        Args:
            coords: (N, 2) 좌표 텐서

        Returns:
            dict:
              mean: (N, 2) 평균 예측
              std:  (N, 2) 표준편차 (불확실도)
              sigma: (N,) 세기 불확실도 (|U|² 기준)
        """
        self.mc_model.enable_mc_dropout()
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.mc_model(coords.to(self.device))
                predictions.append(pred)

        preds = torch.stack(predictions)  # (n_samples, N, 2)
        mean = preds.mean(dim=0)          # (N, 2)
        std = preds.std(dim=0)            # (N, 2)

        # 세기 불확실도: |U|²의 표준편차
        intensities = preds[:, :, 0]**2 + preds[:, :, 1]**2  # (n_samples, N)
        sigma = intensities.std(dim=0)    # (N,)

        self.mc_model.disable_mc_dropout()

        return {
            "mean": mean.cpu(),
            "std": std.cpu(),
            "sigma": sigma.cpu(),
        }

    def estimate_psf_uncertainty(
        self,
        model,
        params: BMDesignParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        7-OPD PSF의 불확실도 추정.

        Returns:
            (psf7_mean, psf7_sigma): 각각 (7,) 배열
        """
        from backend.core.pinn_trainer import Z_OPD, OPD_PITCH, N_PITCHES

        coords = []
        for i in range(N_PITCHES):
            x = (i - 3) * OPD_PITCH + OPD_PITCH / 2
            coords.append([x, Z_OPD])

        coords_t = torch.tensor(coords, dtype=torch.float32)
        result = self.estimate_uncertainty(coords_t)

        # mean intensity
        mean_re = result["mean"][:, 0]
        mean_im = result["mean"][:, 1]
        psf_mean = (mean_re**2 + mean_im**2).numpy()

        psf_sigma = result["sigma"].numpy()

        return psf_mean, psf_sigma

    def needs_lt_verification(self, sigma: torch.Tensor) -> bool:
        """σ > 0.05 이면 LT 검증 필요."""
        return bool(sigma.max().item() > self.SIGMA_THRESHOLD)

    def get_priority_indices(self, sigmas: torch.Tensor) -> list[int]:
        """σ 상위 20% 인덱스 반환 (LT 우선 검증 대상)."""
        n_top = max(1, int(len(sigmas) * self.TOP_PERCENT))
        _, indices = sigmas.topk(n_top)
        return indices.tolist()
