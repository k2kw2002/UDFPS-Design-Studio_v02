"""
dataset_manager.py - (p, PSF×7) 쌍 관리
=========================================
PINN 학습, FNO 증류, BoTorch 평가에 사용되는
설계변수-PSF 데이터 쌍을 통합 관리합니다.

데이터 소스:
  1. LightTools 시뮬레이션 (최대 200개)
  2. PINN 예측 (FNO 증류용 10,000개)
  3. Active Learning 추가 데이터

저장 형식:
  - params: (N, 4) tensor [delta_bm1, delta_bm2, w1, w2]
  - psf7:   (N, 7) tensor [OPD#0 ~ OPD#6 세기]
  - source: 데이터 출처 ("lt", "pinn", "measured")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from backend.api.schemas import BMDesignParams

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "datasets"


class PINNDataset(Dataset):
    """
    PyTorch Dataset for (params, psf7) pairs.
    PINN 학습, FNO 증류에서 DataLoader로 사용.
    """

    def __init__(
        self,
        params: torch.Tensor,
        psf7: torch.Tensor,
        sources: Optional[list[str]] = None,
    ):
        assert params.shape[0] == psf7.shape[0], "params와 psf7 개수 불일치"
        assert params.shape[1] == 4, "params는 4차원이어야 합니다"
        assert psf7.shape[1] == 7, "psf7은 7개 OPD 값이어야 합니다"

        self.params = params.float()
        self.psf7 = psf7.float()
        self.sources = sources or ["unknown"] * len(params)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.psf7[idx]


class DatasetManager:
    """
    데이터셋 통합 관리자.
    LT 데이터, PINN 예측, 측정 데이터를 합쳐서 관리.
    """

    def __init__(self):
        self._params_list: list[np.ndarray] = []  # each (4,)
        self._psf7_list: list[np.ndarray] = []     # each (7,)
        self._sources: list[str] = []

        DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def n_samples(self) -> int:
        return len(self._params_list)

    @property
    def n_lt_samples(self) -> int:
        return sum(1 for s in self._sources if s == "lt")

    def add_sample(
        self,
        params: BMDesignParams,
        psf7: np.ndarray,
        source: str = "lt",
    ):
        """단일 샘플 추가."""
        p = np.array([params.delta_bm1, params.delta_bm2, params.w1, params.w2])
        self._params_list.append(p)
        self._psf7_list.append(psf7.copy())
        self._sources.append(source)

    def add_batch(
        self,
        data: list[tuple[BMDesignParams, np.ndarray]],
        source: str = "lt",
    ):
        """배치 추가."""
        for params, psf7 in data:
            self.add_sample(params, psf7, source)
        logger.info(f"Added {len(data)} samples (source={source}), total={self.n_samples}")

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """전체 데이터를 (params_tensor, psf7_tensor)로 반환."""
        if self.n_samples == 0:
            return torch.zeros(0, 4), torch.zeros(0, 7)
        params = torch.tensor(np.stack(self._params_list), dtype=torch.float32)
        psf7 = torch.tensor(np.stack(self._psf7_list), dtype=torch.float32)
        return params, psf7

    def get_dataset(self) -> PINNDataset:
        """PyTorch Dataset 반환."""
        params, psf7 = self.get_tensors()
        return PINNDataset(params, psf7, self._sources.copy())

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """PyTorch DataLoader 반환."""
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def save(self, name: str = "default"):
        """데이터셋을 .pt 파일로 저장."""
        if self.n_samples == 0:
            logger.warning("No data to save")
            return

        params, psf7 = self.get_tensors()
        filepath = DATA_DIR / f"{name}.pt"
        torch.save({
            "params": params,
            "psf7": psf7,
            "sources": self._sources,
        }, filepath)
        logger.info(f"Saved {self.n_samples} samples to {filepath}")

    def load(self, name: str = "default") -> bool:
        """저장된 데이터셋 로드."""
        filepath = DATA_DIR / f"{name}.pt"
        if not filepath.exists():
            logger.warning(f"Dataset not found: {filepath}")
            return False

        data = torch.load(filepath, weights_only=False)
        params = data["params"].numpy()
        psf7 = data["psf7"].numpy()
        sources = data.get("sources", ["unknown"] * len(params))

        self._params_list = [params[i] for i in range(len(params))]
        self._psf7_list = [psf7[i] for i in range(len(psf7))]
        self._sources = list(sources)

        logger.info(f"Loaded {self.n_samples} samples from {filepath}")
        return True

    def split(self, train_ratio: float = 0.8) -> tuple["DatasetManager", "DatasetManager"]:
        """학습/검증 분할."""
        n = self.n_samples
        indices = np.random.permutation(n)
        split_idx = int(n * train_ratio)

        train_mgr = DatasetManager()
        val_mgr = DatasetManager()

        for i in indices[:split_idx]:
            train_mgr._params_list.append(self._params_list[i])
            train_mgr._psf7_list.append(self._psf7_list[i])
            train_mgr._sources.append(self._sources[i])

        for i in indices[split_idx:]:
            val_mgr._params_list.append(self._params_list[i])
            val_mgr._psf7_list.append(self._psf7_list[i])
            val_mgr._sources.append(self._sources[i])

        return train_mgr, val_mgr

    def summary(self) -> dict:
        """데이터셋 요약 통계."""
        source_counts = {}
        for s in self._sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "total_samples": self.n_samples,
            "sources": source_counts,
            "lt_remaining": 200 - self.n_lt_samples,
        }
