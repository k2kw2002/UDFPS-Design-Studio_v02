"""
lighttools_runner.py - LightTools Python API 래퍼
===================================================
LightTools COM API를 통해 7피치 도메인 PSF를 수집합니다.

실행 조건:
  - Windows + LightTools 설치 필수
  - LightTools Python API (ltapi) 사용 가능해야 함
  - 없으면 CSV 파일에서 기존 데이터 로드

사용처:
  - L_I 손실함수용 정답 데이터 (200개)
  - Active Learning 검증 데이터
  - UQ 임계값 초과 시 강제 검증
"""

import os
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from backend.api.schemas import BMDesignParams

logger = logging.getLogger(__name__)

# 데이터 저장 경로
DATA_DIR = Path(__file__).parent / "lt_results"


class LightToolsRunner:
    """
    LightTools COM API 래퍼.
    7피치(504um) 도메인에서 PSF 7개 OPD 세기를 수집.

    LightTools가 없는 환경에서는 CSV 로드/저장만 동작.
    """

    MAX_LT_CALLS = 200  # LT 사용 한도 (AGENTS.md)

    def __init__(self, lt_model_path: Optional[str] = None):
        self.lt_model_path = lt_model_path
        self.lt_api = None
        self.call_count = 0

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # LightTools COM 연결 시도
        try:
            import win32com.client
            self.lt_api = win32com.client.Dispatch("LightTools.LTAPI")
            if lt_model_path:
                self.lt_api.Open(lt_model_path)
            logger.info("LightTools API connected")
        except Exception as e:
            logger.warning(f"LightTools not available: {e}. Using CSV mode only.")

    def run_single(self, params: BMDesignParams) -> Optional[np.ndarray]:
        """
        단일 설계변수로 LightTools 시뮬레이션 실행.

        Args:
            params: BM 설계변수

        Returns:
            np.ndarray: 7개 OPD PSF 세기 (shape: 7) 또는 None
        """
        if self.lt_api is None:
            logger.error("LightTools API not available")
            return None

        if self.call_count >= self.MAX_LT_CALLS:
            logger.warning(f"LT call limit reached ({self.MAX_LT_CALLS})")
            return None

        try:
            # BM1 아퍼처 설정 (7피치 반복)
            for i in range(7):
                x_center = (i - 3) * params.opd_pitch + params.delta_bm1
                self.lt_api.SetParameter(
                    f"BM1.Aperture[{i}].Center", x_center
                )
                self.lt_api.SetParameter(
                    f"BM1.Aperture[{i}].Width", params.w1
                )

            # BM2 아퍼처 설정
            for i in range(7):
                x_center = (i - 3) * params.opd_pitch + params.delta_bm2
                self.lt_api.SetParameter(
                    f"BM2.Aperture[{i}].Center", x_center
                )
                self.lt_api.SetParameter(
                    f"BM2.Aperture[{i}].Width", params.w2
                )

            # Ray trace 실행
            self.lt_api.RunRayTrace(NumRays=1_000_000)

            # OPD 7개 세기 수집
            psf7 = np.zeros(7)
            for i in range(7):
                psf7[i] = self.lt_api.GetReceiverPower(f"OPD[{i}]")

            self.call_count += 1
            logger.info(
                f"LT run #{self.call_count}: "
                f"delta_bm1={params.delta_bm1:.2f}, w1={params.w1:.2f} "
                f"-> peak={psf7.max():.4f}"
            )
            return psf7

        except Exception as e:
            logger.error(f"LightTools simulation failed: {e}")
            return None

    def run_batch(
        self, samples: list[BMDesignParams]
    ) -> list[tuple[BMDesignParams, np.ndarray]]:
        """
        배치 시뮬레이션. 성공한 (params, psf7) 쌍만 반환.
        """
        results = []
        for i, params in enumerate(samples):
            if self.call_count >= self.MAX_LT_CALLS:
                logger.warning(f"LT limit reached at sample {i}/{len(samples)}")
                break
            psf7 = self.run_single(params)
            if psf7 is not None:
                results.append((params, psf7))
        return results

    def save_to_csv(
        self,
        data: list[tuple[BMDesignParams, np.ndarray]],
        filename: str = "lt_data.csv",
    ):
        """(params, psf7) 쌍을 CSV로 저장."""
        filepath = DATA_DIR / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "delta_bm1", "delta_bm2", "w1", "w2",
                "psf0", "psf1", "psf2", "psf3", "psf4", "psf5", "psf6",
            ])
            for params, psf7 in data:
                writer.writerow([
                    params.delta_bm1, params.delta_bm2,
                    params.w1, params.w2,
                    *psf7.tolist(),
                ])
        logger.info(f"Saved {len(data)} samples to {filepath}")

    def load_from_csv(
        self, filename: str = "lt_data.csv"
    ) -> list[tuple[BMDesignParams, np.ndarray]]:
        """CSV에서 기존 데이터 로드."""
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.warning(f"No CSV found at {filepath}")
            return []

        data = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                params = BMDesignParams(
                    delta_bm1=float(row["delta_bm1"]),
                    delta_bm2=float(row["delta_bm2"]),
                    w1=float(row["w1"]),
                    w2=float(row["w2"]),
                )
                psf7 = np.array([
                    float(row[f"psf{i}"]) for i in range(7)
                ])
                data.append((params, psf7))
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data

    @property
    def remaining_calls(self) -> int:
        """남은 LT 호출 횟수."""
        return max(0, self.MAX_LT_CALLS - self.call_count)
