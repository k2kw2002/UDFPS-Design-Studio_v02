"""
flywheel.py - 재학습 파이프라인 (플라이휠)
============================================
데이터 수집 → PINN 학습 → FNO 증류 → BO 최적화의
전체 사이클을 자동으로 관리합니다.

플라이휠 사이클:
  1. LHS 샘플링 → LT 시뮬레이션 (초기 데이터)
  2. PINN 학습 (Adam → L-BFGS)
  3. PINN → FNO 증류 (10,000 teacher 샘플)
  4. BoTorch qNEHVI 최적화
  5. Active Learning → LT 검증 → 데이터 편입
  6. 드리프트 감지 → 재학습 트리거 → 2번으로 복귀
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

import numpy as np
import torch

from backend.api.schemas import BMDesignParams, BMDesignSpec
from backend.data.dataset_manager import DatasetManager
from backend.data.lhs_sampler import generate_lhs_samples
from backend.harness.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


class FlywheelStage(Enum):
    IDLE = "idle"
    SAMPLING = "sampling"
    PINN_TRAINING = "pinn_training"
    FNO_DISTILLATION = "fno_distillation"
    OPTIMIZATION = "optimization"
    ACTIVE_LEARNING = "active_learning"
    COMPLETE = "complete"


@dataclass
class FlywheelStatus:
    """플라이휠 현재 상태."""
    stage: FlywheelStage
    cycle: int
    progress: float  # 0.0 ~ 1.0
    message: str


class Flywheel:
    """
    전체 재학습 파이프라인 오케스트레이터.

    사용법:
        flywheel = Flywheel(
            pinn_trainer=trainer,
            fno_distiller=distiller,
            optimizer=optimizer,
            al_pipeline=al_pipeline,
        )
        flywheel.run_cycle()
    """

    def __init__(
        self,
        pinn_trainer=None,
        fno_model=None,
        optimizer=None,
        al_pipeline=None,
        dataset_mgr: Optional[DatasetManager] = None,
        drift_detector: Optional[DriftDetector] = None,
    ):
        self.pinn_trainer = pinn_trainer
        self.fno_model = fno_model
        self.optimizer = optimizer
        self.al_pipeline = al_pipeline
        self.dataset_mgr = dataset_mgr or DatasetManager()
        self.drift_detector = drift_detector or DriftDetector()

        self._cycle_count = 0
        self._stage = FlywheelStage.IDLE
        self._callbacks: list[Callable] = []

    @property
    def status(self) -> FlywheelStatus:
        return FlywheelStatus(
            stage=self._stage,
            cycle=self._cycle_count,
            progress=self._estimate_progress(),
            message=self._stage_message(),
        )

    def add_callback(self, fn: Callable[[FlywheelStatus], None]):
        """상태 변경 콜백 등록."""
        self._callbacks.append(fn)

    def _notify(self):
        """상태 변경 알림."""
        for fn in self._callbacks:
            try:
                fn(self.status)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def run_initial_setup(self, n_lhs: int = 200):
        """
        초기 설정: LHS 샘플링.
        LT가 없는 환경에서는 샘플만 생성.
        """
        self._stage = FlywheelStage.SAMPLING
        self._notify()

        logger.info(f"Generating {n_lhs} LHS samples...")
        samples = generate_lhs_samples(n_samples=n_lhs)
        logger.info(f"Generated {len(samples)} valid samples")

        self._stage = FlywheelStage.IDLE
        self._notify()
        return samples

    def run_cycle(
        self,
        spec: Optional[BMDesignSpec] = None,
        skip_pinn: bool = False,
        skip_fno: bool = False,
    ):
        """
        플라이휠 1사이클 실행.

        Args:
            spec: 역설계 목표 사양
            skip_pinn: PINN 학습 건너뛰기 (이미 학습된 경우)
            skip_fno: FNO 증류 건너뛰기
        """
        self._cycle_count += 1
        logger.info(f"=== Flywheel Cycle {self._cycle_count} ===")

        # Stage 1: PINN 학습
        if not skip_pinn and self.pinn_trainer is not None:
            self._stage = FlywheelStage.PINN_TRAINING
            self._notify()
            logger.info("Stage 1: PINN Training")
            self.pinn_trainer.train()

        # Stage 2: FNO 증류
        if not skip_fno and self.fno_model is not None and self.pinn_trainer is not None:
            self._stage = FlywheelStage.FNO_DISTILLATION
            self._notify()
            logger.info("Stage 2: FNO Distillation")
            self._distill_fno()

        # Stage 3: BO 최적화
        if self.optimizer is not None:
            self._stage = FlywheelStage.OPTIMIZATION
            self._notify()
            logger.info("Stage 3: BoTorch Optimization")
            result = self.optimizer.optimize()
            logger.info(
                f"Optimization complete: {len(result.top5)} candidates, "
                f"HV={result.hypervolume:.4f}"
            )

        # Stage 4: Active Learning
        if self.al_pipeline is not None:
            self._stage = FlywheelStage.ACTIVE_LEARNING
            self._notify()
            logger.info("Stage 4: Active Learning")
            samples = generate_lhs_samples(n_samples=50)
            self.al_pipeline.run(samples)

        # 드리프트 검사
        if self.drift_detector.needs_retrain:
            logger.warning("Drift detected! Scheduling next cycle.")

        self._stage = FlywheelStage.COMPLETE
        self._notify()
        logger.info(f"Cycle {self._cycle_count} complete")

    def _distill_fno(self, n_teacher_samples: int = 10_000):
        """PINN → FNO 증류."""
        if self.pinn_trainer is None or self.fno_model is None:
            return

        logger.info(f"Generating {n_teacher_samples} teacher samples...")
        samples = generate_lhs_samples(n_samples=n_teacher_samples, validate=True)

        teacher_X = []
        teacher_Y = []
        for s in samples:
            psf7 = self.pinn_trainer.predict_psf7(s)
            teacher_X.append([s.delta_bm1, s.delta_bm2, s.w1, s.w2])
            teacher_Y.append(psf7)

        X = torch.tensor(np.array(teacher_X), dtype=torch.float32)
        Y = torch.tensor(np.array(teacher_Y), dtype=torch.float32)

        # FNO 학습
        optimizer = torch.optim.Adam(self.fno_model.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.fno_model.train()
        for epoch in range(100):
            total_loss = 0.0
            for xb, yb in loader:
                pred = self.fno_model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                logger.info(f"FNO distillation epoch {epoch}: loss={total_loss/len(loader):.6f}")

        self.fno_model.eval()
        logger.info("FNO distillation complete")

    def _estimate_progress(self) -> float:
        """현재 진행률 추정."""
        stage_progress = {
            FlywheelStage.IDLE: 0.0,
            FlywheelStage.SAMPLING: 0.1,
            FlywheelStage.PINN_TRAINING: 0.3,
            FlywheelStage.FNO_DISTILLATION: 0.5,
            FlywheelStage.OPTIMIZATION: 0.7,
            FlywheelStage.ACTIVE_LEARNING: 0.9,
            FlywheelStage.COMPLETE: 1.0,
        }
        return stage_progress.get(self._stage, 0.0)

    def _stage_message(self) -> str:
        """현재 단계 메시지."""
        messages = {
            FlywheelStage.IDLE: "대기 중",
            FlywheelStage.SAMPLING: "LHS 샘플링 중...",
            FlywheelStage.PINN_TRAINING: "PINN 학습 중 (Adam → L-BFGS)...",
            FlywheelStage.FNO_DISTILLATION: "FNO 증류 중 (PINN → FNO)...",
            FlywheelStage.OPTIMIZATION: "BoTorch 3목적 최적화 중...",
            FlywheelStage.ACTIVE_LEARNING: "능동 학습 루프 실행 중...",
            FlywheelStage.COMPLETE: "사이클 완료",
        }
        return messages.get(self._stage, "알 수 없음")
