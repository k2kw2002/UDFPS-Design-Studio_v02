"""
agents_config.py - AGENTS.md 파싱 및 도메인 규칙 제공
=====================================================
AGENTS.md에 정의된 BM 도메인 규칙을 파싱하여
코드에서 사용할 수 있는 상수/제약으로 변환합니다.

용도:
  - 에이전트(Planner, Generator, Evaluator)가 규칙 참조
  - 물리 상수/범위를 코드 내에서 일관되게 사용
  - AGENTS.md 변경 시 코드 자동 반영
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

AGENTS_MD_PATH = Path(__file__).parent.parent / "AGENTS.md"


@dataclass
class OpticalConstants:
    """광학 상수 (AGENTS.md '광학 상수' 섹션)."""
    wavelength_nm: float = 520.0
    cg_critical_angle: float = 41.1
    n_cg: float = 1.52
    crosstalk_angle: float = 22.5
    crosstalk_distance_um: float = 191.0
    crosstalk_pitches: float = 2.65
    dphi_30deg_520nm: float = -10.6


@dataclass
class PINNDomain:
    """PINN 도메인 설정 (AGENTS.md 'PINN 도메인' 섹션)."""
    x_um: float = 504.0      # 7피치
    z_um: float = 570.0      # CG + Encap
    theta_range: tuple = (-41.1, 41.1)
    n_pitches: int = 7
    pitch_um: float = 72.0


@dataclass
class DesignBounds:
    """설계변수 범위 (AGENTS.md '최적화 설계변수' 섹션)."""
    delta_bm1: tuple = (-10.0, 10.0)
    delta_bm2: tuple = (-10.0, 10.0)
    w1: tuple = (5.0, 20.0)
    w2: tuple = (5.0, 20.0)
    d_fixed: float = 20.0


@dataclass
class OptimizationTargets:
    """최적화 목표 (AGENTS.md '최적화 목표' 섹션)."""
    mtf_ridge_min: float = 0.60
    skewness_max: float = 0.10
    throughput_min: float = 0.60


@dataclass
class BoTorchRules:
    """BoTorch 탐색 규칙."""
    n_initial_lhs: int = 20
    convergence_window: int = 5
    convergence_threshold: float = 0.001  # 0.1%
    uq_sigma_threshold: float = 0.05


@dataclass
class EvaluatorRules:
    """Evaluator 채점 기준."""
    score_dimensions: int = 5
    points_per_dimension: int = 20
    total_points: int = 100
    retry_threshold: int = 70


@dataclass
class AgentsConfig:
    """전체 도메인 설정."""
    optical: OpticalConstants = field(default_factory=OpticalConstants)
    domain: PINNDomain = field(default_factory=PINNDomain)
    bounds: DesignBounds = field(default_factory=DesignBounds)
    targets: OptimizationTargets = field(default_factory=OptimizationTargets)
    botorch: BoTorchRules = field(default_factory=BoTorchRules)
    evaluator: EvaluatorRules = field(default_factory=EvaluatorRules)


def load_agents_config(path: Optional[Path] = None) -> AgentsConfig:
    """
    AGENTS.md를 파싱하여 AgentsConfig 반환.
    파싱 실패 시 기본값 사용.
    """
    path = path or AGENTS_MD_PATH
    config = AgentsConfig()

    if not path.exists():
        logger.warning(f"AGENTS.md not found at {path}, using defaults")
        return config

    try:
        text = path.read_text(encoding="utf-8")
        _parse_agents_md(text, config)
        logger.info(f"Loaded AGENTS.md from {path}")
    except Exception as e:
        logger.warning(f"Failed to parse AGENTS.md: {e}, using defaults")

    return config


def _parse_agents_md(text: str, config: AgentsConfig):
    """AGENTS.md 텍스트에서 값을 추출."""
    # 파장
    m = re.search(r"평가 파장:\s*(\d+)nm", text)
    if m:
        config.optical.wavelength_nm = float(m.group(1))

    # 임계각
    m = re.search(r"CG 임계각:\s*([\d.]+)도", text)
    if m:
        config.optical.cg_critical_angle = float(m.group(1))

    # 크로스토크 각도
    m = re.search(r"크로스토크 각도:\s*([\d.]+)도", text)
    if m:
        config.optical.crosstalk_angle = float(m.group(1))

    # PINN 도메인
    m = re.search(r"가로 x:\s*(\d+)um", text)
    if m:
        config.domain.x_um = float(m.group(1))

    m = re.search(r"깊이 z:\s*(\d+)um", text)
    if m:
        config.domain.z_um = float(m.group(1))

    # 최적화 목표
    m = re.search(r"MTF@ridge\s*>=?\s*([\d.]+)%", text)
    if m:
        config.targets.mtf_ridge_min = float(m.group(1)) / 100.0

    m = re.search(r"skewness\s*<=?\s*([\d.]+)", text)
    if m:
        config.targets.skewness_max = float(m.group(1))

    # Evaluator 기준
    m = re.search(r"(\d+)점 미만.*재시도", text)
    if m:
        config.evaluator.retry_threshold = int(m.group(1))
