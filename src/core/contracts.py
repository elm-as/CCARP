"""Immutable domain data contracts transferred across CCARP pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import ModelingConfig


@dataclass(frozen=True)
class JoinAuditStats:
    """Audit statistics from deduplicating and inner-joining raw credit tables."""

    app_raw_rows: int
    app_unique_ids: int
    app_duplicate_ids: int
    credit_raw_rows: int
    credit_unique_ids: int
    joined_ids: int
    joined_credit_rows: int
    app_only_ids: int
    credit_only_ids: int


@dataclass(frozen=True)
class JoinedTables:
    """Validated and inner-joined application and credit history tables."""

    application_df: pd.DataFrame
    credit_df: pd.DataFrame
    audit: JoinAuditStats


@dataclass(frozen=True)
class TemporalPartition:
    """Strict causal split of credit history into observation and performance slices."""

    t_obs: int
    obs_df: pd.DataFrame
    perf_df: pd.DataFrame
    cohort_ids: np.ndarray
    target_series: pd.Series
    cohort_bounds: pd.DataFrame
    obs_prior_bad_count: int


@dataclass(frozen=True)
class FeatureSchema:
    """Explicit numerical and categorical column lists for model pipelines."""

    numeric_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]


@dataclass(frozen=True)
class FeatureDataset:
    """Model-ready feature matrix paired with targets and demographic group keys."""

    features_df: pd.DataFrame
    target: np.ndarray
    group_keys: np.ndarray
    client_ids: np.ndarray
    schema: FeatureSchema


@dataclass(frozen=True)
class ModelRunSpec:
    """Bundled model identifier, CV/seed configuration, and feature schema."""

    model_name: str
    modeling_cfg: ModelingConfig
    schema: FeatureSchema


@dataclass(frozen=True)
class FoldSplitSet:
    """Immutable container of cross-validation train/validation index pairs."""

    scheme_name: str
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class CalibrationBundle:
    """Out-of-fold raw, Platt-calibrated, and Isotonic-calibrated predictions."""

    model_name: str
    y_true: np.ndarray
    oof_raw: np.ndarray
    oof_platt: np.ndarray
    oof_isotonic: np.ndarray


@dataclass(frozen=True)
class RiskMetrics:
    """Standard discrimination and calibration quality metrics."""

    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss_val: float
    ece_10bin: float
    ks_statistic: float


@dataclass(frozen=True)
class AblationRow:
    """Single row of the temporal and demographic-group leakage ablation study."""

    regime_label: str
    feature_scope: str
    cv_scheme: str
    n_samples: int
    n_positives: int
    metrics: RiskMetrics


@dataclass(frozen=True)
class ProfitPoint:
    """Expected unit-economics performance at a specific approval PD cutoff."""

    threshold: float
    approval_rate: float
    n_approved: int
    approved_bad_rate: float
    expected_profit_per_applicant: float
    total_portfolio_profit: float


@dataclass(frozen=True)
class SensitivityRow:
    """Optimal expected profit and cutoff across a (margin_good, LGD) grid cell."""

    margin_good: float
    lgd: float
    tau_theory: float
    tau_optimal: float
    approval_rate_optimal: float
    expected_profit_optimal: float
    profit_lift_vs_approve_all: float


@dataclass(frozen=True)
class ProfitCurveResult:
    """Complete threshold optimization curve and policy benchmark comparison."""

    best_model_name: str
    calibration_variant: str
    tau_theory: float
    optimal_point: ProfitPoint
    theory_point: ProfitPoint
    approve_all_point: ProfitPoint
    tau_050_point: ProfitPoint
    curve_points: tuple[ProfitPoint, ...]
    sensitivity_grid: tuple[SensitivityRow, ...]


@dataclass(frozen=True)
class VisualizationBundle:
    """Aggregated experimental outputs consumed by static figures and HTML dashboard."""

    cohort_summary: dict[str, Any]
    cutoff_grid: tuple[dict[str, Any], ...]
    ablation_rows: tuple[AblationRow, ...]
    calibration_bundles: tuple[CalibrationBundle, ...]
    model_metrics_rows: tuple[dict[str, Any], ...]
    reliability_curves: dict[str, dict[str, list[float]]]
    profit_result: ProfitCurveResult
    feature_drivers: tuple[dict[str, Any], ...]
    hyperparameters: tuple[dict[str, Any], ...]
