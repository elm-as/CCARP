"""Leakage ablation auditing and calibrated model selection."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from src.core.contracts import AblationRow, CalibrationBundle
from src.evaluation.risk_metrics import compute_risk_metrics


def evaluate_ablation_regime(
    regime_spec: tuple[str, str, str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> AblationRow:
    """Compute standardized risk metrics for one temporal/CV ablation regime."""
    regime_label, feature_scope, cv_scheme = regime_spec
    metrics = compute_risk_metrics(y_true, y_prob)
    return AblationRow(
        regime_label=regime_label,
        feature_scope=feature_scope,
        cv_scheme=cv_scheme,
        n_samples=int(len(y_true)),
        n_positives=int(np.sum(y_true)),
        metrics=metrics,
    )


def summarize_calibration_comparison(
    bundles: tuple[CalibrationBundle, ...],
) -> tuple[dict[str, Any], ...]:
    """Produce a flat comparison table of Raw, Platt, and Isotonic OOF metrics."""
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        variants = (
            ("raw", bundle.oof_raw),
            ("platt", bundle.oof_platt),
            ("isotonic", bundle.oof_isotonic),
        )
        for variant_name, probs in variants:
            metrics = compute_risk_metrics(bundle.y_true, probs)
            row = {
                "model_name": bundle.model_name,
                "calibration": variant_name,
            }
            row.update(asdict(metrics))
            rows.append(row)
    return tuple(rows)


def select_best_calibrated_variant(
    bundles: tuple[CalibrationBundle, ...],
) -> tuple[CalibrationBundle, str]:
    """Select the calibrated (bundle, variant) achieving lowest OOF Log-Loss."""
    if not bundles:
        raise ValueError("Cannot select best model from empty bundles tuple.")

    best_score: tuple[float, float, float] | None = None
    best_pair: tuple[CalibrationBundle, str] | None = None

    for bundle in bundles:
        for variant, probs in (("platt", bundle.oof_platt), ("isotonic", bundle.oof_isotonic)):
            metrics = compute_risk_metrics(bundle.y_true, probs)
            candidate_key = (
                metrics.log_loss_val,
                metrics.brier_score,
                -metrics.roc_auc,
            )
            if best_score is None or candidate_key < best_score:
                best_score = candidate_key
                best_pair = (bundle, variant)

    if best_pair is None:
        raise RuntimeError("Failed to select a calibrated model variant.")
    return best_pair
