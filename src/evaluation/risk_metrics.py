"""Classification discrimination, calibration error, and KS risk metrics."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from src.core.contracts import RiskMetrics

PROB_CLIP: float = 1e-6


def _validate_binary_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary targets and finite predicted probabilities."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    if len(y_arr) != len(p_arr) or len(y_arr) == 0:
        raise ValueError("y_true and y_prob must be non-empty and equal length.")
    if not np.isfinite(p_arr).all():
        raise ValueError("y_prob contains non-finite values.")
    if len(np.unique(y_arr)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1).")
    return y_arr, np.clip(p_arr, 0.0, 1.0)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov separation between positive and negative PDs."""
    y_arr, p_arr = _validate_binary_inputs(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_arr, p_arr)
    return float(np.max(np.abs(tpr - fpr)))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute uniform-bin Expected Calibration Error (ECE)."""
    y_arr, p_arr = _validate_binary_inputs(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, n_bins - 1)
    n_total = float(len(y_arr))
    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        emp_rate = float(np.mean(y_arr[mask]))
        mean_pred = float(np.mean(p_arr[mask]))
        ece += (float(np.sum(mask)) / n_total) * abs(emp_rate - mean_pred)
    return float(ece)


def compute_risk_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> RiskMetrics:
    """Compute ROC-AUC, PR-AUC, Brier score, Log-Loss, 10-bin ECE, and KS."""
    y_arr, p_arr = _validate_binary_inputs(y_true, y_prob)
    p_clipped = np.clip(p_arr, PROB_CLIP, 1.0 - PROB_CLIP)
    return RiskMetrics(
        roc_auc=round(float(roc_auc_score(y_arr, p_arr)), 6),
        pr_auc=round(float(average_precision_score(y_arr, p_arr)), 6),
        brier_score=round(float(brier_score_loss(y_arr, p_arr)), 6),
        log_loss_val=round(float(log_loss(y_arr, p_clipped)), 6),
        ece_10bin=round(compute_ece(y_arr, p_arr, n_bins=10), 6),
        ks_statistic=round(compute_ks_statistic(y_arr, p_arr), 6),
    )


def build_reliability_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Build quantile-binned reliability curve coordinates for calibration plots."""
    y_arr, p_arr = _validate_binary_inputs(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(
        y_arr, p_arr, n_bins=n_bins, strategy="quantile"
    )
    return {
        "mean_predicted_prob": [round(float(v), 6) for v in prob_pred],
        "empirical_default_rate": [round(float(v), 6) for v in prob_true],
    }
