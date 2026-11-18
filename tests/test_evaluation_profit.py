"""Unit tests for risk metrics, break-even cutoff, and expected profit optimization."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import EconomicsConfig
from src.core.contracts import CalibrationBundle
from src.evaluation.profit_optimizer import (
    compute_theoretical_cutoff,
    optimize_profit_threshold,
)
from src.evaluation.risk_metrics import compute_risk_metrics


def test_compute_risk_metrics_separable_predictions() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    metrics = compute_risk_metrics(y_true, y_prob)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.ks_statistic == pytest.approx(1.0)
    assert metrics.brier_score < 0.02


def test_theoretical_cutoff_and_profit_optimum_beat_baselines() -> None:
    econ = EconomicsConfig(margin_good=180.0, lgd=0.75, exposure_at_default=2000.0)
    tau_theory = compute_theoretical_cutoff(econ)
    assert tau_theory == pytest.approx(180.0 / 1680.0)

    y_true = np.array([0] * 160 + [1] * 25 + [0] * 15)
    probs = np.concatenate(
        [
            np.full(160, 0.02),
            np.full(25, 0.35),
            np.full(15, 0.25),
        ]
    )
    bundle = CalibrationBundle(
        model_name="synthetic",
        y_true=y_true,
        oof_raw=probs,
        oof_platt=probs,
        oof_isotonic=probs,
    )
    res = optimize_profit_threshold(bundle, econ, variant="platt")

    approvals = [pt.approval_rate for pt in res.curve_points]
    assert all(b >= a for a, b in zip(approvals, approvals[1:]))
    assert res.optimal_point.expected_profit_per_applicant > res.approve_all_point.expected_profit_per_applicant
    assert res.optimal_point.expected_profit_per_applicant > res.tau_050_point.expected_profit_per_applicant
    assert len(res.sensitivity_grid) == len(econ.sensitivity_margins) * len(econ.sensitivity_lgds)
