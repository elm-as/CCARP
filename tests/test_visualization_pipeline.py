"""Smoke tests for 300 DPI static figure generation and Plotly HTML5 dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.config import EconomicsConfig, PlotConfig
from src.core.contracts import (
    AblationRow,
    CalibrationBundle,
    RiskMetrics,
    VisualizationBundle,
)
from src.evaluation.profit_optimizer import optimize_profit_threshold
from src.evaluation.risk_metrics import build_reliability_curve_data, compute_risk_metrics
from src.visualization.dashboard_builder import build_plotly_dashboard
from src.visualization.static_figures import render_static_figures


def test_visualization_smoke_outputs(tmp_path: Path) -> None:
    y_true = np.array([0] * 90 + [1] * 10)
    probs = np.concatenate([np.linspace(0.01, 0.08, 90), np.linspace(0.20, 0.70, 10)])
    cal = CalibrationBundle("lightgbm", y_true, probs, probs, probs)
    econ = EconomicsConfig()
    profit_res = optimize_profit_threshold(cal, econ, "platt")
    metrics = compute_risk_metrics(y_true, probs)

    bundle = VisualizationBundle(
        cohort_summary={
            "joined_ids": 100,
            "cohort_n": 100,
            "t_obs": -12,
            "default_rate_pct": 10.0,
        },
        cutoff_grid=(
            {"t_obs": -18, "cohort_n": 80, "default_rate_pct": 11.0},
            {"t_obs": -12, "cohort_n": 100, "default_rate_pct": 10.0},
        ),
        ablation_rows=(
            AblationRow("Leak-Free", "Obs <= -12", "StratifiedGroupKFold(5)", 100, 10, metrics),
            AblationRow("Naive", "All [-60, 0]", "StratifiedGroupKFold(5)", 100, 10, RiskMetrics(0.99, 0.95, 0.01, 0.05, 0.01, 0.95)),
        ),
        calibration_bundles=(cal,),
        model_metrics_rows=(
            {
                "model_name": "lightgbm",
                "calibration": "platt",
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
                "brier_score": metrics.brier_score,
                "log_loss_val": metrics.log_loss_val,
                "ece_10bin": metrics.ece_10bin,
                "ks_statistic": metrics.ks_statistic,
            },
        ),
        reliability_curves={
            "raw": build_reliability_curve_data(y_true, probs),
            "platt": build_reliability_curve_data(y_true, probs),
            "isotonic": build_reliability_curve_data(y_true, probs),
        },
        profit_result=profit_res,
        feature_drivers=(
            {"feature": "obs_dpd60_count", "feature_family": "Behavioral (<= T_obs)", "gain_share_pct": 42.0},
            {"feature": "age_years", "feature_family": "Demographic", "gain_share_pct": 15.0},
        ),
        hyperparameters=(),
    )
    plot_cfg = PlotConfig(
        images_dir=tmp_path / "images",
        dashboard_path=tmp_path / "artifacts" / "dashboard.html",
        results_json_path=tmp_path / "artifacts" / "results.json",
    )

    fig_names = render_static_figures(bundle, plot_cfg)
    assert len(fig_names) == 4
    for name in fig_names:
        fig_path = plot_cfg.images_dir / name
        assert fig_path.is_file() and fig_path.stat().st_size > 1000

    dash_path = build_plotly_dashboard(bundle, plot_cfg)
    html = dash_path.read_text(encoding="utf-8")
    assert "tabular-nums" in html
    assert "plotly" in html.lower()
    assert "bg-indigo" not in html and "#6366F1" not in html
