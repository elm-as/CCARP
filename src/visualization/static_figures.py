"""300 DPI Matplotlib figures for temporal leakage ablation and model calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.core.config import PlotConfig
from src.core.contracts import VisualizationBundle
from src.visualization.profit_driver_figures import (
    render_feature_drivers_figure,
    render_profit_curve_figure,
)


def _style_axis(ax: plt.Axes, plot_config: PlotConfig) -> None:
    """Apply clean institutional light styling to a Matplotlib axis."""
    ax.set_facecolor(plot_config.bg_surface)
    ax.grid(True, linestyle="--", linewidth=0.6, color=plot_config.border_color, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(plot_config.border_color)
    ax.spines["bottom"].set_color(plot_config.border_color)
    ax.tick_params(colors=plot_config.ink_primary, labelsize=9)


def _render_temporal_leakage_figure(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> Path:
    """Render cutoff retention grid and 5-regime temporal/group leakage ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), facecolor=plot_config.bg_page)
    _style_axis(axes[0], plot_config)
    _style_axis(axes[1], plot_config)

    grid = bundle.cutoff_grid
    cutoffs = [f"T={row['t_obs']}" for row in grid]
    cohorts = [row["cohort_n"] for row in grid]
    rates = [row["default_rate_pct"] for row in grid]
    bar_colors = [
        plot_config.emerald if row["t_obs"] == bundle.cohort_summary["t_obs"] else plot_config.navy
        for row in grid
    ]
    axes[0].bar(cutoffs, cohorts, color=bar_colors, width=0.52, alpha=0.88)
    axes[0].set_ylabel("Eligible Cohort Accounts (N)", fontsize=10, color=plot_config.navy)
    axes[0].set_title("Temporal Cutoff Grid (Observation vs Performance)", fontsize=11, fontweight="bold", color=plot_config.navy)

    ax_rate = axes[0].twinx()
    ax_rate.plot(cutoffs, rates, color=plot_config.crimson, marker="o", linewidth=2.0)
    ax_rate.set_ylabel("Forward Default Rate (%)", fontsize=10, color=plot_config.crimson)
    ax_rate.spines["top"].set_visible(False)

    ablation = list(bundle.ablation_rows)
    labels = [row.regime_label for row in ablation]
    rocs = [row.metrics.roc_auc for row in ablation]
    prs = [row.metrics.pr_auc for row in ablation]
    y_pos = np.arange(len(labels))
    axes[1].barh(y_pos + 0.18, rocs, height=0.34, color=plot_config.navy, label="ROC-AUC")
    axes[1].barh(y_pos - 0.18, prs, height=0.34, color=plot_config.amber, label="PR-AUC")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8.5)
    axes[1].set_xlim(0.0, 1.08)
    axes[1].set_xlabel("Out-of-Fold Metric Score", fontsize=10, color=plot_config.ink_primary)
    axes[1].set_title("Leakage Ablation: Temporal & Demographic-Group Isolation", fontsize=11, fontweight="bold", color=plot_config.navy)
    axes[1].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_path = plot_config.images_dir / "ccarp_temporal_window_leakage.png"
    fig.savefig(out_path, dpi=plot_config.dpi)
    plt.close(fig)
    return out_path


def _render_roc_pr_calibration_figure(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> Path:
    """Render 3-panel ROC, Precision-Recall, and Reliability calibration figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), facecolor=plot_config.bg_page)
    for ax in axes:
        _style_axis(ax, plot_config)

    palette = [plot_config.navy, plot_config.emerald, plot_config.steel_blue, plot_config.amber]
    for idx, cal_bundle in enumerate(bundle.calibration_bundles):
        color = palette[idx % len(palette)]
        fpr, tpr, _ = roc_curve(cal_bundle.y_true, cal_bundle.oof_platt)
        ks_val = float(np.max(np.abs(tpr - fpr)))
        axes[0].plot(fpr, tpr, color=color, linewidth=1.8, label=f"{cal_bundle.model_name} (KS={ks_val:.3f})")

        prec, rec, _ = precision_recall_curve(cal_bundle.y_true, cal_bundle.oof_platt)
        axes[1].plot(rec, prec, color=color, linewidth=1.8, label=cal_bundle.model_name)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color=plot_config.ink_muted, linewidth=1.0)
    axes[0].set_title("OOF ROC Curves (StratifiedGroupKFold)", fontsize=10.5, fontweight="bold", color=plot_config.navy)
    axes[0].set_xlabel("False Positive Rate", fontsize=9.5)
    axes[0].set_ylabel("True Positive Rate", fontsize=9.5)
    axes[0].legend(loc="lower right", fontsize=8)

    base_rate = float(np.mean(bundle.calibration_bundles[0].y_true))
    axes[1].axhline(base_rate, linestyle="--", color=plot_config.crimson, linewidth=1.0, label=f"Baseline ({base_rate:.3%})")
    axes[1].set_title("OOF Precision-Recall Curves", fontsize=10.5, fontweight="bold", color=plot_config.navy)
    axes[1].set_xlabel("Recall", fontsize=9.5)
    axes[1].set_ylabel("Precision", fontsize=9.5)
    axes[1].legend(loc="upper right", fontsize=8)

    rel_styles = (
        ("raw", plot_config.crimson, "--", "Raw"),
        ("platt", plot_config.navy, "-", "Platt (Sigmoid)"),
        ("isotonic", plot_config.emerald, "-", "Isotonic"),
    )
    max_coord = 0.05
    for key, color, style, label in rel_styles:
        curve = bundle.reliability_curves[key]
        xs = curve["mean_predicted_prob"]
        ys = curve["empirical_default_rate"]
        max_coord = max(max_coord, max(xs + ys))
        axes[2].plot(xs, ys, marker="o", linestyle=style, color=color, linewidth=1.7, markersize=4.5, label=label)

    bound = min(1.0, max_coord * 1.12)
    axes[2].plot([0, bound], [0, bound], linestyle=":", color=plot_config.ink_muted, linewidth=1.2, label="Perfect Calibration")
    axes[2].set_xlim(0.0, bound)
    axes[2].set_ylim(0.0, bound)
    axes[2].set_title(f"Reliability Diagram ({bundle.profit_result.best_model_name})", fontsize=10.5, fontweight="bold", color=plot_config.navy)
    axes[2].set_xlabel("Mean Predicted PD", fontsize=9.5)
    axes[2].set_ylabel("Empirical Default Rate", fontsize=9.5)
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = plot_config.images_dir / "ccarp_roc_pr_calibration.png"
    fig.savefig(out_path, dpi=plot_config.dpi)
    plt.close(fig)
    return out_path


def render_static_figures(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> tuple[str, ...]:
    """Generate all four 300 DPI CCARP PNG figures in plot_config.images_dir."""
    plot_config.images_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        _render_temporal_leakage_figure(bundle, plot_config),
        _render_roc_pr_calibration_figure(bundle, plot_config),
        render_profit_curve_figure(bundle, plot_config),
        render_feature_drivers_figure(bundle, plot_config),
    )
    return tuple(path.name for path in paths)
