"""300 DPI Matplotlib figures for expected profit optimization and risk drivers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.config import PlotConfig
from src.core.contracts import VisualizationBundle


def _style_axis(ax: plt.Axes, plot_config: PlotConfig) -> None:
    """Apply clean institutional light styling to a Matplotlib axis."""
    ax.set_facecolor(plot_config.bg_surface)
    ax.grid(True, linestyle="--", linewidth=0.6, color=plot_config.border_color, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(plot_config.border_color)
    ax.spines["bottom"].set_color(plot_config.border_color)
    ax.tick_params(colors=plot_config.ink_primary, labelsize=9)


def _plot_profit_curve_panel(
    ax: plt.Axes,
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> None:
    """Plot expected profit per applicant and approval rate across PD thresholds."""
    _style_axis(ax, plot_config)
    profit_res = bundle.profit_result
    pts = [pt for pt in profit_res.curve_points if pt.threshold <= 0.60]
    taus = [pt.threshold for pt in pts]
    profits = [pt.expected_profit_per_applicant for pt in pts]
    approvals = [100.0 * pt.approval_rate for pt in pts]

    ax.plot(
        taus, profits, color=plot_config.navy, linewidth=2.2,
        label="Expected Profit / Applicant ($)",
    )
    opt = profit_res.optimal_point
    ax.axvline(
        opt.threshold, color=plot_config.emerald, linestyle="--", linewidth=1.6,
        label=f"Optimal tau* = {opt.threshold:.3f} (${opt.expected_profit_per_applicant:.2f})",
    )
    ax.axvline(
        profit_res.tau_theory, color=plot_config.amber, linestyle=":", linewidth=1.6,
        label=f"Theory tau = {profit_res.tau_theory:.4f}",
    )
    all_pnl = profit_res.approve_all_point.expected_profit_per_applicant
    ax.axhline(
        all_pnl, color=plot_config.crimson, linestyle="-.", linewidth=1.3,
        label=f"Approve All = ${all_pnl:.2f}",
    )
    ax.scatter([opt.threshold], [opt.expected_profit_per_applicant], color=plot_config.emerald, s=45, zorder=5)
    ax.set_xlabel("Approval PD Cutoff (tau)", fontsize=10, color=plot_config.ink_primary)
    ax.set_ylabel("Expected Profit / Applicant (USD)", fontsize=10, color=plot_config.navy)
    ax.set_title("Expected Profit vs Approval Cutoff (tau)", fontsize=11, fontweight="bold", color=plot_config.navy)

    ax_right = ax.twinx()
    ax_right.plot(
        taus, approvals, color=plot_config.steel_blue, linewidth=1.4,
        alpha=0.65, linestyle="-", label="Approval Rate (%)",
    )
    ax_right.set_ylabel("Approval Rate (%)", fontsize=10, color=plot_config.steel_blue)
    ax_right.spines["top"].set_visible(False)
    ax_right.tick_params(axis="y", colors=plot_config.steel_blue, labelsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=plot_config.bg_surface)


def _plot_sensitivity_heatmap(
    ax: plt.Axes,
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> None:
    """Plot 2D (margin_good x LGD) optimal expected profit sensitivity grid."""
    rows = [
        {"margin": r.margin_good, "lgd": r.lgd, "profit": r.expected_profit_optimal, "tau": r.tau_optimal}
        for r in bundle.profit_result.sensitivity_grid
    ]
    df = pd.DataFrame(rows)
    piv_profit = df.pivot(index="margin", columns="lgd", values="profit")
    piv_tau = df.pivot(index="margin", columns="lgd", values="tau")

    im = ax.imshow(piv_profit.to_numpy(), cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(piv_profit.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv_profit.columns], fontsize=9)
    ax.set_yticks(range(len(piv_profit.index)))
    ax.set_yticklabels([f"${int(m)}" for m in piv_profit.index], fontsize=9)
    ax.set_xlabel("Loss Given Default (LGD)", fontsize=10, color=plot_config.ink_primary)
    ax.set_ylabel("Good Account Margin (USD)", fontsize=10, color=plot_config.ink_primary)
    ax.set_title("Sensitivity: Optimal Profit ($) & tau*", fontsize=11, fontweight="bold", color=plot_config.navy)

    vals = piv_profit.to_numpy()
    taus = piv_tau.to_numpy()
    median_val = float(np.median(vals))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            text_color = "#FFFFFF" if vals[i, j] > median_val else plot_config.ink_primary
            ax.text(
                j, i, f"${vals[i, j]:.1f}\nt={taus[i, j]:.3f}",
                ha="center", va="center", fontsize=8, fontweight="bold", color=text_color,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_profit_curve_figure(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> Path:
    """Render and save the 300 DPI expected profit curve and sensitivity figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), facecolor=plot_config.bg_page)
    _plot_profit_curve_panel(axes[0], bundle, plot_config)
    _plot_sensitivity_heatmap(axes[1], bundle, plot_config)
    fig.suptitle(
        f"CCARP Unit-Economics Optimization ({bundle.profit_result.best_model_name} / {bundle.profit_result.calibration_variant})",
        fontsize=12.5, fontweight="bold", color=plot_config.navy, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = plot_config.images_dir / "ccarp_expected_profit_curve.png"
    fig.savefig(out_path, dpi=plot_config.dpi)
    plt.close(fig)
    return out_path


def render_feature_drivers_figure(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> Path:
    """Render and save top risk drivers and approved-vs-rejected bad rate figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), facecolor=plot_config.bg_page)
    _style_axis(axes[0], plot_config)
    _style_axis(axes[1], plot_config)

    drivers = list(bundle.feature_drivers[:12])[::-1]
    labels = [d["feature"] for d in drivers]
    shares = [d["gain_share_pct"] for d in drivers]
    colors = [
        plot_config.emerald if str(d["feature_family"]).startswith("Behavioral") else plot_config.navy
        for d in drivers
    ]
    axes[0].barh(labels, shares, color=colors, height=0.68)
    axes[0].set_xlabel("LightGBM Mean Gain Share (%)", fontsize=10, color=plot_config.ink_primary)
    axes[0].set_title("Top Risk Drivers (Behavioral <= T_obs vs Demographic)", fontsize=11, fontweight="bold", color=plot_config.navy)

    opt = bundle.profit_result.optimal_point
    all_pt = bundle.profit_result.approve_all_point
    n_total = max(all_pt.n_approved, 1)
    n_rej = max(n_total - opt.n_approved, 1)
    rej_bad = max(all_pt.approved_bad_rate * n_total - opt.approved_bad_rate * opt.n_approved, 0.0) / n_rej

    seg_names = [f"Approved\n(PD <= {opt.threshold:.3f})", "Overall\nPortfolio", f"Rejected\n(PD > {opt.threshold:.3f})"]
    seg_rates = [100.0 * opt.approved_bad_rate, 100.0 * all_pt.approved_bad_rate, 100.0 * rej_bad]
    seg_colors = [plot_config.emerald, plot_config.navy, plot_config.crimson]
    bars = axes[1].bar(seg_names, seg_rates, color=seg_colors, width=0.52)
    for bar, rate in zip(bars, seg_rates):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0, rate + 0.15,
            f"{rate:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=plot_config.ink_primary,
        )
    axes[1].set_ylabel("Observed Default Rate (%)", fontsize=10, color=plot_config.ink_primary)
    axes[1].set_title("Underwriting Risk Separation at Optimal Cutoff", fontsize=11, fontweight="bold", color=plot_config.navy)

    plt.tight_layout()
    out_path = plot_config.images_dir / "ccarp_feature_risk_drivers.png"
    fig.savefig(out_path, dpi=plot_config.dpi)
    plt.close(fig)
    return out_path
