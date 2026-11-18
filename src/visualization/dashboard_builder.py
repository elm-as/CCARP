"""Standalone interactive Plotly HTML5 underwriting-desk dashboard builder."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import PlotConfig
from src.core.contracts import VisualizationBundle


def _build_plotly_figure(bundle: VisualizationBundle, plot_config: PlotConfig) -> go.Figure:
    """Build 2x2 interactive Plotly figure for ablation, calibration, profit, and drivers."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Temporal & Group Leakage Ablation (OOF AUC)",
            "2. Probability Reliability Diagram (Deciles)",
            "3. Expected Profit ($) vs PD Cutoff tau",
            "4. Top Risk Drivers (Behavioral <= T_obs vs Demographic)",
        ),
        horizontal_spacing=0.11,
        vertical_spacing=0.16,
    )

    ab_labels = [row.regime_label for row in bundle.ablation_rows]
    fig.add_trace(
        go.Bar(
            x=ab_labels,
            y=[row.metrics.roc_auc for row in bundle.ablation_rows],
            name="ROC-AUC",
            marker_color=plot_config.navy,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ab_labels,
            y=[row.metrics.pr_auc for row in bundle.ablation_rows],
            name="PR-AUC",
            marker_color=plot_config.amber,
        ),
        row=1, col=1,
    )

    rel_specs = (
        ("raw", plot_config.crimson, "dash", "Raw OOF"),
        ("platt", plot_config.navy, "solid", "Platt (Sigmoid)"),
        ("isotonic", plot_config.emerald, "solid", "Isotonic"),
    )
    max_rel = 0.05
    for key, color, dash, label in rel_specs:
        xs = bundle.reliability_curves[key]["mean_predicted_prob"]
        ys = bundle.reliability_curves[key]["empirical_default_rate"]
        max_rel = max(max_rel, max(xs + ys))
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=label,
                line=dict(color=color, dash=dash, width=2),
            ),
            row=1, col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0, min(1.0, max_rel * 1.1)],
            y=[0.0, min(1.0, max_rel * 1.1)],
            mode="lines",
            name="Ideal Calibration",
            line=dict(color=plot_config.ink_muted, dash="dot", width=1.5),
        ),
        row=1, col=2,
    )

    pts = [p for p in bundle.profit_result.curve_points if p.threshold <= 0.60]
    fig.add_trace(
        go.Scatter(
            x=[p.threshold for p in pts],
            y=[p.expected_profit_per_applicant for p in pts],
            mode="lines",
            name="Expected Profit ($)",
            line=dict(color=plot_config.emerald, width=2.5),
            customdata=[[100.0 * p.approval_rate, 100.0 * p.approved_bad_rate] for p in pts],
            hovertemplate="tau=%{x:.3f}<br>Profit=$%{y:.2f}<br>Approval=%{customdata[0]:.1f}%<br>Bad Rate=%{customdata[1]:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    opt = bundle.profit_result.optimal_point
    fig.add_trace(
        go.Scatter(
            x=[opt.threshold],
            y=[opt.expected_profit_per_applicant],
            mode="markers",
            name=f"Optimal tau*={opt.threshold:.3f}",
            marker=dict(color=plot_config.amber, size=11, symbol="diamond"),
        ),
        row=2, col=1,
    )

    drivers = list(bundle.feature_drivers[:10])[::-1]
    fig.add_trace(
        go.Bar(
            x=[d["gain_share_pct"] for d in drivers],
            y=[d["feature"] for d in drivers],
            orientation="h",
            name="Gain Share (%)",
            marker_color=[
                plot_config.emerald if str(d["feature_family"]).startswith("Behavioral") else plot_config.navy
                for d in drivers
            ],
        ),
        row=2, col=2,
    )

    fig.update_layout(
        height=820,
        template="plotly_white",
        paper_bgcolor=plot_config.bg_surface,
        plot_bgcolor=plot_config.bg_surface,
        font=dict(family="IBM Plex Sans, Inter, system-ui, sans-serif", color=plot_config.ink_primary),
        barmode="group",
        margin=dict(l=55, r=30, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.0),
    )
    return fig


def _render_policy_rows_html(bundle: VisualizationBundle) -> str:
    """Render policy comparison HTML table rows with tabular numeric cells."""
    pr = bundle.profit_result
    policies = (
        ("Optimal Empirical (tau*)", pr.optimal_point),
        ("Analytical Break-Even (tau_theory)", pr.theory_point),
        ("Naive Cutoff (tau = 0.50)", pr.tau_050_point),
        ("Approve All (tau = 1.00)", pr.approve_all_point),
    )
    base_profit = pr.approve_all_point.expected_profit_per_applicant
    rows_html: list[str] = []
    for name, pt in policies:
        lift = pt.expected_profit_per_applicant - base_profit
        rows_html.append(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{pt.threshold:.4f}</td>"
            f"<td class='num'>{100.0 * pt.approval_rate:.2f}%</td>"
            f"<td class='num'>{100.0 * pt.approved_bad_rate:.3f}%</td>"
            f"<td class='num'>${pt.expected_profit_per_applicant:,.2f}</td>"
            f"<td class='num'>{lift:+.2f}</td></tr>"
        )
    return "\n".join(rows_html)


def _render_model_rows_html(bundle: VisualizationBundle) -> str:
    """Render OOF model calibration comparison HTML table rows."""
    rows_html: list[str] = []
    for row in bundle.model_metrics_rows:
        rows_html.append(
            f"<tr><td>{row['model_name']}</td><td>{row['calibration']}</td>"
            f"<td class='num'>{row['roc_auc']:.4f}</td>"
            f"<td class='num'>{row['pr_auc']:.4f}</td>"
            f"<td class='num'>{row['brier_score']:.5f}</td>"
            f"<td class='num'>{row['log_loss_val']:.5f}</td>"
            f"<td class='num'>{row['ece_10bin']:.5f}</td>"
            f"<td class='num'>{row['ks_statistic']:.4f}</td></tr>"
        )
    return "\n".join(rows_html)


def build_plotly_dashboard(
    bundle: VisualizationBundle,
    plot_config: PlotConfig,
) -> Path:
    """Write standalone institutional underwriting-desk HTML5 dashboard."""
    plot_config.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    fig = _build_plotly_figure(bundle, plot_config)
    plotly_fragment = fig.to_html(full_html=False, include_plotlyjs="cdn")

    cs = bundle.cohort_summary
    pr = bundle.profit_result
    opt = pr.optimal_point
    lift = opt.expected_profit_per_applicant - pr.approve_all_point.expected_profit_per_applicant
    best_row = next(
        r for r in bundle.model_metrics_rows
        if r["model_name"] == pr.best_model_name and r["calibration"] == pr.calibration_variant
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CCARP — Credit Card Approval Risk &amp; Profit Desk</title>
<style>
  body {{ margin: 0; padding: 24px; background: {plot_config.bg_page}; color: {plot_config.ink_primary}; font-family: 'IBM Plex Sans', 'Inter', system-ui, sans-serif; }}
  .container {{ max-width: 1380px; margin: 0 auto; }}
  .header {{ background: {plot_config.navy}; color: #FFFFFF; padding: 20px 24px; border-radius: 6px; margin-bottom: 18px; }}
  .header h1 {{ margin: 0 0 6px 0; font-size: 20px; font-weight: 700; }}
  .header p {{ margin: 0; font-size: 13px; color: #CBD5E1; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 18px; }}
  .kpi-card {{ background: {plot_config.bg_surface}; border: 1px solid {plot_config.border_color}; border-radius: 6px; padding: 14px 16px; }}
  .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: {plot_config.ink_muted}; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 21px; font-weight: 700; color: {plot_config.navy}; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}
  .panel {{ background: {plot_config.bg_surface}; border: 1px solid {plot_config.border_color}; border-radius: 6px; padding: 16px; margin-bottom: 18px; }}
  .panel h2 {{ margin: 0 0 12px 0; font-size: 15px; color: {plot_config.navy}; border-bottom: 1px solid {plot_config.border_color}; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid {plot_config.border_color}; text-align: left; }}
  th {{ background: {plot_config.bg_page}; color: {plot_config.ink_muted}; font-weight: 600; font-size: 12px; }}
  .num, td.num, th.num {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; text-align: right; font-family: 'IBM Plex Mono', monospace; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>CCARP — Credit Card Approval Risk &amp; Expected Profit Desk</h1>
    <p>Temporal Cutoff T_obs = {cs['t_obs']} | StratifiedGroupKFold(5) | Model: {pr.best_model_name} ({pr.calibration_variant})</p>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Joined / Cohort IDs</div><div class="kpi-value">{cs['joined_ids']:,} / {cs['cohort_n']:,}</div></div>
    <div class="kpi-card"><div class="kpi-label">Forward Default Rate</div><div class="kpi-value">{cs['default_rate_pct']:.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Calibrated ROC-AUC</div><div class="kpi-value">{best_row['roc_auc']:.4f}</div></div>
    <div class="kpi-card"><div class="kpi-label">Brier / KS Stat</div><div class="kpi-value">{best_row['brier_score']:.4f} / {best_row['ks_statistic']:.3f}</div></div>
    <div class="kpi-card"><div class="kpi-label">Optimal Cutoff tau*</div><div class="kpi-value">{opt.threshold:.3f} ({100.0 * opt.approval_rate:.1f}%)</div></div>
    <div class="kpi-card"><div class="kpi-label">Expected Profit / Lift</div><div class="kpi-value">${opt.expected_profit_per_applicant:.2f} (+${lift:.2f})</div></div>
  </div>
  <div class="panel">{plotly_fragment}</div>
  <div class="panel">
    <h2>Underwriting Policy Benchmark Comparison</h2>
    <table><thead><tr><th>Policy</th><th class="num">PD Cutoff</th><th class="num">Approval Rate</th><th class="num">Approved Bad Rate</th><th class="num">Expected Profit / App</th><th class="num">Lift vs Approve-All ($)</th></tr></thead>
    <tbody>{_render_policy_rows_html(bundle)}</tbody></table>
  </div>
  <div class="panel">
    <h2>Out-of-Fold Discrimination &amp; Calibration Audit (StratifiedGroupKFold)</h2>
    <table><thead><tr><th>Model</th><th>Calibration</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Brier</th><th class="num">Log-Loss</th><th class="num">ECE (10-bin)</th><th class="num">KS</th></tr></thead>
    <tbody>{_render_model_rows_html(bundle)}</tbody></table>
  </div>
</div>
</body>
</html>"""
    plot_config.dashboard_path.write_text(html, encoding="utf-8")
    return plot_config.dashboard_path
