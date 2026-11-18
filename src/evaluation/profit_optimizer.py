"""Unit-economics expected profit optimization and sensitivity surface."""

from __future__ import annotations

import numpy as np

from src.core.config import EconomicsConfig
from src.core.contracts import (
    CalibrationBundle,
    ProfitCurveResult,
    ProfitPoint,
    SensitivityRow,
)


def _select_probability_array(bundle: CalibrationBundle, variant: str) -> np.ndarray:
    """Return the requested probability vector from a CalibrationBundle."""
    if variant == "raw":
        return bundle.oof_raw
    if variant == "platt":
        return bundle.oof_platt
    if variant == "isotonic":
        return bundle.oof_isotonic
    raise ValueError(f"Unsupported calibration variant: {variant}")


def compute_theoretical_cutoff(economics: EconomicsConfig) -> float:
    """Compute analytical break-even PD threshold M_good / (M_good + LGD * EAD)."""
    if economics.margin_good <= 0 or economics.exposure_at_default <= 0:
        raise ValueError("margin_good and exposure_at_default must be positive.")
    if not 0.0 < economics.lgd <= 1.0:
        raise ValueError("lgd must lie in (0, 1].")
    loss_default = economics.lgd * economics.exposure_at_default
    return float(economics.margin_good / (economics.margin_good + loss_default))


def evaluate_profit_at_cutoff(
    y_bundle: tuple[np.ndarray, np.ndarray],
    economics: EconomicsConfig,
    cutoff: float,
) -> ProfitPoint:
    """Evaluate approval rate, bad rate, and expected profit per applicant at cutoff."""
    y_true, y_prob = y_bundle
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("y_true and y_prob must be non-empty with equal length.")

    approved_mask = y_prob <= cutoff
    n_total = len(y_true)
    n_approved = int(np.sum(approved_mask))
    approval_rate = float(n_approved / n_total)
    bad_rate = float(np.mean(y_true[approved_mask])) if n_approved > 0 else 0.0

    loss_default = economics.lgd * economics.exposure_at_default
    pnl = np.where(
        approved_mask,
        np.where(y_true == 1, -loss_default, economics.margin_good),
        0.0,
    )
    expected_profit = float(np.mean(pnl))
    total_profit = float(np.sum(pnl))

    return ProfitPoint(
        threshold=round(float(cutoff), 6),
        approval_rate=round(approval_rate, 6),
        n_approved=n_approved,
        approved_bad_rate=round(bad_rate, 6),
        expected_profit_per_applicant=round(expected_profit, 4),
        total_portfolio_profit=round(total_profit, 2),
    )


def _build_threshold_grid(economics: EconomicsConfig, tau_theory: float) -> tuple[float, ...]:
    """Generate sorted dense PD cutoff grid including tau_theory, 0.50, and 1.00."""
    dense = np.round(np.arange(economics.threshold_step, 0.996, economics.threshold_step), 4)
    all_vals = set(dense.tolist()) | {round(tau_theory, 6), 0.50, 1.00}
    return tuple(sorted(all_vals))


def build_sensitivity_grid(
    bundle: CalibrationBundle,
    economics: EconomicsConfig,
    variant: str = "platt",
) -> tuple[SensitivityRow, ...]:
    """Sweep optimal profit cutoff across (margin_good, LGD) economic assumptions."""
    y_prob = _select_probability_array(bundle, variant)
    y_bundle = (bundle.y_true, y_prob)
    rows: list[SensitivityRow] = []

    for margin in economics.sensitivity_margins:
        for lgd in economics.sensitivity_lgds:
            econ_cell = EconomicsConfig(
                margin_good=margin,
                lgd=lgd,
                exposure_at_default=economics.exposure_at_default,
                threshold_step=economics.threshold_step,
            )
            tau_theory = compute_theoretical_cutoff(econ_cell)
            grid = _build_threshold_grid(econ_cell, tau_theory)
            points = [evaluate_profit_at_cutoff(y_bundle, econ_cell, t) for t in grid]
            best = max(points, key=lambda p: (p.expected_profit_per_applicant, p.approval_rate))
            approve_all = evaluate_profit_at_cutoff(y_bundle, econ_cell, 1.00)
            lift = best.expected_profit_per_applicant - approve_all.expected_profit_per_applicant
            rows.append(
                SensitivityRow(
                    margin_good=float(margin),
                    lgd=float(lgd),
                    tau_theory=round(tau_theory, 6),
                    tau_optimal=best.threshold,
                    approval_rate_optimal=best.approval_rate,
                    expected_profit_optimal=best.expected_profit_per_applicant,
                    profit_lift_vs_approve_all=round(float(lift), 4),
                )
            )
    return tuple(rows)


def optimize_profit_threshold(
    bundle: CalibrationBundle,
    economics: EconomicsConfig,
    variant: str = "platt",
) -> ProfitCurveResult:
    """Find profit-maximizing cutoff tau* and compare against policy baselines."""
    y_prob = _select_probability_array(bundle, variant)
    y_bundle = (bundle.y_true, y_prob)
    tau_theory = compute_theoretical_cutoff(economics)
    thresholds = _build_threshold_grid(economics, tau_theory)

    curve_points = tuple(
        evaluate_profit_at_cutoff(y_bundle, economics, t) for t in thresholds
    )
    optimal_point = max(
        curve_points,
        key=lambda p: (p.expected_profit_per_applicant, p.approval_rate),
    )
    theory_point = evaluate_profit_at_cutoff(y_bundle, economics, tau_theory)
    approve_all_point = evaluate_profit_at_cutoff(y_bundle, economics, 1.00)
    tau_050_point = evaluate_profit_at_cutoff(y_bundle, economics, 0.50)
    sensitivity = build_sensitivity_grid(bundle, economics, variant)

    return ProfitCurveResult(
        best_model_name=bundle.model_name,
        calibration_variant=variant,
        tau_theory=round(tau_theory, 6),
        optimal_point=optimal_point,
        theory_point=theory_point,
        approve_all_point=approve_all_point,
        tau_050_point=tau_050_point,
        curve_points=curve_points,
        sensitivity_grid=sensitivity,
    )
