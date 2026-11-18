"""End-to-end orchestration entrypoint for the CCARP credit risk & profit study."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.core.config import PipelineConfig
from src.core.contracts import (
    AblationRow,
    CalibrationBundle,
    FeatureDataset,
    JoinedTables,
    ModelRunSpec,
    ProfitCurveResult,
    TemporalPartition,
    VisualizationBundle,
)
from src.evaluation.ablation_auditor import (
    evaluate_ablation_regime,
    select_best_calibrated_variant,
    summarize_calibration_comparison,
)
from src.evaluation.profit_optimizer import optimize_profit_threshold
from src.evaluation.risk_metrics import build_reliability_curve_data
from src.features.behavioral_aggregator import (
    aggregate_behavioral_history,
    assemble_feature_dataset,
    build_demographic_only_dataset,
)
from src.features.demographic_cleaner import clean_demographic_records
from src.features.profile_grouping import (
    build_demographic_group_keys,
    summarize_profile_duplication,
)
from src.features.temporal_splitter import (
    compute_cutoff_retention_grid,
    partition_temporal_windows,
)
from src.ingestion.kaggle_loader import load_raw_tables
from src.ingestion.relational_join import join_application_credit
from src.modeling.credit_models import extract_feature_importances, train_oof_model
from src.modeling.cv_splitter import (
    create_group_splits,
    create_stratified_group_splits,
    create_stratified_splits,
)
from src.modeling.probability_calibrator import fit_and_apply_oof_calibrators
from src.visualization.dashboard_builder import build_plotly_dashboard
from src.visualization.static_figures import render_static_figures


def _hyperparameter_records() -> tuple[dict[str, Any], ...]:
    """Return structured model hyperparameter documentation for results and README."""
    return (
        {
            "model": "logistic_regression",
            "family": "Linear / GLM",
            "hyperparameters": "C=0.5, max_iter=1000, StandardScaler + Median Imputer",
            "calibration": "3-fold inner StratifiedGroupKFold Platt & Isotonic",
            "role": "Interpretable linear scorecard baseline",
        },
        {
            "model": "lightgbm",
            "family": "Histogram GBDT",
            "hyperparameters": "n_estimators=300, lr=0.04, num_leaves=15, max_depth=4, min_child_samples=50, reg_lambda=5.0",
            "calibration": "3-fold inner StratifiedGroupKFold Platt & Isotonic",
            "role": "Primary non-linear behavioral risk scorer",
        },
        {
            "model": "xgboost",
            "family": "Exact/Hist GBDT",
            "hyperparameters": "n_estimators=300, lr=0.04, max_depth=4, min_child_weight=20.0, reg_lambda=5.0, hist",
            "calibration": "3-fold inner StratifiedGroupKFold Platt & Isotonic",
            "role": "Cross-library gradient boosting benchmark",
        },
        {
            "model": "catboost",
            "family": "Symmetric Oblivious Trees",
            "hyperparameters": "iterations=400, lr=0.04, depth=4, l2_leaf_reg=8.0, allow_writing_files=False",
            "calibration": "3-fold inner StratifiedGroupKFold Platt & Isotonic",
            "role": "Regularized symmetric-tree benchmark",
        },
    )


def _run_ablation_suite(
    joined: JoinedTables,
    partition: TemporalPartition,
    config: PipelineConfig,
) -> tuple[FeatureDataset, tuple[AblationRow, ...]]:
    """Execute the 5-regime temporal and demographic-group leakage ablation study."""
    app_with_keys = joined.application_df.copy()
    app_with_keys["demo_group_key"] = build_demographic_group_keys(app_with_keys)
    cleaned_app = clean_demographic_records(app_with_keys)

    demo_ds = build_demographic_only_dataset(cleaned_app, partition)
    obs_beh = aggregate_behavioral_history(partition.obs_df, partition.t_obs)
    leak_free_ds = assemble_feature_dataset(cleaned_app, obs_beh, partition)

    naive_slice = joined.credit_df[joined.credit_df["ID"].isin(partition.cohort_ids)].copy()
    naive_beh = aggregate_behavioral_history(naive_slice, reference_month=0)
    naive_ds = assemble_feature_dataset(cleaned_app, naive_beh, partition)

    skf_demo = create_stratified_splits(demo_ds, config.modeling)
    sgkf_demo = create_stratified_group_splits(demo_ds, config.modeling)
    skf_full = create_stratified_splits(leak_free_ds, config.modeling)
    sgkf_full = create_stratified_group_splits(leak_free_ds, config.modeling)
    gkf_full = create_group_splits(leak_free_ds, config.modeling)

    demo_spec = ModelRunSpec("lightgbm", config.modeling, demo_ds.schema)
    full_spec = ModelRunSpec("lightgbm", config.modeling, leak_free_ds.schema)

    regimes = (
        ("Demo Only + StratifiedKFold", "Demographics Only", skf_demo.scheme_name, demo_ds, skf_demo, demo_spec),
        ("Demo Only + StratifiedGroupKFold", "Demographics Only", sgkf_demo.scheme_name, demo_ds, sgkf_demo, demo_spec),
        ("Leak-Free + StratifiedKFold", f"Demo + Obs <= {partition.t_obs}", skf_full.scheme_name, leak_free_ds, skf_full, full_spec),
        ("Leak-Free + StratifiedGroupKFold", f"Demo + Obs <= {partition.t_obs}", sgkf_full.scheme_name, leak_free_ds, sgkf_full, full_spec),
        ("Leak-Free + GroupKFold", f"Demo + Obs <= {partition.t_obs}", gkf_full.scheme_name, leak_free_ds, gkf_full, full_spec),
        ("Naive Full-History + StratifiedGroupKFold", "Demo + All Months [-60, 0]", sgkf_full.scheme_name, naive_ds, sgkf_full, full_spec),
    )

    rows: list[AblationRow] = []
    for label, scope, cv_name, ds_item, split_item, spec_item in regimes:
        preds = train_oof_model(ds_item, split_item, spec_item)
        rows.append(evaluate_ablation_regime((label, scope, cv_name), ds_item.target, preds))
    return leak_free_ds, tuple(rows)


def _train_calibrated_models(
    dataset: FeatureDataset,
    config: PipelineConfig,
) -> tuple[tuple[CalibrationBundle, ...], tuple[dict[str, Any], ...]]:
    """Train all 4 model families with fold-internal Platt and Isotonic calibration."""
    splits = create_stratified_group_splits(dataset, config.modeling)
    bundles: list[CalibrationBundle] = []

    for model_name in config.modeling.model_names:
        spec = ModelRunSpec(model_name, config.modeling, dataset.schema)
        bundles.append(fit_and_apply_oof_calibrators(dataset, splits, spec))

    lgbm_spec = ModelRunSpec("lightgbm", config.modeling, dataset.schema)
    drivers = extract_feature_importances(dataset, splits, lgbm_spec)
    return tuple(bundles), drivers


def _serialize_results(
    bundle: VisualizationBundle,
    figure_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Persist structured JSON summary of metrics, ablation, profit, and figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pr: ProfitCurveResult = bundle.profit_result
    payload = {
        "cohort_summary": bundle.cohort_summary,
        "cutoff_grid": list(bundle.cutoff_grid),
        "ablation_rows": [
            {
                "regime_label": r.regime_label,
                "feature_scope": r.feature_scope,
                "cv_scheme": r.cv_scheme,
                "n_samples": r.n_samples,
                "n_positives": r.n_positives,
                **asdict(r.metrics),
            }
            for r in bundle.ablation_rows
        ],
        "model_metrics": list(bundle.model_metrics_rows),
        "reliability_curves": bundle.reliability_curves,
        "profit_optimization": {
            "best_model_name": pr.best_model_name,
            "calibration_variant": pr.calibration_variant,
            "tau_theory": pr.tau_theory,
            "optimal_point": asdict(pr.optimal_point),
            "theory_point": asdict(pr.theory_point),
            "approve_all_point": asdict(pr.approve_all_point),
            "tau_050_point": asdict(pr.tau_050_point),
            "sensitivity_grid": [asdict(s) for s in pr.sensitivity_grid],
        },
        "feature_drivers": list(bundle.feature_drivers),
        "hyperparameters": list(bundle.hyperparameters),
        "figure_filenames": list(figure_names),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(root_dir: Path) -> Path:
    """Execute the full CCARP credit-risk and expected-profit pipeline."""
    config = PipelineConfig.create_default(root_dir)
    app_raw, credit_raw = load_raw_tables(config.dataset)
    joined = join_application_credit(app_raw, credit_raw)
    partition = partition_temporal_windows(joined.credit_df, config.temporal)
    cutoff_grid = compute_cutoff_retention_grid(joined, config.temporal)

    leak_free_ds, ablation_rows = _run_ablation_suite(joined, partition, config)
    dup_stats = summarize_profile_duplication(
        pd_series_from_array(leak_free_ds.group_keys),
        pd_series_from_array(leak_free_ds.target),
    )

    cal_bundles, drivers = _train_calibrated_models(leak_free_ds, config)
    model_rows = summarize_calibration_comparison(cal_bundles)
    best_bundle, best_variant = select_best_calibrated_variant(cal_bundles)
    profit_res = optimize_profit_threshold(best_bundle, config.economics, best_variant)

    reliability = {
        "raw": build_reliability_curve_data(best_bundle.y_true, best_bundle.oof_raw),
        "platt": build_reliability_curve_data(best_bundle.y_true, best_bundle.oof_platt),
        "isotonic": build_reliability_curve_data(best_bundle.y_true, best_bundle.oof_isotonic),
    }
    cohort_summary = {
        **asdict(joined.audit),
        "t_obs": partition.t_obs,
        "min_obs_months": config.temporal.min_obs_months,
        "min_perf_months": config.temporal.min_perf_months,
        "cohort_n": int(len(partition.cohort_ids)),
        "positives": int(partition.target_series.sum()),
        "default_rate_pct": round(100.0 * float(partition.target_series.mean()), 3),
        "obs_prior_bad_count": partition.obs_prior_bad_count,
        "duplication_audit": dup_stats,
    }

    vis_bundle = VisualizationBundle(
        cohort_summary=cohort_summary,
        cutoff_grid=cutoff_grid,
        ablation_rows=ablation_rows,
        calibration_bundles=cal_bundles,
        model_metrics_rows=model_rows,
        reliability_curves=reliability,
        profit_result=profit_res,
        feature_drivers=drivers,
        hyperparameters=_hyperparameter_records(),
    )
    figure_names = render_static_figures(vis_bundle, config.plots)
    build_plotly_dashboard(vis_bundle, config.plots)
    _serialize_results(vis_bundle, figure_names, config.plots.results_json_path)
    return config.plots.results_json_path


def pd_series_from_array(values: Any) -> Any:
    """Wrap a NumPy array into a pandas Series for summary helpers."""
    import pandas as pd

    return pd.Series(values)


if __name__ == "__main__":
    results_path = run_pipeline(Path(__file__).resolve().parent)
    print(f"CCARP pipeline completed successfully. Results written to: {results_path}")
