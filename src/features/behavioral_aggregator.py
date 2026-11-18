"""Pure pre-cutoff behavioral credit-history aggregations and feature assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.contracts import FeatureDataset, FeatureSchema, TemporalPartition

SEVERITY_MAP: dict[str, int] = {
    "C": 0,
    "X": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
}

DEMOGRAPHIC_NUMERIC_COLS: tuple[str, ...] = (
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
    "is_unemployed_or_retired",
    "employment_years",
    "age_years",
    "is_occupation_missing",
    "income_per_family_member",
    "children_ratio",
)

DEMOGRAPHIC_CATEGORICAL_COLS: tuple[str, ...] = (
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "occupation_clean",
)

BEHAVIORAL_NUMERIC_COLS: tuple[str, ...] = (
    "obs_months_count",
    "obs_span_months",
    "obs_recency_months",
    "obs_dpd_any_count",
    "obs_dpd_any_ratio",
    "obs_dpd30_count",
    "obs_dpd30_ratio",
    "obs_dpd60_count",
    "obs_dpd60_ratio",
    "obs_paid_off_ratio",
    "obs_no_loan_ratio",
    "obs_max_severity",
    "obs_recent_6m_dpd_ratio",
    "obs_dpd_trend",
)


def aggregate_behavioral_history(
    credit_slice_df: pd.DataFrame,
    reference_month: int,
) -> pd.DataFrame:
    """Aggregate pure credit-history risk features relative to reference_month."""
    if credit_slice_df.empty:
        raise ValueError("Cannot aggregate behavioral features on an empty slice.")

    work = credit_slice_df.loc[:, ["ID", "MONTHS_BALANCE", "STATUS"]].copy()
    status_str = work["STATUS"].astype(str)
    work["is_dpd_any"] = status_str.isin({"0", "1", "2", "3", "4", "5"}).astype(float)
    work["is_dpd30"] = status_str.isin({"1", "2", "3", "4", "5"}).astype(float)
    work["is_dpd60"] = status_str.isin({"2", "3", "4", "5"}).astype(float)
    work["is_paid_off"] = (status_str == "C").astype(float)
    work["is_no_loan"] = (status_str == "X").astype(float)
    work["severity"] = status_str.map(SEVERITY_MAP).fillna(0).astype(float)

    agg = work.groupby("ID", as_index=False).agg(
        obs_months_count=("MONTHS_BALANCE", "size"),
        min_month=("MONTHS_BALANCE", "min"),
        max_month=("MONTHS_BALANCE", "max"),
        obs_dpd_any_count=("is_dpd_any", "sum"),
        obs_dpd_any_ratio=("is_dpd_any", "mean"),
        obs_dpd30_count=("is_dpd30", "sum"),
        obs_dpd30_ratio=("is_dpd30", "mean"),
        obs_dpd60_count=("is_dpd60", "sum"),
        obs_dpd60_ratio=("is_dpd60", "mean"),
        obs_paid_off_ratio=("is_paid_off", "mean"),
        obs_no_loan_ratio=("is_no_loan", "mean"),
        obs_max_severity=("severity", "max"),
    )
    agg["obs_span_months"] = agg["max_month"] - agg["min_month"] + 1
    agg["obs_recency_months"] = np.maximum(reference_month - agg["max_month"], 0)

    recent_slice = work[work["MONTHS_BALANCE"] >= (reference_month - 5)]
    recent_rates = (
        recent_slice.groupby("ID")["is_dpd_any"]
        .mean()
        .rename("obs_recent_6m_dpd_ratio")
    )
    agg = agg.merge(recent_rates, on="ID", how="left")
    agg["obs_recent_6m_dpd_ratio"] = agg["obs_recent_6m_dpd_ratio"].fillna(0.0)
    agg["obs_dpd_trend"] = agg["obs_recent_6m_dpd_ratio"] - agg["obs_dpd_any_ratio"]
    return agg.drop(columns=["min_month", "max_month"]).sort_values("ID")


def assemble_feature_dataset(
    cleaned_app: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    partition: TemporalPartition,
) -> FeatureDataset:
    """Align cleaned demographics, behavioral features, targets, and group keys."""
    if "demo_group_key" not in cleaned_app.columns:
        raise ValueError("cleaned_app must include 'demo_group_key' column.")

    cohort_frame = pd.DataFrame({"ID": partition.cohort_ids})
    merged = cohort_frame.merge(cleaned_app, on="ID", how="inner")
    merged = merged.merge(behavioral_df, on="ID", how="inner")
    merged = merged.sort_values("ID").reset_index(drop=True)

    if len(merged) != len(partition.cohort_ids):
        raise ValueError("Feature merge dropped or duplicated cohort client IDs.")

    client_ids = merged["ID"].to_numpy(dtype=np.int64)
    group_keys = merged["demo_group_key"].to_numpy()
    target = partition.target_series.reindex(client_ids).to_numpy(dtype=np.int64)

    schema = FeatureSchema(
        numeric_cols=DEMOGRAPHIC_NUMERIC_COLS + BEHAVIORAL_NUMERIC_COLS,
        categorical_cols=DEMOGRAPHIC_CATEGORICAL_COLS,
    )
    feature_cols = list(schema.numeric_cols) + list(schema.categorical_cols)
    return FeatureDataset(
        features_df=merged.loc[:, feature_cols].copy(),
        target=target,
        group_keys=group_keys,
        client_ids=client_ids,
        schema=schema,
    )


def build_demographic_only_dataset(
    cleaned_app: pd.DataFrame,
    partition: TemporalPartition,
) -> FeatureDataset:
    """Construct the demographics-only feature dataset for leakage ablation."""
    if "demo_group_key" not in cleaned_app.columns:
        raise ValueError("cleaned_app must include 'demo_group_key' column.")

    cohort_frame = pd.DataFrame({"ID": partition.cohort_ids})
    merged = cohort_frame.merge(cleaned_app, on="ID", how="inner")
    merged = merged.sort_values("ID").reset_index(drop=True)

    client_ids = merged["ID"].to_numpy(dtype=np.int64)
    group_keys = merged["demo_group_key"].to_numpy()
    target = partition.target_series.reindex(client_ids).to_numpy(dtype=np.int64)

    schema = FeatureSchema(
        numeric_cols=DEMOGRAPHIC_NUMERIC_COLS,
        categorical_cols=DEMOGRAPHIC_CATEGORICAL_COLS,
    )
    feature_cols = list(schema.numeric_cols) + list(schema.categorical_cols)
    return FeatureDataset(
        features_df=merged.loc[:, feature_cols].copy(),
        target=target,
        group_keys=group_keys,
        client_ids=client_ids,
        schema=schema,
    )
