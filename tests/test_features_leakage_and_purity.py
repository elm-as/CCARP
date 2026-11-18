"""Unit tests for sentinel cleaning, temporal separation, and future-leakage invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import TemporalWindowConfig
from src.features.behavioral_aggregator import aggregate_behavioral_history
from src.features.demographic_cleaner import clean_demographic_records
from src.features.temporal_splitter import (
    is_valid_temporal_partition,
    partition_temporal_windows,
)


def _synthetic_app() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "CODE_GENDER": ["F", "M", "F"],
            "FLAG_OWN_CAR": ["N", "Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N"],
            "CNT_CHILDREN": [0, 1, 0],
            "AMT_INCOME_TOTAL": [100000.0, 200000.0, 150000.0],
            "NAME_INCOME_TYPE": ["Pensioner", "Working", "Working"],
            "NAME_EDUCATION_TYPE": ["Secondary / secondary special"] * 3,
            "NAME_FAMILY_STATUS": ["Married"] * 3,
            "NAME_HOUSING_TYPE": ["House / apartment"] * 3,
            "DAYS_BIRTH": [-23000, -14000, -11000],
            "DAYS_EMPLOYED": [365243, -2000, -900],
            "FLAG_MOBIL": [1, 1, 1],
            "FLAG_WORK_PHONE": [0, 1, 0],
            "FLAG_PHONE": [1, 0, 1],
            "FLAG_EMAIL": [0, 0, 1],
            "OCCUPATION_TYPE": [np.nan, "Managers", np.nan],
            "CNT_FAM_MEMBERS": [1.0, 3.0, 2.0],
        }
    )


def _synthetic_credit() -> pd.DataFrame:
    rows = []
    for m in range(-20, -11):
        rows.append((1, m, "0"))
        rows.append((2, m, "C"))
    for m in range(-11, 0):
        rows.append((1, m, "3" if m == -2 else "0"))
        rows.append((2, m, "C"))
    rows.append((3, -15, "0"))
    rows.append((3, -1, "0"))
    return pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])


def test_clean_demographic_records_handles_sentinel_and_preserves_input() -> None:
    app = _synthetic_app()
    orig = app.copy(deep=True)
    cleaned = clean_demographic_records(app)

    pd.testing.assert_frame_equal(app, orig)
    assert "FLAG_MOBIL" not in cleaned.columns
    assert cleaned.loc[cleaned["ID"] == 1, "is_unemployed_or_retired"].iloc[0] == 1
    assert np.isnan(cleaned.loc[cleaned["ID"] == 1, "employment_years"].iloc[0])
    assert cleaned.loc[cleaned["ID"] == 1, "occupation_clean"].iloc[0] == "Pensioner_or_Retired"
    assert cleaned.loc[cleaned["ID"] == 3, "occupation_clean"].iloc[0] == "Missing_Occupation"
    assert cleaned["occupation_clean"].isna().sum() == 0


def test_partition_temporal_windows_enforces_causal_separation() -> None:
    cfg = TemporalWindowConfig(t_obs=-12, min_obs_months=3, min_perf_months=6)
    part = partition_temporal_windows(_synthetic_credit(), cfg)

    assert is_valid_temporal_partition(part.obs_df, part.perf_df, cfg.t_obs)
    assert int(part.obs_df["MONTHS_BALANCE"].max()) <= cfg.t_obs
    assert int(part.perf_df["MONTHS_BALANCE"].min()) > cfg.t_obs
    assert list(part.cohort_ids) == [1, 2]
    assert int(part.target_series.loc[1]) == 1
    assert int(part.target_series.loc[2]) == 0


def test_behavioral_features_invariant_to_future_perturbation() -> None:
    cfg = TemporalWindowConfig(t_obs=-12, min_obs_months=3, min_perf_months=6)
    credit_a = _synthetic_credit()
    credit_b = credit_a.copy()
    credit_b.loc[credit_b["MONTHS_BALANCE"] > cfg.t_obs, "STATUS"] = "5"

    part_a = partition_temporal_windows(credit_a, cfg)
    part_b = partition_temporal_windows(credit_b, cfg)

    feat_a = aggregate_behavioral_history(part_a.obs_df, cfg.t_obs)
    feat_b = aggregate_behavioral_history(part_b.obs_df, cfg.t_obs)
    pd.testing.assert_frame_equal(feat_a, feat_b)

    naive_a = aggregate_behavioral_history(credit_a[credit_a["ID"].isin(part_a.cohort_ids)], 0)
    naive_b = aggregate_behavioral_history(credit_b[credit_b["ID"].isin(part_b.cohort_ids)], 0)
    assert not naive_a.equals(naive_b)
