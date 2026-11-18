"""Strict causal temporal window splitting for observation and performance periods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.config import TemporalWindowConfig
from src.core.contracts import JoinedTables, TemporalPartition
from src.features.profile_grouping import build_demographic_group_keys


def is_valid_temporal_partition(
    obs_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    t_obs: int,
) -> bool:
    """Return True if observation and performance slices obey strict causal ordering."""
    if obs_df.empty or perf_df.empty:
        return False
    obs_max = int(obs_df["MONTHS_BALANCE"].max())
    perf_min = int(perf_df["MONTHS_BALANCE"].min())
    return obs_max <= t_obs < perf_min


def _compute_window_bounds(
    obs_all: pd.DataFrame,
    perf_all: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-ID observation and performance month boundaries."""
    obs_bounds = obs_all.groupby("ID", as_index=False).agg(
        obs_months_count=("MONTHS_BALANCE", "size"),
        obs_min_month=("MONTHS_BALANCE", "min"),
        obs_max_month=("MONTHS_BALANCE", "max"),
    )
    perf_bounds = perf_all.groupby("ID", as_index=False).agg(
        perf_months_count=("MONTHS_BALANCE", "size"),
        perf_min_month=("MONTHS_BALANCE", "min"),
        perf_max_month=("MONTHS_BALANCE", "max"),
    )
    return obs_bounds.merge(perf_bounds, on="ID", how="inner")


def partition_temporal_windows(
    credit_df: pd.DataFrame,
    window_cfg: TemporalWindowConfig,
) -> TemporalPartition:
    """Partition credit history at T_obs and derive forward 60+ DPD default targets."""
    if window_cfg.t_obs >= 0:
        raise ValueError(f"t_obs must be strictly negative, got {window_cfg.t_obs}.")
    if credit_df.empty:
        raise ValueError("Cannot partition an empty credit_df.")

    obs_all = credit_df[credit_df["MONTHS_BALANCE"] <= window_cfg.t_obs].copy()
    perf_all = credit_df[credit_df["MONTHS_BALANCE"] > window_cfg.t_obs].copy()
    if not is_valid_temporal_partition(obs_all, perf_all, window_cfg.t_obs):
        raise ValueError(f"Temporal partition invalid at t_obs={window_cfg.t_obs}.")

    bounds = _compute_window_bounds(obs_all, perf_all)
    eligible = bounds[
        (bounds["obs_months_count"] >= window_cfg.min_obs_months)
        & (bounds["perf_months_count"] >= window_cfg.min_perf_months)
    ].sort_values("ID").reset_index(drop=True)

    if eligible.empty:
        raise ValueError("No accounts satisfy min_obs_months and min_perf_months.")

    cohort_ids = eligible["ID"].to_numpy(dtype=np.int64)
    obs_cohort = obs_all[obs_all["ID"].isin(cohort_ids)].copy()
    perf_cohort = perf_all[perf_all["ID"].isin(cohort_ids)].copy()

    default_set = set(window_cfg.default_statuses)
    perf_is_bad = perf_cohort["STATUS"].astype(str).isin(default_set)
    perf_bad_by_id = perf_is_bad.groupby(perf_cohort["ID"]).any().astype(np.int64)
    target_series = perf_bad_by_id.reindex(cohort_ids, fill_value=0)

    obs_is_bad = obs_cohort["STATUS"].astype(str).isin(default_set)
    obs_prior_bad_count = int(obs_is_bad.groupby(obs_cohort["ID"]).any().sum())

    return TemporalPartition(
        t_obs=window_cfg.t_obs,
        obs_df=obs_cohort.reset_index(drop=True),
        perf_df=perf_cohort.reset_index(drop=True),
        cohort_ids=cohort_ids,
        target_series=target_series,
        cohort_bounds=eligible,
        obs_prior_bad_count=obs_prior_bad_count,
    )


def compute_cutoff_retention_grid(
    joined_tables: JoinedTables,
    window_cfg: TemporalWindowConfig,
) -> tuple[dict[str, Any], ...]:
    """Evaluate cohort retention, default rate, and group count across T_obs cutoffs."""
    app_df = joined_tables.application_df
    group_map = pd.Series(
        build_demographic_group_keys(app_df).to_numpy(),
        index=app_df["ID"].to_numpy(),
    )
    total_joined = len(app_df)
    rows: list[dict[str, Any]] = []

    for cutoff in window_cfg.cutoff_grid:
        cfg_i = TemporalWindowConfig(
            t_obs=cutoff,
            min_obs_months=window_cfg.min_obs_months,
            min_perf_months=window_cfg.min_perf_months,
            default_statuses=window_cfg.default_statuses,
            cutoff_grid=window_cfg.cutoff_grid,
        )
        part = partition_temporal_windows(joined_tables.credit_df, cfg_i)
        n_cohort = int(len(part.cohort_ids))
        n_pos = int(part.target_series.sum())
        n_groups = int(group_map.reindex(part.cohort_ids).nunique())
        rows.append(
            {
                "t_obs": int(cutoff),
                "perf_horizon_months": int(abs(cutoff)),
                "cohort_n": n_cohort,
                "retention_pct": round(100.0 * n_cohort / total_joined, 2),
                "positives": n_pos,
                "default_rate_pct": round(100.0 * n_pos / n_cohort, 3),
                "obs_prior_bad_count": part.obs_prior_bad_count,
                "n_demo_groups": n_groups,
            }
        )
    return tuple(rows)
