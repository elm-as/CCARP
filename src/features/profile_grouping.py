"""Demographic profile signature hashing and cluster duplication diagnostics."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

DEFAULT_SIGNATURE_COLS: tuple[str, ...] = (
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
)


def _hash_signature_row(row_text: str) -> str:
    """Return a deterministic 16-character SHA-256 hex digest for a profile string."""
    return hashlib.sha256(row_text.encode("utf-8")).hexdigest()[:16]


def build_demographic_group_keys(
    app_df: pd.DataFrame,
    signature_cols: tuple[str, ...] = DEFAULT_SIGNATURE_COLS,
) -> pd.Series:
    """Build deterministic SHA-256 group keys over demographic attributes."""
    if app_df.empty:
        raise ValueError("Cannot build demographic group keys for an empty DataFrame.")
    missing = set(signature_cols) - set(app_df.columns)
    if missing:
        raise ValueError(f"Missing signature columns: {sorted(missing)}")

    subset = app_df.loc[:, list(signature_cols)].copy()
    for col in signature_cols:
        subset[col] = subset[col].fillna("__MISSING__").astype(str)

    joined = subset.agg("|".join, axis=1)
    hashed = joined.map(_hash_signature_row)
    return pd.Series(hashed.to_numpy(), index=app_df.index, name="demo_group_key")


def summarize_profile_duplication(
    group_keys: pd.Series,
    targets: pd.Series,
) -> dict[str, Any]:
    """Quantify demographic clone clusters and within-cluster default concentration."""
    if len(group_keys) != len(targets):
        raise ValueError("group_keys and targets must have equal length.")
    if group_keys.empty:
        raise ValueError("Cannot summarize empty group_keys.")

    frame = pd.DataFrame(
        {"group_key": group_keys.to_numpy(), "target": targets.to_numpy()}
    )
    stats = frame.groupby("group_key", as_index=False).agg(
        cluster_size=("target", "size"),
        n_defaults=("target", "sum"),
        default_rate=("target", "mean"),
    )
    multi = stats[stats["cluster_size"] > 1]
    pos_multi = multi[multi["n_defaults"] > 0]
    n_clients = int(len(frame))
    n_clustered = int(multi["cluster_size"].sum())

    return {
        "n_clients": n_clients,
        "n_unique_groups": int(len(stats)),
        "n_singleton_groups": int((stats["cluster_size"] == 1).sum()),
        "n_multi_id_groups": int(len(multi)),
        "n_clients_in_multi_id_groups": n_clustered,
        "pct_clients_in_multi_id_groups": round(100.0 * n_clustered / n_clients, 2),
        "max_cluster_size": int(stats["cluster_size"].max()),
        "n_positive_multi_id_groups": int(len(pos_multi)),
        "mean_within_positive_cluster_default_rate": (
            round(float(pos_multi["default_rate"].mean()), 4)
            if not pos_multi.empty
            else 0.0
        ),
    }
