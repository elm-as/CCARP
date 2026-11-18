"""Pure demographic cleaning with sentinel handling and missingness indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL: int = 365243
DAYS_PER_YEAR: float = 365.25


def clean_demographic_records(app_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of application records without mutating the input."""
    if app_df.empty:
        raise ValueError("Cannot clean an empty application DataFrame.")

    cleaned = app_df.copy()
    is_sentinel = cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    cleaned["is_unemployed_or_retired"] = is_sentinel.astype(np.int8)

    days_emp_valid = cleaned["DAYS_EMPLOYED"].where(~is_sentinel, np.nan)
    cleaned["employment_years"] = (-days_emp_valid) / DAYS_PER_YEAR
    cleaned["age_years"] = (-cleaned["DAYS_BIRTH"]) / DAYS_PER_YEAR

    occ_missing = cleaned["OCCUPATION_TYPE"].isna()
    cleaned["is_occupation_missing"] = occ_missing.astype(np.int8)
    cleaned["occupation_clean"] = np.where(
        ~occ_missing,
        cleaned["OCCUPATION_TYPE"].astype(str),
        np.where(is_sentinel, "Pensioner_or_Retired", "Missing_Occupation"),
    )

    fam_size = np.maximum(cleaned["CNT_FAM_MEMBERS"].fillna(1.0), 1.0)
    cleaned["income_per_family_member"] = cleaned["AMT_INCOME_TOTAL"] / fam_size
    cleaned["children_ratio"] = cleaned["CNT_CHILDREN"] / fam_size

    cols_to_drop = [
        col
        for col in ("FLAG_MOBIL", "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE")
        if col in cleaned.columns
    ]
    cleaned = cleaned.drop(columns=cols_to_drop)
    return cleaned.sort_values("ID").reset_index(drop=True)
