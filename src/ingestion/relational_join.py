"""Schema validation, deduplication, and relational inner join for CCARP tables."""

from __future__ import annotations

import pandas as pd

from src.core.contracts import JoinAuditStats, JoinedTables

REQUIRED_APP_COLUMNS: tuple[str, ...] = (
    "ID",
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
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
)

REQUIRED_CREDIT_COLUMNS: tuple[str, ...] = ("ID", "MONTHS_BALANCE", "STATUS")
ALLOWED_CREDIT_STATUSES: frozenset[str] = frozenset(
    {"0", "1", "2", "3", "4", "5", "C", "X"}
)


def validate_table_schemas(app_df: pd.DataFrame, credit_df: pd.DataFrame) -> None:
    """Validate required columns, non-emptiness, and credit history uniqueness."""
    if app_df.empty or credit_df.empty:
        raise ValueError("Both application and credit tables must be non-empty.")

    missing_app = set(REQUIRED_APP_COLUMNS) - set(app_df.columns)
    if missing_app:
        raise ValueError(f"Missing application columns: {sorted(missing_app)}")

    missing_credit = set(REQUIRED_CREDIT_COLUMNS) - set(credit_df.columns)
    if missing_credit:
        raise ValueError(f"Missing credit columns: {sorted(missing_credit)}")

    if app_df["ID"].isna().any() or credit_df["ID"].isna().any():
        raise ValueError("Client ID columns must not contain null values.")

    if credit_df.duplicated(subset=["ID", "MONTHS_BALANCE"]).any():
        raise ValueError("credit_record contains duplicate (ID, MONTHS_BALANCE) rows.")

    observed_statuses = set(credit_df["STATUS"].astype(str).unique())
    unknown_statuses = observed_statuses - ALLOWED_CREDIT_STATUSES
    if unknown_statuses:
        raise ValueError(f"Unexpected credit STATUS values: {sorted(unknown_statuses)}")


def join_application_credit(
    app_df: pd.DataFrame,
    credit_df: pd.DataFrame,
) -> JoinedTables:
    """Deduplicate application IDs and inner-join with credit_record with full audit."""
    validate_table_schemas(app_df, credit_df)

    app_raw_rows = int(len(app_df))
    app_dedup = app_df.drop_duplicates(subset=["ID"], keep="first").copy()
    app_unique_ids = int(len(app_dedup))
    app_duplicate_ids = int(app_df["ID"].duplicated(keep=False).sum())

    credit_clean = credit_df.copy()
    credit_clean["STATUS"] = credit_clean["STATUS"].astype(str)
    credit_ids = set(credit_clean["ID"].unique())
    app_ids = set(app_dedup["ID"].unique())
    matched_ids = app_ids & credit_ids

    if not matched_ids:
        raise ValueError("Inner join on ID produced zero matched clients.")

    joined_app = app_dedup[app_dedup["ID"].isin(matched_ids)].copy()
    joined_credit = credit_clean[credit_clean["ID"].isin(matched_ids)].copy()
    joined_app = joined_app.sort_values("ID").reset_index(drop=True)
    joined_credit = joined_credit.sort_values(
        ["ID", "MONTHS_BALANCE"]
    ).reset_index(drop=True)

    audit = JoinAuditStats(
        app_raw_rows=app_raw_rows,
        app_unique_ids=app_unique_ids,
        app_duplicate_ids=app_duplicate_ids,
        credit_raw_rows=int(len(credit_clean)),
        credit_unique_ids=len(credit_ids),
        joined_ids=len(matched_ids),
        joined_credit_rows=int(len(joined_credit)),
        app_only_ids=len(app_ids - credit_ids),
        credit_only_ids=len(credit_ids - app_ids),
    )
    return JoinedTables(
        application_df=joined_app,
        credit_df=joined_credit,
        audit=audit,
    )
