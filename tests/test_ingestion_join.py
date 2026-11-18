"""Unit tests for schema validation and relational inner join."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.relational_join import join_application_credit, validate_table_schemas


def _sample_app_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [101, 101, 102, 103],
            "CODE_GENDER": ["F", "F", "M", "F"],
            "FLAG_OWN_CAR": ["N", "N", "Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y", "Y", "N"],
            "CNT_CHILDREN": [0, 0, 1, 0],
            "AMT_INCOME_TOTAL": [120000.0, 120000.0, 200000.0, 95000.0],
            "NAME_INCOME_TYPE": ["Working", "Working", "Commercial associate", "Pensioner"],
            "NAME_EDUCATION_TYPE": ["Higher education"] * 4,
            "NAME_FAMILY_STATUS": ["Married"] * 4,
            "NAME_HOUSING_TYPE": ["House / apartment"] * 4,
            "DAYS_BIRTH": [-12000, -12000, -14000, -22000],
            "DAYS_EMPLOYED": [-1500, -1500, -2500, 365243],
            "FLAG_MOBIL": [1, 1, 1, 1],
            "FLAG_WORK_PHONE": [0, 0, 1, 0],
            "FLAG_PHONE": [1, 1, 0, 1],
            "FLAG_EMAIL": [0, 0, 0, 1],
            "OCCUPATION_TYPE": ["Core staff", "Core staff", "Managers", None],
            "CNT_FAM_MEMBERS": [2.0, 2.0, 3.0, 1.0],
        }
    )


def _sample_credit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [101, 101, 102, 999],
            "MONTHS_BALANCE": [-15, -5, -12, -3],
            "STATUS": ["0", "C", "2", "0"],
        }
    )


def test_validate_table_schemas_raises_on_missing_column() -> None:
    app_bad = _sample_app_df().drop(columns=["AMT_INCOME_TOTAL"])
    with pytest.raises(ValueError, match="Missing application columns"):
        validate_table_schemas(app_bad, _sample_credit_df())


def test_validate_table_schemas_raises_on_duplicate_credit_month() -> None:
    cred_dup = pd.DataFrame(
        {
            "ID": [101, 101],
            "MONTHS_BALANCE": [-10, -10],
            "STATUS": ["0", "1"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_table_schemas(_sample_app_df(), cred_dup)


def test_join_application_credit_deduplicates_and_audits() -> None:
    app_df = _sample_app_df()
    credit_df = _sample_credit_df()
    app_orig = app_df.copy(deep=True)
    cred_orig = credit_df.copy(deep=True)

    joined = join_application_credit(app_df, credit_df)

    pd.testing.assert_frame_equal(app_df, app_orig)
    pd.testing.assert_frame_equal(credit_df, cred_orig)
    assert joined.audit.app_raw_rows == 4
    assert joined.audit.app_unique_ids == 3
    assert joined.audit.app_duplicate_ids == 2
    assert joined.audit.joined_ids == 2
    assert joined.audit.app_only_ids == 1
    assert joined.audit.credit_only_ids == 1
    assert list(joined.application_df["ID"]) == [101, 102]
