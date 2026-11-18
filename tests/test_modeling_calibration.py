"""Unit tests for demographic grouping, GroupKFold disjointness, and zero-leakage calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import ModelingConfig
from src.core.contracts import FeatureDataset, FeatureSchema, ModelRunSpec
from src.features.profile_grouping import build_demographic_group_keys
from src.modeling.cv_splitter import create_stratified_group_splits, has_disjoint_groups
from src.modeling.probability_calibrator import fit_and_apply_oof_calibrators


def _make_synthetic_dataset(n_groups: int = 24, clones_per_group: int = 3) -> FeatureDataset:
    rng = np.random.default_rng(42)
    rows = []
    targets = []
    keys = []
    ids = []
    for grp in range(n_groups):
        base_x1 = float(rng.normal(0.0, 1.0))
        base_x2 = float(rng.normal(0.0, 1.0))
        cat_val = "A" if grp % 2 == 0 else "B"
        y_grp = 1 if grp < (n_groups // 3) else 0
        for clone in range(clones_per_group):
            ids.append(grp * 10 + clone)
            keys.append(f"grp_{grp}")
            targets.append(y_grp)
            rows.append(
                {
                    "x_num1": base_x1 + 0.05 * clone,
                    "x_num2": base_x2 + (1.2 if y_grp == 1 else -0.5),
                    "x_cat": cat_val,
                }
            )
    schema = FeatureSchema(numeric_cols=("x_num1", "x_num2"), categorical_cols=("x_cat",))
    return FeatureDataset(
        features_df=pd.DataFrame(rows),
        target=np.array(targets, dtype=np.int64),
        group_keys=np.array(keys),
        client_ids=np.array(ids, dtype=np.int64),
        schema=schema,
    )


def test_build_demographic_group_keys_maps_clones_identically() -> None:
    df = pd.DataFrame(
        {
            "CODE_GENDER": ["F", "F", "M"],
            "FLAG_OWN_CAR": ["N", "N", "Y"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N"],
            "CNT_CHILDREN": [0, 0, 1],
            "AMT_INCOME_TOTAL": [100000.0, 100000.0, 150000.0],
            "NAME_INCOME_TYPE": ["Working", "Working", "Working"],
            "NAME_EDUCATION_TYPE": ["Higher education"] * 3,
            "NAME_FAMILY_STATUS": ["Married"] * 3,
            "NAME_HOUSING_TYPE": ["House / apartment"] * 3,
            "DAYS_BIRTH": [-12000, -12000, -14000],
            "DAYS_EMPLOYED": [-1000, -1000, -2000],
            "OCCUPATION_TYPE": [np.nan, np.nan, "Managers"],
            "CNT_FAM_MEMBERS": [2.0, 2.0, 3.0],
        }
    )
    keys = build_demographic_group_keys(df)
    assert keys.iloc[0] == keys.iloc[1]
    assert keys.iloc[0] != keys.iloc[2]


def test_group_cv_disjoint_and_zero_validation_calibration_leakage() -> None:
    ds = _make_synthetic_dataset()
    cfg = ModelingConfig(n_splits=4, inner_cal_splits=3, random_seed=42)
    splits = create_stratified_group_splits(ds, cfg)
    assert has_disjoint_groups(splits, ds.group_keys)

    spec = ModelRunSpec("logistic_regression", cfg, ds.schema)
    bundle_orig = fit_and_apply_oof_calibrators(ds, splits, spec)

    for arr in (bundle_orig.oof_raw, bundle_orig.oof_platt, bundle_orig.oof_isotonic):
        assert len(arr) == len(ds.target)
        assert np.isfinite(arr).all()
        assert ((arr >= 0.0) & (arr <= 1.0)).all()

    val_idx_0 = splits.folds[0][1]
    permuted_target = ds.target.copy()
    permuted_target[val_idx_0] = 1 - permuted_target[val_idx_0]
    ds_perm = FeatureDataset(
        features_df=ds.features_df,
        target=permuted_target,
        group_keys=ds.group_keys,
        client_ids=ds.client_ids,
        schema=ds.schema,
    )
    bundle_perm = fit_and_apply_oof_calibrators(ds_perm, splits, spec)

    np.testing.assert_allclose(bundle_orig.oof_raw[val_idx_0], bundle_perm.oof_raw[val_idx_0])
    np.testing.assert_allclose(bundle_orig.oof_platt[val_idx_0], bundle_perm.oof_platt[val_idx_0])
    np.testing.assert_allclose(bundle_orig.oof_isotonic[val_idx_0], bundle_perm.oof_isotonic[val_idx_0])
