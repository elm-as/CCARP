"""Zero-leakage fold-internal OOF Platt and Isotonic probability calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.core.contracts import (
    CalibrationBundle,
    FeatureDataset,
    FoldSplitSet,
    ModelRunSpec,
)
from src.modeling.credit_models import build_model_pipeline

PROB_EPS: float = 1e-6


def _to_logits(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities in (0, 1) to 2D logit column array."""
    clipped = np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return logits.reshape(-1, 1)


def _build_inner_folds(
    train_subset: FeatureDataset,
    model_spec: ModelRunSpec,
    fold_seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create inner CV folds strictly inside an outer training fold."""
    n_groups = int(len(np.unique(train_subset.group_keys)))
    n_pos = int(train_subset.target.sum())
    n_splits = min(model_spec.modeling_cfg.inner_cal_splits, n_groups, max(n_pos, 2))
    if n_splits < 2:
        raise ValueError("Insufficient positive samples or groups for inner calibration.")

    inner_sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=fold_seed
    )
    folds = list(
        inner_sgkf.split(
            train_subset.features_df, train_subset.target, train_subset.group_keys
        )
    )
    valid = all(len(np.unique(train_subset.target[tr])) == 2 for tr, _ in folds)
    if valid:
        return tuple((tr.astype(np.int64), va.astype(np.int64)) for tr, va in folds)

    fallback = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    return tuple(
        (tr.astype(np.int64), va.astype(np.int64))
        for tr, va in fallback.split(train_subset.features_df, train_subset.target)
    )


def _compute_inner_oof_probs(
    train_subset: FeatureDataset,
    model_spec: ModelRunSpec,
    fold_seed: int,
) -> np.ndarray:
    """Generate leak-free inner OOF raw probabilities on an outer training fold."""
    inner_folds = _build_inner_folds(train_subset, model_spec, fold_seed)
    inner_oof = np.full(len(train_subset.target), np.nan, dtype=np.float64)

    for in_tr, in_va in inner_folds:
        pipe = build_model_pipeline(model_spec)
        pipe.fit(train_subset.features_df.iloc[in_tr], train_subset.target[in_tr])
        inner_oof[in_va] = pipe.predict_proba(train_subset.features_df.iloc[in_va])[:, 1]

    if np.isnan(inner_oof).any():
        raise RuntimeError("Inner calibration OOF predictions contain NaN entries.")
    return inner_oof


def _fit_platt_and_isotonic(
    inner_probs: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LogisticRegression, IsotonicRegression]:
    """Fit Platt (logistic on logits) and Isotonic calibrators on inner OOF probs."""
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(_to_logits(inner_probs), y_train)

    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(inner_probs, y_train)
    return platt, isotonic


def fit_and_apply_oof_calibrators(
    dataset: FeatureDataset,
    splits: FoldSplitSet,
    model_spec: ModelRunSpec,
) -> CalibrationBundle:
    """Run outer CV with strictly fold-internal OOF Platt and Isotonic calibration."""
    n_samples = len(dataset.target)
    oof_raw = np.full(n_samples, np.nan, dtype=np.float64)
    oof_platt = np.full(n_samples, np.nan, dtype=np.float64)
    oof_iso = np.full(n_samples, np.nan, dtype=np.float64)
    base_seed = model_spec.modeling_cfg.random_seed

    for fold_idx, (train_idx, val_idx) in enumerate(splits.folds):
        train_subset = FeatureDataset(
            features_df=dataset.features_df.iloc[train_idx].reset_index(drop=True),
            target=dataset.target[train_idx],
            group_keys=dataset.group_keys[train_idx],
            client_ids=dataset.client_ids[train_idx],
            schema=dataset.schema,
        )
        inner_probs = _compute_inner_oof_probs(
            train_subset, model_spec, base_seed + fold_idx + 1
        )
        platt_cal, iso_cal = _fit_platt_and_isotonic(inner_probs, train_subset.target)

        outer_pipe = build_model_pipeline(model_spec)
        outer_pipe.fit(train_subset.features_df, train_subset.target)
        raw_val = outer_pipe.predict_proba(dataset.features_df.iloc[val_idx])[:, 1]

        oof_raw[val_idx] = raw_val
        oof_platt[val_idx] = platt_cal.predict_proba(_to_logits(raw_val))[:, 1]
        oof_iso[val_idx] = iso_cal.predict(raw_val)

    return CalibrationBundle(
        model_name=model_spec.model_name,
        y_true=dataset.target.copy(),
        oof_raw=oof_raw,
        oof_platt=oof_platt,
        oof_isotonic=oof_iso,
    )
