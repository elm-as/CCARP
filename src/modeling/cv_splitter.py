"""Cross-validation splitters with strict demographic-group isolation audits."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from src.core.config import ModelingConfig
from src.core.contracts import FeatureDataset, FoldSplitSet


def has_disjoint_groups(splits: FoldSplitSet, group_keys: np.ndarray) -> bool:
    """Return True if no demographic group appears in both train and val of any fold."""
    for train_idx, val_idx in splits.folds:
        train_groups = set(group_keys[train_idx])
        val_groups = set(group_keys[val_idx])
        if not train_groups.isdisjoint(val_groups):
            return False
    return True


def create_stratified_group_splits(
    dataset: FeatureDataset,
    modeling_cfg: ModelingConfig,
) -> FoldSplitSet:
    """Build StratifiedGroupKFold splits guaranteeing zero demographic-profile overlap."""
    splitter = StratifiedGroupKFold(
        n_splits=modeling_cfg.n_splits,
        shuffle=True,
        random_state=modeling_cfg.random_seed,
    )
    folds = tuple(
        (tr.astype(np.int64), va.astype(np.int64))
        for tr, va in splitter.split(
            dataset.features_df, dataset.target, dataset.group_keys
        )
    )
    split_set = FoldSplitSet(
        scheme_name=f"StratifiedGroupKFold({modeling_cfg.n_splits})",
        folds=folds,
    )
    if not has_disjoint_groups(split_set, dataset.group_keys):
        raise AssertionError("StratifiedGroupKFold violated demographic group disjointness.")
    return split_set


def create_group_splits(
    dataset: FeatureDataset,
    modeling_cfg: ModelingConfig,
) -> FoldSplitSet:
    """Build deterministic GroupKFold splits keyed by demographic signature."""
    splitter = GroupKFold(n_splits=modeling_cfg.n_splits)
    folds = tuple(
        (tr.astype(np.int64), va.astype(np.int64))
        for tr, va in splitter.split(
            dataset.features_df, dataset.target, dataset.group_keys
        )
    )
    split_set = FoldSplitSet(
        scheme_name=f"GroupKFold({modeling_cfg.n_splits})",
        folds=folds,
    )
    if not has_disjoint_groups(split_set, dataset.group_keys):
        raise AssertionError("GroupKFold violated demographic group disjointness.")
    return split_set


def create_stratified_splits(
    dataset: FeatureDataset,
    modeling_cfg: ModelingConfig,
) -> FoldSplitSet:
    """Build naive StratifiedKFold splits (baseline for duplicate-profile ablation)."""
    splitter = StratifiedKFold(
        n_splits=modeling_cfg.n_splits,
        shuffle=True,
        random_state=modeling_cfg.random_seed,
    )
    folds = tuple(
        (tr.astype(np.int64), va.astype(np.int64))
        for tr, va in splitter.split(dataset.features_df, dataset.target)
    )
    return FoldSplitSet(
        scheme_name=f"StratifiedKFold({modeling_cfg.n_splits})",
        folds=folds,
    )
