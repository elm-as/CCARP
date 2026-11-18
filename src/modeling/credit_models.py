"""Credit risk model pipelines, OOF training, and feature driver extraction."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.core.contracts import FeatureDataset, FoldSplitSet, ModelRunSpec

BEHAVIORAL_PREFIX: str = "obs_"


def _create_estimator(model_name: str, seed: int) -> Any:
    """Instantiate a regularized credit scoring estimator by canonical name."""
    if model_name == "logistic_regression":
        return LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=20.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=400,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=8.0,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model_pipeline(model_spec: ModelRunSpec) -> Pipeline:
    """Build a fold-safe preprocessing + estimator Pipeline from ModelRunSpec."""
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_spec.model_name == "logistic_regression":
        num_steps.append(("scaler", StandardScaler()))

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), list(model_spec.schema.numeric_cols)),
            ("cat", cat_transformer, list(model_spec.schema.categorical_cols)),
        ],
        remainder="drop",
    )
    estimator = _create_estimator(
        model_spec.model_name, model_spec.modeling_cfg.random_seed
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])


def train_oof_model(
    dataset: FeatureDataset,
    splits: FoldSplitSet,
    model_spec: ModelRunSpec,
) -> np.ndarray:
    """Train a model across CV folds and return aligned out-of-fold PD predictions."""
    n_samples = len(dataset.target)
    oof_preds = np.full(n_samples, np.nan, dtype=np.float64)

    for train_idx, val_idx in splits.folds:
        pipeline = build_model_pipeline(model_spec)
        x_train = dataset.features_df.iloc[train_idx]
        y_train = dataset.target[train_idx]
        x_val = dataset.features_df.iloc[val_idx]

        pipeline.fit(x_train, y_train)
        oof_preds[val_idx] = pipeline.predict_proba(x_val)[:, 1]

    if np.isnan(oof_preds).any():
        raise RuntimeError("OOF prediction array contains unassigned validation rows.")
    return oof_preds


def _resolve_feature_family(transformed_name: str) -> tuple[str, str]:
    """Strip ColumnTransformer prefixes and classify feature as Behavioral or Demographic."""
    clean_name = transformed_name.replace("num__", "").replace("cat__", "")
    family = (
        "Behavioral (<= T_obs)"
        if clean_name.startswith(BEHAVIORAL_PREFIX)
        else "Demographic"
    )
    return clean_name, family


def extract_feature_importances(
    dataset: FeatureDataset,
    splits: FoldSplitSet,
    model_spec: ModelRunSpec,
) -> tuple[dict[str, Any], ...]:
    """Compute mean fold-level LightGBM split/gain importances across CV folds."""
    importance_acc: dict[str, float] = {}
    n_folds = len(splits.folds)

    for train_idx, _ in splits.folds:
        pipeline = build_model_pipeline(model_spec)
        pipeline.fit(dataset.features_df.iloc[train_idx], dataset.target[train_idx])
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        estimator = pipeline.named_steps["estimator"]
        raw_imp = estimator.booster_.feature_importance(importance_type="gain")
        total_gain = float(np.sum(raw_imp)) or 1.0
        for name, val in zip(names, raw_imp):
            importance_acc[name] = importance_acc.get(name, 0.0) + float(val) / (
                total_gain * n_folds
            )

    ranked = sorted(importance_acc.items(), key=lambda kv: kv[1], reverse=True)
    rows: list[dict[str, Any]] = []
    for raw_name, share in ranked[:15]:
        clean_name, family = _resolve_feature_family(raw_name)
        rows.append(
            {
                "feature": clean_name,
                "feature_family": family,
                "gain_share_pct": round(100.0 * share, 2),
            }
        )
    return tuple(rows)
