"""Final LightGBM failure model and shared cross-validation utilities."""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import FINAL_SEEDS, RANDOM_STATE, TUNED_LGBM_PARAMS
from src.cv_utils import (
    get_prediction_probabilities,
    prepare_training_data,
    validate_folds,
)
from src.feature_analysis import align_test_sensors, find_unavailable_in_test
from src.preprocessing import CONSTANT_SENSORS, UNRELIABLE_SENSORS

DROP_FEATURES = [
    "s027", "s038", "s047", "s055", "s056",
    "s057", "s059", "s062", "s063", "s113",
    "s120", "s122", "s132", "s154",
]

def get_base_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_already_aligned: bool = False,
) -> list:
    """Return reliable sensor features available in both train and test."""
    X, _ = prepare_training_data(train_df)
    sensors = list(X.columns)

    # Align only the original, uncorrected test data.
    test_features = (
        test_df
        if test_already_aligned
        else align_test_sensors(test_df, train_df)
    )

    excluded = (
        set(find_unavailable_in_test(train_df, test_features, sensors))
        | set(CONSTANT_SENSORS)
        | set(UNRELIABLE_SENSORS)
    )

    return [column for column in sensors if column not in excluded]


def build_lgbm(**overrides):
    """Build a LightGBM classifier."""
    import lightgbm as lgb

    params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(overrides)

    return lgb.LGBMClassifier(**params)


class SeedAveragedLGBM(ClassifierMixin, BaseEstimator):
    """Average predictions from LightGBM models trained with different seeds."""

    def __init__(self, params=None, seeds=FINAL_SEEDS):
        self.params = params
        self.seeds = seeds

    def fit(self, X, y):
        self.models_ = [
            build_lgbm(**{**(self.params or {}), "random_state": seed}).fit(X, y)
            for seed in self.seeds
        ]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        predictions = [model.predict_proba(X) for model in self.models_]
        return np.mean(predictions, axis=0)


def build_final_lgbm() -> Pipeline:
    """Build the final tuned, five-seed averaged LightGBM pipeline."""
    return Pipeline([
        (
            "model",
            SeedAveragedLGBM(
                params=TUNED_LGBM_PARAMS,
                seeds=FINAL_SEEDS,
            ),
        )
    ])


def run_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    name: str = "model",
    verbose: bool = True,
) -> dict:
    """Generate out-of-fold predictions using the shared CV folds."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if verbose:
        validate_folds(folds, len(X))

    oof = np.full(len(X), np.nan)
    models = []
    rows = []

    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        start = time.time()

        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        probabilities = get_prediction_probabilities(
            fold_model,
            X.iloc[valid_idx],
        )
        oof[valid_idx] = probabilities
        models.append(fold_model)

        fold_result = {
            "model": name,
            "fold": fold_number,
            "roc_auc": roc_auc_score(y.iloc[valid_idx], probabilities),
            "pr_auc": average_precision_score(y.iloc[valid_idx], probabilities),
            "fit_seconds": time.time() - start,
        }
        rows.append(fold_result)

        if verbose:
            print(
                f"{name} | fold {fold_number}: "
                f"ROC-AUC {fold_result['roc_auc']:.4f} | "
                f"PR-AUC {fold_result['pr_auc']:.4f} | "
                f"{fold_result['fit_seconds']:.1f}s"
            )

    if np.isnan(oof).any():
        raise ValueError("Some samples are missing OOF predictions.")

    return {
        "name": name,
        "oof": oof,
        "folds": pd.DataFrame(rows),
        "models": models,
    }
