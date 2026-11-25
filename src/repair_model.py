
"""Repair-budget regression for Fleet Maintenance Planning."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.preprocessing import fit_preprocessor, transform_features


def build_repair_regressor():
    """Build the LightGBM repair-budget model selected through nested CV."""
    return LGBMRegressor(
        n_estimators=100,
        num_leaves=8,
        learning_rate=0.05,
        verbosity=-1,
        random_state=42,
        n_jobs=2,
    )


def generate_oof_repair_budgets(train_df, features, folds):
    """Generate out-of-fold repair-budget predictions without leakage."""

    train_df = train_df.reset_index(drop=True)
    oof_budgets = np.full(len(train_df), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"Repair-budget OOF fold {fold}/{len(folds)}")

        train_part = train_df.iloc[train_idx]
        valid_part = train_df.iloc[valid_idx]

        # Fit preprocessing using only this fold's training data.
        prep = fit_preprocessor(
            train_part[features],
            scale=True,
        )

        X_train = transform_features(prep, train_part[features])
        X_valid = transform_features(prep, valid_part[features])

        # Train only on failed vehicles with known repair costs.
        known = (
            train_part["failure"].eq(1)
            & train_part["repair_cost"].notna()
        )

        if not known.any():
            raise ValueError(f"No known repair costs in fold {fold}.")

        model = build_repair_regressor()
        model.fit(
            X_train.loc[known],
            train_part.loc[known, "repair_cost"],
        )

        oof_budgets[valid_idx] = np.maximum(
            model.predict(X_valid),
            0.0,
        )

    if not np.isfinite(oof_budgets).all():
        raise ValueError("OOF repair-budget predictions are incomplete.")

    return oof_budgets


def fit_final_repair_model(train_df, test_df, features):
    """Fit the final repair model and predict budgets for test vehicles."""

    known = (
        train_df["failure"].eq(1)
        & train_df["repair_cost"].notna()
    )

    prep = fit_preprocessor(
        train_df[features],
        scale=True,
    )

    X_train = transform_features(prep, train_df[features])
    X_test = transform_features(prep, test_df[features])

    model = build_repair_regressor()
    model.fit(
        X_train.loc[known],
        train_df.loc[known, "repair_cost"],
    )

    predictions = np.maximum(model.predict(X_test), 0.0)

    return model, prep, predictions
