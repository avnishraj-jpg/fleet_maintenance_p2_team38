
"""Repair-budget and repair-cost regression for Fleet Maintenance Planning.

V2:
    Raw repair-cost LightGBM model with fold-fitted preprocessing.

V3:
    Log-transformed LightGBM repair-cost model with Duan smearing
    back-transformation, trained only on failed vehicles.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.preprocessing import fit_preprocessor, transform_features


# ============================================================
# V2 MODEL -- PRESERVED
# ============================================================

def build_repair_regressor():
    """Build the V2 LightGBM repair-budget model."""
    return LGBMRegressor(
        n_estimators=100,
        num_leaves=8,
        learning_rate=0.05,
        verbosity=-1,
        random_state=42,
        n_jobs=2,
    )


def generate_oof_repair_budgets(train_df, features, folds):
    """Generate V2 out-of-fold repair-budget predictions."""

    train_df = train_df.reset_index(drop=True)
    oof_budgets = np.full(len(train_df), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"Repair-budget OOF fold {fold}/{len(folds)}")

        train_part = train_df.iloc[train_idx]
        valid_part = train_df.iloc[valid_idx]

        prep = fit_preprocessor(
            train_part[features],
            scale=True,
        )

        X_train = transform_features(prep, train_part[features])
        X_valid = transform_features(prep, valid_part[features])

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
        raise ValueError("V2 OOF predictions are incomplete.")

    return oof_budgets


def fit_final_repair_model(train_df, test_df, features):
    """Fit the final V2 repair model and predict test budgets."""

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


# ============================================================
# V3 MODEL -- LOG-TRANSFORMED REPAIR COST
# ============================================================

def build_log_repair_regressor():
    """Build the V3 LightGBM model for log-transformed repair costs."""

    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        random_state=42,
        verbosity=-1,
    )


def fit_log_repair_model(X_train, y_train):
    """Fit the V3 model on log1p repair costs.

    Returns
    -------
    model : fitted LGBMRegressor
    smear : float
        Duan smearing correction factor estimated from training residuals.
    """

    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) == 0:
        raise ValueError("Cannot fit repair model on an empty dataset.")

    if not np.isfinite(y_train).all() or (y_train < 0).any():
        raise ValueError("Repair costs must be finite and non-negative.")

    y_log = np.log1p(y_train)

    model = build_log_repair_regressor()
    model.fit(X_train, y_log)

    # Duan smearing correction, following Farhan's experiment.
    fitted_log = model.predict(X_train)
    smear = float(np.mean(np.exp(y_log - fitted_log)))

    return model, smear


def predict_log_repair_cost(model, X, smear):
    """Predict repair costs and reverse the log transformation."""

    predicted_log = model.predict(X)

    predictions = np.exp(predicted_log) * smear - 1.0

    return np.maximum(predictions, 0.0)


def generate_oof_log_repair_costs(train_df, features, folds):
    """Generate V3 OOF repair-cost predictions.

    Each fold model is trained only on failed vehicles in that
    fold's training partition.
    """

    train_df = train_df.reset_index(drop=True)
    oof_costs = np.full(len(train_df), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"V3 log repair-cost OOF fold {fold}/{len(folds)}")

        train_part = train_df.iloc[train_idx]
        valid_part = train_df.iloc[valid_idx]

        known = (
            train_part["failure"].eq(1)
            & train_part["repair_cost"].notna()
        )

        if not known.any():
            raise ValueError(f"No known repair costs in fold {fold}.")

        # Farhan's V3 log-cost model uses raw features.
        X_train = train_part.loc[known, features]
        y_train = train_part.loc[known, "repair_cost"]
        X_valid = valid_part[features]

        model, smear = fit_log_repair_model(
            X_train,
            y_train,
        )

        oof_costs[valid_idx] = predict_log_repair_cost(
            model,
            X_valid,
            smear,
        )

    if not np.isfinite(oof_costs).all():
        raise ValueError("V3 OOF repair-cost predictions are incomplete.")

    return oof_costs


def fit_final_log_repair_model(train_df, test_df, features):
    """Fit the final V3 model and predict repair costs for test vehicles."""

    known = (
        train_df["failure"].eq(1)
        & train_df["repair_cost"].notna()
    )

    if not known.any():
        raise ValueError("No known failed-vehicle repair costs.")

    X_train = train_df.loc[known, features]
    y_train = train_df.loc[known, "repair_cost"]
    X_test = test_df[features]

    model, smear = fit_log_repair_model(
        X_train,
        y_train,
    )

    predictions = predict_log_repair_cost(
        model,
        X_test,
        smear,
    )

    return model, smear, predictions
