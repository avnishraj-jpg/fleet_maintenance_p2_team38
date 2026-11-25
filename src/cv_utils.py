"""
cv_utils.py

Shared cross-validation utilities for the Fleet Maintenance Planning project.

Responsibilities:
1. Generate stratified folds.
2. Save and load fold indices.
3. Evaluate classification models using identical folds.
4. Generate OOF predictions and fold-wise metrics.

Important:
- All preprocessing must be inside the estimator Pipeline.
- Never fit preprocessing on the complete dataset before CV.
- All team members must use the same saved folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    FOLDS_PATH,
    RANDOM_STATE,
    N_SPLITS,
)


# --------------------------------------------------
# 1. Prepare training features and target
# --------------------------------------------------

def prepare_training_data(train_df):
    """
    Separate classifier features and target.

    Excluded columns:
    - id: identifier, not a predictive feature.
    - followup_days: unavailable in test data.
    - repair_cost: used for the repair-cost modelling task,
      not as a failure-classification feature.
    - failure: target variable.

    Returns
    -------
    X : pandas DataFrame
        Sensor features.
    y : pandas Series
        Binary failure target.
    """

    excluded_columns = [
        "id",
        "followup_days",
        "repair_cost",
        "failure",
    ]

    if "failure" not in train_df.columns:
        raise ValueError("Training data must contain the 'failure' column.")

    X = train_df.drop(
        columns=excluded_columns,
        errors="ignore",
    ).copy()

    y = train_df["failure"].copy()

    if X.empty:
        raise ValueError("No feature columns found.")

    if y.isna().any():
        raise ValueError("The failure target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The failure target must contain only 0 and 1.")

    return X, y


# --------------------------------------------------
# 2. Generate stratified folds
# --------------------------------------------------

def create_folds(
    X,
    y,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Generate stratified CV folds.

    The same folds must be used by all team members.

    Returns
    -------
    folds : list of tuples
        Each tuple contains:
        (training_indices, validation_indices)
    """

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []

    for train_idx, val_idx in splitter.split(X, y):
        folds.append((train_idx, val_idx))

    return folds


# --------------------------------------------------
# 3. Save folds
# --------------------------------------------------

def save_folds(folds, path=FOLDS_PATH):
    """
    Save fold indices to an NPZ file.

    This allows all team members to reuse exactly
    the same train-validation splits.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fold_data = {}

    for i, (train_idx, val_idx) in enumerate(folds):
        fold_data[f"train_{i}"] = train_idx
        fold_data[f"val_{i}"] = val_idx

    np.savez_compressed(path, **fold_data)

    print(f"Saved {len(folds)} folds to: {path}")


# --------------------------------------------------
# 4. Load saved folds
# --------------------------------------------------

def load_folds(path=FOLDS_PATH):
    """
    Load previously saved CV folds.

    Returns
    -------
    folds : list of tuples
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CV folds not found at {path}. "
            "Run create_folds() and save_folds() first."
        )

    data = np.load(path)

    folds = []
    i = 0

    while f"train_{i}" in data:
        train_idx = data[f"train_{i}"]
        val_idx = data[f"val_{i}"]

        folds.append((train_idx, val_idx))
        i += 1

    if not folds:
        raise ValueError("No folds found in the saved file.")

    return folds


# --------------------------------------------------
# 5. Validate folds
# --------------------------------------------------

def validate_folds(folds, n_samples):
    """
    Check that:
    - Every validation index is valid.
    - No training and validation indices overlap.
    - Every sample appears in validation exactly once.
    """

    validation_counts = np.zeros(n_samples, dtype=int)

    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):

        if np.any(train_idx < 0) or np.any(train_idx >= n_samples):
            raise ValueError(f"Invalid training index in fold {fold_number}.")

        if np.any(val_idx < 0) or np.any(val_idx >= n_samples):
            raise ValueError(f"Invalid validation index in fold {fold_number}.")

        if np.intersect1d(train_idx, val_idx).size > 0:
            raise ValueError(f"Training-validation overlap in fold {fold_number}.")

        validation_counts[val_idx] += 1

    if not np.all(validation_counts == 1):
        raise ValueError(
            "Each sample must appear in validation exactly once."
        )

    print("Fold validation successful.")


# --------------------------------------------------
# 6. Get classification probabilities
# --------------------------------------------------

def get_prediction_probabilities(model, X):
    """
    Obtain positive-class probabilities.

    Supports:
    - Models with predict_proba()
    - Models with decision_function()

    For decision_function-only models, scores are
    converted to probabilities using a sigmoid.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        probabilities = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

    else:
        raise AttributeError(
            "Model must support predict_proba() "
            "or decision_function()."
        )

    return np.asarray(probabilities)


# --------------------------------------------------
# 7. Evaluate one classification model
# --------------------------------------------------

def evaluate_classifier(
    model,
    X,
    y,
    folds,
    model_name="model",
    threshold=0.5,
):
    """
    Evaluate a classification model using predefined folds.

    Parameters
    ----------
    model:
        An sklearn-compatible estimator or Pipeline.

    X:
        Training features.

    y:
        Binary failure target.

    folds:
        Saved or generated CV folds.

    model_name:
        Name used in result tables.

    threshold:
        Classification threshold for binary metrics.

    Returns
    -------
    results_df:
        Fold-wise evaluation metrics.

    oof_df:
        Out-of-fold predictions for every training sample.
    """

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    validate_folds(folds, len(X))

    oof_predictions = np.full(len(X), np.nan)

    fold_results = []

    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):

        print(f"\n{model_name} | Fold {fold_number}/{len(folds)}")

        # Split data
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Clone model to ensure independent training per fold
        fold_model = clone(model)

        # Fit only on training fold
        fold_model.fit(X_train, y_train)

        # Predict validation probabilities
        probabilities = get_prediction_probabilities(
            fold_model,
            X_val,
        )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                f"Non-finite predictions in fold {fold_number}."
            )

        # Save OOF predictions
        oof_predictions[val_idx] = probabilities

        # Convert probabilities to binary predictions
        predictions = (probabilities >= threshold).astype(int)

        # Calculate metrics
        fold_metrics = {
            "model": model_name,
            "fold": fold_number,
            "roc_auc": roc_auc_score(y_val, probabilities),
            "pr_auc": average_precision_score(y_val, probabilities),
            "accuracy": accuracy_score(y_val, predictions),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
        }

        fold_results.append(fold_metrics)

        print(
            f"ROC-AUC: {fold_metrics['roc_auc']:.4f} | "
            f"PR-AUC: {fold_metrics['pr_auc']:.4f} | "
            f"F1: {fold_metrics['f1']:.4f}"
        )

    # Ensure every sample received an OOF prediction
    if np.isnan(oof_predictions).any():
        raise ValueError("Some samples are missing OOF predictions.")

    results_df = pd.DataFrame(fold_results)

    # Create OOF prediction dataframe
    oof_df = pd.DataFrame({
        "y_true": y,
        "failure_probability": oof_predictions,
    })

    oof_df["model"] = model_name

    return results_df, oof_df


# --------------------------------------------------
# 8. Summarise fold-wise results
# --------------------------------------------------

def summarise_results(results_df):
    """
    Calculate mean and standard deviation
    of the fold-wise evaluation metrics.
    """

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    summary = results_df[metric_columns].agg(["mean", "std"]).T

    summary.columns = ["mean", "std"]

    return summary


# --------------------------------------------------
# 9. Save evaluation results
# --------------------------------------------------

def save_evaluation_results(
    results_df,
    oof_df,
    model_name,
    results_dir=RESULTS_DIR,
):
    """
    Save fold-wise metrics and OOF predictions.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{model_name}_fold_results.csv"
    oof_path = results_dir / f"{model_name}_oof_predictions.csv"

    results_df.to_csv(results_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"\nSaved fold results: {results_path}")
    print(f"Saved OOF predictions: {oof_path}")
