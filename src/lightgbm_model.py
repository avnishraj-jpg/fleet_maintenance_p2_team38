"""LightGBM failure model for Fleet Maintenance Planning (Member 3 - Shaik Mohd Aatiq).

All experiments use the team's shared 5-fold CV indices (results/cv_folds.npz,
loaded with cv_utils.load_folds). Every model is a scikit-learn Pipeline that is
cloned and fitted on the training part of each fold only, so any preprocessing
or supervised feature selection inside it never sees the validation fold.

Evaluation
----------
The primary metric is the official competition cost:
- out-of-fold (OOF) probabilities from all 5 folds are pooled;
- one inspection threshold is chosen on the pooled OOF probabilities (the same
  threshold would then be applied to test), with the constant budget fitted on
  the full training data;
- per-fold costs at that threshold use a budget fitted on that fold's training
  part only, and show how stable the cost is across folds.
ROC-AUC and PR-AUC are reported as supplementary metrics.

Features
--------
All sensors except those that cannot be used at prediction time: s063 and s122
are unavailable in the aligned test set, and s146 is constant (see
notebooks/eda.ipynb, Section 6).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_model import fit_constant_budget  # noqa: E402
from cv_utils import (  # noqa: E402
    get_prediction_probabilities,
    load_folds,
    prepare_training_data,
    validate_folds,
)
from evaluation import evaluate_predictions, select_threshold  # noqa: E402
from failure_model import CHOSEN_CONFIG, build_model  # noqa: E402
from feature_analysis import (  # noqa: E402
    align_test_sensors,
    find_unavailable_in_test,
    load_data,
)
from preprocessing import CONSTANT_SENSORS, fit_preprocessor, transform_features  # noqa: E402

RANDOM_STATE = 42


# --------------------------------------------------
# Features
# --------------------------------------------------

def get_base_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """Sensor features usable at prediction time.

    Drops sensors that are fully missing in the aligned test set (s063, s122)
    and constant sensors (s146).
    """
    X, _ = prepare_training_data(train_df)
    sensors = list(X.columns)
    test_aligned = align_test_sensors(test_df, train_df)
    excluded = set(find_unavailable_in_test(train_df, test_aligned, sensors)) | set(CONSTANT_SENSORS)
    return [c for c in sensors if c not in excluded]


# --------------------------------------------------
# Logistic Regression benchmark (V1 configuration)
# --------------------------------------------------

class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn wrapper around src/preprocessing.py so it is fitted per CV fold."""

    def __init__(self, imputation="median", missing_indicators=None, scale=True):
        self.imputation = imputation
        self.missing_indicators = missing_indicators
        self.scale = scale

    def fit(self, X, y=None):
        self.preprocessor_ = fit_preprocessor(
            X, imputation=self.imputation, missing_indicators=self.missing_indicators, scale=self.scale
        )
        return self

    def transform(self, X):
        return transform_features(self.preprocessor_, X)


def build_lr_benchmark() -> Pipeline:
    """Aatiq's V1 model: median imputation + scaling + LR with class weight 1:50."""
    return Pipeline([
        ("preprocess", FoldPreprocessor(imputation="median", scale=True)),
        ("model", build_model(class_weight=CHOSEN_CONFIG["class_weight"])),
    ])


# --------------------------------------------------
# LightGBM
# --------------------------------------------------

# Baseline configuration (before tuning). No early stopping: stopping on the
# validation fold would let it influence training. Missing values are passed
# through unchanged; LightGBM learns a default direction for them at each split,
# which keeps the missingness signal found in the EDA.
DEFAULT_LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def build_lgbm(**overrides):
    """LightGBM classifier with the baseline parameters (overridable)."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **overrides})


def build_lgbm_all_features(**overrides) -> Pipeline:
    """Task 6: LightGBM on all base features, no preprocessing needed."""
    return Pipeline([("model", build_lgbm(**overrides))])


# --------------------------------------------------
# Cost-based threshold selection
# --------------------------------------------------

def fast_select_threshold(probabilities, repair_costs, baseline_budget: float):
    """Same result as evaluation.select_threshold, computed in O(n log n).

    Flagging vehicle i changes the total cost by
    delta_i = inspected_cost_i - actual_cost_i. Sorting by probability
    (descending) and taking cumulative sums of delta gives the cost of every
    candidate threshold at once. Candidates are the distinct probabilities
    plus "inspect nobody", exactly as in select_threshold.
    """
    p = np.asarray(probabilities, dtype=float)
    actual = np.nan_to_num(np.asarray(repair_costs, dtype=float), nan=0.0)
    b = float(baseline_budget)
    inspected = 10.0 + 0.20 * np.maximum(actual - b, 0.0) + 0.02 * np.maximum(b - actual, 0.0)
    delta = inspected - actual

    order = np.argsort(-p, kind="stable")
    p_sorted = p[order]
    cum = np.cumsum(delta[order])
    # Last position of each distinct probability (threshold t flags all p >= t)
    last = np.r_[np.flatnonzero(np.diff(p_sorted) != 0), len(p_sorted) - 1]
    costs = actual.sum() + cum[last]

    no_inspection = actual.sum()
    best = int(np.argmin(costs))
    if costs[best] < no_inspection:
        return float(p_sorted[last[best]]), float(costs[best])
    return float(np.nextafter(p.max(), np.inf)), float(no_inspection)


# --------------------------------------------------
# Cross-validation on the shared folds
# --------------------------------------------------

def run_cv(model, X: pd.DataFrame, y: pd.Series, folds, name: str = "model", verbose: bool = True) -> dict:
    """Fit `model` on each training fold and predict its validation fold.

    Mirrors cv_utils.evaluate_classifier, but also keeps the fitted fold
    models (needed for feature importance) and the fit time.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    validate_folds(folds, len(X)) if verbose else None

    oof = np.full(len(X), np.nan)
    models, rows = [], []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        start = time.time()
        fold_model = clone(model).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = get_prediction_probabilities(fold_model, X.iloc[va_idx])
        oof[va_idx] = proba
        models.append(fold_model)
        rows.append({
            "model": name, "fold": k,
            "roc_auc": roc_auc_score(y.iloc[va_idx], proba),
            "pr_auc": average_precision_score(y.iloc[va_idx], proba),
            "fit_seconds": time.time() - start,
        })
        if verbose:
            print(f"{name} | fold {k}: ROC-AUC {rows[-1]['roc_auc']:.4f} | PR-AUC {rows[-1]['pr_auc']:.4f} "
                  f"| {rows[-1]['fit_seconds']:.1f}s")

    if np.isnan(oof).any():
        raise ValueError("Some samples are missing OOF predictions.")
    return {"name": name, "oof": oof, "folds": pd.DataFrame(rows), "models": models}


def evaluate_cost(cv_result: dict, train_df: pd.DataFrame, folds) -> dict:
    """Official cost for a CV result (see module docstring for the protocol)."""
    oof = cv_result["oof"]
    repair = train_df["repair_cost"].reset_index(drop=True)
    y = train_df["failure"].reset_index(drop=True)

    budget = fit_constant_budget(train_df)
    threshold, total_cost = fast_select_threshold(oof, repair, budget)
    flags = (oof >= threshold).astype(int)
    overall = evaluate_predictions(y, oof, repair, flags, np.where(flags == 1, budget, 0.0), threshold)

    per_fold = []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_budget = fit_constant_budget(train_df.iloc[tr_idx])
        f = flags[va_idx]
        m = evaluate_predictions(y.iloc[va_idx], oof[va_idx], repair.iloc[va_idx], f,
                                 np.where(f == 1, fold_budget, 0.0), threshold)
        per_fold.append({"fold": k, "cost": m["total_cost"], "tp": m["tp"], "fn": m["fn"], "fp": m["fp"],
                         "flagged": m["vehicles_inspected"]})
    per_fold = pd.DataFrame(per_fold)

    fold_metrics = cv_result["folds"]
    summary = {
        "model": cv_result["name"],
        "oof_cost": total_cost,
        "cost_per_1000_vehicles": 1000 * total_cost / len(oof),
        "fold_cost_mean": per_fold["cost"].mean(),
        "fold_cost_std": per_fold["cost"].std(),
        "threshold": threshold,
        "budget": budget,
        "flagged": overall["vehicles_inspected"],
        "tp": overall["tp"], "fn": overall["fn"], "fp": overall["fp"],
        "recall": overall["recall"], "precision": overall["precision"],
        "roc_auc_mean": fold_metrics["roc_auc"].mean(), "roc_auc_std": fold_metrics["roc_auc"].std(),
        "pr_auc_mean": fold_metrics["pr_auc"].mean(), "pr_auc_std": fold_metrics["pr_auc"].std(),
    }
    return {"summary": summary, "per_fold": per_fold}


def no_inspection_cost(train_df: pd.DataFrame) -> float:
    """Cost of inspecting nobody (every failure pays its full repair cost)."""
    return float(train_df["repair_cost"].fillna(0).sum())


def comparison_table(evaluations: list) -> pd.DataFrame:
    """One row per model, sorted by OOF cost (lower is better)."""
    cols = ["oof_cost", "cost_per_1000_vehicles", "fold_cost_mean", "fold_cost_std", "flagged", "tp", "fn", "fp",
            "recall", "precision", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "threshold"]
    table = pd.DataFrame([e["summary"] for e in evaluations]).set_index("model")[cols]
    return table.sort_values("oof_cost")


def per_fold_costs(evaluations: list) -> pd.DataFrame:
    """Fold-by-fold cost for each model (same folds, so directly comparable)."""
    return pd.DataFrame({e["summary"]["model"]: e["per_fold"].set_index("fold")["cost"] for e in evaluations})


# --------------------------------------------------
# Command line
# --------------------------------------------------

def main():
    train_df, test_df = load_data()
    X_all, y = prepare_training_data(train_df)
    features = get_base_features(train_df, test_df)
    X = X_all[features]
    folds = load_folds()
    print(f"Training rows: {len(X):,} | features: {len(features)} | folds: {len(folds)}")
    print(f"Cost of inspecting nobody: {no_inspection_cost(train_df):,.2f}\n")

    experiments = {
        "LR benchmark (1:50)": build_lr_benchmark(),
        "LGBM all features": build_lgbm_all_features(),
    }
    evaluations = []
    for name, model in experiments.items():
        cv = run_cv(model, X, y, folds, name=name)
        evaluations.append(evaluate_cost(cv, train_df, folds))
        print()

    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 200):
        print("Comparison (sorted by OOF cost):")
        print(comparison_table(evaluations).T.to_string())
        print("\nCost per fold:")
        print(per_fold_costs(evaluations).to_string())


if __name__ == "__main__":
    main()
