
"""Compare alternative failure classifiers using shared OOF folds."""

import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from src.preprocessing import fit_preprocessor, transform_features
from src.evaluation import tune_policy, calculate_cost


def build_experiment_models(y_train):
    """Create fresh classifiers for each CV fold."""

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = negatives / max(positives, 1)

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_lambda=5,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }


def run_classifier_experiments(
    train_df,
    features,
    folds,
    oof_repair_budgets,
    baseline_probabilities=None,
):
    """
    Generate OOF probabilities and compare official cost.

    Preprocessing is fitted separately inside each training fold.
    The same OOF repair-budget predictions are used for every classifier.
    """

    train_df = train_df.reset_index(drop=True)
    X = train_df[features]
    y = train_df["failure"].to_numpy()

    budgets = np.asarray(oof_repair_budgets, dtype=float)

    if len(budgets) != len(train_df):
        raise ValueError("Repair-budget OOF predictions have wrong length.")

    if not np.isfinite(budgets).all():
        raise ValueError("Repair-budget predictions contain invalid values.")

    results = []
    predictions = {}

    models = build_experiment_models(y)

    # Include current LightGBM as the comparison baseline.
    if baseline_probabilities is not None:
        predictions["lightgbm"] = np.asarray(
            baseline_probabilities, dtype=float
        )

    for name, model in models.items():
        print(f"\n{'=' * 55}")
        print(f"Running {name}")
        print(f"{'=' * 55}")

        oof = np.full(len(train_df), np.nan)
        fold_metrics = []

        for fold_number, (train_idx, valid_idx) in enumerate(
            folds, start=1
        ):
            start = time.time()

            train_part = X.iloc[train_idx]
            valid_part = X.iloc[valid_idx]

            y_train = y[train_idx]
            y_valid = y[valid_idx]

            # Fit preprocessing on this fold's training data only.
            prep = fit_preprocessor(
                train_part,
                imputation="median",
                scale=False,
            )

            X_train = transform_features(prep, train_part)
            X_valid = transform_features(prep, valid_part)

            # Rebuild XGBoost with the fold-specific class ratio.
            if name == "xgboost":
                fold_model = build_experiment_models(y_train)[name]
            else:
                fold_model = build_experiment_models(y_train)[name]

            fold_model.fit(X_train, y_train)

            probabilities = fold_model.predict_proba(X_valid)[:, 1]
            oof[valid_idx] = probabilities

            fold_metrics.append({
                "model": name,
                "fold": fold_number,
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "pr_auc": average_precision_score(y_valid, probabilities),
                "seconds": time.time() - start,
            })

            print(
                f"Fold {fold_number}/{len(folds)} | "
                f"ROC-AUC: {fold_metrics[-1]['roc_auc']:.4f} | "
                f"PR-AUC: {fold_metrics[-1]['pr_auc']:.4f} | "
                f"Time: {fold_metrics[-1]['seconds']:.1f}s"
            )

        if not np.isfinite(oof).all():
            raise ValueError(f"Incomplete OOF predictions for {name}.")

        predictions[name] = oof

        print(f"{name} OOF generation complete.")

    # Compare every classifier using the same repair-budget predictions.
    for name, probabilities in predictions.items():

        policy = tune_policy(
            probabilities=probabilities,
            actual_costs=train_df["repair_cost"],
            predicted_budgets=budgets,
        )

        flags = (probabilities >= policy["threshold"]).astype(int)

        scaled_budgets = np.where(
            flags == 1,
            budgets * policy["multiplier"],
            0.0,
        )

        cost = calculate_cost(
            repair_costs=train_df["repair_cost"],
            flags=flags,
            budgets=scaled_budgets,
        )

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y, probabilities),
            "pr_auc": average_precision_score(y, probabilities),
            "oof_cost": cost,
            "threshold": policy["threshold"],
            "budget_multiplier": policy["multiplier"],
            "inspections": int(flags.sum()),
            "inspection_rate": float(flags.mean()),
        })

        print(
            f"\n{name}: cost={cost:,.2f}, "
            f"ROC-AUC={results[-1]['roc_auc']:.4f}, "
            f"PR-AUC={results[-1]['pr_auc']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("oof_cost")

    return results_df, predictions
