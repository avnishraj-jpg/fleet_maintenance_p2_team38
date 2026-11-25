
"""Nested CV evaluation of vehicle-specific repair budgets.

Outer folds estimate generalisation performance.
Inner OOF predictions select the inspection threshold and budget multiplier.
The outer validation fold is never used for policy selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.cv_utils import PROJECT_ROOT, RESULTS_DIR, load_folds
from src.cost_model import fit_constant_budget
from src.evaluation import calculate_cost
from src.preprocessing import fit_preprocessor, transform_features
from src.feature_analysis import load_data
from src.lightgbm_model import build_final_lgbm, get_base_features


MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
INNER_SPLITS = 4
RANDOM_STATE = 42


def make_repair_models():
    """Create fresh repair-cost regressors."""
    return {
        "forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=2,
        ),
        "xgb": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=2,
        ),
        "lgbm": LGBMRegressor(
            n_estimators=100,
            num_leaves=8,
            learning_rate=0.05,
            verbosity=-1,
            random_state=42,
            n_jobs=2,
        ),
    }


def predict_budgets(train_part, valid_part, features):
    """Fit repair-budget models on training failures and predict validation."""

    # Fit preprocessing only on this training partition.
    X_train_raw = train_part[features]
    X_valid_raw = valid_part[features]

    prep = fit_preprocessor(X_train_raw, scale=True)

    X_train = transform_features(prep, X_train_raw)
    X_valid = transform_features(prep, X_valid_raw)

    known = (
        train_part["failure"].eq(1)
        & train_part["repair_cost"].notna()
    )

    repair_targets = train_part.loc[known, "repair_cost"]

    if len(repair_targets) == 0:
        raise ValueError("No known repair costs in a training partition.")

    predictions = {
        "constant": np.full(
            len(valid_part),
            fit_constant_budget(train_part),
        ),
        "median": np.full(
            len(valid_part),
            repair_targets.median(),
        ),
    }

    for name, model in make_repair_models().items():
        model.fit(X_train.loc[known], repair_targets)

        predictions[name] = np.maximum(
            model.predict(X_valid),
            0.0,
        )

    return predictions


def predict_failure_probabilities(X_train, y_train, X_valid):
    """Fit the final tuned failure classifier and predict probabilities."""
    model = build_final_lgbm()
    model.fit(X_train, y_train)

    return model.predict_proba(X_valid)[:, 1]


def tune_policy(probabilities, actual_costs, predicted_budgets):
    """Select threshold and budget multiplier using inner OOF predictions."""

    p = np.asarray(probabilities, dtype=float)
    actual = np.nan_to_num(
        np.asarray(actual_costs, dtype=float),
        nan=0.0,
    )
    budgets = np.maximum(
        np.asarray(predicted_budgets, dtype=float),
        0.0,
    )

    if not (
        len(p) == len(actual) == len(budgets)
    ):
        raise ValueError("Policy inputs have different lengths.")

    best = {
        "cost": np.inf,
        "threshold": None,
        "multiplier": None,
    }

    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]

    # Last index of each group of equal probabilities.
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_p) != 0),
        len(sorted_p) - 1,
    ]

    for multiplier in MULTIPLIERS:

        scaled_budgets = budgets * multiplier

        inspected_cost = (
            10.0
            + 0.20 * np.maximum(actual - scaled_budgets, 0.0)
            + 0.02 * np.maximum(scaled_budgets - actual, 0.0)
        )

        delta = inspected_cost - actual
        cumulative_delta = np.cumsum(delta[order])

        candidate_costs = actual.sum() + cumulative_delta[group_ends]

        # Include the option of inspecting nobody.
        no_inspection_cost = actual.sum()

        candidate_idx = int(np.argmin(candidate_costs))

        if candidate_costs[candidate_idx] < no_inspection_cost:
            threshold = float(sorted_p[group_ends[candidate_idx]])
            cost = float(candidate_costs[candidate_idx])
        else:
            threshold = float(np.nextafter(p.max(), np.inf))
            cost = float(no_inspection_cost)

        if cost < best["cost"]:
            best = {
                "cost": cost,
                "threshold": threshold,
                "multiplier": multiplier,
            }

    return best


def run_nested_evaluation():

    print("Loading training data and shared folds...")

    train_df, test_df = load_data()

    features = get_base_features(train_df, test_df)

    X = train_df[features].reset_index(drop=True)
    y = train_df["failure"].reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)

    outer_folds = load_folds()

    results = []

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(
        outer_folds, start=1
    ):

        print(f"\n{'=' * 55}")
        print(f"OUTER FOLD {outer_fold}/{len(outer_folds)}")
        print(f"{'=' * 55}")

        outer_train = train_df.iloc[outer_train_idx].reset_index(drop=True)
        outer_valid = train_df.iloc[outer_valid_idx].reset_index(drop=True)

        X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
        X_outer_valid = X.iloc[outer_valid_idx].reset_index(drop=True)

        y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)

        # ------------------------------------------------
        # 1. Generate inner OOF failure probabilities
        # ------------------------------------------------

        inner_probabilities = np.full(len(outer_train), np.nan)

        inner_budgets = {
            name: np.full(len(outer_train), np.nan)
            for name in ["constant", "median", "forest", "xgb", "lgbm"]
        }

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
            inner_cv.split(X_outer_train, y_outer_train),
            start=1,
        ):

            print(f"  Inner fold {inner_fold}/{INNER_SPLITS}")

            inner_train = outer_train.iloc[inner_train_idx]
            inner_valid = outer_train.iloc[inner_valid_idx]

            X_inner_train = X_outer_train.iloc[inner_train_idx]
            X_inner_valid = X_outer_train.iloc[inner_valid_idx]

            y_inner_train = y_outer_train.iloc[inner_train_idx]

            # Failure classifier: trained only on inner training rows.
            inner_probabilities[inner_valid_idx] = (
                predict_failure_probabilities(
                    X_inner_train,
                    y_inner_train,
                    X_inner_valid,
                )
            )

            # Repair-budget models: also trained only on inner training rows.
            budget_predictions = predict_budgets(
                inner_train,
                inner_valid,
                features,
            )

            for name, values in budget_predictions.items():
                inner_budgets[name][inner_valid_idx] = values

        if np.isnan(inner_probabilities).any():
            raise ValueError("Missing inner OOF failure probabilities.")

        for name, values in inner_budgets.items():
            if np.isnan(values).any():
                raise ValueError(f"Missing inner OOF budgets for {name}.")

        # ------------------------------------------------
        # 2. Select each strategy's policy using inner OOF
        # ------------------------------------------------

        policies = {}

        for name, budgets in inner_budgets.items():

            policies[name] = tune_policy(
                inner_probabilities,
                outer_train["repair_cost"],
                budgets,
            )

            print(
                f"  {name:>8}: "
                f"threshold={policies[name]['threshold']:.6f}, "
                f"multiplier={policies[name]['multiplier']}, "
                f"inner cost={policies[name]['cost']:,.2f}"
            )

        # ------------------------------------------------
        # 3. Fit on full outer training data
        # ------------------------------------------------

        print("  Fitting models on full outer training fold...")

        outer_probabilities = predict_failure_probabilities(
            X_outer_train,
            y_outer_train,
            X_outer_valid,
        )

        outer_budgets = predict_budgets(
            outer_train,
            outer_valid,
            features,
        )

        # ------------------------------------------------
        # 4. Evaluate policies on untouched outer fold
        # ------------------------------------------------

        for name, budgets in outer_budgets.items():

            policy = policies[name]

            threshold = policy["threshold"]
            multiplier = policy["multiplier"]

            flags = (
                outer_probabilities >= threshold
            ).astype(int)

            final_budgets = np.where(
                flags == 1,
                budgets * multiplier,
                0.0,
            )

            total_cost = calculate_cost(
                outer_valid["repair_cost"],
                flags,
                final_budgets,
            )

            row = {
                "fold": outer_fold,
                "strategy": name,
                "threshold": threshold,
                "multiplier": multiplier,
                "inspected": int(flags.sum()),
                "total_cost": total_cost,
                "vehicles": len(outer_valid),
            }

            results.append(row)

            print(
                f"  OUTER {name:>8}: "
                f"inspected={flags.sum():>4}, "
                f"cost={total_cost:,.2f}"
            )

    # ------------------------------------------------
    # 5. Aggregate outer-fold results
    # ------------------------------------------------

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby("strategy")
        .agg(
            total_cost=("total_cost", "sum"),
            total_inspected=("inspected", "sum"),
            vehicles=("vehicles", "sum"),
            mean_fold_cost=("total_cost", "mean"),
            std_fold_cost=("total_cost", "std"),
        )
        .sort_values("total_cost")
    )

    summary["avg_cost_per_vehicle"] = (
        summary["total_cost"] / summary["vehicles"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        RESULTS_DIR / "nested_budget_fold_results.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "nested_budget_comparison.csv",
    )

    print("\n" + "=" * 65)
    print("NESTED CV RESULTS")
    print("=" * 65)

    print(summary.round(3).to_string())

    print("\nSaved:")
    print("results/nested_budget_fold_results.csv")
    print("results/nested_budget_comparison.csv")

    return results_df, summary


if __name__ == "__main__":
    run_nested_evaluation()
