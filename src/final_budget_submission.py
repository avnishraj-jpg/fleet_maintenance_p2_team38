
"""Final LightGBM failure + repair-budget submission pipeline."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.cv_utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    load_folds,
    validate_folds,
)
from src.feature_analysis import load_data, align_test_sensors
from src.lightgbm_model import (
    build_final_lgbm,
    get_base_features,
    run_cv,
)
from src.evaluation import tune_policy
from src.preprocessing import fit_preprocessor, transform_features


SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def build_repair_regressor():
    """Same LightGBM repair model used in nested CV."""
    return LGBMRegressor(
        n_estimators=100,
        num_leaves=8,
        learning_rate=0.05,
        verbosity=-1,
        random_state=42,
        n_jobs=2,
    )


def generate_oof_repair_budgets(train_df, features, folds):
    """Generate leakage-safe OOF repair-budget predictions."""

    oof_budgets = np.full(len(train_df), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"Repair-budget OOF fold {fold}/{len(folds)}")

        train_part = train_df.iloc[train_idx].reset_index(drop=True)
        valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

        # Fit preprocessing on this fold's training data only.
        prep = fit_preprocessor(
            train_part[features],
            scale=True,
        )

        X_train = transform_features(
            prep,
            train_part[features],
        )

        X_valid = transform_features(
            prep,
            valid_part[features],
        )

        # Only failed vehicles with known repair costs train the regressor.
        known = (
            train_part["failure"].eq(1)
            & train_part["repair_cost"].notna()
        )

        y_repair = train_part.loc[known, "repair_cost"]

        if len(y_repair) == 0:
            raise ValueError(f"No known repair costs in fold {fold}.")

        model = build_repair_regressor()

        model.fit(
            X_train.loc[known],
            y_repair,
        )

        predictions = np.maximum(
            model.predict(X_valid),
            0.0,
        )

        oof_budgets[valid_idx] = predictions

    if not np.isfinite(oof_budgets).all():
        raise ValueError("OOF repair-budget predictions are incomplete.")

    return oof_budgets


def main():

    print("=" * 65)
    print("FINAL LIGHTGBM REPAIR-BUDGET PIPELINE")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Load data and shared folds
    # --------------------------------------------------

    train_df, test_df = load_data()

    train_df = train_df.reset_index(drop=True)

    features = get_base_features(train_df, test_df)

    X = train_df[features]
    y = train_df["failure"]

    folds = load_folds()
    validate_folds(folds, len(X))

    print(f"\nTraining vehicles: {len(train_df):,}")
    print(f"Test vehicles: {len(test_df):,}")
    print(f"Features: {len(features)}")
    print(f"CV folds: {len(folds)}")

    # --------------------------------------------------
    # 2. Generate OOF failure probabilities
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("GENERATING FAILURE PROBABILITIES")
    print("-" * 65)

    failure_cv = run_cv(
        build_final_lgbm(),
        X,
        y,
        folds,
        name="Final tuned 5-seed LightGBM",
        verbose=True,
    )

    oof_probabilities = failure_cv["oof"]

    del failure_cv
    gc.collect()

    # --------------------------------------------------
    # 3. Generate OOF repair-budget predictions
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("GENERATING REPAIR-BUDGET PREDICTIONS")
    print("-" * 65)

    oof_budgets = generate_oof_repair_budgets(
        train_df,
        features,
        folds,
    )

    # --------------------------------------------------
    # 4. Select threshold and budget multiplier
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("SELECTING FINAL INSPECTION POLICY")
    print("-" * 65)

    policy = tune_policy(
        probabilities=oof_probabilities,
        actual_costs=train_df["repair_cost"],
        predicted_budgets=oof_budgets,
    )

    threshold = policy["threshold"]
    multiplier = policy["multiplier"]

    print(f"Selected threshold: {threshold:.8f}")
    print(f"Selected multiplier: {multiplier}")
    print(f"OOF policy-selection cost: {policy['cost']:,.2f}")

    print(
        "\nNote: This OOF policy-selection cost is optimised on these "
        "predictions. Use nested CV as the generalisation estimate."
    )

    # Save policy and OOF predictions for reproducibility.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([policy]).to_csv(
        RESULTS_DIR / "final_budget_policy.csv",
        index=False,
    )

    pd.DataFrame({
        "id": train_df["id"],
        "failure": train_df["failure"],
        "repair_cost": train_df["repair_cost"],
        "oof_failure_probability": oof_probabilities,
        "oof_repair_budget": oof_budgets,
    }).to_csv(
        RESULTS_DIR / "final_budget_oof_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Align test sensors
    # --------------------------------------------------

    test_aligned = align_test_sensors(
        test_df,
        train_df,
    )

    X_test = test_aligned[features]

    # --------------------------------------------------
    # 6. Fit final failure classifier on all training data
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("FITTING FINAL FAILURE CLASSIFIER")
    print("-" * 65)

    failure_model = build_final_lgbm()
    failure_model.fit(X, y)

    test_probabilities = failure_model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # 7. Fit final repair-budget regressor on all known failures
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("FITTING FINAL REPAIR-BUDGET MODEL")
    print("-" * 65)

    prep = fit_preprocessor(
        train_df[features],
        scale=True,
    )

    X_train_repair = transform_features(
        prep,
        train_df[features],
    )

    X_test_repair = transform_features(
        prep,
        X_test,
    )

    known = (
        train_df["failure"].eq(1)
        & train_df["repair_cost"].notna()
    )

    repair_model = build_repair_regressor()

    repair_model.fit(
        X_train_repair.loc[known],
        train_df.loc[known, "repair_cost"],
    )

    test_repair_budgets = np.maximum(
        repair_model.predict(X_test_repair),
        0.0,
    )

    # --------------------------------------------------
    # 8. Generate final submission
    # --------------------------------------------------

    flags = (
        test_probabilities >= threshold
    ).astype(int)

    budgets = np.where(
        flags == 1,
        test_repair_budgets * multiplier,
        0.0,
    )

    submission = pd.DataFrame({
        "id": test_df["id"].to_numpy(),
        "flag": flags,
        "budget": budgets,
    })

    # --------------------------------------------------
    # 9. Validate submission
    # --------------------------------------------------

    assert len(submission) == len(test_df)
    assert submission["id"].is_unique
    assert submission["id"].equals(test_df["id"])
    assert submission["flag"].isin([0, 1]).all()
    assert np.isfinite(submission["budget"]).all()
    assert (submission["budget"] >= 0).all()
    assert submission.notna().all().all()

    sample_path = PROJECT_ROOT / "data" / "sample_submission.csv"

    if sample_path.exists():
        sample = pd.read_csv(sample_path)

        assert list(submission.columns) == list(sample.columns)
        assert submission["id"].equals(sample["id"])

    # --------------------------------------------------
    # 10. Save
    # --------------------------------------------------

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        SUBMISSIONS_DIR / "v3_lgbm_repair_budget.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("SUBMISSION GENERATED SUCCESSFULLY")
    print("=" * 65)

    print(f"File: {output_path}")
    print(f"Vehicles: {len(submission):,}")
    print(f"Inspections: {int(flags.sum()):,}")
    print(f"Inspection rate: {flags.mean():.2%}")
    print(f"Mean predicted failure probability: {test_probabilities.mean():.6f}")
    print(f"Mean budget among inspected: "
          f"{budgets[flags == 1].mean() if flags.sum() else 0:.2f}")

    print("\nSubmission validation passed.")

    return submission


if __name__ == "__main__":
    main()
