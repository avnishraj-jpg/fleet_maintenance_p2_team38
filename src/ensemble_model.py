"""V3 failure-probability ensemble for Fleet Maintenance Planning.

Three gradient-boosting families are trained on the same data:

- CatBoost
- XGBoost
- HistGradientBoosting (scikit-learn)

Each family is trained once per seed in SEEDS and its probabilities are
averaged over the seeds. The ensemble probability is the equal-weight average
of the three family probabilities:

    P_ensemble = (P_CatBoost + P_XGBoost + P_HGB) / 3

Interface
---------
FailureProbabilityEnsemble is a scikit-learn compatible classifier:

- fit(X, y) trains every family and seed on the supplied training data.
- predict_proba(X) returns the usual two-column array
  [P(no failure), P(failure)], so it works unchanged with
  lightgbm_model.run_cv and cv_utils.get_prediction_probabilities, which read
  predict_proba(X)[:, 1].
- predict_failure_probability(X) returns the one-dimensional failure
  probabilities.
- predict_family_proba(X) returns the seed-averaged probability of each family,
  for comparing the individual models with the ensemble.

All three libraries handle missing values natively, so the 167 base features
are used without imputation (missingness is predictive; see
notebooks/eda.ipynb, Section 6.1). Predictions keep the row order of X.

This module contains no data loading or preprocessing. Running it as a script
gives an independent fitting and prediction example on the shared CV folds:

    python -m src.ensemble_model
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import RANDOM_STATE

SEEDS = (42, 7, 123)

CATBOOST_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5,
}

XGBOOST_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}

HIST_GRADIENT_BOOSTING_PARAMS = {
    "max_iter": 600,
    "learning_rate": 0.04,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


# --------------------------------------------------
# Model builders
# --------------------------------------------------

def build_catboost(seed: int = RANDOM_STATE):
    """CatBoost classifier with the V3 configuration."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        **CATBOOST_PARAMS,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def build_xgboost(seed: int = RANDOM_STATE):
    """XGBoost classifier with the V3 configuration."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        **XGBOOST_PARAMS,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_hist_gradient_boosting(seed: int = RANDOM_STATE):
    """HistGradientBoosting classifier with the V3 configuration."""
    return HistGradientBoostingClassifier(
        **HIST_GRADIENT_BOOSTING_PARAMS,
        random_state=seed,
    )


MODEL_BUILDERS = {
    "catboost": build_catboost,
    "xgboost": build_xgboost,
    "hist_gradient_boosting": build_hist_gradient_boosting,
}


# --------------------------------------------------
# Ensemble
# --------------------------------------------------

def _positive_class_probability(model, X) -> np.ndarray:
    """Probability of class 1 from a fitted binary classifier."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


class FailureProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Seed-averaged CatBoost, XGBoost and HistGradientBoosting ensemble.

    Parameters
    ----------
    families : tuple of str
        Model families to include (keys of MODEL_BUILDERS). The default uses
        all three; a single family gives that model on its own.
    seeds : tuple of int
        Random seeds; every family is trained once per seed.
    """

    def __init__(self, families=tuple(MODEL_BUILDERS), seeds=SEEDS):
        self.families = families
        self.seeds = seeds

    def fit(self, X, y):
        unknown = [f for f in self.families if f not in MODEL_BUILDERS]
        if unknown:
            raise ValueError(f"Unknown model families: {unknown}")
        if len(self.families) == 0 or len(self.seeds) == 0:
            raise ValueError("At least one family and one seed are required.")

        y = np.asarray(y).astype(int)
        if not set(np.unique(y)) <= {0, 1} or len(np.unique(y)) < 2:
            raise ValueError("y must be binary and contain both classes 0 and 1.")
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows.")

        self.models_ = {}
        self.fit_seconds_ = {}
        for family in self.families:
            start = time.time()
            self.models_[family] = [MODEL_BUILDERS[family](seed).fit(X, y) for seed in self.seeds]
            self.fit_seconds_[family] = time.time() - start

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _check_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("Feature columns differ from those used in fit().")

    def predict_family_proba(self, X) -> pd.DataFrame:
        """Seed-averaged failure probability of each family (one column per family)."""
        self._check_features(X)
        return pd.DataFrame({
            family: np.mean([_positive_class_probability(m, X) for m in models], axis=0)
            for family, models in self.models_.items()
        })

    def predict_failure_probability(self, X) -> np.ndarray:
        """Ensemble failure probability: equal-weight mean of the family probabilities."""
        p = self.predict_family_proba(X).mean(axis=1).to_numpy()
        return np.clip(p, 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        """Two-column probabilities [P(no failure), P(failure)]."""
        p = self.predict_failure_probability(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        """Class labels at a 0.5 threshold (cost-based thresholds are tuned elsewhere)."""
        return (self.predict_failure_probability(X) >= 0.5).astype(int)


# --------------------------------------------------
# Comparison helpers
# --------------------------------------------------

def family_oof_probabilities(cv_result: dict, X: pd.DataFrame, folds) -> pd.DataFrame:
    """OOF probabilities of each family and of the ensemble, from one CV run.

    cv_result is the output of lightgbm_model.run_cv for a
    FailureProbabilityEnsemble. The fitted fold models are reused, so the
    individual families are compared without retraining.
    """
    X = X.reset_index(drop=True)
    families = list(cv_result["models"][0].models_)
    oof = pd.DataFrame(np.nan, index=X.index, columns=families + ["ensemble"])
    for model, (_, valid_idx) in zip(cv_result["models"], folds):
        family_p = model.predict_family_proba(X.iloc[valid_idx])
        oof.loc[valid_idx, families] = family_p.to_numpy()
        oof.loc[valid_idx, "ensemble"] = family_p.mean(axis=1).to_numpy()
    if oof.isna().any().any():
        raise ValueError("Some samples are missing OOF predictions.")
    return oof


def compare_oof_probabilities(oof: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC, PR-AUC and official cost (V2 constant-budget policy) per column.

    The cost uses evaluation.select_threshold with the constant budget from
    cost_model.fit_constant_budget, i.e. the same yardstick as the V2 results.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.cost_model import fit_constant_budget
    from src.evaluation import select_threshold

    y = train_df["failure"].to_numpy()
    repair = train_df["repair_cost"].to_numpy()
    budget = fit_constant_budget(train_df)
    rows = []
    for name in oof.columns:
        p = oof[name].to_numpy()
        threshold, cost = select_threshold(p, repair, budget)
        flags = p >= threshold
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "oof_cost": cost,
            "threshold": threshold,
            "flagged": int(flags.sum()),
            "tp": int((flags & (y == 1)).sum()),
            "fp": int((flags & (y == 0)).sum()),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("oof_cost")


# --------------------------------------------------
# Independent fitting and prediction example
# --------------------------------------------------

def main():
    from src.cv_utils import load_folds, prepare_training_data
    from src.feature_analysis import align_test_sensors, load_data
    from src.lightgbm_model import build_final_lgbm, get_base_features, run_cv

    train_df, test_df = load_data()
    features = get_base_features(train_df, test_df)
    X_all, y = prepare_training_data(train_df)
    X = X_all[features]
    X_test = align_test_sensors(test_df, train_df)[features]
    folds = load_folds()
    print(f"Training rows: {len(X):,} | test rows: {len(X_test):,} | features: {len(features)}")

    # 1. OOF predictions on the shared folds
    start = time.time()
    ensemble_cv = run_cv(FailureProbabilityEnsemble(), X, y, folds, name="V3 ensemble")
    cv_seconds = time.time() - start
    oof = family_oof_probabilities(ensemble_cv, X, folds)

    v2_cv = run_cv(build_final_lgbm(), X, y, folds, name="V2 LightGBM", verbose=False)
    oof["v2_lightgbm"] = v2_cv["oof"]

    # 2. Final fit on all training data and test probabilities
    start = time.time()
    final = FailureProbabilityEnsemble().fit(X, y)
    fit_seconds = time.time() - start
    test_p = final.predict_failure_probability(X_test)

    # 3. Checks
    assert np.allclose(oof["ensemble"], ensemble_cv["oof"]), "Ensemble OOF mismatch"
    assert ((oof >= 0) & (oof <= 1)).all().all(), "OOF probabilities outside [0, 1]"
    assert ((test_p >= 0) & (test_p <= 1)).all(), "Test probabilities outside [0, 1]"
    assert len(test_p) == len(test_df), "Test row count changed"
    shuffled = X_test.sample(frac=1.0, random_state=0)
    assert np.allclose(final.predict_failure_probability(shuffled),
                       pd.Series(test_p, index=X_test.index).loc[shuffled.index]), "Row order not preserved"
    print("Checks passed: probabilities in [0, 1], row order preserved.")

    # 4. Results
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 200):
        print("\nOOF comparison (shared 5-fold CV, V2 constant-budget policy):")
        print(compare_oof_probabilities(oof, train_df).to_string())
        print("\nEnsemble fold metrics:")
        print(ensemble_cv["folds"][["fold", "roc_auc", "pr_auc", "fit_seconds"]].to_string(index=False))
        print(f"\nRuntime: CV {cv_seconds / 60:.1f} min | final fit {fit_seconds / 60:.1f} min "
              f"(per family: {', '.join(f'{k} {v / 60:.1f} min' for k, v in final.fit_seconds_.items())})")
        print(f"Test probabilities: mean {test_p.mean():.4f}, "
              f"share >= 0.5: {(test_p >= 0.5).mean():.4f}")


if __name__ == "__main__":
    main()
