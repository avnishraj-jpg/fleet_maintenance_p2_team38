"""L1 feature selection and preprocessing comparison (Member 2 - Mohd Farhan Ali).

Uses the shared 5-fold CV indices. Every imputer and selector is a Pipeline
step, so it is fitted on the training part of each fold only. Cost: one
threshold on the pooled OOF probabilities with the constant budget (the V2
LightGBM protocol). Results are saved in results/.

Run: python -m src.feature_selection [imputation | l1 | lgbm | features | all]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, RESULTS_DIR, TUNED_LGBM_PARAMS
from src.cost_model import fit_constant_budget
from src.evaluation import evaluate_predictions, tune_policy
from src.preprocessing import PREPROCESSING_EXPERIMENTS, SensorPreprocessor, get_sensor_columns

LR_CLASS_WEIGHT = {0: 1, 1: 50}  # V1 Logistic Regression weight 1:50

# Smaller C = fewer features. C >= 0.1 keeps almost all sensors and is very slow.
L1_C_VALUES = (0.0005, 0.001, 0.003, 0.01, 0.03)

L1_PREPROCESSING = dict(imputation="median", scale=True)  # L1 needs scaled inputs

COST_NOISE = 3000.0  # OOF cost differences below this are seed noise (README)

_MISSING_SUFFIX = "_missing"


# --------------------------------------------------
# L1 Logistic Regression
# --------------------------------------------------

def _sklearn_at_least(major: int, minor: int) -> bool:
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


def build_lr_model(class_weight=LR_CLASS_WEIGHT, random_state: int = RANDOM_STATE) -> LogisticRegression:
    """V1 Logistic Regression (same settings as the V1 failure model)."""
    return LogisticRegression(class_weight=class_weight, max_iter=2000, random_state=random_state)


def build_l1_model(C: float = 0.01, class_weight=LR_CLASS_WEIGHT,
                   random_state: int = RANDOM_STATE) -> LogisticRegression:
    """L1 Logistic Regression. scikit-learn >= 1.8 uses l1_ratio=1, older versions penalty="l1"."""
    params = dict(C=C, solver="liblinear", class_weight=class_weight,
                  max_iter=2000, random_state=random_state)
    if _sklearn_at_least(1, 8):
        params["l1_ratio"] = 1.0
    else:
        params["penalty"] = "l1"
    return LogisticRegression(**params)


class L1FeatureSelector(BaseEstimator, TransformerMixin):
    """Keep sensors with a non-zero L1 Logistic Regression coefficient.

    transform() returns the selected raw columns (missing values kept), so the
    next Pipeline step does its own preprocessing.
    """

    def __init__(self, C=0.01, class_weight=LR_CLASS_WEIGHT, imputation="median",
                 missing_indicators=None, log_transform=False, tol=1e-10):
        self.C = C
        self.class_weight = class_weight
        self.imputation = imputation
        self.missing_indicators = missing_indicators
        self.log_transform = log_transform
        self.tol = tol

    def fit(self, X, y):
        sensors = get_sensor_columns(X)
        self.preprocess_ = SensorPreprocessor(
            imputation=self.imputation,
            missing_indicators=self.missing_indicators,
            log_transform=self.log_transform,
            scale=True,
        ).fit(X[sensors])
        Z = self.preprocess_.transform(X[sensors])

        self.model_ = build_l1_model(C=self.C, class_weight=self.class_weight).fit(Z, y)
        self.coef_ = pd.Series(self.model_.coef_.ravel(), index=Z.columns)

        nonzero = self.coef_.index[self.coef_.abs() > self.tol]
        keep = {c[: -len(_MISSING_SUFFIX)] if c.endswith(_MISSING_SUFFIX) else c for c in nonzero}

        candidates = self.preprocess_.preprocessor_["feature_cols"]
        self.selected_features_ = [c for c in candidates if c in keep]
        self.zero_coef_features_ = [c for c in candidates if c not in keep]
        self.dropped_before_selection_ = list(self.preprocess_.preprocessor_["dropped_cols"])

        if not self.selected_features_:
            raise ValueError(f"L1 with C={self.C} kept no features; use a larger C.")
        return self

    def transform(self, X):
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)


# --------------------------------------------------
# Pipelines (cloned and fitted per CV fold)
# --------------------------------------------------

def build_lr_pipeline(**preprocessing) -> Pipeline:
    """V1 Logistic Regression (1:50) with the given preprocessing options."""
    options = {"scale": True, **preprocessing}
    return Pipeline([
        ("preprocess", SensorPreprocessor(**options)),
        ("model", build_lr_model()),
    ])


def build_l1_lr(C: float = 0.01, **preprocessing) -> Pipeline:
    """L1 Logistic Regression used directly as the failure classifier."""
    options = {**L1_PREPROCESSING, **preprocessing}
    return Pipeline([
        ("preprocess", SensorPreprocessor(**options)),
        ("model", build_l1_model(C=C)),
    ])


def build_l1_selected_lr(C: float = 0.01, **preprocessing) -> Pipeline:
    """L1 selection inside the fold, then the V1 Logistic Regression on the kept sensors."""
    options = {"scale": True, **preprocessing}
    return Pipeline([
        ("select", L1FeatureSelector(C=C)),
        ("preprocess", SensorPreprocessor(**options)),
        ("model", build_lr_model()),
    ])


def build_l1_selected_lgbm(C: float = 0.01, **lgbm_params) -> Pipeline:
    """L1 selection inside the fold, then LightGBM on the raw selected sensors."""
    from src.lightgbm_model import build_lgbm

    return Pipeline([
        ("select", L1FeatureSelector(C=C)),
        ("model", build_lgbm(**{**TUNED_LGBM_PARAMS, **lgbm_params})),
    ])


def build_all_features_lgbm(**lgbm_params) -> Pipeline:
    """Tuned LightGBM (single seed) on all base features, for a like-for-like comparison."""
    from src.lightgbm_model import build_lgbm

    return Pipeline([("model", build_lgbm(**{**TUNED_LGBM_PARAMS, **lgbm_params}))])


# --------------------------------------------------
# Selection summaries
# --------------------------------------------------

def fold_selection_table(cv_result: dict, step: str = "select") -> pd.DataFrame:
    """Per-fold number of selected / zero-coefficient sensors."""
    rows = []
    for k, model in enumerate(cv_result["models"], start=1):
        sel = model.named_steps[step]
        rows.append({"fold": k, "C": sel.C, "n_selected": len(sel.selected_features_),
                     "n_zero_coef": len(sel.zero_coef_features_)})
    return pd.DataFrame(rows)


def selection_frequency(cv_result: dict, step: str = "select") -> pd.Series:
    """How many of the folds selected each sensor (0 to number of folds)."""
    selectors = [m.named_steps[step] for m in cv_result["models"]]
    candidates = selectors[0].selected_features_ + selectors[0].zero_coef_features_
    counts = pd.Series(0, index=sorted(candidates), dtype=int)
    for sel in selectors:
        counts[sel.selected_features_] += 1
    return counts


def mean_pairwise_jaccard(cv_result: dict, step: str = "select") -> float:
    """Average overlap of the selected sets between folds (1 = identical)."""
    sets = [set(m.named_steps[step].selected_features_) for m in cv_result["models"]]
    scores = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(scores))


def final_feature_table(X: pd.DataFrame, y: pd.Series, C: float,
                        fold_counts: pd.Series | None = None) -> pd.DataFrame:
    """Fit L1 on all training data: coefficient, selected flag and folds selected per sensor."""
    selector = L1FeatureSelector(C=C).fit(X, y)
    sensor_coef = selector.coef_[[c for c in selector.coef_.index if not c.endswith(_MISSING_SUFFIX)]]
    table = pd.DataFrame({
        "feature": sensor_coef.index,
        "l1_coefficient": sensor_coef.values,
        "selected": sensor_coef.index.isin(selector.selected_features_),
    })
    if fold_counts is not None:
        table["folds_selected"] = table["feature"].map(fold_counts).fillna(0).astype(int)
    table["abs_coefficient"] = table["l1_coefficient"].abs()
    table = table.sort_values(["selected", "abs_coefficient"], ascending=[False, False])
    return table.drop(columns="abs_coefficient").reset_index(drop=True)


# --------------------------------------------------
# Experiment runners
# --------------------------------------------------

def evaluate_cost(cv_result: dict, train_df: pd.DataFrame) -> dict:
    """Official cost of pooled OOF probabilities at the cost-minimising threshold."""
    oof = cv_result["oof"]
    repair = train_df["repair_cost"].reset_index(drop=True)
    y = train_df["failure"].reset_index(drop=True)

    budget = fit_constant_budget(train_df)
    policy = tune_policy(oof, repair, np.full(len(oof), budget), multipliers=(1.0,))
    flags = (oof >= policy["threshold"]).astype(int)
    m = evaluate_predictions(y, oof, repair, flags, np.where(flags == 1, budget, 0.0), policy["threshold"])

    folds = cv_result["folds"]
    return {
        "model": cv_result["name"],
        "oof_cost": m["total_cost"],
        "cost_per_1000_vehicles": 1000 * m["total_cost"] / len(oof),
        "threshold": policy["threshold"],
        "flagged": m["vehicles_inspected"],
        "tp": m["tp"], "fn": m["fn"], "fp": m["fp"],
        "recall": m["recall"], "precision": m["precision"],
        "roc_auc_mean": folds["roc_auc"].mean(), "roc_auc_std": folds["roc_auc"].std(),
        "pr_auc_mean": folds["pr_auc"].mean(), "pr_auc_std": folds["pr_auc"].std(),
    }


def _evaluate(model, X, y, folds, train_df, name: str) -> dict:
    """Shared-fold CV (lightgbm_model.run_cv) followed by the official cost."""
    from src.lightgbm_model import run_cv

    start = time.time()
    cv = run_cv(model, X, y, folds, name=name, verbose=False)
    s = evaluate_cost(cv, train_df)
    s["seconds"] = round(time.time() - start, 1)
    print(f"{name:45s} OOF cost {s['oof_cost']:>10,.0f} | ROC-AUC {s['roc_auc_mean']:.4f} "
          f"| flagged {s['flagged']:>5} | {s['seconds']:.0f}s")
    return {"cv": cv, "summary": s}


def _table(evaluations: list, extra: dict | None = None) -> pd.DataFrame:
    table = pd.DataFrame([e["summary"] for e in evaluations]).set_index("model").sort_values("oof_cost")
    if extra:
        for col, values in extra.items():
            table[col] = table.index.map(values)
    best = table["oof_cost"].min()
    table["vs_best"] = table["oof_cost"] - best
    table["within_noise_of_best"] = table["vs_best"] < COST_NOISE
    return table


def run_imputation_comparison(X, y, folds, train_df, include_knn: bool = False) -> pd.DataFrame:
    """Compare preprocessing settings with the V1 Logistic Regression."""
    print("\n=== Imputation / missingness-indicator comparison (LR 1:50) ===")
    configs = dict(PREPROCESSING_EXPERIMENTS)
    if include_knn:
        configs["knn (k=5)"] = dict(imputation="knn", knn_neighbors=5, scale=True)
    results = [_evaluate(build_lr_pipeline(**opts), X, y, folds, train_df, name)
               for name, opts in configs.items()]
    n_features = {}
    for name, r in zip(configs, results):
        n_features[name] = len(r["cv"]["models"][0].named_steps["preprocess"].get_feature_names_out())
    table = _table(results, {"n_model_inputs": n_features})
    table.to_csv(RESULTS_DIR / "preprocessing_cv_comparison.csv")
    return table


def run_l1_comparison(X, y, folds, train_df, C_values=L1_C_VALUES):
    """All features vs L1-selected features, for a range of C."""
    print("\n=== L1 feature selection (selector fitted inside each training fold) ===")
    results = [_evaluate(build_lr_pipeline(imputation="median"), X, y, folds, train_df,
                         "LR all features (benchmark)")]
    benchmark_prep = results[0]["cv"]["models"][0].named_steps["preprocess"].preprocessor_
    n_selected = {"LR all features (benchmark)": float(len(benchmark_prep["feature_cols"]))}
    fold_tables, selected_runs = [], {}

    for C in C_values:
        l1 = _evaluate(build_l1_lr(C=C), X, y, folds, train_df, f"L1 LR (C={C})")
        sel = _evaluate(build_l1_selected_lr(C=C), X, y, folds, train_df, f"LR on L1-selected (C={C})")
        results += [l1, sel]

        ft = fold_selection_table(sel["cv"])
        ft["jaccard_mean"] = mean_pairwise_jaccard(sel["cv"])
        fold_tables.append(ft)
        n_selected[f"L1 LR (C={C})"] = ft["n_selected"].mean()
        n_selected[f"LR on L1-selected (C={C})"] = ft["n_selected"].mean()
        selected_runs[C] = sel

    table = _table(results, {"mean_features_used": n_selected})
    table.to_csv(RESULTS_DIR / "l1_cv_comparison.csv")
    folds_df = pd.concat(fold_tables, ignore_index=True)
    folds_df.to_csv(RESULTS_DIR / "l1_fold_selection.csv", index=False)
    return table, folds_df, selected_runs


def choose_C(l1_table: pd.DataFrame) -> float:
    """Smallest feature set whose 'LR on L1-selected' cost is within noise of the best one."""
    rows = l1_table[l1_table.index.str.startswith("LR on L1-selected")].copy()
    rows["C"] = rows.index.str.extract(r"C=([0-9.]+)", expand=False).astype(float)
    best = rows["oof_cost"].min()
    ok = rows[rows["oof_cost"] - best < COST_NOISE]
    return float(ok.sort_values(["mean_features_used", "oof_cost"]).iloc[0]["C"])


def run_lgbm_comparison(X, y, folds, train_df, C: float) -> pd.DataFrame:
    """Tuned LightGBM (one seed) on all features vs on L1-selected features."""
    print(f"\n=== LightGBM: all features vs L1-selected (C={C}) ===")
    results = [
        _evaluate(build_all_features_lgbm(), X, y, folds, train_df, "LightGBM all features"),
        _evaluate(build_l1_selected_lgbm(C=C), X, y, folds, train_df, f"LightGBM on L1-selected (C={C})"),
    ]
    table = _table(results)
    table.to_csv(RESULTS_DIR / "l1_lgbm_comparison.csv")
    return table


def save_feature_lists(X, y, C: float, selected_run: dict | None = None) -> pd.DataFrame:
    """Write the final selected / zero-coefficient feature lists."""
    counts = selection_frequency(selected_run["cv"]) if selected_run else None
    table = final_feature_table(X, y, C, counts)
    table.insert(0, "C", C)
    table.to_csv(RESULTS_DIR / "l1_selected_features.csv", index=False)
    n_sel = int(table["selected"].sum())
    print(f"\nC={C}: {n_sel} selected, {len(table) - n_sel} zero-coefficient sensors "
          f"-> results/l1_selected_features.csv")
    return table


def _load():
    from src.cv_utils import load_folds, validate_folds
    from src.feature_analysis import load_data
    from src.lightgbm_model import get_base_features

    train_df, test_df = load_data()
    train_df = train_df.reset_index(drop=True)
    features = get_base_features(train_df, test_df)
    folds = load_folds()
    validate_folds(folds, len(train_df))
    return train_df, train_df[features], train_df["failure"], folds


def main():
    parser = argparse.ArgumentParser(description="Preprocessing and L1 feature-selection experiments.")
    parser.add_argument("part", nargs="?", default="all",
                        choices=["imputation", "l1", "lgbm", "features", "all"])
    parser.add_argument("--C", type=float, default=None, help="L1 strength for lgbm/features (default: chosen)")
    parser.add_argument("--knn", action="store_true", help="also run KNN imputation (slow)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, X, y, folds = _load()
    print(f"Training vehicles: {len(X):,} | base features: {X.shape[1]} | folds: {len(folds)}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    show = ["oof_cost", "vs_best", "within_noise_of_best", "flagged", "tp", "fp", "roc_auc_mean", "pr_auc_mean"]

    if args.part in ("imputation", "all"):
        table = run_imputation_comparison(X, y, folds, train_df, include_knn=args.knn)
        print(table[["n_model_inputs"] + show].round(4).to_string())

    C, selected_run = args.C, None
    if args.part in ("l1", "all"):
        l1_table, _, runs = run_l1_comparison(X, y, folds, train_df)
        print(l1_table[["mean_features_used"] + show].round(4).to_string())
        if C is None:
            C = choose_C(l1_table)
            print(f"\nChosen C = {C} (fewest features within {COST_NOISE:,.0f} of the best selected-feature cost)")
        selected_run = runs.get(C)

    if C is None:
        C = 0.01
    if args.part in ("lgbm", "all"):
        print(run_lgbm_comparison(X, y, folds, train_df, C)[show].round(4).to_string())

    if args.part in ("features", "l1", "all"):
        save_feature_lists(X, y, C, selected_run)


if __name__ == "__main__":
    main()
