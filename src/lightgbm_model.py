"""LightGBM failure model for Fleet Maintenance Planning.

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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TUNED_LGBM_PARAMS, FINAL_SEEDS
from src.cost_model import fit_constant_budget, predict_budgets
from src.cv_utils import (
    get_prediction_probabilities,
    load_folds,
    prepare_training_data,
    validate_folds,
)
from src.evaluation import evaluate_predictions
from src.failure_model import CHOSEN_CONFIG, build_model
from src.feature_analysis import (
    align_test_sensors,
    find_correlated_groups,
    find_low_variance_features,
    find_unavailable_in_test,
    load_data,
)
from src.preprocessing import CONSTANT_SENSORS, fit_preprocessor, transform_features


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
    """V1 model: median imputation + scaling + LR with class weight 1:50."""
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
# Fold-wise feature selection (task 7)
# --------------------------------------------------

class FoldFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection fitted on the training fold only.

    Inside a Pipeline that run_cv clones per fold, `fit` only ever sees the
    training part of a fold, so the validation fold never influences which
    features are kept. Steps, in order:

    1. Low-variance filter (unsupervised): drop constant and quasi-constant
       sensors (one value in >= 99% of observed rows).
    2. Correlation filter (unsupervised): within each group of sensors with
       |Spearman rho| > corr_threshold keep one (the one with the fewest
       missing values).
    3. Importance ranking (supervised): fit a LightGBM model on the remaining
       sensors and keep the top_k by total gain.

    Set corr_threshold=None or top_k=None to skip a step.
    """

    def __init__(self, drop_low_variance=True, corr_threshold=0.95, top_k=None, ranking_params=None):
        self.drop_low_variance = drop_low_variance
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.ranking_params = ranking_params

    def fit(self, X, y):
        cols = list(X.columns)
        self.dropped_ = {}

        if self.drop_low_variance:
            lv = find_low_variance_features(X[cols])
            self.dropped_["low_variance"] = lv["constant"] + lv["quasi_constant"]
            cols = [c for c in cols if c not in self.dropped_["low_variance"]]

        if self.corr_threshold is not None:
            _, to_drop = find_correlated_groups(X[cols], threshold=self.corr_threshold)
            self.dropped_["correlated"] = to_drop
            cols = [c for c in cols if c not in to_drop]

        if self.top_k is not None and self.top_k < len(cols):
            params = {"n_estimators": 300, "learning_rate": 0.05, **(self.ranking_params or {})}
            ranker = build_lgbm(**params).fit(X[cols], y)
            gain = pd.Series(ranker.booster_.feature_importance("gain"), index=cols)
            keep = gain.sort_values(ascending=False).index[: self.top_k]
            self.dropped_["low_importance"] = [c for c in cols if c not in keep]
            cols = [c for c in cols if c in keep]

        self.selected_features_ = cols
        return self

    def transform(self, X):
        return X[self.selected_features_]


def build_lgbm_selected(top_k=None, corr_threshold=0.95, **overrides) -> Pipeline:
    """Task 7: fold-wise feature selection followed by LightGBM."""
    return Pipeline([
        ("select", FoldFeatureSelector(corr_threshold=corr_threshold, top_k=top_k)),
        ("model", build_lgbm(**overrides)),
    ])


def selection_stability(cv_result: dict) -> dict:
    """How consistently the fold-wise selector picked the same features."""
    selected = [set(m.named_steps["select"].selected_features_) for m in cv_result["models"]]
    counts = pd.Series([f for s in selected for f in s]).value_counts()
    jaccard = [len(a & b) / len(a | b) for i, a in enumerate(selected) for b in selected[i + 1:]]
    per_fold = pd.DataFrame([
        {"fold": k, "n_selected": len(s),
         **{f"dropped_{r}": len(v) for r, v in m.named_steps["select"].dropped_.items()}}
        for k, (s, m) in enumerate(zip(selected, cv_result["models"]), start=1)
    ]).set_index("fold")
    return {
        "per_fold": per_fold,
        "times_selected": counts,
        "in_all_folds": sorted(counts[counts == len(selected)].index),
        "mean_pairwise_jaccard": float(np.mean(jaccard)),
    }


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


# --------------------------------------------------
# Feature importance (task 8)
# --------------------------------------------------

def _final_estimator(model):
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def builtin_importance(cv_result: dict) -> pd.DataFrame:
    """LightGBM gain and split importance, averaged over the fold models.

    Gain is normalised to a share of the total within each fold before
    averaging, so every fold counts equally.
    """
    gains, splits = [], []
    for model in cv_result["models"]:
        est = _final_estimator(model)
        names = est.booster_.feature_name()
        gain = pd.Series(est.booster_.feature_importance("gain"), index=names)
        gains.append(gain / gain.sum())
        splits.append(pd.Series(est.booster_.feature_importance("split"), index=names))
    gains, splits = pd.concat(gains, axis=1).fillna(0), pd.concat(splits, axis=1).fillna(0)
    table = pd.DataFrame({
        "gain_share_mean": gains.mean(axis=1), "gain_share_std": gains.std(axis=1),
        "split_mean": splits.mean(axis=1),
    })
    table["gain_rank"] = table["gain_share_mean"].rank(ascending=False, method="min").astype(int)
    table["split_rank"] = table["split_mean"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("gain_share_mean", ascending=False)


def _permutation_drop(model, X_val, y_val, columns, baseline, rng, n_repeats):
    """Mean drop in PR-AUC when `columns` are shuffled together (same row order)."""
    drops = []
    for _ in range(n_repeats):
        X_perm = X_val.copy()
        perm = rng.permutation(len(X_val))
        X_perm[columns] = X_val[columns].to_numpy()[perm]
        drops.append(baseline - average_precision_score(y_val, get_prediction_probabilities(model, X_perm)))
    return float(np.mean(drops))


def permutation_importance_cv(cv_result: dict, X: pd.DataFrame, y: pd.Series, folds,
                              groups: list | None = None, n_repeats: int = 3,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance on each validation fold, averaged over folds.

    Score: PR-AUC (average precision), which suits the 0.7% failure rate.
    Importance = drop in PR-AUC when a feature (or a whole group) is shuffled.

    groups=None: every feature is permuted on its own.
    groups=list of lists: each group is permuted together (features not in a
    group are permuted alone). Use this for correlated sensors: permuting one
    sensor of a near-duplicate group understates its importance because the
    others still carry the same information.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    in_group = {c for g in (groups or []) for c in g}
    units = {", ".join(g) if len(g) <= 4 else f"{g[0]} + {len(g) - 1} correlated": list(g) for g in (groups or [])}
    units.update({c: [c] for c in X.columns if c not in in_group})

    rng = np.random.default_rng(random_state)
    per_fold = []
    for model, (_, va_idx) in zip(cv_result["models"], folds):
        X_val, y_val = X.iloc[va_idx].reset_index(drop=True), y.iloc[va_idx]
        baseline = average_precision_score(y_val, get_prediction_probabilities(model, X_val))
        per_fold.append(pd.Series({name: _permutation_drop(model, X_val, y_val, cols, baseline, rng, n_repeats)
                                   for name, cols in units.items()}))
    drops = pd.concat(per_fold, axis=1)
    table = pd.DataFrame({
        "pr_auc_drop_mean": drops.mean(axis=1),
        "pr_auc_drop_std": drops.std(axis=1),
        "n_sensors": pd.Series({name: len(cols) for name, cols in units.items()}),
        "folds_positive": (drops > 0).sum(axis=1),
    })
    table["rank"] = table["pr_auc_drop_mean"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("pr_auc_drop_mean", ascending=False)


def importance_report(builtin: pd.DataFrame, perm: pd.DataFrame, groups: list, top: int = 20) -> dict:
    """Combine gain, split and permutation importance; measure their agreement."""
    group_of = {c: (g[0] if len(g) <= 4 else f"{g[0]} + {len(g) - 1} correlated")
                for g in groups for c in g}
    combined = builtin.join(perm[["pr_auc_drop_mean", "folds_positive", "rank"]]
                            .rename(columns={"rank": "perm_rank"}), how="left")
    combined["correlated_group"] = [group_of.get(c, "") for c in combined.index]

    both = combined.dropna(subset=["pr_auc_drop_mean"])
    agreement = pd.Series({
        "gain vs split (Spearman)": builtin["gain_share_mean"].corr(builtin["split_mean"], method="spearman"),
        "gain vs permutation (Spearman)": both["gain_share_mean"].corr(both["pr_auc_drop_mean"], method="spearman"),
        f"top-{top} overlap: gain vs permutation": len(set(builtin.index[:top]) & set(perm.index[:top])) / top,
        "features with no positive permutation effect": int((perm["pr_auc_drop_mean"] <= 0).sum()),
    })
    return {"combined": combined.head(top), "agreement": agreement}


# --------------------------------------------------
# Tuning (task 9)
# --------------------------------------------------

TUNING_SPACE = {
    "learning_rate": (0.01, 0.1),            # log-uniform
    "n_estimators": [300, 500, 800, 1200],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [10, 20, 40, 80, 150],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
    "reg_lambda": (0.1, 20.0),               # log-uniform
    "scale_pos_weight": [1, 3, 10, 30],
}


def sample_params(rng) -> dict:
    """One random configuration from TUNING_SPACE."""
    params = {}
    for name, space in TUNING_SPACE.items():
        if isinstance(space, tuple):
            lo, hi = np.log(space[0]), np.log(space[1])
            params[name] = float(np.round(np.exp(rng.uniform(lo, hi)), 4))
        else:
            params[name] = space[int(rng.integers(len(space)))]
            params[name] = params[name].item() if hasattr(params[name], "item") else params[name]
    return params


def threshold_sensitivity(oof, train_df: pd.DataFrame, threshold: float) -> dict:
    """Cost when the chosen threshold is halved or doubled (lower = more robust)."""
    repair = train_df["repair_cost"].to_numpy()
    budget = fit_constant_budget(train_df)
    out = {}
    for label, factor in [("cost_thr_x0.5", 0.5), ("cost_thr_x2", 2.0)]:
        flags = (oof >= threshold * factor).astype(int)
        out[label] = evaluate_predictions(train_df["failure"], oof, repair, flags,
                                          np.where(flags == 1, budget, 0.0))["total_cost"]
    return out


def tune_lgbm(X: pd.DataFrame, y: pd.Series, folds, train_df: pd.DataFrame,
              n_trials: int = 30, random_state: int = RANDOM_STATE, log=print) -> pd.DataFrame:
    """Random search over TUNING_SPACE on the shared folds.

    Trial 0 is the default configuration, so every trial can be compared with
    it fold by fold ("folds_beating_default"). Ranked by OOF cost.
    """
    rng = np.random.default_rng(random_state)
    trials = [{}] + [sample_params(rng) for _ in range(n_trials - 1)]
    rows, default_fold_costs = [], None
    for i, params in enumerate(trials):
        start = time.time()
        cv = run_cv(build_lgbm_all_features(**params), X, y, folds, name=f"trial {i}", verbose=False)
        ev = evaluate_cost(cv, train_df, folds)
        fold_costs = ev["per_fold"]["cost"].to_numpy()
        if i == 0:
            default_fold_costs = fold_costs
        s = ev["summary"]
        rows.append({
            "trial": i,
            **{k: params.get(k, DEFAULT_LGBM_PARAMS.get(k, 1)) for k in TUNING_SPACE},
            "oof_cost": s["oof_cost"], "fold_cost_std": s["fold_cost_std"],
            "folds_beating_default": int((fold_costs < default_fold_costs).sum()),
            **threshold_sensitivity(cv["oof"], train_df, s["threshold"]),
            "threshold": s["threshold"], "tp": s["tp"], "fp": s["fp"],
            "roc_auc": s["roc_auc_mean"], "pr_auc": s["pr_auc_mean"],
            "seconds": time.time() - start,
        })
        log(f"trial {i:>2}: OOF cost {s['oof_cost']:>10,.0f} | PR-AUC {s['pr_auc_mean']:.4f} | "
            f"{rows[-1]['seconds']:.0f}s | {params or 'default'}")
    return pd.DataFrame(rows).set_index("trial").sort_values("oof_cost")


# Tuning notes (30-trial random search on the shared folds, seed 42):
# - Best OOF cost: trial 10 (70,305), trial 28 (70,347), trial 2 (71,331);
#   default (trial 0) 73,828. The top 10 trials lie within 70.3k-72.1k.
# - Re-running with seeds 7 / 123 / 2024 shows that seed noise is about
#   +/-3k: default averages 70,251, trial 2 69,752, trial 10 70,286. The
#   default's 73,828 at seed 42 was an unlucky seed, so most of the apparent
#   tuning gain was noise.
# - Trial 2 is chosen: lowest seed-averaged cost, beats the default on 4 of 5
#   folds, low fold-cost std (2,644), and it is the most robust to the
#   threshold (cost 75,251 / 72,629 when the threshold is halved / doubled,
#   versus 75,837 / 78,083 for the default and 72,985 / 79,997 for trial 10).


class SeedAveragedLGBM(BaseEstimator, ClassifierMixin):
    """Average the probabilities of LightGBM models trained with different seeds.

    A single model's OOF cost moves by about +/-3k with the random seed;
    averaging several seeds reduces that variance at no risk of leakage
    (every member is fitted on the same training data).
    """

    def __init__(self, params=None, seeds=FINAL_SEEDS):
        self.params = params
        self.seeds = seeds

    def fit(self, X, y):
        self.models_ = [build_lgbm(**{**(self.params or {}), "random_state": s}).fit(X, y) for s in self.seeds]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def build_final_lgbm() -> Pipeline:
    """Final model: tuned LightGBM averaged over FINAL_SEEDS."""
    return Pipeline([("model", SeedAveragedLGBM(params=TUNED_LGBM_PARAMS, seeds=FINAL_SEEDS))])


def make_submission(test_df: pd.DataFrame, probabilities, threshold: float, budget: float) -> pd.DataFrame:
    """Kaggle submission (id, flag, budget) with the same checks as V1."""
    flags = (np.asarray(probabilities) >= threshold).astype(int)
    submission = pd.DataFrame({"id": test_df["id"].to_numpy(), "flag": flags,
                               "budget": predict_budgets(flags, budget)})
    assert len(submission) == len(test_df)
    assert submission["id"].is_unique
    assert submission["flag"].isin([0, 1]).all()
    assert (submission["budget"] >= 0).all()
    assert submission.notna().all().all()
    return submission


def seed_stability(configs: dict, X, y, folds, train_df, seeds=(7, 123, 2024)) -> pd.DataFrame:
    """OOF cost of each configuration under several random seeds.

    Differences between configurations smaller than the spread across seeds
    are not meaningful.
    """
    rows = []
    for seed in seeds:
        for name, params in configs.items():
            cv = run_cv(build_lgbm_all_features(**{**params, "random_state": seed}), X, y, folds,
                        name=name, verbose=False)
            s = evaluate_cost(cv, train_df, folds)["summary"]
            rows.append({"seed": seed, "config": name, "oof_cost": s["oof_cost"], "pr_auc": s["pr_auc_mean"]})
    return pd.DataFrame(rows).pivot(index="seed", columns="config", values="oof_cost")


def per_fold_costs(evaluations: list) -> pd.DataFrame:
    """Fold-by-fold cost for each model (same folds, so directly comparable)."""
    return pd.DataFrame({e["summary"]["model"]: e["per_fold"].set_index("fold")["cost"] for e in evaluations})


# --------------------------------------------------
# Command line
# --------------------------------------------------

def _load():
    train_df, test_df = load_data()
    X_all, y = prepare_training_data(train_df)
    features = get_base_features(train_df, test_df)
    folds = load_folds()
    print(f"Training rows: {len(X_all):,} | features: {len(features)} | folds: {len(folds)}")
    return train_df, test_df, X_all[features], y, folds


def run_importance():
    """Task 8: feature importance of the all-features LightGBM."""
    _, _, X, y, folds = _load()
    cv = run_cv(build_lgbm_all_features(), X, y, folds, name="LGBM all features")
    # Unsupervised groups on the training data, used only to interpret importance
    groups, _ = find_correlated_groups(X, threshold=0.95)

    builtin = builtin_importance(cv)
    print("\nPermutation importance (single sensors) ...")
    perm_single = permutation_importance_cv(cv, X, y, folds)
    print("Permutation importance (correlated groups shuffled together) ...")
    perm_grouped = permutation_importance_cv(cv, X, y, folds, groups=groups)
    report = importance_report(builtin, perm_single, groups)

    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 200,
                           "display.max_colwidth", 40):
        print("\nTop 20 features (gain) with split and permutation importance:")
        print(report["combined"].to_string())
        print("\nAgreement between methods:")
        print(report["agreement"].to_string())
        print("\nTop 15 by grouped permutation importance:")
        print(perm_grouped.head(15).to_string())
        print("\nCorrelated groups:")
        for g in groups:
            print(f"  {len(g):>2} sensors: {', '.join(g)}")


def run_tuning(n_trials: int = 30):
    """Task 9: random search for LightGBM on the shared folds."""
    train_df, _, X, y, folds = _load()
    results = tune_lgbm(X, y, folds, train_df, n_trials=n_trials, log=lambda m: print(m, flush=True))
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 250):
        print("\nTop 10 trials (sorted by OOF cost):")
        print(results.head(10).to_string())
        print("\nDefault configuration (trial 0):")
        print(results.loc[[0]].to_string())


def run_seed_check():
    """Default vs tuned configuration across random seeds."""
    train_df, _, X, y, folds = _load()
    table = seed_stability({"default": {}, "tuned": TUNED_LGBM_PARAMS}, X, y, folds, train_df)
    print(table.round(0).to_string())
    print("\nMean:", table.mean().round(0).to_dict(), "| std:", table.std().round(0).to_dict())


# --------------------------------------------------
# Diagnostics and charts (used by notebooks/lightgbm.ipynb)
# --------------------------------------------------

def per_fold_best_costs(cv_result: dict, train_df: pd.DataFrame, folds) -> pd.DataFrame:
    """Per-fold cost at the pooled OOF threshold vs at each fold's own best threshold.

    The gap shows how much cost comes from the threshold not transferring
    between folds, rather than from worse ranking.
    """
    oof = cv_result["oof"]
    repair = train_df["repair_cost"].to_numpy()
    pooled_thr, _ = fast_select_threshold(oof, repair, fit_constant_budget(train_df))
    rows = []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        b = fit_constant_budget(train_df.iloc[tr_idx])
        best_thr, best_cost = fast_select_threshold(oof[va_idx], repair[va_idx], b)
        flags = (oof[va_idx] >= pooled_thr).astype(int)
        pooled_cost = evaluate_predictions(train_df["failure"].iloc[va_idx], oof[va_idx], repair[va_idx], flags,
                                           np.where(flags == 1, b, 0.0))["total_cost"]
        rows.append({"fold": k, "cost_at_pooled_threshold": pooled_cost, "cost_at_fold_best_threshold": best_cost,
                     "fold_best_threshold": best_thr,
                     "median_risk_of_failed": float(np.median(oof[va_idx][train_df["failure"].to_numpy()[va_idx] == 1]))})
    return pd.DataFrame(rows).set_index("fold")


def cost_curve(oof, train_df: pd.DataFrame, max_flagged: int = 4000) -> pd.Series:
    """OOF cost as a function of how many vehicles are flagged (highest risk first)."""
    actual = train_df["repair_cost"].fillna(0).to_numpy()
    b = fit_constant_budget(train_df)
    inspected = 10.0 + 0.20 * np.maximum(actual - b, 0.0) + 0.02 * np.maximum(b - actual, 0.0)
    delta = (inspected - actual)[np.argsort(-np.asarray(oof), kind="stable")]
    costs = actual.sum() + np.r_[0.0, np.cumsum(delta)]
    return pd.Series(costs[: max_flagged + 1], index=np.arange(min(max_flagged, len(delta)) + 1), name="cost")


def plot_cost_curves(curves: dict, title: str = "OOF cost vs number of vehicles flagged"):
    """One line per model; the minimum of each curve is marked."""
    import matplotlib.pyplot as plt
    from src.feature_analysis import INK_2, SERIES, SURFACE

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Lowest curve's label goes below its point, the others above, so labels never overlap
    lowest = min(curves, key=lambda n: curves[n].min())
    for colour, (name, curve) in zip(SERIES, curves.items()):
        ax.plot(curve.index, curve.values / 1000, color=colour, label=name)
        best = curve.idxmin()
        ax.scatter([best], [curve.min() / 1000], s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"min {curve.min() / 1000:,.1f}k at {best:,} flagged", (best, curve.min() / 1000),
                    textcoords="offset points", xytext=(8, -16 if name == lowest else 10), fontsize=8,
                    color=INK_2)
    ax.set(xlabel="Vehicles flagged for inspection (highest predicted risk first)",
           ylabel="Total OOF cost (thousands)", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fold_costs(fold_costs: pd.DataFrame, title: str = "Cost per fold (same shared folds)"):
    """Dot plot: one row per fold, one coloured dot per model."""
    import matplotlib.pyplot as plt
    from src.feature_analysis import AXIS, SERIES, SURFACE

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(fold_costs) + 1.5))
    ypos = np.arange(len(fold_costs))
    ax.hlines(ypos, fold_costs.min(axis=1) / 1000, fold_costs.max(axis=1) / 1000, color=AXIS, lw=1.5)
    for colour, model in zip(SERIES, fold_costs.columns):
        ax.scatter(fold_costs[model] / 1000, ypos, s=60, color=colour, label=model,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(ypos, [f"fold {f}" for f in fold_costs.index])
    ax.invert_yaxis()
    ax.set(xlabel="Validation-fold cost (thousands)", title=title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, fold_costs.shape[1]))
    fig.tight_layout()
    return fig


def plot_importance_bars(values: pd.Series, title: str, xlabel: str, top: int = 20):
    """Horizontal bars for the `top` largest values."""
    import matplotlib.pyplot as plt
    from src.feature_analysis import BLUE

    d = values.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(d) + 1.2))
    ax.barh([str(i) for i in d.index], d.values, height=0.6, color=BLUE)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_tuning(trials: pd.DataFrame):
    """OOF cost of every tuning trial vs its worst cost when the threshold is halved or doubled."""
    import matplotlib.pyplot as plt
    from src.feature_analysis import BLUE, INK_2, ORANGE, SURFACE

    worst = trials[["cost_thr_x0.5", "cost_thr_x2"]].max(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(trials["oof_cost"] / 1000, worst / 1000, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               label="tuning trial")
    for trial, colour in [(0, ORANGE)]:
        ax.scatter(trials.loc[trial, "oof_cost"] / 1000, worst.loc[trial] / 1000, s=70, color=colour,
                   edgecolor=SURFACE, linewidth=2, label="default (trial 0)", zorder=3)
    for trial in trials.index[:3].tolist() + [2]:
        ax.annotate(f"trial {trial}", (trials.loc[trial, "oof_cost"] / 1000, worst.loc[trial] / 1000),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_2)
    ax.set(xlabel="OOF cost at the chosen threshold (thousands)",
           ylabel="Worst cost if threshold halved or doubled (thousands)",
           title="Tuning trials: cost vs robustness (lower-left is better)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_test_vs_cv(test_vs_cv: pd.DataFrame):
    """Share of vehicles flagged: CV vs each set of test predictions."""
    import matplotlib.pyplot as plt
    from src.feature_analysis import BLUE, ORANGE

    d = test_vs_cv
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(d))
    ax.barh(y - 0.18, d["cv_flag_rate"] * 100, height=0.34, color=BLUE, label="CV (out-of-fold)")
    ax.barh(y + 0.18, d["test_flag_rate"] * 100, height=0.34, color=ORANGE, label="test")
    ax.set_yticks(y, d.index)
    ax.invert_yaxis()
    ax.set(xlabel="Vehicles flagged (%)", title="Inspection rate: test vs CV")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


FINAL_NAME, LR_NAME = "LGBM tuned, 5-seed average", "LR benchmark (1:50)"


def final_report(train_df, test_df, X, y, folds, cv_results: dict | None = None,
                 write_submissions: bool = True, verbose: bool = True) -> dict:
    """Final model: CV of the seed-averaged tuned LightGBM, test predictions, submissions.

    cv_results: optional {name: run_cv result} to reuse CV runs that already
    exist (keys FINAL_NAME / LR_NAME).
    """
    cv_results = dict(cv_results or {})
    features = list(X.columns)
    test_aligned = align_test_sensors(test_df, train_df)
    budget = fit_constant_budget(train_df)

    # 1. CV: thresholds come from the same kind of model that predicts test
    for name, builder in [(FINAL_NAME, build_final_lgbm), (LR_NAME, build_lr_benchmark)]:
        if name not in cv_results:
            cv_results[name] = run_cv(builder(), X, y, folds, name=name, verbose=verbose)
    evals = {name: evaluate_cost(cv_results[name], train_df, folds) for name in (FINAL_NAME, LR_NAME)}

    # 2. Fit on all training data and predict test
    lgbm = build_final_lgbm().fit(X, y)
    lr = build_lr_benchmark().fit(X, y)
    p = {
        "lgbm_aligned": lgbm.predict_proba(test_aligned[features])[:, 1],
        "lgbm_raw": lgbm.predict_proba(test_df[features])[:, 1],
        "lr_aligned": lr.predict_proba(test_aligned[features])[:, 1],
        "lr_raw": lr.predict_proba(test_df[features])[:, 1],
    }
    thr = {"lgbm": evals[FINAL_NAME]["summary"]["threshold"], "lr": evals[LR_NAME]["summary"]["threshold"]}
    cv_rate = {k: evals[n]["summary"]["flagged"] / len(X) for k, n in [("lgbm", FINAL_NAME), ("lr", LR_NAME)]}

    # 3. Diagnostics: inspection rate on test vs CV, raw vs aligned test
    rows = []
    for key, probs in p.items():
        model = key.split("_")[0]
        flags = probs >= thr[model]
        rows.append({"predictions": key, "threshold": thr[model], "flagged": int(flags.sum()),
                     "test_flag_rate": flags.mean(), "cv_flag_rate": cv_rate[model],
                     "mean_probability": probs.mean()})
    diag = pd.DataFrame(rows).set_index("predictions")
    agreement = {}
    for model in ("lgbm", "lr"):
        fa_, fr_ = p[f"{model}_aligned"] >= thr[model], p[f"{model}_raw"] >= thr[model]
        agreement[model.upper()] = {
            "Spearman(aligned, raw) probabilities":
                pd.Series(p[f"{model}_aligned"]).corr(pd.Series(p[f"{model}_raw"]), method="spearman"),
            "flagged in both": int((fa_ & fr_).sum()),
            "flagged only with aligned test": int((fa_ & ~fr_).sum()),
            "flagged only with raw test": int((~fa_ & fr_).sum()),
        }
    agreement = pd.DataFrame(agreement)

    # 4. Submissions (submissions/ is ignored by git)
    saved = []
    if write_submissions:
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        for key, model in [("lgbm_aligned", "lgbm"), ("lgbm_raw", "lgbm"), ("lr_aligned", "lr")]:
            path = SUBMISSIONS_DIR / f"v2_{key}.csv"
            make_submission(test_df, p[key], thr[model], budget).to_csv(path, index=False)
            saved.append(path.name)

    return {"cv_results": cv_results, "evaluations": evals, "probabilities": p, "thresholds": thr,
            "budget": budget, "test_vs_cv": diag, "raw_vs_aligned": agreement, "saved": saved}


def run_final():
    """Command-line wrapper around final_report()."""
    train_df, test_df, X, y, folds = _load()
    r = final_report(train_df, test_df, X, y, folds)
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 200):
        print("\nCV results:")
        print(comparison_table(list(r["evaluations"].values())).T.to_string())
        print(f"\nBudget for inspected vehicles: {r['budget']:,.2f}")
        print("\nTest predictions vs CV:")
        print(r["test_vs_cv"].to_string())
        print("\nAligned vs raw test:")
        print(r["raw_vs_aligned"].to_string())
    print("Saved:", ", ".join(r["saved"]))


def main():
    train_df, test_df, X, y, folds = _load()
    print(f"Cost of inspecting nobody: {no_inspection_cost(train_df):,.2f}\n")

    experiments = {
        "LR benchmark (1:50)": build_lr_benchmark(),
        "LGBM all features": build_lgbm_all_features(),
        "LGBM selected: filters only": build_lgbm_selected(top_k=None),
        "LGBM selected: top 80": build_lgbm_selected(top_k=80),
        "LGBM selected: top 50": build_lgbm_selected(top_k=50),
        "LGBM selected: top 30": build_lgbm_selected(top_k=30),
    }
    evaluations, stability = [], {}
    for name, model in experiments.items():
        cv = run_cv(model, X, y, folds, name=name)
        evaluations.append(evaluate_cost(cv, train_df, folds))
        if "select" in cv["models"][0].named_steps:
            stability[name] = selection_stability(cv)
        print()

    print("Fold-wise selection:")
    for name, s in stability.items():
        print(f"\n{name}: mean pairwise Jaccard {s['mean_pairwise_jaccard']:.3f}, "
              f"{len(s['in_all_folds'])} features selected in all 5 folds")
        print(s["per_fold"].to_string())

    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}", "display.width", 200):
        print("Comparison (sorted by OOF cost):")
        print(comparison_table(evaluations).T.to_string())
        print("\nCost per fold:")
        print(per_fold_costs(evaluations).to_string())


if __name__ == "__main__":
    # python src/lightgbm_model.py [compare|importance|tune|seeds|final]
    step = sys.argv[1] if len(sys.argv) > 1 else "compare"
    {"compare": main, "importance": run_importance, "tune": run_tuning, "seeds": run_seed_check,
     "final": run_final}[step]()
