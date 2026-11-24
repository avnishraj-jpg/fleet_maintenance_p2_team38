import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42


def build_model(class_weight="balanced", random_state=RANDOM_STATE):

    return LogisticRegression(
        class_weight=class_weight,
        max_iter=2000,
        random_state=random_state,
    )


# Apply to TRAINING data only. Never oversample validation or test data.
def random_oversample(X, y, random_state=RANDOM_STATE):

    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    failure_idx = np.flatnonzero(y == 1)
    healthy_idx = np.flatnonzero(y == 0)
    if len(failure_idx) == 0:
        raise ValueError("No failure rows to oversample.")

    rng = np.random.default_rng(random_state)
    extra_idx = rng.choice(failure_idx, size=len(healthy_idx) - len(failure_idx), replace=True)
    all_idx = np.concatenate([np.arange(len(y)), extra_idx])
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def train_failure_model(X_train, y_train, class_weight="balanced", oversample=False,
                        random_state=RANDOM_STATE):

    if np.isnan(np.asarray(X_train, dtype=float)).any():
        raise ValueError("X_train contains missing values. Run preprocessing first.")

    X_fit, y_fit = X_train, np.asarray(y_train).astype(int)
    if oversample:
        X_fit, y_fit = random_oversample(X_fit, y_fit, random_state)

    model = build_model(class_weight=class_weight, random_state=random_state)
    model.fit(X_fit, y_fit)
    return model

def predict_failure_proba(model, X):
    return model.predict_proba(X)[:, 1]

def evaluate_classifier(y_true, proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    flags = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, flags, zero_division=0),
        "recall": recall_score(y_true, flags, zero_division=0),
        "f1": f1_score(y_true, flags, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def describe_result(metrics):
    n_fail = metrics["tp"] + metrics["fn"]
    n_healthy = metrics["tn"] + metrics["fp"]
    return (
        f"Detected {metrics['tp']} of {n_fail} failures (missed {metrics['fn']}); "
        f"wrongly flagged {metrics['fp']} of {n_healthy} healthy vehicles."
    )


IMBALANCE_EXPERIMENTS = {
    "LR balanced": {"class_weight": "balanced"},
    "LR no weighting": {"class_weight": None},
    "LR weight 1:10": {"class_weight": {0: 1, 1: 10}},
    "LR weight 1:25": {"class_weight": {0: 1, 1: 25}},
    "LR weight 1:50": {"class_weight": {0: 1, 1: 50}},
    "LR random oversampling": {"class_weight": None, "oversample": True},
}

# Chosen model (see comparison at the bottom). Use in the notebook as:
#   train_failure_model(X_train, y_train, **CHOSEN_CONFIG)
CHOSEN_CONFIG = IMBALANCE_EXPERIMENTS["LR weight 1:50"]


# ---------------------------------------------------------------------------
# Experiment runner:  python src/failure_model.py
#
# TEMPORARY: src/preprocessing.py (Farhan) and src/evaluation.py (Avnish) are
# not merged yet, so _preprocess, _calculate_cost and _select_threshold are
# minimal versions copied from the team guide. Replace them with imports from
# those files once they are on main. Budgets come from src/cost_model.py.
# ---------------------------------------------------------------------------
def _run_experiments():
    import os

    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from cost_model import fit_constant_budget, predict_budgets

    def _preprocess(train_df, val_df):
        cols = [c for c in train_df.columns
                if c not in ("id", "failure", "repair_cost", "followup_days", "s146")]
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(imputer.fit_transform(train_df[cols]))
        X_va = scaler.transform(imputer.transform(val_df[cols]))
        return X_tr, X_va

    def _calculate_cost(repair_cost, flags, budgets):
        actual = repair_cost.fillna(0).to_numpy()
        flags, budgets = np.asarray(flags), np.asarray(budgets)
        inspected = (10 + 0.20 * np.maximum(actual - budgets, 0)
                     + 0.02 * np.maximum(budgets - actual, 0))
        return np.where(flags == 1, inspected, actual).sum()

    def _select_threshold(proba, repair_cost, budget):
        best_t, best_cost = None, np.inf
        for t in np.unique(np.quantile(proba, np.linspace(0, 1, 2001))):
            flags = (proba >= t).astype(int)
            cost = _calculate_cost(repair_cost, flags, predict_budgets(flags, budget))
            if cost < best_cost:
                best_t, best_cost = t, cost
        return best_t, best_cost

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "train.csv")
    train = pd.read_csv(data_path)
    train_idx, val_idx = train_test_split(
        train.index, test_size=0.20, random_state=RANDOM_STATE, stratify=train["failure"]
    )
    tr, va = train.loc[train_idx], train.loc[val_idx]
    X_tr, X_va = _preprocess(tr, va)
    y_tr, y_va = tr["failure"].to_numpy(), va["failure"].to_numpy()
    budget = fit_constant_budget(tr)

    rows = []
    for name, kwargs in IMBALANCE_EXPERIMENTS.items():
        model = train_failure_model(X_tr, y_tr, **kwargs)
        proba = predict_failure_proba(model, X_va)
        t, cost = _select_threshold(proba, va["repair_cost"], budget)
        m = evaluate_classifier(y_va, proba, t)
        print(f"{name:<24} {describe_result(m)}")
        rows.append({"model": name, "val_cost": cost, "threshold": t,
                     "roc_auc": m["roc_auc"], "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"],
                     "TP": m["tp"], "FN": m["fn"], "FP": m["fp"], "TN": m["tn"]})

    results = pd.DataFrame(rows).sort_values("val_cost")
    print("\nRanked by official validation cost:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    _run_experiments()


# ---------------------------------------------------------------------------
# Comparison of imbalance-handling approaches
# ---------------------------------------------------------------------------
# Setup: 80:20 stratified split (seed 42), 79 failures among 11,200 validation
# vehicles. Each model uses the threshold that minimises its official
# validation cost (budgets from cost_model.fit_constant_budget).
#
# Model                   Val cost  Thresh  ROC-AUC  Prec   Recall  F1     TP  FN  FP
# LR weight 1:50          12,349    0.533   0.955    0.271  0.848   0.411  67  12  180
# LR random oversampling  12,349    0.770   0.948    0.271  0.848   0.411  67  12  180
# LR balanced             12,399    0.756   0.956    0.266  0.848   0.405  67  12  185
# LR weight 1:10          12,458    0.137   0.944    0.245  0.886   0.384  70   9  216
# LR weight 1:25          12,519    0.315   0.951    0.254  0.848   0.391  67  12  197
# LR no weighting         13,686    0.023   0.938    0.270  0.823   0.406  65  14  176
#
# - Every imbalance-handling approach beats no weighting (about 1,300 lower cost).
# - The weighted and oversampled models are close to each other (within ~200).
# - Chosen model: LR with class weight 1:50. It ties with random oversampling
#   for the lowest validation cost and has the higher ROC-AUC.
#   It detects 67 of 79 failures (missed 12) and wrongly flags 180 of 11,121
#   healthy vehicles.
