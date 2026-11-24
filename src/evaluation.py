"""Evaluation utilities for Fleet Maintenance Planning (Member 1 - Avnish Raj).

The competition objective is to minimise the total maintenance cost, not to
maximise classification accuracy. These functions implement the official cost
and select an inspection threshold using labelled validation data.

Typical use
-----------
    from src.evaluation import (
        calculate_cost,
        select_threshold,
        evaluate_predictions,
    )

    best_threshold, best_cost = select_threshold(
        probabilities=val_probabilities,
        repair_costs=val_part["repair_cost"],
        baseline_budget=baseline_budget,
    )

    val_flags = (val_probabilities >= best_threshold).astype(int)
    val_budgets = predict_budgets(val_flags, baseline_budget)

    metrics = evaluate_predictions(
        y_true=val_part["failure"],
        probabilities=val_probabilities,
        repair_costs=val_part["repair_cost"],
        flags=val_flags,
        budgets=val_budgets,
        threshold=best_threshold,
    )

All threshold selection must be performed on the validation set, never on test
labels or test outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_1d_array(values, name: str, dtype=float) -> np.ndarray:
    """Convert input to a one-dimensional NumPy array and validate it."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _validated_cost_inputs(repair_costs, flags, budgets):
    """Validate and align actual costs, inspection flags and budgets."""
    # NaN repair costs represent healthy vehicles and therefore have actual cost 0.
    if isinstance(repair_costs, (pd.Series, pd.DataFrame)):
        costs = pd.Series(repair_costs).fillna(0).to_numpy(dtype=float)
    else:
        costs = np.asarray(repair_costs, dtype=float)
        costs = np.where(np.isnan(costs), 0.0, costs)

    flags = _to_1d_array(flags, "flags", dtype=float)
    budgets = _to_1d_array(budgets, "budgets", dtype=float)

    if costs.ndim != 1 or costs.size == 0:
        raise ValueError("repair_costs must be a non-empty one-dimensional array.")
    if not (len(costs) == len(flags) == len(budgets)):
        raise ValueError("repair_costs, flags and budgets must have equal lengths.")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Repair costs must be finite and non-negative after filling NaNs.")
    if not np.isfinite(flags).all() or not np.isin(flags, [0, 1]).all():
        raise ValueError("flags must contain only 0 and 1.")
    if not np.isfinite(budgets).all() or (budgets < 0).any():
        raise ValueError("budgets must be finite and non-negative.")

    return costs, flags.astype(int), budgets


def calculate_cost(repair_costs, flags, budgets, return_per_vehicle: bool = False):
    """Calculate the official competition cost.

    Parameters
    ----------
    repair_costs : array-like
        Actual repair costs. NaN denotes a healthy vehicle and is treated as 0.
    flags : array-like
        1 means inspect; 0 means do not inspect.
    budgets : array-like
        Parts budget for each vehicle. Budget is ignored when flag is 0.
    return_per_vehicle : bool, default=False
        If True, return (total_cost, per_vehicle_costs). Otherwise return total.

    Returns
    -------
    float or tuple
        Total official cost, optionally with the per-vehicle costs.
    """
    actual, flags, budgets = _validated_cost_inputs(repair_costs, flags, budgets)

    inspected_cost = (
        10.0
        + 0.20 * np.maximum(actual - budgets, 0.0)
        + 0.02 * np.maximum(budgets - actual, 0.0)
    )
    per_vehicle_cost = np.where(flags == 1, inspected_cost, actual)
    total_cost = float(per_vehicle_cost.sum())

    if return_per_vehicle:
        return total_cost, per_vehicle_cost
    return total_cost


def select_threshold(
    probabilities,
    repair_costs,
    baseline_budget: float,
):
    """Find the probability threshold that minimises official validation cost.

    A vehicle is inspected when its predicted failure probability is greater
    than or equal to the threshold. The same constant budget is assigned to
    every inspected vehicle; uninspected vehicles receive budget 0.

    The search considers every distinct predicted probability, plus a
    threshold just above the maximum probability. The latter explicitly
    evaluates the valid option of inspecting no vehicles.

    Returns
    -------
    tuple[float, float]
        (best_threshold, minimum_total_validation_cost)
    """
    probabilities = _to_1d_array(probabilities, "probabilities", dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must all be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1.")
    if not np.isfinite(baseline_budget) or baseline_budget < 0:
        raise ValueError("baseline_budget must be finite and non-negative.")

    costs = np.asarray(repair_costs, dtype=float)
    if costs.ndim != 1 or len(costs) != len(probabilities):
        raise ValueError("repair_costs and probabilities must have equal lengths.")
    if not np.isfinite(np.where(np.isnan(costs), 0.0, costs)).all():
        raise ValueError("repair_costs must be finite or NaN.")
    if (np.where(np.isnan(costs), 0.0, costs) < 0).any():
        raise ValueError("repair_costs must be non-negative.")

    # Every distinct probability creates a distinct set of inspected vehicles.
    # nextafter(max) adds the no-inspection candidate.
    thresholds = np.unique(
        np.concatenate(
            [
                np.unique(probabilities),
                [np.nextafter(probabilities.max(), np.inf)],
            ]
        )
    )

    best_threshold = None
    best_cost = np.inf

    for threshold in thresholds:
        flags = (probabilities >= threshold).astype(int)
        budgets = np.where(flags == 1, float(baseline_budget), 0.0)
        cost = calculate_cost(costs, flags, budgets)

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold, float(best_cost)


def evaluate_predictions(
    y_true,
    probabilities,
    repair_costs,
    flags,
    budgets,
    threshold: float | None = None,
):
    """Report official cost and classification metrics for a set of predictions.

    Classification metrics are supplementary. Model selection should primarily
    use total_cost because that is the competition's objective.
    """
    y_true = _to_1d_array(y_true, "y_true", dtype=float)
    probabilities = _to_1d_array(probabilities, "probabilities", dtype=float)
    if not np.isfinite(y_true).all() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    y_true = y_true.astype(int)
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must all be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1.")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths.")

    actual, flags, budgets = _validated_cost_inputs(repair_costs, flags, budgets)
    if len(actual) != len(y_true):
        raise ValueError("All input arrays must have equal lengths.")

    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()
    total_cost = calculate_cost(actual, flags, budgets)

    result = {
        "threshold": None if threshold is None else float(threshold),
        "total_cost": total_cost,
        "mean_cost_per_vehicle": float(total_cost / len(y_true)),
        "vehicles_inspected": int(flags.sum()),
        "inspection_rate": float(flags.mean()),
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "f1": float(f1_score(y_true, flags, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if np.unique(y_true).size == 2
            else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result
