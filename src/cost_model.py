import numpy as np
import pandas as pd


def fit_constant_budget(train: pd.DataFrame) -> float:
    costs = train["repair_cost"].dropna().to_numpy(dtype=float)

    if len(costs) == 0 or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Invalid repair costs.")

    q = 0.20 / (0.20 + 0.02)
    budget = np.quantile(costs, q)

    return float(budget)


def predict_budgets(inspection_flags, baseline_budget: float) -> np.ndarray:
    flags = np.asarray(inspection_flags)

    if flags.ndim != 1 or not np.isin(flags, [0, 1]).all():
        raise ValueError("Flags must contain only 0 or 1.")

    if not np.isfinite(baseline_budget) or baseline_budget < 0:
        raise ValueError("Invalid budget.")

    budgets = np.where(flags == 1, baseline_budget, 0.0)

    return budgets
