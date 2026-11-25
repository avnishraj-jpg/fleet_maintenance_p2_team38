
import numpy as np
import pandas as pd
from .cv_utils import RESULTS_DIR
from .evaluation import calculate_cost


def compare_policies():
    oof = pd.read_csv(RESULTS_DIR / "repair_oof.csv")
    zero = np.zeros(len(oof))
    actual = oof.repair_cost
    skip = calculate_cost(actual, zero, zero)
    rows = [{"model": "skip_all", "threshold": None, "budget": "zero",
             "inspections": 0, "total_cost": skip, "saving_vs_skip": 0}]
    for model in ["median", "forest", "xgb", "lgbm"]:
        if model not in oof:
            continue
        risk = oof.failure_probability * oof[model]
        for threshold in [5, 10, 20, 40, 80]:
            inspect = risk >= threshold
            budgets = {"constant": oof.constant_budget, "zero": zero,
                       "half": .5 * oof[model], "full": oof[model],
                       "one_and_half": 1.5 * oof[model]}
            for strategy, amount in budgets.items():
                total = calculate_cost(actual, inspect, np.where(inspect, amount, 0))
                rows.append({"model": model, "threshold": threshold, "budget": strategy,
                             "inspections": inspect.sum(), "total_cost": total,
                             "saving_vs_skip": skip - total})
    results = pd.DataFrame(rows).sort_values("total_cost")
    results.to_csv(RESULTS_DIR / "budget_experiments.csv", index=False)
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    compare_policies()
