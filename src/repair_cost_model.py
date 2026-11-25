
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .cv_utils import PROJECT_ROOT, RESULTS_DIR, load_folds
from .cost_model import fit_constant_budget
from .failure_model import train_failure_model, predict_failure_proba
from .preprocessing import fit_preprocessor, transform_features


def compare_models(boosters=False):
    train = pd.read_csv(PROJECT_ROOT / "data/train.csv")
    known = train.failure.eq(1) & train.repair_cost.notna()
    print(train.loc[known, "repair_cost"].describe(percentiles=[.5, .9, .99]))
    models = {"forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=4,
                                               random_state=42, n_jobs=2)}
    if boosters:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        models["xgb"] = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.05,
                                     random_state=42, n_jobs=2)
        models["lgbm"] = LGBMRegressor(n_estimators=100, num_leaves=8, learning_rate=.05,
                                       verbosity=-1, random_state=42, n_jobs=2)

    folds = load_folds()
    positions = np.arange(len(train))
    if not np.array_equal(np.sort(np.concatenate([v for _, v in folds])), positions):
        raise ValueError("Validation folds must cover every vehicle once.")

    oof = train[["id", "failure", "repair_cost"]].copy()
    for fold, (t, v) in enumerate(folds):
        if not np.array_equal(np.sort(t), np.setdiff1d(positions, v)):
            raise ValueError(f"Invalid training indices in fold {fold}")
        training, validation = train.iloc[t], train.iloc[v]
        prep = fit_preprocessor(training, scale=True)
        x = transform_features(prep, training)
        xv = transform_features(prep, validation)
        classifier = train_failure_model(x, training.failure, class_weight=None)
        oof.loc[v, "failure_probability"] = predict_failure_proba(classifier, xv)
        failed = known.iloc[t]
        y = training.loc[failed, "repair_cost"]
        oof.loc[v, "fold"] = fold
        oof.loc[v, "constant_budget"] = fit_constant_budget(training)
        oof.loc[v, "median"] = y.median()
        for name, model in models.items():
            model.fit(x.loc[failed], y)
            oof.loc[v, name] = np.maximum(model.predict(xv), 0)
    rows = []
    for name in ["median", *models]:
        y, pred = oof.loc[known, "repair_cost"], oof.loc[known, name]
        rows.append({"model": name, "MAE": mean_absolute_error(y, pred),
                     "RMSE": mean_squared_error(y, pred) ** .5})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "repair_comparison.csv", index=False)
    oof.to_csv(RESULTS_DIR / "repair_oof.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boosters", action="store_true")
    compare_models(parser.parse_args().boosters)