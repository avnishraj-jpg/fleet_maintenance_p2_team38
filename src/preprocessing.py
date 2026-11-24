"""Data preprocessing for Fleet Maintenance Planning (Member 2 - Mohd Farhan Ali).

Every transformation (column drops, imputer, missingness indicators, scaler) is
fitted on the TRAINING split only, then applied unchanged to validation and
test data, so no statistics leak from validation/test into training.

Typical use in the notebook (after the train-validation split):

    from src.preprocessing import preprocess_data

    X_train, X_val, X_test, prep = preprocess_data(
        train_part, val_part, test_df,
        imputation="median", missing_indicators=None, scale=False,
    )
"""

import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

# EDA: s146 has a single observed value (~939138.7) in both train and test.
CONSTANT_SENSORS = ["s146"]

# EDA: eight sensors have more than 50% missing values in training.
HIGH_MISSING_THRESHOLD = 0.50

IMPUTATION_OPTIONS = ("median", "mean", "constant", "knn")
INDICATOR_OPTIONS = (None, "high_missing", "all")

_SENSOR_PATTERN = re.compile(r"^s\d{3}$")


def get_sensor_columns(df: pd.DataFrame) -> list:
    """Return sensor feature columns (s000-s169) in their original order.

    Only sensor columns are used as features. Targets (failure, repair_cost),
    IDs and followup_days (absent from test) are excluded automatically.
    """
    cols = [c for c in df.columns if _SENSOR_PATTERN.match(str(c))]
    if not cols:
        raise ValueError("No sensor columns (s000-s169) found.")
    return cols


def find_high_missing_sensors(X_train: pd.DataFrame,
                              threshold: float = HIGH_MISSING_THRESHOLD) -> list:
    """Return sensors whose missing fraction in the TRAINING data exceeds threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    sensors = get_sensor_columns(X_train)
    missing_frac = X_train[sensors].isna().mean()
    return missing_frac[missing_frac > threshold].index.tolist()


def fit_preprocessor(train: pd.DataFrame,
                     imputation: str = "median",
                     fill_value: float = 0.0,
                     knn_neighbors: int = 5,
                     missing_indicators=None,
                     drop_high_missing: bool = False,
                     scale: bool = False,
                     high_missing_threshold: float = HIGH_MISSING_THRESHOLD) -> dict:
    """Fit all preprocessing steps on the training data only.

    Parameters
    ----------
    train : training rows (may include target columns; only sensors are used).
    imputation : "median" (baseline), "mean", "constant" or "knn".
    fill_value : value used when imputation="constant". Note that some sensors
        have genuine zero readings, so 0 is not a "missing" marker.
    knn_neighbors : neighbours for KNNImputer (slow on the full data set).
    missing_indicators : None, "high_missing" (only the >50%-missing sensors)
        or "all" (every sensor). Adds 0/1 columns named "<sensor>_missing".
    drop_high_missing : drop the >50%-missing sensors (for the with/without
        comparison; keep them in the first baseline).
    scale : standardise imputed sensor values (useful for Logistic Regression
        and distance-based models; not needed for tree models).

    Returns
    -------
    dict holding the fitted objects and column lists, passed to transform_features().
    """
    if imputation not in IMPUTATION_OPTIONS:
        raise ValueError(f"imputation must be one of {IMPUTATION_OPTIONS}.")
    if missing_indicators not in INDICATOR_OPTIONS:
        raise ValueError(f"missing_indicators must be one of {INDICATOR_OPTIONS}.")
    if len(train) == 0:
        raise ValueError("Training data is empty.")

    sensors = get_sensor_columns(train)
    high_missing = find_high_missing_sensors(train[sensors], high_missing_threshold)

    # 1. Remove constant sensor(s); optionally remove highly missing sensors.
    dropped = [c for c in CONSTANT_SENSORS if c in sensors]
    if drop_high_missing:
        dropped += [c for c in high_missing if c not in dropped]
    feature_cols = [c for c in sensors if c not in dropped]

    # 2. Missingness indicators (decided from the raw, un-imputed values).
    if missing_indicators == "all":
        indicator_cols = list(feature_cols)
    elif missing_indicators == "high_missing":
        indicator_cols = [c for c in high_missing if c in feature_cols]
    else:
        indicator_cols = []

    X = train[feature_cols].astype(float)

    # 3. Imputer, fitted on training data only.
    if imputation == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    elif imputation == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value,
                                keep_empty_features=True)
    else:
        imputer = SimpleImputer(strategy=imputation, keep_empty_features=True)
    X_imputed = imputer.fit_transform(X)

    # 4. Optional scaler, fitted on the imputed training data only.
    scaler = StandardScaler().fit(X_imputed) if scale else None

    return {
        "imputation": imputation,
        "feature_cols": feature_cols,
        "dropped_cols": dropped,
        "high_missing_cols": high_missing,
        "indicator_cols": indicator_cols,
        "imputer": imputer,
        "scaler": scaler,
    }


def transform_features(preprocessor: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor to train, validation or test data.

    Row order and index are preserved, so model outputs line up with the input rows.
    """
    feature_cols = preprocessor["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    X = df[feature_cols].astype(float)
    flags = X[preprocessor["indicator_cols"]].isna().astype(np.int8)

    values = preprocessor["imputer"].transform(X)
    if preprocessor["scaler"] is not None:
        values = preprocessor["scaler"].transform(values)

    out = pd.DataFrame(values, columns=feature_cols, index=df.index)
    if len(flags.columns):
        flags.columns = [f"{c}_missing" for c in flags.columns]
        out = pd.concat([out, flags], axis=1)

    if out.isna().any().any():
        raise ValueError("Missing values remain after preprocessing.")
    return out


def preprocess_data(train: pd.DataFrame, val: pd.DataFrame,
                    test: pd.DataFrame = None, **options):
    """Fit on train, then transform train, validation and (optionally) test.

    `options` are passed to fit_preprocessor().
    Returns (X_train, X_val, X_test, preprocessor); X_test is None if test is None.
    """
    preprocessor = fit_preprocessor(train, **options)
    X_train = transform_features(preprocessor, train)
    X_val = transform_features(preprocessor, val)
    X_test = transform_features(preprocessor, test) if test is not None else None
    return X_train, X_val, X_test, preprocessor


# Settings to compare on the validation set (median baseline first).
PREPROCESSING_EXPERIMENTS = {
    "median (baseline)": dict(imputation="median"),
    "mean": dict(imputation="mean"),
    "constant 0": dict(imputation="constant", fill_value=0.0),
    "median + high-missing indicators": dict(imputation="median",
                                             missing_indicators="high_missing"),
    "median + all indicators": dict(imputation="median", missing_indicators="all"),
    "median, drop high-missing": dict(imputation="median", drop_high_missing=True),
    "median + scaling": dict(imputation="median", scale=True),
    # Optional and slow on 56k rows - run only after the baseline works:
    # "knn (k=5)": dict(imputation="knn", knn_neighbors=5),
}
