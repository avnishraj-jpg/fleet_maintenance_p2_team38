"""Feature analysis for Fleet Maintenance Planning (Member 3 - Shaik Mohd Aatiq).

Sensor-level analysis of the 170 sensors (s000-s169). The general EDA (target,
repair cost, basic missing values) is in notebooks/eda.ipynb sections 1-5.
This module covers:

1. Predictive signal of each sensor (value and missingness).
2. Low-variance and potentially noisy sensors.
3. Sensor correlations and groups of redundant sensors.
4. Train vs test sensor distributions (KS statistic, PSI).
5. Test-set sensor misalignment, and `align_test_sensors()` to fix it.
6. Potential depot distribution shifts (adversarial validation, latent
   groups, predicted-risk shift).
7. Feature recommendations for modelling.

Every function returns tables (pandas) and/or matplotlib figures; nothing is
written to disk. Results are shown in notebooks/eda.ipynb (section 6).
`python src/feature_analysis.py` prints the key findings.

Everything here is descriptive and computed on the full training data. Any
supervised feature selection used for modelling is re-fitted inside each CV
training fold (see lightgbm_model.py), so these results never leak
validation information into model selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import CONSTANT_SENSORS, get_sensor_columns  # noqa: E402

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Thresholds used to flag sensors.
QUASI_CONSTANT_TOP_FREQ = 0.99   # one value covers >= 99% of observed rows
HIGH_MISSING_FRAC = 0.50
CORRELATION_THRESHOLD = 0.95     # |Spearman rho| above which sensors are redundant
WEAK_SIGNAL_AUC = 0.52           # direction-free univariate AUC below this
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25


# --------------------------------------------------
# Plot style (light-mode reference palette)
# --------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[0], SERIES[1]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#104281", BLUE, NEUTRAL, "#e34948", "#8f1d1d"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", SURFACE])


def set_plot_style():
    """Consistent, recessive chart chrome for all figures in this module."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "lines.linewidth": 2,
    })


set_plot_style()


# --------------------------------------------------
# Data
# --------------------------------------------------

def load_data(data_dir: Path = DATA_DIR):
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def signed_log(X):
    """log1p for heavily right-skewed, non-negative sensor values."""
    return np.sign(X) * np.log1p(np.abs(X))


# --------------------------------------------------
# 1. Predictive signal per sensor
# --------------------------------------------------

def sensor_signal(train: pd.DataFrame, sensors: list) -> pd.DataFrame:
    """Univariate signal of each sensor's value and of its missingness.

    auc_strength = max(AUC, 1 - AUC) of the raw value (missing filled by the
    median), so 0.5 = no signal and 1.0 = perfect separation either way.
    The missingness columns compare the failure rate when the sensor is
    missing vs present.
    """
    y = train["failure"]
    rows = []
    for col in sensors:
        x = train[col]
        miss = x.isna()
        auc = roc_auc_score(y, x.fillna(x.median())) if x.nunique() > 1 else 0.5
        fr_miss = y[miss].mean() if miss.any() else np.nan
        fr_pres = y[~miss].mean() if (~miss).any() else np.nan
        rows.append({
            "sensor": col,
            "auc_strength": max(auc, 1 - auc),
            "direction": "higher = failure" if auc >= 0.5 else "lower = failure",
            "missing_frac": miss.mean(),
            "fail_rate_if_missing": fr_miss,
            "fail_rate_if_present": fr_pres,
        })
    df = pd.DataFrame(rows).set_index("sensor")
    df["missing_risk_ratio"] = df["fail_rate_if_missing"] / df["fail_rate_if_present"]
    return df.sort_values("auc_strength", ascending=False)


def plot_signal(signal: pd.DataFrame, top: int = 25):
    top_df = signal.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.22 * top + 1))
    ax.barh(top_df.index, top_df["auc_strength"] - 0.5, left=0.5, height=0.6, color=BLUE)
    ax.set(xlim=(0.5, 1.0), xlabel="Univariate AUC, max(AUC, 1 - AUC)",
           title=f"Top {top} sensors by univariate signal")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_class_distributions(train: pd.DataFrame, sensors: list, signal: pd.DataFrame | None = None):
    """log1p distribution of each sensor for healthy vs failed vehicles."""
    n = len(sensors)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), sensors):
        vals = np.log1p(train[col])
        bins = np.linspace(vals.min(), vals.max(), 40)
        for cls, colour, label in [(0, BLUE, "healthy"), (1, ORANGE, "failed")]:
            ax.hist(vals[train["failure"] == cls].dropna(), bins=bins, density=True,
                    histtype="step", lw=2, color=colour, label=label)
        title = col if signal is None else f"{col}  (AUC {signal.loc[col, 'auc_strength']:.3f})"
        ax.set(title=title, xlabel="log1p(value)", yticks=[])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


def plot_missingness_signal(signal: pd.DataFrame, n: int = 8, min_missing: float = 0.01):
    """Failure rate when missing vs present, for the most informative sensors."""
    s = signal[(signal["missing_frac"] > min_missing) & (signal["missing_frac"] < 1)].copy()
    s["log_ratio"] = np.log(s["missing_risk_ratio"])
    pick = pd.concat([s.nsmallest(n // 2, "log_ratio"), s.nlargest(n // 2, "log_ratio")])
    pick = pick.sort_values("log_ratio")
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(pick) + 1.2))
    y = np.arange(len(pick))
    ax.hlines(y, pick["fail_rate_if_missing"] * 100, pick["fail_rate_if_present"] * 100, color=AXIS, lw=1.5)
    ax.scatter(pick["fail_rate_if_present"] * 100, y, s=60, color=BLUE, label="sensor present", zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.scatter(pick["fail_rate_if_missing"] * 100, y, s=60, color=ORANGE, label="sensor missing", zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [f"{c}  ({m:.0%} missing)" for c, m in pick["missing_frac"].items()])
    ax.set_xscale("log")
    ax.set(xlabel="Failure rate (%), log scale", title="Missingness is predictive: failure rate when missing vs present")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 2. Low-variance and noisy sensors
# --------------------------------------------------

def find_low_variance_features(X: pd.DataFrame,
                               top_freq_threshold: float = QUASI_CONSTANT_TOP_FREQ) -> dict:
    """Constant (<= 1 distinct observed value) and quasi-constant columns.

    Quasi-constant: the most frequent observed value covers at least
    `top_freq_threshold` of the observed rows. Unsupervised, so it can be
    re-fitted cheaply inside each CV training fold.
    """
    constant, quasi = [], []
    for col in X.columns:
        observed = X[col].dropna()
        if observed.nunique() <= 1:
            constant.append(col)
        elif observed.value_counts(normalize=True).iloc[0] >= top_freq_threshold:
            quasi.append(col)
    return {"constant": constant, "quasi_constant": quasi}


def feature_quality(train: pd.DataFrame, sensors: list, signal: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor with variance / missingness / signal diagnostics and flags."""
    X = train[sensors]
    lv = find_low_variance_features(X)
    q = pd.DataFrame({
        "n_unique": X.nunique(),
        "top_value_freq": X.apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 1.0),
        "missing_frac": X.isna().mean(),
        "skew": X.skew(),
        "auc_strength": signal["auc_strength"],
    })
    q["constant"] = q.index.isin(lv["constant"])
    q["quasi_constant"] = q.index.isin(lv["quasi_constant"])
    q["high_missing"] = q["missing_frac"] > HIGH_MISSING_FRAC
    q["weak_signal"] = q["auc_strength"] < WEAK_SIGNAL_AUC
    flags = ["constant", "quasi_constant", "high_missing", "weak_signal"]
    q["n_flags"] = q[flags].sum(axis=1)
    q["category"] = np.select(
        [q["constant"], q["quasi_constant"], q["weak_signal"], q["high_missing"]],
        ["constant", "quasi-constant", "weak signal", "high missing"], default="ok")
    return q.sort_values(["n_flags", "auc_strength"], ascending=[False, True])


def plot_feature_quality(quality: pd.DataFrame):
    """Missing fraction vs univariate signal, coloured by flag category."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = ["ok", "high missing", "weak signal", "quasi-constant", "constant"]
    colours = dict(zip(order, [BLUE, ORANGE, SERIES[2], SERIES[3], SERIES[7]]))
    for cat in order:
        d = quality[quality["category"] == cat]
        if len(d):
            ax.scatter(d["missing_frac"], d["auc_strength"], s=36, color=colours[cat],
                       edgecolor=SURFACE, linewidth=1.5, label=f"{cat} ({len(d)})")
    ax.axhline(WEAK_SIGNAL_AUC, color=MUTED, lw=1, ls="--")
    ax.axvline(HIGH_MISSING_FRAC, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="Missing fraction (train)", ylabel="Univariate AUC strength",
           title="Sensor quality: missingness vs signal")
    ax.legend(loc="upper right", title="category (n sensors)", title_fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 3. Sensor correlations
# --------------------------------------------------

def spearman_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation using pairwise-complete observations."""
    return X.rank().corr(method="pearson", min_periods=100)


def find_correlated_groups(X: pd.DataFrame | None = None,
                           threshold: float = CORRELATION_THRESHOLD,
                           corr: pd.DataFrame | None = None,
                           priority: pd.Series | None = None) -> tuple[list, list]:
    """Group sensors linked by |rho| > threshold (connected components).

    One representative per group is kept: the sensor with the highest
    `priority` (default: fewest missing values). Unsupervised, so it can be
    re-fitted inside each CV training fold.

    Returns (groups, to_drop): groups of size >= 2, and sensors that can be
    removed while keeping one per group.
    """
    if corr is None:
        corr = spearman_matrix(X)
    cols = list(corr.columns)
    parent = {c: c for c in cols}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    values = corr.abs().to_numpy()
    rows, cols_idx = np.where(np.triu(values > threshold, k=1))
    for i, j in zip(rows, cols_idx):
        parent[find(cols[i])] = find(cols[j])

    components: dict[str, list] = {}
    for c in cols:
        components.setdefault(find(c), []).append(c)
    groups = [g for g in components.values() if len(g) >= 2]

    if priority is None:
        priority = X.notna().mean() if X is not None else pd.Series(0.0, index=cols)
    to_drop = []
    for g in groups:
        keep = max(g, key=lambda c: (priority.get(c, 0), -cols.index(c)))
        to_drop += [c for c in g if c != keep]
    return groups, to_drop


def correlation_analysis(train: pd.DataFrame, sensors: list,
                         threshold: float = CORRELATION_THRESHOLD,
                         exclude: list | None = None) -> dict:
    """Spearman matrix, most correlated pairs and redundant groups.

    `exclude` lists sensors that should never be chosen as a group's
    representative (for example sensors unavailable in test).
    """
    exclude = set(exclude or [])
    usable = [c for c in sensors if train[c].nunique() > 1]
    corr = spearman_matrix(train[usable]).fillna(0)
    np.fill_diagonal(corr.values, 1.0)

    priority = train[usable].notna().mean()
    priority[list(exclude & set(usable))] = -1
    groups, to_drop = find_correlated_groups(corr=corr, threshold=threshold, priority=priority)

    upper = corr.abs().where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("abs_rho").reset_index()
    pairs.columns = ["sensor_a", "sensor_b", "abs_rho"]
    pairs = pairs.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    group_table = pd.DataFrame([
        {"group": i, "size": len(g),
         "kept": next(c for c in g if c not in to_drop),
         "dropped": ", ".join(c for c in g if c in to_drop)}
        for i, g in enumerate(sorted(groups, key=len, reverse=True), 1)
    ]).set_index("group")
    return {"corr": corr, "pairs": pairs, "groups": groups, "group_table": group_table, "to_drop": to_drop}


def plot_correlation_heatmap(corr: pd.DataFrame):
    """Spearman correlation, ordered by hierarchical clustering on 1 - |rho|."""
    dist = squareform(np.clip(1 - corr.abs().values, 0, None), checks=False)
    order = leaves_list(linkage(dist, method="average"))
    ordered = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(ordered.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set(xticks=[], yticks=[], title=f"Spearman correlation of {len(corr)} sensors (clustered order)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 4. Train vs test distributions
# --------------------------------------------------

def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI of `actual` against `expected`, quantile bins from `expected`.

    Missing values form their own bin, so a change in missing rate also counts.
    """
    exp_obs = expected.dropna()
    if exp_obs.nunique() <= 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = np.unique(np.quantile(exp_obs, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

    def proportions(s):
        counts = np.append(np.histogram(s.dropna(), bins=edges)[0].astype(float), s.isna().sum())
        return np.clip(counts / counts.sum(), 1e-6, None)

    e, a = proportions(expected), proportions(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame, sensors: list) -> pd.DataFrame:
    """KS statistic, PSI and missing-rate gap for every sensor."""
    rows = []
    for col in sensors:
        a, b = train[col], test[col]
        a_obs, b_obs = a.dropna(), b.dropna()
        ks = ks_2samp(a_obs, b_obs, method="asymp").statistic if a_obs.nunique() > 1 and len(b_obs) else np.nan
        rows.append({
            "sensor": col, "psi": population_stability_index(a, b), "ks_stat": ks,
            "train_missing": a.isna().mean(), "test_missing": b.isna().mean(),
            "train_median": a_obs.median() if len(a_obs) else np.nan,
            "test_median": b_obs.median() if len(b_obs) else np.nan,
        })
    shift = pd.DataFrame(rows).set_index("sensor")
    shift["missing_gap"] = shift["test_missing"] - shift["train_missing"]
    shift["psi_band"] = pd.cut(shift["psi"], [-np.inf, PSI_MODERATE, PSI_MAJOR, np.inf],
                               labels=["stable", "moderate", "major"])
    return shift.sort_values("psi", ascending=False)


def plot_psi(shift: pd.DataFrame, top: int = 20, title: str = "Train vs test shift (PSI)"):
    d = shift.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.24 * top + 1.2))
    ax.barh(d.index, d["psi"], height=0.6, color=BLUE)
    ax.axvline(PSI_MAJOR, color=MUTED, ls="--", lw=1)
    ax.text(PSI_MAJOR, len(d) - 0.3, " major shift (0.25)", color=INK_2, fontsize=8, va="bottom")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set(xlabel="PSI (symmetric log scale)", title=title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_train_test_distributions(train: pd.DataFrame, test: pd.DataFrame, sensors: list,
                                  test_label: str = "test"):
    n = len(sensors)
    nrows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.2 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), sensors):
        a, b = np.log1p(train[col].dropna()), np.log1p(test[col].dropna())
        lo, hi = min(a.min(), b.min() if len(b) else a.min()), max(a.max(), b.max() if len(b) else a.max())
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        ax.hist(a, bins=bins, density=True, histtype="step", lw=2, color=BLUE, label="train")
        if len(b):
            ax.hist(b, bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label=test_label)
        ax.set(title=col, xlabel="log1p(value)", yticks=[])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 5. Test-set sensor misalignment
# --------------------------------------------------
# Matching every test column against every training column (KS statistic)
# and checking row-level relationships shows three systematic changes that
# affect every test vehicle:
#   1. Block shift: test s055..s063 hold the values of train s054..s062;
#      test s054 is all zero (train s063 is almost always 0, little is lost).
#   2. s047 moved: test s047 is 100% missing and test s122 holds s047's values
#      (its ratio to s012 matches train s047, not train s122). True s122 is lost.
#   3. s120 offset: train s120 = 1.7416 * s012 (residual about +/-500), but
#      test s120 is about 16.02 million higher for every vehicle.

SHIFTED_BLOCK = [f"s{i:03d}" for i in range(54, 64)]   # s054..s063
S047_SOURCE_IN_TEST = "s122"


def find_unavailable_in_test(train: pd.DataFrame, test: pd.DataFrame, sensors: list) -> list:
    """Sensors observed in training but entirely missing in test."""
    return [c for c in sensors if test[c].isna().all() and train[c].notna().any()]


def detect_column_mismatches(train: pd.DataFrame, test: pd.DataFrame, sensors: list,
                             sample: int = 5000, margin: float = 0.02) -> pd.DataFrame:
    """For each test column, the training column whose distribution matches best.

    `mismatch` is True when another training column matches clearly better
    (by at least `margin` in KS statistic) than the column's own.
    """
    samples = {c: train[c].dropna().sample(min(sample, train[c].notna().sum()), random_state=RANDOM_STATE)
               for c in sensors if train[c].notna().any()}
    rows = []
    for col in sensors:
        b = test[col].dropna()
        if b.empty:
            rows.append({"test_sensor": col, "best_train_match": None, "ks_best": np.nan,
                         "ks_own": np.nan, "mismatch": True, "note": "all missing in test"})
            continue
        b = b.sample(min(sample, len(b)), random_state=RANDOM_STATE)
        ks = {t: ks_2samp(a, b, method="asymp").statistic for t, a in samples.items()}
        best = min(ks, key=ks.get)
        own = ks.get(col, np.nan)
        rows.append({"test_sensor": col, "best_train_match": best, "ks_best": ks[best], "ks_own": own,
                     "mismatch": best != col and own - ks[best] > margin, "note": ""})
    return pd.DataFrame(rows).set_index("test_sensor")


def block_ks_matrix(train: pd.DataFrame, test: pd.DataFrame, block: list = SHIFTED_BLOCK,
                    sample: int = 5000) -> pd.DataFrame:
    """KS statistic between every test column (rows) and train column (columns) in a block."""
    tr = {c: train[c].dropna().sample(min(sample, train[c].notna().sum()), random_state=RANDOM_STATE)
          for c in block}
    te = {c: test[c].dropna().sample(min(sample, test[c].notna().sum()), random_state=RANDOM_STATE)
          for c in block}
    return pd.DataFrame([[ks_2samp(tr[j], te[i], method="asymp").statistic for j in block] for i in block],
                        index=[f"test {c}" for c in block], columns=[f"train {c}" for c in block])


def plot_block_ks(ks: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(ks.values, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(ks.shape[1]), ks.columns, rotation=60, ha="right")
    ax.set_yticks(range(ks.shape[0]), ks.index)
    for i in range(ks.shape[0]):
        for j in range(ks.shape[1]):
            v = ks.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v < 0.35 else INK_2)
    ax.grid(False)
    ax.set_title("KS distance, test vs train columns (dark = same distribution)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="KS statistic")
    fig.tight_layout()
    return fig


def estimate_s120_offset(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Constant added to s120 in test, from the train relation s120 = a * s012 + b."""
    m = train[["s012", "s120"]].dropna()
    a, b = np.polyfit(m["s012"], m["s120"], 1)
    t = test[["s012", "s120"]].dropna()
    return float(np.median(t["s120"] - (a * t["s012"] + b)))


def plot_s120_offset(train: pd.DataFrame, test: pd.DataFrame, n: int = 3000):
    """s120 against s012 for train and test (test sits on a parallel, offset line)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for df, colour, label in [(train, BLUE, "train"), (test, ORANGE, "test")]:
        d = df[["s012", "s120"]].dropna()
        d = d.sample(min(n, len(d)), random_state=RANDOM_STATE)
        ax.scatter(d["s012"] + 1, d["s120"] + 1, s=8, color=colour, alpha=0.5, label=label, linewidth=0)
    ax.set(xscale="log", yscale="log", xlabel="s012 + 1 (log)", ylabel="s120 + 1 (log)",
           title=f"s120 vs s012: test is offset by ~{estimate_s120_offset(train, test) / 1e6:.2f} million")
    ax.legend(markerscale=2)
    fig.tight_layout()
    return fig


def ratio_to_s012(df: pd.DataFrame, col: str, q=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.Series:
    """Quantiles of col / s012, used to identify which sensor a column really holds."""
    return (df[col] / df["s012"].replace(0, np.nan)).dropna().quantile(list(q))


def align_test_sensors(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `test` with the misaligned sensors mapped back.

    - s054..s062 <- test s055..s063; s063 set to missing (value lost).
    - s047 <- test s122; s122 set to missing (value lost).
    - s120 <- test s120 minus the estimated constant offset.

    s063 and s122 are then fully missing in test and should be excluded from
    models, because they are never observed at prediction time.
    """
    out = test.copy()
    block = [c for c in SHIFTED_BLOCK if c in out.columns]
    out[block[:-1]] = test[block[1:]].to_numpy()
    out[block[-1]] = np.nan
    out["s047"] = test[S047_SOURCE_IN_TEST]
    out[S047_SOURCE_IN_TEST] = np.nan
    out["s120"] = test["s120"] - estimate_s120_offset(train, test)
    return out


# --------------------------------------------------
# 6. Depot / latent-group distribution shift
# --------------------------------------------------

def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, features: list,
                           n_splits: int = 5) -> dict:
    """Train a classifier to tell train rows from test rows.

    AUC near 0.5 = the sets look alike; a high AUC = a shift, and the gain
    importances show which sensors carry it. Also reports how "test-like"
    healthy vs failed training vehicles look.
    """
    import lightgbm as lgb

    X = pd.concat([train[features], test[features]], ignore_index=True)
    is_test = np.r_[np.zeros(len(train)), np.ones(len(test))].astype(int)
    oof = np.zeros(len(X))
    gains = pd.Series(0.0, index=features)
    for tr_idx, va_idx in StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X, is_test):
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                   verbose=-1)
        model.fit(X.iloc[tr_idx], is_test[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        gains += pd.Series(model.booster_.feature_importance("gain"), index=features)

    train_scores = pd.Series(oof[: len(train)]).groupby(train["failure"].to_numpy()).mean()
    return {
        "auc": float(roc_auc_score(is_test, oof)),
        "importance": (gains / gains.sum()).sort_values(ascending=False).rename("gain_share"),
        "testlikeness_healthy": float(train_scores.get(0, np.nan)),
        "testlikeness_failed": float(train_scores.get(1, np.nan)),
    }


def latent_groups(train: pd.DataFrame, test: pd.DataFrame, sensors: list,
                  k_range=range(2, 9), sample_size: int = 8000) -> dict:
    """Cluster vehicles into latent groups as a stand-in for depots.

    Unsupervised (labels unused), fitted on train + test. Two views:
    - "values": log1p sensor values, median-imputed and standardised;
    - "missingness": which sensors are missing (depots may run different
      sensor kits).
    k is chosen by silhouette score on a sample. For each view the cluster
    shares in train vs test are compared (Cramer's V; chi-square p-values are
    always tiny at this sample size), with failure rate per cluster.
    """
    usable = [c for c in sensors if train[c].nunique() > 1 and test[c].notna().any()]
    both = pd.concat([train[usable], test[usable]], ignore_index=True)
    source = np.r_[np.repeat("train", len(train)), np.repeat("test", len(test))]
    rng = np.random.default_rng(RANDOM_STATE)
    views = {
        "values": StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(signed_log(both))),
        "missingness": both.isna().astype(float).to_numpy(),
    }

    results = {}
    for view, Z in views.items():
        sample = rng.choice(len(Z), size=min(sample_size, len(Z)), replace=False)
        sil = {}
        for k in k_range:
            labels = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit_predict(Z[sample])
            if len(np.unique(labels)) > 1:
                sil[k] = silhouette_score(Z[sample], labels, random_state=RANDOM_STATE)
        best_k = max(sil, key=sil.get)
        labels = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)

        table = pd.crosstab(labels, source)
        chi2 = chi2_contingency(table)[0]
        shares = table / table.sum()
        train_labels = labels[: len(train)]
        profile = pd.DataFrame({
            "train_share": shares["train"],
            "test_share": shares["test"],
            "failure_rate": train["failure"].groupby(train_labels).mean(),
            "n_failures": train["failure"].groupby(train_labels).sum(),
            "median_repair_cost": train["repair_cost"].groupby(train_labels).median(),
        }).sort_values("failure_rate")
        profile.index.name = "cluster"
        results[view] = {
            "best_k": best_k,
            "silhouette": pd.Series(sil, name="silhouette"),
            "cramers_v": float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))),
            "profile": profile,
        }
    return results


def predicted_risk_shift(train: pd.DataFrame, test: pd.DataFrame, features: list,
                         n_splits: int = 5) -> dict:
    """Compare predicted failure risk on train (out-of-fold) and test.

    A simple LightGBM failure model is trained on the training data. A much
    higher mean predicted risk on test means the test population contains more
    failure-like vehicles, which changes how many vehicles should be inspected.
    """
    import lightgbm as lgb

    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                  random_state=RANDOM_STATE, verbose=-1)
    X, y = train[features], train["failure"].to_numpy()
    oof = np.zeros(len(X))
    for tr_idx, va_idx in StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X, y):
        oof[va_idx] = lgb.LGBMClassifier(**params).fit(X.iloc[tr_idx], y[tr_idx]).predict_proba(X.iloc[va_idx])[:, 1]
    p_test = lgb.LGBMClassifier(**params).fit(X, y).predict_proba(test[features])[:, 1]

    table = pd.DataFrame({
        "mean_predicted_risk": [oof.mean(), p_test.mean()],
        "share_risk_above_0.1": [(oof > 0.1).mean(), (p_test > 0.1).mean()],
        "share_risk_above_0.5": [(oof > 0.5).mean(), (p_test > 0.5).mean()],
    }, index=["train (out-of-fold)", "test (aligned)"])
    return {"table": table, "oof_auc": float(roc_auc_score(y, oof)),
            "ratio": float(p_test.mean() / oof.mean()), "p_train_oof": oof, "p_test": p_test}


def plot_risk_distribution(risk: dict):
    """Share of vehicles whose predicted risk is at or above each level (train vs test)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = np.logspace(-4, 0, 200)
    for p, colour, label in [(risk["p_train_oof"], BLUE, "train (out-of-fold)"),
                             (risk["p_test"], ORANGE, "test (aligned)")]:
        share = np.array([(p >= t).mean() for t in levels]) * 100
        ax.plot(levels, share, color=colour, label=label)
    ax.set(xscale="log", yscale="log", xlabel="Predicted failure risk threshold (log)",
           ylabel="Vehicles at or above threshold (%, log)",
           title="Test vehicles score higher at every risk level")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 7. Feature recommendations
# --------------------------------------------------

def feature_recommendations(quality: pd.DataFrame, corr: dict, unavailable: list) -> pd.DataFrame:
    """Recommended action per sensor for modelling.

    - drop (unavailable in test): never observed at prediction time.
    - drop (constant): carries no information.
    - candidate (quasi-constant / redundant): removal is tested in CV, with
      the selection re-fitted inside each training fold.
    - keep: everything else.
    """
    rec = pd.Series("keep", index=quality.index, name="action")
    rec[rec.index.isin(corr["to_drop"])] = "candidate: redundant (|rho| > 0.95)"
    rec[quality["quasi_constant"]] = "candidate: quasi-constant"
    rec[quality["constant"]] = "drop: constant"
    rec[rec.index.isin(unavailable)] = "drop: unavailable in test"
    return rec.to_frame().join(quality[["missing_frac", "auc_strength"]])


# --------------------------------------------------
# Command-line summary
# --------------------------------------------------

def main():
    train, test = load_data()
    sensors = get_sensor_columns(train)

    signal = sensor_signal(train, sensors)
    quality = feature_quality(train, sensors, signal)
    test_aligned = align_test_sensors(test, train)
    lost = find_unavailable_in_test(train, test_aligned, sensors)
    corr = correlation_analysis(train, sensors, exclude=lost)
    shift_raw = train_test_shift(train, test, sensors)
    shift_aligned = train_test_shift(train, test_aligned, sensors).drop(index=lost)
    usable = [c for c in sensors if c not in lost and c not in CONSTANT_SENSORS]
    adv_raw = adversarial_validation(train, test, sensors)
    adv_aligned = adversarial_validation(train, test_aligned, usable)
    risk = predicted_risk_shift(train, test_aligned, usable)
    rec = feature_recommendations(quality, corr, lost)

    print("Top sensors by univariate AUC:", ", ".join(f"{s} {a:.3f}" for s, a in signal["auc_strength"].head(6).items()))
    print("Constant:", list(quality.index[quality["constant"]]))
    print("Quasi-constant:", list(quality.index[quality["quasi_constant"]]))
    print(f"Weak-signal sensors (AUC < {WEAK_SIGNAL_AUC}):", int(quality["weak_signal"].sum()))
    print(f"Correlated groups (|rho| > {CORRELATION_THRESHOLD}): {len(corr['groups'])}, redundant sensors: {len(corr['to_drop'])}")
    print(f"Major-PSI sensors: raw test {int((shift_raw['psi'] > PSI_MAJOR).sum())}, "
          f"aligned test {int((shift_aligned['psi'] > PSI_MAJOR).sum())}")
    print(f"s120 test offset: {estimate_s120_offset(train, test):,.0f}; lost in aligned test: {lost}")
    print(f"Adversarial AUC: raw {adv_raw['auc']:.3f}, aligned {adv_aligned['auc']:.3f}")
    print(f"Predicted risk test / train: {risk['ratio']:.2f}x")
    print("\nRecommended actions:\n", rec["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
