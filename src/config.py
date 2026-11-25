"""Central configuration for Fleet Maintenance Planning."""

from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Reproducibility
RANDOM_STATE = 42
N_SPLITS = 5

# Shared cross-validation artefact
FOLDS_PATH = RESULTS_DIR / "cv_folds.npz"

# Final model configuration
FINAL_SEEDS = (42, 7, 123, 2024, 99)

TUNED_LGBM_PARAMS = {
    "learning_rate": 0.0945,
    "n_estimators": 800,
    "num_leaves": 63,
    "min_child_samples": 80,
    "subsample": 1.0,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0873,
    "scale_pos_weight": 1,
}
