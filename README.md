# Fleet Maintenance Planning (P2)

**Course:** Machine Learning - ML (CSE343)  
**Team Number:** 38

## Team Members

| Name | Roll Number |
|---|---|
| Avnish Raj | 2024141 |
| Mohd Farhan Ali | 2024352 |
| Shaik Mohd Aatiq | 2024518 |
| Saransh Kumar Gupta | 2024508 |

## 1. Project Overview

The objective of this project is to develop machine learning models for fleet maintenance planning using vehicle sensor data.

The project focuses on two prediction tasks:

- **Failure prediction:** Estimate the probability of vehicle failure.
- **Repair-budget prediction:** Estimate the repair budget required for a vehicle.

These predictions are combined with a cost-based inspection policy to support maintenance decisions.

## 2. Important Dataset Considerations

### Test sensor misalignment

The test file is misaligned. Several test columns hold another sensor's values:

- Test `s055`–`s063` contain train `s054`–`s062`.
- `s047`'s values are in `s122`.
- `s120` has a constant +16.02M offset.

Before predicting on test data, apply the alignment function:

```python
from src.feature_analysis import align_test_sensors

test_aligned = align_test_sensors(test_df, train_df)
```

Then exclude `s063` and `s122` as features because they are unavailable in the aligned test set.

Linear and distance-based models, such as Logistic Regression, are strongly affected by the misalignment; LightGBM is much less affected.

### Test population and cross-validation

- **Higher-risk test population:** The test population has approximately 3x the predicted failure risk of the training population. CV cost should therefore be treated as an optimistic estimate of test cost.
- **CV variability:** With the shared five-fold CV, differences in out-of-fold cost below approximately 3,000 may fall within random-seed noise and should not automatically be treated as meaningful improvements.

**Evidence:** `notebooks/eda.ipynb`, Sections 6.5, 6.6 and 7.

## 3. Repository Structure

```text
fleet_maintenance_p2_team38/
│
├── data/
│   └── Local training, test and sample submission data
│
├── notebooks/
│   ├── eda.ipynb
│   ├── final_notebook.ipynb
│   └── lightgbm.ipynb
│
├── results/
│   ├── cv_folds.npz
│   ├── budget_experiments.csv
│   ├── final_budget_oof_predictions.csv
│   ├── final_budget_policy.csv
│   ├── nested_budget_comparison.csv
│   ├── nested_budget_fold_results.csv
│   ├── repair_comparison.csv
│   └── repair_oof.csv
│
├── src/
│   ├── config.py
│   ├── cost_model.py
│   ├── cv_utils.py
│   ├── evaluation.py
│   ├── feature_analysis.py
│   ├── lightgbm_model.py
│   ├── nested_budget_evaluation.py
│   ├── preprocessing.py
│   └── repair_model.py
│
├── submissions/
│   └── Generated submission CSV files
│
├── CONTRIBUTIONS.md
├── SUBMISSIONS.md
└── README.md
```

## 4. Main Workflow

The main end-to-end workflow is implemented in `notebooks/final_notebook.ipynb`.

It covers the following stages:

1. Load and validate the training, test and sample submission data.
2. Load and validate the shared five-fold CV splits.
3. Prepare the sensor features available for both training and test.
4. Generate out-of-fold failure probabilities using the final LightGBM classifier.
5. Generate out-of-fold repair-budget predictions.
6. Tune the inspection threshold and repair-budget multiplier using pooled OOF predictions.
7. Fit the final models on the complete training dataset.
8. Generate test predictions.
9. Validate and export the final submission CSV.

Run the cells in `final_notebook.ipynb` sequentially to reproduce the workflow.

## 5. Source Modules

| Module | Responsibility |
|---|---|
| `config.py` | Shared paths, random seeds and model configuration |
| `cost_model.py` | Constant repair-budget baseline |
| `cv_utils.py` | Shared folds, fold validation and CV utilities |
| `evaluation.py` | Cost calculation, threshold selection and policy evaluation |
| `feature_analysis.py` | Data loading, sensor alignment and feature analysis |
| `lightgbm_model.py` | Final LightGBM classifier and failure-prediction CV |
| `nested_budget_evaluation.py` | Nested cross-validation for repair-budget evaluation |
| `preprocessing.py` | Sensor selection, imputation and preprocessing |
| `repair_model.py` | Repair-budget model training and OOF predictions |

## 6. Notebooks

- **`eda.ipynb`:** Exploratory data analysis, dataset investigation and sensor-alignment findings.
- **`final_notebook.ipynb`:** Main end-to-end modelling, policy selection and submission-generation workflow.
- **`lightgbm.ipynb`:** Supporting LightGBM experiments and historical modelling analysis.

## 7. Results and Submissions

- `results/` stores cross-validation splits, OOF predictions, model comparisons and experiment evidence.
- `submissions/` stores generated submission CSV files.
- `SUBMISSIONS.md` records submission history.
- `CONTRIBUTIONS.md` records individual team contributions.

Training and test datasets are stored locally and are not tracked by Git. Generated submission files are also kept locally unless explicitly added to version control.

## 8. Reproducibility

The project uses shared five-fold CV splits stored in `results/cv_folds.npz`. The same folds are reused across model evaluations to make comparisons consistent.

Random seeds and shared model settings are maintained in `src/config.py`.

For the final workflow, use `notebooks/final_notebook.ipynb` rather than running individual experiment scripts.