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

## Project Overview

The objective of this project is to develop machine learning models for fleet maintenance planning using vehicle sensor data. The project focuses on predicting vehicle failures and estimating repair costs to support maintenance decisions.

## Important Notes for All Models

- **The test file is misaligned.** Several test columns hold another sensor's values:
  - test s055–s063 contain train s054–s062;
  - s047's values are in s122;
  - s120 has a constant +16.02M offset.

  Before predicting test, apply:

  ```python
  from src.feature_analysis import align_test_sensors
  test_aligned = align_test_sensors(test_df, train_df)
  ```

  Then drop `s063` and `s122` as features, because they are not available in the aligned test set. Linear and distance-based models (for example Logistic Regression) are strongly affected by the misalignment; LightGBM much less so.
- **The test population is higher-risk** than train: about 3x the predicted failure risk. CV cost is therefore an optimistic estimate of test cost, and a larger share of test vehicles may need inspection.
- **CV noise:** with the shared 5-fold CV, differences in out-of-fold cost below about 3,000 are within random-seed noise and are not meaningful.

Evidence: `notebooks/eda.ipynb` Sections 6.5, 6.6 and 7.

## Repository Structure

- `data/` – Training and test datasets along with the sample submission file (stored locally, not tracked by Git).
- `notebooks/` – Main project notebook.
- `src/` – Reusable Python code for preprocessing, feature engineering and modelling.
- `submissions/` – Generated Kaggle submission files (stored locally, not tracked by Git).
- `CONTRIBUTIONS.md` – Individual contribution records.
- `SUBMISSIONS.md` – Kaggle submission history.


*Project methodology, model details and evaluation results will be added as the project progresses.*