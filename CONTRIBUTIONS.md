# Individual Contributions

This document tracks the individual contributions of all team members.

### Avnish Raj (2024141)

- Set up the initial project notebook, imports and configuration.
- Loaded the training and test datasets and performed initial data inspection.
- Conducted exploratory data analysis (EDA), including:
  - Failure class distribution.
  - Repair-cost distribution.
  - Sensor missing-value analysis.
  - Identification of highly missing and constant value sensors.
  - Train and test sensor distribution comparison.

### Mohd Farhan Ali (2024352)

- To be updated.

### Shaik Mohd Aatiq (2024518)

- Implemented `src/failure_model.py`: Logistic Regression baseline, training function, probability-prediction function, random oversampling (training data only) and evaluation (precision, recall, F1-score, ROC-AUC, confusion matrix).
- Compared imbalance-handling approaches (balanced weights, no weighting, custom weights 1:10/1:25/1:50, random oversampling) using official validation cost after threshold selection.
- Selected Logistic Regression with class weight 1:50 (validation cost 12,349; detects 67 of 79 failures, wrongly flags 180 healthy vehicles).

### Saransh Kumar Gupta (2024508)

- Implemented fit_constant_budget() to calculate a baseline repair budget using the 90.91th percentile of training repair costs, treating missing repair costs as zero.
- Implemented predict_budgets() to assign the baseline budget to inspected vehicles and zero budget to uninspected vehicles.
- Added input validation to handle invalid repair costs, inspection flags, and budget values.
