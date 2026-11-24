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
- To be updated.

### Saransh Kumar Gupta (2024508)
- Implemented fit_constant_budget() to calculate a baseline repair budget using the 90.91th percentile of training repair costs, treating missing repair costs as zero.
- Implemented predict_budgets() to assign the baseline budget to inspected vehicles and zero budget to uninspected vehicles.
- Added input validation to handle invalid repair costs, inspection flags, and budget values.