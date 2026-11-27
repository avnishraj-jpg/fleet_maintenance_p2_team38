# Individual Contributions

This document tracks the individual contributions of all team members across data exploration, preprocessing, feature engineering, modeling, cost optimization, and final submission generation.

### Avnish Raj (2024141)

- Set up the initial repository layout, notebooks, execution environment, and shared project configuration.
- Conducted comprehensive exploratory data analysis (EDA), examining failure class imbalance, right-skewed repair-cost distributions, missing-value patterns across sensor groups, and train–test distribution shifts.
- Addressed right-censoring in the training set by analyzing observation windows (`followup_days`), filtering short-follow-up negatives, and introducing exposure-based sample weighting (`followup_days / 365`).
- Resolved test-set sensor corruption by implementing cyclic realignment for the shifted `s054–s063` sensor block and within-split percentile rank normalization for the rescaled `s122` sensor, alongside engineered missingness-block indicators and key sensor wear ratios.
- Designed, validated, and executed the self-contained end-to-end modeling and submission pipelines (`notebooks/model_1.25_1.6.ipynb` and `notebooks/standalone_final.ipynb`), integrating Platt-calibrated failure probability ensembles with log-cost quantile budgeting to produce the team's top three Kaggle submissions (`~20,000.00`, `20,614.22`, and `~20,700.00`).

### Mohd Farhan Ali (2024352)

- Built the modular data cleaning and preprocessing pipeline (`src/preprocessing.py`), removing constant/corrupted columns (`s146`) and implementing leak-free `fit` / `transform` utilities (`SensorPreprocessor`) fitted strictly within training folds.
- Benchmarked multiple imputation strategies (median, mean, constant, and KNN) alongside missingness indicators, feature scaling (`RobustScaler`), and signed-log transformations (`np.sign(x) * np.log1p(|x|)`).
- Implemented L1-regularized Logistic Regression feature selection (`src/feature_selection.py`), evaluating subset stability across cross-validation folds and comparing reduced sensor sets against the full feature space on both linear and tree-based models.
- Contributed leak-free fold-wise preprocessing, missingness-block aggregations (`NA_BLOCK_70`, `NA_BLOCK_20`, `NA_BLOCK_9A`, `NA_BLOCK_9B`), and wear-ratio feature transformations used in the final linear calibration and Ridge log-cost components.

### Shaik Mohd Aatiq (2024518)

- Implemented baseline and class-weighted Logistic Regression pipelines (`src/failure_model.py`), systematically evaluating class-imbalance strategies (class weights and random oversampling) directly against the competition cost metric.
- Led deep sensor-level statistical analysis (`src/feature_analysis.py` and `notebooks/eda.ipynb`), quantifying univariate predictive power, missingness-signal relationships, high-correlation sensor clusters (`|rho| > 0.95`), and Kolmogorov–Smirnov / PSI distribution shifts.
- Identified the test-set sensor misalignment (`s054–s063` shift, `s120` +16.02M offset, and `s047`/`s122` discrepancy) and quantified the ~3x higher failure risk in the test depot using adversarial validation and latent-group analysis.
- Developed and tuned multi-seed LightGBM and gradient-boosted ensemble models (`src/lightgbm_model.py`, `src/ensemble_model.py`, and `notebooks/lightgbm.ipynb`), establishing seed-noise bounds and informing the Bayes prior-odds / covariate-shift probability calibration used in the final standalone models.

### Saransh Kumar Gupta (2024508)

- Implemented the baseline percentile repair-budget estimator (`src/cost_model.py`), modeling conditional repair costs on failed vehicles and enforcing strict input validation for flags, repair costs, and budget allocations.
- Formulated the Bayes-optimal asymmetric quantile decision rule ($F_C(B^*) = 1 - \frac{1 - L}{p}$ with $L = 0.20 / 0.22 \approx 0.9091$) to balance the 20% under-budget shortfall penalty against the 2% unused-budget penalty and fixed $10 inspection cost.
- Evaluated nested cross-validation policies (`src/nested_budget_evaluation.py` and `src/evaluation.py`), comparing constant-budget baselines against conditional log-cost distribution modeling.
- Co-designed the decoupled inspection and budget probability multipliers (`P_FLAG_MULT = 1.25`, `P_BUDG_MULT = 1.60`), upper-tail dollar scaling (`BUDGET_SCALE = 1.25`), outlier budget capping (`$8,500`), and automated submission integrity assertions.
