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

---

## 1. Project Overview

The goal of this project is to build an end-to-end machine learning and decision-optimization pipeline for predictive fleet maintenance using multi-sensor telemetry data. For each vehicle in the test fleet, the system outputs two decisions:

- **`flag` (`0` or `1`):** Whether to bring the vehicle in for preventive maintenance inspection (incurring a fixed `$10` inspection fee).
- **`budget` (`>= 0`):** How much spare-parts budget to allocate when `flag = 1`.

Predictions are optimized directly against the competition's asymmetric cost structure:
- **Unflagged failure (`flag = 0, failure = 1`):** Full repair cost incurred.
- **Flagged vehicle (`flag = 1`):** Fixed `$10` inspection cost, plus a **`20%` penalty** on any repair shortfall (`max(repair_cost - budget, 0)`) and a **`2%` penalty** on any unused budget (`max(budget - repair_cost, 0)` for failures, or `budget` for healthy vehicles).

Because under-budgeting is penalized **10x more heavily** than over-budgeting (`20%` vs. `2%`), the solution decouples calibrated failure-risk estimation from conditional repair-cost distribution modeling to select Bayes-optimal inspection flags and budget quantiles.

---

## 2. Key Dataset Insights & Preprocessing Decisions

- **Class Imbalance & Right-Censoring:** Only `0.71%` of raw training records represent failures, and non-failing vehicles with short observation windows (`followup_days < 365`) are right-censored. Filtering short-follow-up negatives or weighting observations by exposure (`followup_days / 365`) removes downward bias in predicted failure probabilities.
- **Test Sensor Misalignment & Corruption:**
  - **Cyclic Rotation (`s054–s063`):** Test columns `s055–s063` hold training sensors `s054–s062`, while test `s063` wraps around to hold `s054`. Cyclically shifting `s054–s063` in the test set restores all 10 sensors.
  - **Scale Shift (`s122`) & Missingness (`s047`):** Sensor `s047` is completely missing in test, while `s122` is rescaled by `~2.68x` relative to train. Within-split percentile rank normalization recovers `s122` cleanly without train–test leakage.
  - **Constant Offset (`s120`) & Constant Column (`s146`):** `s120` is shifted by `+16.02M` in test (duplicate of `s012`) and `s146` is constant; both are excluded.
- **Depot Covariate Shift:** Adversarial validation and risk-distribution analysis show that the test fleet has roughly **3x higher failure risk** than the training depot, requiring Bayes prior-odds / covariate-shift adjustment and calibrated decision multipliers.
- **Structured Missingness & Cost Predictors:** Missing values occur in correlated subsystem blocks (`NA_BLOCK_70`, `NA_BLOCK_20`, `NA_BLOCK_9A`, `NA_BLOCK_9B`). Aggregating block missingness rates alongside log-transformed wear sensors and key ratios (`s038/s012`, `s083/s012`, `s033/s097`) improves both failure discrimination and repair-cost estimation.

---