# Kaggle Submission History

| Tag | Date | Version | Description | CV Result | Kaggle Result |
|---|---|---|---|---:|---:|
| `v1-baseline-2026-11-25` | 25-Nov-2026 | V1 | Baseline Logistic Regression | 12,210.07 | 84,066.16000 |
| `submission_1.25_1.6_2` | 27-Nov-2026 | V2 (3rd Best) | Cost-weighted HGB + censoring filter + prior-shift scaling (`1.25`, `1.60`), 1,165 flags, `$809,378.28` budget | — | **20757.10920** |
| `submission_standalone_final` | 27-Nov-2026 | V3 Final (2nd Best / Standalone) | End-to-end exposure-weighted GBDT+Linear Platt ensemble, realigned `s054–s063` + rank-normalized `s122`, Ridge+HGB log-cost (`1.25`, `1.60`, `1.25x` budget scale, `$8,500` cap), 1,169 flags, `$801,972.38` budget | — | **20,614.21860** |
| `submission_1.25_1.6` | 27-Nov-2026 | V2 (1st Best) | Cost-weighted HGB ensemble + censoring filter + Platt calibration + multiplier policy (`1.25`, `1.60`), 1,176 flags, `$843,884.89` budget | — | **20545.47320** |