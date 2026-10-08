# Results: all 22 TPC-H templates

420 queries from 22 templates. Runtime 26 to 2809 ms. Within a template the slowest query is at most 1.63x the fastest (median 1.18x). Repeat spread: the three back-to-back timed runs lie within 1.030x of their median (median over queries), 1.096x at the 90th percentile. This is a lower bound on label noise; scripts/compare_passes.py measures agreement between independent passes.

Predictions clipped at a bound (see LOG_MIN, LOG_MAX): E4/Cost + RF residual 1153, E4/Cost-only 1155, E4/Ridge 126.

E1: mean ± std over 5 random 70/30 splits. E2: every query's held-out prediction pooled once; the 95% interval resamples whole templates (2000 draws).

| Model | E1 median Q | E1 P90 Q | E1 log-MAE | E2 median Q [95% CI] | E2 P90 Q | E2 log-MAE | E2 within 2x |
|---|---|---|---|---|---|---|---|
| Cost-only | 1.66 ± 0.01 | 3.63 ± 0.05 | 0.63 ± 0.00 | 1.75 [1.56, 2.32] | 3.98 | 0.69 | 57% |
| Ridge | 1.09 ± 0.01 | 1.37 ± 0.01 | 0.13 ± 0.01 | 1.71 [1.41, 2.24] | 4.47 | 0.65 | 60% |
| Random Forest | 1.03 ± 0.00 | 1.08 ± 0.01 | 0.04 ± 0.00 | 1.77 [1.33, 2.25] | 3.43 | 0.65 | 55% |
| Cost + RF residual | 1.03 ± 0.00 | 1.08 ± 0.01 | 0.04 ± 0.00 | 1.56 [1.26, 2.36] | 3.34 | 0.59 | 59% |
| Template mean | 1.02 ± 0.00 | 1.08 ± 0.01 | 0.04 ± 0.00 | n/a | n/a | n/a | n/a |
| Global mean | n/a | n/a | n/a | 2.22 [1.42, 2.82] | 6.05 | 0.80 | 48% |

E2 median over held-out templates of each template's median Q-error: Cost-only 1.78, Ridge 1.75, Random Forest 2.03, Cost + RF residual 1.63, Global mean 2.29.

Held-out templates where the model's median Q-error beats Cost-only: Ridge 11/22, Random Forest 11/22, Cost + RF residual 17/22, Global mean 9/22. Worst held-out template: Cost-only 5.98, Ridge 4.65, Random Forest 5.74, Cost + RF residual 9.37, Global mean 14.75.

Paired with Cost-only on the same held-out templates. Primary: geometric mean over templates of the ratio of per-template median Q-errors, with a 95% interval from resampling templates. Secondary: ratio of pooled median Q-errors (same resamples). Then the two-sided sign test and the Wilcoxon signed-rank test over templates (both treat the overlapping folds as independent): Ridge GM 0.95 [0.80, 1.12], pooled 0.98 [0.69, 1.26], 11/22 lower, sign p = 1.000, Wilcoxon p = 0.775; Random Forest GM 0.97 [0.85, 1.11], pooled 1.02 [0.70, 1.29], 11/22 lower, sign p = 1.000, Wilcoxon p = 0.775; Cost + RF residual GM 0.90 [0.79, 1.03], pooled 0.89 [0.69, 1.26], 17/22 lower, sign p = 0.017, Wilcoxon p = 0.074; Global mean GM 1.16 [0.96, 1.43], pooled 1.27 [0.80, 1.45], 9/22 lower, sign p = 0.523, Wilcoxon p = 0.353.

Ridge penalty chosen (min, median, max over fits; grid 1e-6 to 1e5): E1 (leave one query out) 5.6e-06, 1.8e-05, 5.6e-05 (5 fits, 0 at a grid end); E2 (leave one template out) 32, 1e+02, 1.8e+02 (22 fits, 0 at a grid end); E3 (leave one query out) 1e-06, 3.2e-05, 0.001 (25 fits, 3 at a grid end); E4 k=1 (leave one query out) 1e-06, 1e+02, 1e+05 (22 fits, 13 at a grid end); E4 k=2 (leave one template out) 1e-06, 4.4e+02, 1e+05 (20 fits, 10 at a grid end); E4 k=3 (leave one template out) 5.6e-05, 1.4e+03, 1e+05 (20 fits, 9 at a grid end); E4 k=4 (leave one template out) 0.00032, 1e+03, 1e+05 (20 fits, 5 at a grid end); E4 k=6 (leave one template out) 5.6e-05, 56, 1e+05 (20 fits, 3 at a grid end); E4 k=8 (leave one template out) 0.0056, 1e+02, 1e+05 (20 fits, 2 at a grid end); E4 k=12 (leave one template out) 0.01, 56, 5.6e+02 (20 fits, 0 at a grid end); E4 k=16 (leave one template out) 3.2, 1e+02, 3.2e+02 (20 fits, 0 at a grid end); E4 k=21 (leave one template out) 32, 1e+02, 1.8e+02 (22 fits, 0 at a grid end).

## E2 median Q-error per held-out template

| Template | Cost-only | Ridge | Random Forest | Cost + RF residual | Global mean |
|---|---|---|---|---|---|
| Q1 | 2.94 | 4.57 | 3.01 | 2.43 | 6.09 |
| Q2 | 1.16 | 1.28 | 2.17 | 1.45 | 1.24 |
| Q3 | 2.24 | 2.05 | 1.72 | 2.48 | 1.16 |
| Q4 | 2.20 | 1.47 | 1.24 | 1.17 | 2.29 |
| Q5 | 1.02 | 1.35 | 1.33 | 1.02 | 1.06 |
| Q6 | 1.54 | 1.24 | 1.69 | 1.19 | 1.03 |
| Q7 | 3.86 | 4.65 | 3.30 | 2.45 | 3.38 |
| Q8 | 1.77 | 1.42 | 2.23 | 3.27 | 1.85 |
| Q9 | 1.79 | 1.11 | 1.15 | 1.48 | 2.29 |
| Q10 | 1.23 | 1.66 | 2.15 | 2.31 | 1.60 |
| Q11 | 2.28 | 1.10 | 1.20 | 1.78 | 2.80 |
| Q12 | 1.49 | 1.31 | 1.24 | 1.02 | 1.24 |
| Q13 | 5.98 | 4.55 | 5.46 | 9.37 | 5.11 |
| Q14 | 3.15 | 1.79 | 2.66 | 2.00 | 2.38 |
| Q15 | 1.57 | 2.30 | 1.29 | 1.25 | 1.13 |
| Q16 | 1.17 | 1.98 | 1.28 | 1.13 | 1.60 |
| Q17 | 1.56 | 2.39 | 1.90 | 1.35 | 3.02 |
| Q18 | 4.74 | 4.07 | 5.74 | 3.33 | 14.75 |
| Q19 | 4.20 | 1.61 | 4.43 | 3.60 | 7.67 |
| Q20 | 1.66 | 2.30 | 1.11 | 1.30 | 1.42 |
| Q21 | 2.64 | 2.98 | 2.42 | 2.50 | 2.45 |
| Q22 | 1.65 | 1.71 | 2.20 | 1.15 | 5.95 |

## E3 learning curve (median Q-error, mean ± std over 5 splits)

| Training queries | Cost-only | Ridge | Random Forest | Cost + RF residual |
|---|---|---|---|---|
| 43 (10%) | 1.71 ± 0.02 | 1.10 ± 0.02 | 1.13 ± 0.01 | 1.12 ± 0.02 |
| 84 (25%) | 1.67 ± 0.02 | 1.11 ± 0.01 | 1.04 ± 0.00 | 1.04 ± 0.01 |
| 148 (50%) | 1.68 ± 0.01 | 1.10 ± 0.01 | 1.03 ± 0.00 | 1.03 ± 0.00 |
| 232 (75%) | 1.67 ± 0.01 | 1.09 ± 0.01 | 1.02 ± 0.00 | 1.03 ± 0.00 |
| 294 (100%) | 1.66 ± 0.01 | 1.09 ± 0.01 | 1.03 ± 0.00 | 1.03 ± 0.00 |

## E4 template-diversity curve

Each unseen template's median Q-error, summarized as median [IQR] over all (training set, unseen template) pairs. Every possible training set for k = 1 and k = T - 1, 20 random ones otherwise.

| Training templates | Cost-only | Ridge | Random Forest | Cost + RF residual | Global mean |
|---|---|---|---|---|---|
| 1 | 4.57 [1.92, 19.69] | 3.82 [1.82, 11.26] | 2.97 [1.71, 6.10] | 4.57 [1.92, 19.66] | 2.97 [1.72, 6.09] |
| 2 | 2.61 [1.51, 5.95] | 2.50 [1.50, 5.15] | 2.21 [1.45, 4.49] | 2.57 [1.50, 5.98] | 2.18 [1.41, 4.50] |
| 3 | 2.74 [1.52, 6.83] | 2.32 [1.50, 4.26] | 2.35 [1.56, 4.24] | 2.57 [1.54, 6.79] | 2.35 [1.56, 4.34] |
| 4 | 2.22 [1.47, 3.65] | 1.98 [1.36, 3.45] | 2.08 [1.36, 3.62] | 2.10 [1.42, 3.62] | 2.12 [1.39, 4.39] |
| 6 | 2.11 [1.50, 3.20] | 2.26 [1.46, 3.92] | 2.16 [1.41, 3.34] | 2.12 [1.59, 3.16] | 2.17 [1.44, 3.78] |
| 8 | 2.01 [1.42, 3.03] | 2.18 [1.44, 3.78] | 2.20 [1.49, 3.48] | 2.10 [1.43, 3.26] | 2.05 [1.36, 3.67] |
| 12 | 1.81 [1.45, 2.91] | 1.92 [1.38, 3.36] | 1.93 [1.27, 3.19] | 1.92 [1.40, 2.82] | 2.04 [1.34, 3.82] |
| 16 | 1.77 [1.48, 3.01] | 1.81 [1.36, 3.12] | 1.91 [1.32, 2.97] | 1.83 [1.25, 2.58] | 2.18 [1.38, 3.40] |
| 21 | 1.78 [1.55, 2.87] | 1.75 [1.37, 2.37] | 2.03 [1.28, 2.60] | 1.63 [1.21, 2.47] | 2.29 [1.29, 3.29] |
