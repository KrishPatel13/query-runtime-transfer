# Results: the originally proposed 5 templates (Q1, Q3, Q5, Q6, Q14)

100 queries from 5 templates. Runtime 79 to 1136 ms. Within a template the slowest query is at most 1.33x the fastest (median 1.17x). Repeat spread: the three back-to-back timed runs lie within 1.022x of their median (median over queries), 1.085x at the 90th percentile. This is a lower bound on label noise; scripts/compare_passes.py measures agreement between independent passes.

Predictions clipped at a bound (see LOG_MIN, LOG_MAX): none.

E1: mean ± std over 5 random 70/30 splits. E2: every query's held-out prediction pooled once; the 95% interval resamples whole templates (2000 draws).

| Model | E1 median Q | E1 P90 Q | E1 log-MAE | E2 median Q [95% CI] | E2 P90 Q | E2 log-MAE | E2 within 2x |
|---|---|---|---|---|---|---|---|
| Cost-only | 1.98 ± 0.01 | 2.15 ± 0.02 | 0.58 ± 0.00 | 2.70 [1.18, 7.03] | 7.18 | 1.16 | 20% |
| Ridge | 1.02 ± 0.00 | 1.06 ± 0.01 | 0.03 ± 0.00 | 3.35 [1.29, 7.35] | 7.51 | 1.21 | 36% |
| Random Forest | 1.02 ± 0.00 | 1.06 ± 0.01 | 0.03 ± 0.00 | 2.20 [1.90, 6.03] | 6.16 | 0.94 | 34% |
| Cost + RF residual | 1.02 ± 0.00 | 1.06 ± 0.01 | 0.03 ± 0.00 | 3.26 [1.25, 10.60] | 10.83 | 1.23 | 40% |
| Template mean | 1.02 ± 0.00 | 1.06 ± 0.01 | 0.03 ± 0.00 | n/a | n/a | n/a | n/a |
| Global mean | n/a | n/a | n/a | 1.46 [1.17, 6.83] | 6.98 | 0.78 | 60% |

E2 median over held-out templates of each template's median Q-error: Cost-only 2.69, Ridge 3.35, Random Forest 2.23, Cost + RF residual 3.26, Global mean 1.46.

Held-out templates where the model's median Q-error beats Cost-only: Ridge 2/5, Random Forest 4/5, Cost + RF residual 2/5, Global mean 3/5. Worst held-out template: Cost-only 7.20, Ridge 7.53, Random Forest 6.18, Cost + RF residual 10.85, Global mean 7.00.

Paired with Cost-only on the same held-out templates. Primary: geometric mean over templates of the ratio of per-template median Q-errors, with a 95% interval from resampling templates. Secondary: ratio of pooled median Q-errors (same resamples). Then the two-sided sign test and the Wilcoxon signed-rank test over templates (both treat the overlapping folds as independent): Ridge GM 1.06 [0.88, 1.26], pooled 1.24 [0.72, 1.36], 2/5 lower, sign p = 1.000, Wilcoxon p = 0.812; Random Forest GM 0.81 [0.46, 1.33], pooled 0.81 [0.31, 1.89], 4/5 lower, sign p = 0.375, Wilcoxon p = 0.438; Cost + RF residual GM 1.08 [0.82, 1.44], pooled 1.21 [0.71, 1.80], 2/5 lower, sign p = 1.000, Wilcoxon p = 0.812; Global mean GM 0.69 [0.35, 1.18], pooled 0.54 [0.19, 1.39], 3/5 lower, sign p = 1.000, Wilcoxon p = 0.625.

Ridge penalty chosen (min, median, max over fits; grid 1e-6 to 1e5): E1 (leave one query out) 0.00032, 0.0018, 0.32 (5 fits, 0 at a grid end); E2 (leave one template out) 0.032, 5.6e+03, 1e+05 (5 fits, 2 at a grid end); E3 (leave one query out) 1e-06, 0.032, 0.32 (25 fits, 2 at a grid end); E4 k=1 (leave one query out) 32, 1e+02, 1e+05 (5 fits, 1 at a grid end); E4 k=2 (leave one template out) 0.032, 2.1e+03, 1e+05 (10 fits, 4 at a grid end); E4 k=3 (leave one template out) 0.32, 2.1e+03, 1e+05 (10 fits, 3 at a grid end); E4 k=4 (leave one template out) 0.032, 5.6e+03, 1e+05 (5 fits, 2 at a grid end).

## E2 median Q-error per held-out template

| Template | Cost-only | Ridge | Random Forest | Cost + RF residual | Global mean |
|---|---|---|---|---|---|
| Q1 | 7.20 | 6.99 | 6.18 | 6.30 | 7.00 |
| Q3 | 2.69 | 1.97 | 1.98 | 3.26 | 1.46 |
| Q5 | 6.04 | 7.53 | 1.85 | 10.85 | 1.14 |
| Q6 | 1.16 | 1.26 | 2.23 | 1.23 | 1.26 |
| Q14 | 2.46 | 3.35 | 2.29 | 1.74 | 3.43 |

## E3 learning curve (median Q-error, mean ± std over 5 splits)

| Training queries | Cost-only | Ridge | Random Forest | Cost + RF residual |
|---|---|---|---|---|
| 10 (10%) | 1.97 ± 0.02 | 1.02 ± 0.00 | 1.17 ± 0.04 | 1.35 ± 0.03 |
| 20 (25%) | 1.97 ± 0.01 | 1.02 ± 0.00 | 1.05 ± 0.00 | 1.06 ± 0.02 |
| 35 (50%) | 1.97 ± 0.01 | 1.02 ± 0.00 | 1.02 ± 0.00 | 1.02 ± 0.01 |
| 55 (75%) | 1.98 ± 0.01 | 1.02 ± 0.00 | 1.02 ± 0.00 | 1.02 ± 0.00 |
| 70 (100%) | 1.98 ± 0.01 | 1.02 ± 0.00 | 1.02 ± 0.00 | 1.02 ± 0.00 |

## E4 template-diversity curve

Each unseen template's median Q-error, summarized as median [IQR] over all (training set, unseen template) pairs. Every possible training set for k = 1 and k = T - 1, 20 random ones otherwise.

| Training templates | Cost-only | Ridge | Random Forest | Cost + RF residual | Global mean |
|---|---|---|---|---|---|
| 1 | 3.65 [1.75, 7.71] | 4.62 [1.79, 13.64] | 2.32 [1.22, 5.73] | 3.64 [1.76, 7.72] | 2.31 [1.22, 5.70] |
| 2 | 3.85 [2.32, 6.84] | 2.58 [1.78, 5.43] | 2.24 [1.53, 4.43] | 3.82 [2.33, 6.70] | 2.21 [1.57, 5.28] |
| 3 | 2.87 [1.59, 6.10] | 2.38 [1.45, 6.40] | 2.23 [1.84, 3.60] | 2.86 [2.02, 6.65] | 1.69 [1.39, 4.00] |
| 4 | 2.69 [2.46, 6.04] | 3.35 [1.97, 6.99] | 2.23 [1.98, 2.29] | 3.26 [1.74, 6.30] | 1.46 [1.26, 3.43] |
