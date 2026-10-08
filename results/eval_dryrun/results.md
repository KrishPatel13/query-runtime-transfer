# Adaptation evaluation: dry run on synthetic machines built from the pilot

- M1: 420 of 420 queries ok (22 templates), status ok 420.
- M2: 419 of 420 queries ok (22 templates), status ok 419, error: synthetic failure 1.
- M3: 418 of 420 queries ok (22 templates), status ok 418, timeout 2.
- Split: 294 pool, 126 test queries in 22 templates.
- M1->M2: 293 pool and 126 test queries ok on both machines, 22 test templates, 0 queries whose plan features differ between the machines.
- M1->M3: 293 pool and 125 test queries ok on both machines, 21 test templates, 0 queries whose plan features differ between the machines.
- M3->M1: 293 pool and 125 test queries ok on both machines, 21 test templates, 0 queries whose plan features differ between the machines.

**Primary statistic (Q1)**: ridge correction over speed factor, M1->M3, k = 25: 0.97 [0.93, 1.00] over 21 test templates; ridge lower on 13 of 21 (sign test p = 0.383, Wilcoxon p = 0.076). Success (interval below 1): True.

**Primary statistic (Q2, C4)**: correction model over cost line: 0.90 [0.79, 1.03] over 22 held-out groups. Success: False.

Run time (wall clock, s): c1 60, pairs 51, c3_stream 1, c4 5, checks 53, evaluate 116, total 170
- task M1->M2: c2 21, hard 29, c3_continued 1
- task M1->M3: c2 21, hard 28, c3_continued 1
- task M3->M1: c2 21, hard 28, c3_continued 1

### C1: source models on each machine

Seen: 70/30 split inside every template, mean ± std over 5 splits. Unseen: one template held out at a time, pooled, 95% intervals from resampling templates. Ratio: geometric mean over held-out templates of the per-template median Q-error over the cost line's.

| Machine | Model | median Q | P90 Q | median Q [95% CI] | P90 Q | ratio to cost line [95% CI] | wins |
|---|---|---|---|---|---|---|---|
| M1 | Cost line | 1.71 ± 0.02 | 3.60 | 1.76 [1.57, 2.35] | 3.99 | -- | -- |
|  | Ridge | 1.08 ± 0.01 | 1.40 | 1.74 [1.41, 2.19] | 4.36 | 0.94 [0.79, 1.11] | 11/22 |
|  | Random forest | 1.03 ± 0.00 | 1.09 | 1.86 [1.32, 2.30] | 3.27 | 0.97 [0.86, 1.11] | 11/22 |
|  | Correction model | 1.03 ± 0.00 | 1.09 | 1.64 [1.24, 2.34] | 3.40 | 0.90 [0.79, 1.03] | 17/22 |
|  | Template mean | 1.03 ± 0.00 | 1.08 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 2.21 [1.41, 2.81] | 6.03 | 1.16 [0.96, 1.42] | 9/22 |
| M2 | Cost line | 1.88 ± 0.07 | 3.53 | 1.95 [1.40, 2.47] | 3.98 | -- | -- |
|  | Ridge | 1.10 ± 0.02 | 1.36 | 1.81 [1.42, 2.19] | 4.23 | 0.94 [0.82, 1.08] | 12/22 |
|  | Random forest | 1.06 ± 0.00 | 1.13 | 1.77 [1.31, 2.17] | 3.63 | 0.93 [0.80, 1.08] | 13/22 |
|  | Correction model | 1.05 ± 0.00 | 1.13 | 2.06 [1.45, 2.35] | 3.13 | 0.95 [0.81, 1.12] | 14/22 |
|  | Template mean | 1.05 ± 0.00 | 1.12 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 2.31 [1.64, 3.35] | 6.52 | 1.22 [1.00, 1.52] | 9/22 |
| M3 | Cost line | 1.83 ± 0.04 | 3.16 | 1.99 [1.62, 2.48] | 3.45 | -- | -- |
|  | Ridge | 1.10 ± 0.02 | 1.36 | 2.11 [1.40, 2.35] | 4.72 | 0.94 [0.75, 1.15] | 11/22 |
|  | Random forest | 1.05 ± 0.01 | 1.13 | 1.94 [1.45, 2.25] | 3.53 | 0.95 [0.84, 1.06] | 12/22 |
|  | Correction model | 1.05 ± 0.01 | 1.13 | 1.78 [1.32, 2.40] | 3.31 | 0.89 [0.78, 1.03] | 17/22 |
|  | Template mean | 1.05 ± 0.01 | 1.11 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 2.09 [1.45, 2.88] | 7.52 | 1.17 [0.95, 1.48] | 10/22 |

### The shift between the machines of each pair

Over all queries of the pair (pool and test), r = log(target label / source label). Template slowdowns: smallest and largest exp(mean r) of a template with at least 3 queries. Template share: share of the variance of r between templates (one-way ANOVA). Cost R^2: of r on log(1 + total cost). Accepted: template share above the 14% seen between two unchanged pilot passes.

| Pair | queries | median slowdown | template slowdowns | template share | ANOVA p | cost R^2 | accepted |
|---|---|---|---|---|---|---|---|
| M1->M2 | 419 | 1.78 | 1.11 to 2.85 | 95% | 8e-243 | 0.001 | yes |
| M1->M3 | 418 | 1.43 | 1.10 to 1.91 | 93% | 1e-216 | 0.149 | yes |
| M3->M1 | 418 | 0.70 | 0.52 to 0.91 | 93% | 1e-216 | 0.149 | yes |

### C2: error against k

Geometric mean over test templates of the per-template median Q-error, averaged in log over 20 arrival orders. Last column: smallest k at which the share of the gap to the oracle closed reaches 0.9.

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 | G >= 0.9 from k |
|---|---|---|---|---|---|---|---|
| M1->M2 | No adaptation | 1.75 | 1.75 | 1.75 | 1.75 | 1.75 | -- |
|  | Speed factor | 1.23 | 1.22 | 1.20 | 1.19 | 1.19 | >100 |
|  | Ridge correction | 1.24 | 1.16 | 1.12 | 1.10 | 1.08 | 50 |
|  | Target only | 2.27 | 1.92 | 1.56 | 1.30 | 1.13 | >100 |
|  | Oracle | 1.06 | 1.06 | 1.06 | 1.06 | 1.06 | -- |
| M1->M3 | No adaptation | 1.45 | 1.45 | 1.45 | 1.45 | 1.45 | -- |
|  | Speed factor | 1.18 | 1.18 | 1.17 | 1.17 | 1.17 | >100 |
|  | Ridge correction | 1.18 | 1.16 | 1.13 | 1.11 | 1.06 | 100 |
|  | Target only | 2.17 | 1.83 | 1.52 | 1.26 | 1.10 | >100 |
|  | Oracle | 1.05 | 1.05 | 1.05 | 1.05 | 1.05 | -- |
| M3->M1 (descr.) | No adaptation | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 | -- |
|  | Speed factor | 1.17 | 1.16 | 1.15 | 1.15 | 1.15 | >100 |
|  | Ridge correction | 1.17 | 1.14 | 1.12 | 1.09 | 1.05 | 100 |
|  | Target only | 2.06 | 1.78 | 1.48 | 1.24 | 1.08 | >100 |
|  | Oracle | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 | -- |

### C2: share of the gap to the oracle closed, G(k)

G = (E_none - E_method) / (E_none - E_oracle). The ratio row is the ridge correction's error over the speed factor's (geometric mean over test templates, 95% interval from 2000 paired resamples).

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 | G >= 0.9 from k |
|---|---|---|---|---|---|---|---|
| M1->M2 | Speed factor | 0.70 | 0.72 | 0.76 | 0.77 | 0.77 | >100 |
|  | Ridge correction | 0.69 | 0.82 | 0.90 | 0.93 | 0.96 | 50 |
|  | Target only | -0.52 | -0.18 | 0.23 | 0.60 | 0.88 | >100 |
|  | Ridge / speed ratio [95% CI] | 1.00 [0.99, 1.02] | 0.95 [0.92, 0.98] | 0.93 [0.88, 0.98] | 0.92 [0.86, 0.98] | 0.91 [0.85, 0.96] |  |
| M1->M3 | Speed factor | 0.64 | 0.65 | 0.66 | 0.66 | 0.66 | >100 |
|  | Ridge correction | 0.65 | 0.69 | 0.77 | 0.84 | 0.96 | 100 |
|  | Target only | -1.24 | -0.71 | -0.15 | 0.44 | 0.87 | >100 |
|  | Ridge / speed ratio [95% CI] | 1.00 [0.98, 1.02] | 0.99 [0.97, 1.00] | 0.97 [0.93, 1.00] | 0.94 [0.90, 0.99] | 0.91 [0.87, 0.94] |  |
| M3->M1 (descr.) | Speed factor | 0.65 | 0.67 | 0.69 | 0.69 | 0.69 | >100 |
|  | Ridge correction | 0.65 | 0.71 | 0.77 | 0.86 | 0.95 | 100 |
|  | Target only | -1.02 | -0.58 | -0.05 | 0.47 | 0.87 | >100 |
|  | Ridge / speed ratio [95% CI] | 1.00 [0.98, 1.02] | 0.99 [0.97, 1.00] | 0.97 [0.93, 1.01] | 0.94 [0.90, 0.98] | 0.92 [0.88, 0.95] |  |

### C2: pooled median / P90 Q-error over all test queries

Geometric mean over the arrival orders.

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| M1->M2 | No adaptation | 1.78 / 2.17 | 1.78 / 2.17 | 1.78 / 2.17 | 1.78 / 2.17 | 1.78 / 2.17 |
|  | Speed factor | 1.18 / 1.55 | 1.17 / 1.54 | 1.14 / 1.54 | 1.13 / 1.56 | 1.14 / 1.56 |
|  | Ridge correction | 1.18 / 1.56 | 1.12 / 1.41 | 1.09 / 1.27 | 1.07 / 1.22 | 1.06 / 1.16 |
|  | Target only | 1.95 / 5.20 | 1.63 / 4.11 | 1.31 / 2.71 | 1.14 / 1.74 | 1.06 / 1.21 |
|  | Oracle | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 |
| M1->M3 | No adaptation | 1.44 / 1.88 | 1.44 / 1.88 | 1.44 / 1.88 | 1.44 / 1.88 | 1.44 / 1.88 |
|  | Speed factor | 1.15 / 1.38 | 1.14 / 1.37 | 1.14 / 1.35 | 1.14 / 1.35 | 1.14 / 1.35 |
|  | Ridge correction | 1.14 / 1.42 | 1.12 / 1.35 | 1.08 / 1.36 | 1.06 / 1.23 | 1.05 / 1.15 |
|  | Target only | 1.86 / 4.76 | 1.55 / 3.57 | 1.30 / 2.62 | 1.13 / 1.80 | 1.06 / 1.21 |
|  | Oracle | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 | 1.05 / 1.13 |
| M3->M1 (descr.) | No adaptation | 1.43 / 1.83 | 1.43 / 1.83 | 1.43 / 1.83 | 1.43 / 1.83 | 1.43 / 1.83 |
|  | Speed factor | 1.16 / 1.36 | 1.15 / 1.35 | 1.15 / 1.32 | 1.15 / 1.31 | 1.16 / 1.32 |
|  | Ridge correction | 1.14 / 1.40 | 1.12 / 1.34 | 1.07 / 1.34 | 1.06 / 1.21 | 1.05 / 1.12 |
|  | Target only | 1.79 / 4.58 | 1.49 / 3.65 | 1.27 / 2.55 | 1.11 / 1.72 | 1.04 / 1.19 |
|  | Oracle | 1.03 / 1.09 | 1.03 / 1.09 | 1.03 / 1.09 | 1.03 / 1.09 | 1.03 / 1.09 |

### C2 harder test: a template held out on both machines (k=25)

Median Q-error of the held-out template's target test queries, geometric mean over templates [95% interval], averaged in log over 5 orders.

| Method | M1->M2 | M1->M3 | M3->M1 |
|---|---|---|---|
| No adaptation | 2.22 [1.72, 2.91] | 2.00 [1.57, 2.65] | 2.05 [1.67, 2.60] |
| Speed factor | 1.99 [1.67, 2.40] | 1.99 [1.63, 2.54] | 1.85 [1.49, 2.34] |
| Ridge correction | 2.00 [1.59, 2.72] | 1.92 [1.55, 2.47] | 2.03 [1.63, 2.66] |
| Target only | 2.09 [1.72, 2.55] | 2.05 [1.70, 2.55] | 1.95 [1.60, 2.38] |
| Oracle | 2.01 [1.67, 2.46] | 1.92 [1.54, 2.49] | 1.87 [1.52, 2.35] |
| Ridge / speed ratio | 1.01 [0.90, 1.17] | 0.96 [0.91, 1.02] | 1.10 [1.02, 1.18] |

### C3 (a): the orders continued to the whole pool

Correction refitted every 50 queries; error as in C2.

| Pair | Method | error after n queries (n:error) |
|---|---|---|
| M1->M2 | Never update | 50:1.75 100:1.75 200:1.75 293:1.75 |
|  | Refit on all so far | 50:1.10 100:1.08 200:1.07 293:1.07 |
|  | Refit on the last 50 | 50:1.10 100:1.10 200:1.12 293:1.10 |
| M1->M3 | Never update | 50:1.45 100:1.45 200:1.45 293:1.45 |
|  | Refit on all so far | 50:1.11 100:1.06 200:1.06 293:1.06 |
|  | Refit on the last 50 | 50:1.11 100:1.09 200:1.09 293:1.08 |
| M3->M1 | Never update | 50:1.46 100:1.46 200:1.46 293:1.46 |
|  | Refit on all so far | 50:1.09 100:1.05 200:1.05 293:1.05 |
|  | Refit on the last 50 | 50:1.09 100:1.08 200:1.08 293:1.08 |

### C3 (b): stream from M1 to M2, test-then-train

Batch median Q-error (batches of 10), geometric mean over 5 stream orders. Level before and final level: mean of the last 5 batches before the switch and of the last 5 of the stream; jump: first batch after the switch over the level before. Recovery: queries after the switch until every later batch is within 10% of the final level, for the aggregate curve and for each order. Never update is flat after the switch, so its count shows the noise of this statistic.

| Method | level before | first batch after | jump | final level | queries to recover | per order |
|---|---|---|---|---|---|---|
| Never update | 1.03 | 1.70 | 1.65 | 1.81 | 0 | 0, 33, 0, 0, 0 |
| Refit on all so far | 1.03 | 1.72 | 1.67 | 1.43 | 44 | 86, 33, 76, 33, 76 |
| Refit on the last 50 | 1.03 | 1.70 | 1.64 | 1.08 | 44 | 33, 44, 44, 44, 44 |

### C4: one template group held out at a time (22 groups)

Sign test p = 0.017, Wilcoxon p = 0.074. Success (interval below 1): False.

| Model | median Q [95% CI] | P90 Q | ratio to cost line [95% CI] | wins |
|---|---|---|---|---|
| Cost line | 1.75 [1.56, 2.32] | 3.98 | -- | -- |
| Correction model | 1.56 [1.26, 2.36] | 3.34 | 0.90 [0.79, 1.03] | 17/22 |

## Dry-run checks

| Check | Value | Pass |
|---|---|---|
| C1's code on all of M1's queries reproduces the pilot's E1 and E2 (largest absolute difference) | 4.6e-10 | yes |
| C4 on M1 reproduces the pilot's correction-model ratio to the cost line (0.90 [0.79, 1.03], largest absolute difference) | 1.1e-16 | yes |
| M1->M2: the shift depends on the template (template share above the pilot's between-pass share) | 95% vs 14% | yes |
| M1->M2: speed factor beats no adaptation at every k | 1.23 < 1.75 | yes |
| M1->M2: ridge correction beats the speed factor for k >= 25 | 1.12 vs 1.20, 1.10 vs 1.19, 1.08 vs 1.19 | yes |
| M1->M2: target only improves from the smallest to the largest k | 2.27 -> 1.13 | yes |
| M1->M2: oracle beats no adaptation | 1.06 < 1.75 | yes |
| M1->M2: ridge correction recovers the per-template slowdown (RMSE ridge vs speed factor, k = 100) | 0.040 vs 0.211, correlation 0.98 | yes |
| M1->M3: the shift depends on the template (template share above the pilot's between-pass share) | 93% vs 14% | yes |
| M1->M3: speed factor beats no adaptation at every k | 1.18 < 1.45 | yes |
| M1->M3: ridge correction beats the speed factor for k >= 25 | 1.13 vs 1.17, 1.11 vs 1.17, 1.06 vs 1.17 | yes |
| M1->M3: target only improves from the smallest to the largest k | 2.17 -> 1.10 | yes |
| M1->M3: oracle beats no adaptation | 1.05 < 1.45 | yes |
| M1->M3: ridge correction recovers the per-template slowdown (RMSE ridge vs speed factor, k = 100) | 0.028 vs 0.171, correlation 0.99 | yes |
| M1->M2: refitting on all queries so far keeps improving to the whole pool | 1.10 -> 1.07 | yes |
| M1->M3: refitting on all queries so far keeps improving to the whole pool | 1.11 -> 1.06 | yes |
| stream: the switch raises the batch error by more than 20% | 1.65 | yes |
| stream: refitting on the last 50 recovers within 60 queries of the switch | 44 | yes |
| stream: refitting on all so far settles higher than on the last 50 (the old machine's queries stay) | 1.43 vs 1.08 | yes |

