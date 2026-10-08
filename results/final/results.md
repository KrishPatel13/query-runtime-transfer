# Adaptation evaluation: labels in results/runs, split queries/tpch_sf1_v2/split.csv

- M1: 680 of 680 queries ok (22 templates), status ok 680.
- M2: 680 of 680 queries ok (22 templates), status ok 680.
- M3: 680 of 680 queries ok (22 templates), status ok 680.
- Split: 476 pool, 204 test queries in 22 templates.
- M1->M2: 476 pool and 204 test queries ok on both machines, 22 test templates, 0 queries whose plan features differ between the machines.
- M1->M3: 476 pool and 204 test queries ok on both machines, 22 test templates, 0 queries whose plan features differ between the machines.
- M3->M1: 476 pool and 204 test queries ok on both machines, 22 test templates, 0 queries whose plan features differ between the machines.

**Primary statistic (Q1)**: ridge correction over speed factor, M1->M3, k = 25: 0.93 [0.89, 0.96] over 22 test templates; ridge lower on 16 of 22 (sign test p = 0.052, Wilcoxon p = 0.002). Success (interval below 1): True.

**Primary statistic (Q2, C4)**: correction model over cost line: 1.04 [0.90, 1.19] over 75 held-out family groups. Success: False.
With one template group held out instead: 0.93 [0.82, 1.06] over 93 groups.

Run time (wall clock, s): c1 76, pairs 50, c3_stream 1, c4 43, evaluate 170, total 170
- task M1->M2: c2 20, hard 28, c3_continued 1
- task M1->M3: c2 20, hard 27, c3_continued 1
- task M3->M1: c2 20, hard 28, c3_continued 1

### C1: source models on each machine

Seen: 70/30 split inside every template, mean ± std over 5 splits. Unseen: one template held out at a time, pooled, 95% intervals from resampling templates. Ratio: to the cost line, the geometric mean over held-out templates of the per-template median Q-error over the cost line's.

| Machine | Model | median Q | P90 Q | median Q [95% CI] | P90 Q | ratio [95% CI] | wins |
|---|---|---|---|---|---|---|---|
| M1 | Cost line | 1.78 ± 0.07 | 3.73 | 1.84 [1.48, 2.59] | 4.44 | -- | -- |
|  | Ridge | 1.20 ± 0.01 | 1.75 | 1.64 [1.37, 1.97] | 3.53 | 0.91 [0.74, 1.09] | 9/22 |
|  | Random forest | 1.06 ± 0.01 | 1.30 | 1.57 [1.39, 1.84] | 4.01 | 0.93 [0.77, 1.09] | 12/22 |
|  | Correction model | 1.05 ± 0.00 | 1.29 | 1.60 [1.40, 2.11] | 3.77 | 0.95 [0.82, 1.10] | 12/22 |
|  | Template mean | 1.16 ± 0.04 | 2.22 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 2.07 [1.59, 2.86] | 6.56 | 1.17 [0.94, 1.48] | 11/22 |
| M2 | Cost line | 1.78 ± 0.04 | 4.37 | 1.98 [1.68, 2.63] | 4.37 | -- | -- |
|  | Ridge | 1.20 ± 0.01 | 1.71 | 1.47 [1.30, 2.12] | 4.09 | 0.83 [0.71, 0.998] | 16/22 |
|  | Random forest | 1.06 ± 0.00 | 1.44 | 1.67 [1.41, 2.10] | 4.65 | 0.89 [0.73, 1.08] | 12/22 |
|  | Correction model | 1.06 ± 0.00 | 1.32 | 1.92 [1.66, 2.49] | 4.70 | 1.03 [0.88, 1.23] | 9/22 |
|  | Template mean | 1.21 ± 0.03 | 2.56 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 2.25 [1.88, 3.67] | 6.87 | 1.22 [1.01, 1.52] | 11/22 |
| M3 | Cost line | 1.48 ± 0.05 | 2.69 | 1.53 [1.32, 1.97] | 2.92 | -- | -- |
|  | Ridge | 1.17 ± 0.02 | 1.54 | 1.51 [1.23, 1.70] | 2.92 | 0.97 [0.84, 1.11] | 12/22 |
|  | Random forest | 1.04 ± 0.00 | 1.27 | 1.42 [1.27, 1.64] | 3.57 | 0.97 [0.84, 1.12] | 10/22 |
|  | Correction model | 1.04 ± 0.00 | 1.20 | 1.45 [1.29, 1.90] | 3.11 | 1.01 [0.89, 1.15] | 12/22 |
|  | Template mean | 1.13 ± 0.01 | 1.92 | -- | -- | -- | -- |
|  | Global mean | -- | -- | 1.81 [1.45, 2.67] | 4.29 | 1.26 [1.05, 1.55] | 9/22 |

### The shift between the machines of each pair

Over all queries of the pair (pool and test), r = log(target label / source label). Serial / parallel plans: median slowdown of the plans without and with parallel workers. Template slowdowns: smallest and largest exp(mean r) of a template with at least 3 queries. Template share: share of the variance of r between templates (one-way ANOVA). Cost R^2: of r on log(1 + total cost). Accepted: template share above the 13.6% seen between two unchanged pilot passes.

| Pair | queries | median slowdown | serial / parallel plans | template slowdowns | template share | ANOVA p | cost R^2 | accepted |
|---|---|---|---|---|---|---|---|---|
| M1->M2 | 680 | 2.48 | 1.12 / 2.66 | 0.97 to 3.45 | 86% | 2e-264 | 0.039 | yes |
| M1->M3 | 680 | 2.01 | 1.81 / 2.07 | 1.25 to 3.10 | 80% | 5e-211 | 0.029 | yes |
| M3->M1 | 680 | 0.50 | 0.55 / 0.48 | 0.32 to 0.80 | 80% | 5e-211 | 0.029 | yes |

### C2: error against k

Geometric mean over test templates of the per-template median Q-error, averaged in log over 20 arrival orders. Last column: smallest k at which the share of the gap to the oracle closed reaches 0.9.

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 | G >= 0.9 from k |
|---|---|---|---|---|---|---|---|
| M1->M2 | No adaptation | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 | -- |
|  | Speed factor | 1.44 | 1.44 | 1.44 | 1.43 | 1.43 | >100 |
|  | Ridge correction | 1.40 | 1.30 | 1.21 | 1.15 | 1.11 | 100 |
|  | Target only | 2.40 | 1.97 | 1.58 | 1.37 | 1.17 | >100 |
|  | Oracle | 1.06 | 1.06 | 1.06 | 1.06 | 1.06 | -- |
| M1->M3 | No adaptation | 1.93 | 1.93 | 1.93 | 1.93 | 1.93 | -- |
|  | Speed factor | 1.26 | 1.26 | 1.25 | 1.25 | 1.24 | >100 |
|  | Ridge correction | 1.29 | 1.24 | 1.16 | 1.12 | 1.09 | 100 |
|  | Target only | 1.83 | 1.61 | 1.39 | 1.25 | 1.12 | >100 |
|  | Oracle | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 | -- |
| M3->M1 (descr.) | No adaptation | 1.95 | 1.95 | 1.95 | 1.95 | 1.95 | -- |
|  | Speed factor | 1.26 | 1.25 | 1.24 | 1.24 | 1.23 | >100 |
|  | Ridge correction | 1.28 | 1.24 | 1.19 | 1.11 | 1.09 | 50 |
|  | Target only | 2.27 | 1.91 | 1.53 | 1.33 | 1.15 | >100 |
|  | Oracle | 1.05 | 1.05 | 1.05 | 1.05 | 1.05 | -- |

### C2: share of the gap to the oracle closed, G(k)

G = (E_none - E_method) / (E_none - E_oracle).

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 | G >= 0.9 from k |
|---|---|---|---|---|---|---|---|
| M1->M2 | Speed factor | 0.54 | 0.55 | 0.55 | 0.56 | 0.56 | >100 |
|  | Ridge correction | 0.59 | 0.70 | 0.80 | 0.88 | 0.93 | 100 |
|  | Target only | -0.19 | 0.10 | 0.41 | 0.62 | 0.85 | >100 |
| M1->M3 | Speed factor | 0.69 | 0.69 | 0.70 | 0.71 | 0.72 | >100 |
|  | Ridge correction | 0.66 | 0.72 | 0.83 | 0.88 | 0.93 | 100 |
|  | Target only | 0.09 | 0.29 | 0.54 | 0.70 | 0.89 | >100 |
| M3->M1 (descr.) | Speed factor | 0.71 | 0.72 | 0.73 | 0.74 | 0.74 | >100 |
|  | Ridge correction | 0.68 | 0.74 | 0.80 | 0.91 | 0.94 | 50 |
|  | Target only | -0.25 | 0.04 | 0.39 | 0.62 | 0.86 | >100 |

### C2: ridge correction over speed factor

The ridge correction's error over the speed factor's: geometric mean over test templates of the ratio of their per-template errors, 95% interval from 2000 paired resamples of templates.

|  | M1->M2 | M1->M3 | M3->M1 (descr.) |
|---|---|---|---|
| k=5 | 0.97 [0.93, 1.001] | 1.02 [0.99, 1.05] | 1.02 [0.99, 1.04] |
| k=10 | 0.90 [0.85, 0.96] | 0.98 [0.95, 1.01] | 0.99 [0.96, 1.02] |
| k=25 | 0.84 [0.79, 0.90] | 0.93 [0.89, 0.96] | 0.96 [0.91, 0.998] |
| k=50 | 0.80 [0.74, 0.87] | 0.90 [0.86, 0.94] | 0.90 [0.85, 0.94] |
| k=100 | 0.78 [0.72, 0.84] | 0.88 [0.83, 0.92] | 0.88 [0.83, 0.93] |

### C2: pooled median / P90 Q-error over all test queries

Geometric mean over the arrival orders.

| Pair | Method | k=5 | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| M1->M2 | No adaptation | 2.47 / 3.26 | 2.47 / 3.26 | 2.47 / 3.26 | 2.47 / 3.26 | 2.47 / 3.26 |
|  | Speed factor | 1.37 / 2.01 | 1.38 / 1.94 | 1.40 / 1.90 | 1.38 / 1.92 | 1.37 / 1.93 |
|  | Ridge correction | 1.29 / 2.07 | 1.20 / 1.86 | 1.14 / 1.61 | 1.11 / 1.39 | 1.09 / 1.31 |
|  | Target only | 2.03 / 5.63 | 1.65 / 4.29 | 1.36 / 2.87 | 1.23 / 2.19 | 1.11 / 1.60 |
|  | Oracle | 1.04 / 1.20 | 1.04 / 1.20 | 1.04 / 1.20 | 1.04 / 1.20 | 1.04 / 1.20 |
| M1->M3 | No adaptation | 2.02 / 2.87 | 2.02 / 2.87 | 2.02 / 2.87 | 2.02 / 2.87 | 2.02 / 2.87 |
|  | Speed factor | 1.23 / 1.58 | 1.22 / 1.55 | 1.22 / 1.54 | 1.23 / 1.53 | 1.22 / 1.51 |
|  | Ridge correction | 1.22 / 1.68 | 1.17 / 1.60 | 1.10 / 1.47 | 1.08 / 1.30 | 1.07 / 1.22 |
|  | Target only | 1.52 / 3.77 | 1.40 / 2.82 | 1.22 / 2.13 | 1.13 / 1.70 | 1.07 / 1.33 |
|  | Oracle | 1.04 / 1.13 | 1.04 / 1.13 | 1.04 / 1.13 | 1.04 / 1.13 | 1.04 / 1.13 |
| M3->M1 (descr.) | No adaptation | 2.00 / 2.96 | 2.00 / 2.96 | 2.00 / 2.96 | 2.00 / 2.96 | 2.00 / 2.96 |
|  | Speed factor | 1.22 / 1.59 | 1.22 / 1.58 | 1.22 / 1.56 | 1.22 / 1.54 | 1.22 / 1.52 |
|  | Ridge correction | 1.23 / 1.67 | 1.17 / 1.64 | 1.12 / 1.53 | 1.09 / 1.29 | 1.07 / 1.24 |
|  | Target only | 1.76 / 5.99 | 1.57 / 3.95 | 1.30 / 2.67 | 1.17 / 2.01 | 1.08 / 1.44 |
|  | Oracle | 1.04 / 1.19 | 1.04 / 1.19 | 1.04 / 1.19 | 1.04 / 1.19 | 1.04 / 1.19 |

### C2 harder test: a template held out on both machines (k=25)

Median Q-error of the held-out template's target test queries, geometric mean over templates [95% interval], averaged in log over 5 orders.

| Method | M1->M2 | M1->M3 | M3->M1 |
|---|---|---|---|
| No adaptation | 2.91 [2.10, 4.07] | 2.62 [2.01, 3.47] | 2.29 [1.94, 2.73] |
| Speed factor | 2.11 [1.72, 2.63] | 1.78 [1.49, 2.16] | 1.92 [1.54, 2.45] |
| Ridge correction | 2.12 [1.76, 2.59] | 1.83 [1.58, 2.18] | 1.88 [1.54, 2.38] |
| Target only | 2.09 [1.74, 2.54] | 1.73 [1.48, 2.05] | 2.04 [1.65, 2.58] |
| Oracle | 2.31 [1.91, 2.83] | 1.73 [1.47, 2.07] | 1.95 [1.57, 2.51] |
| Ridge / speed ratio | 1.003 [0.87, 1.16] | 1.03 [0.96, 1.10] | 0.98 [0.91, 1.06] |

### C3 (a): the orders continued to the whole pool

Correction refitted every 50 queries; error as in C2.

| Pair | Method | error after n queries (n:error) |
|---|---|---|
| M1->M2 | Never update | 50:2.10 100:2.10 200:2.10 300:2.10 400:2.10 476:2.10 |
|  | Refit on all so far | 50:1.15 100:1.11 200:1.09 300:1.09 400:1.09 476:1.08 |
|  | Refit on the last 50 | 50:1.15 100:1.17 200:1.15 300:1.16 400:1.17 476:1.16 |
| M1->M3 | Never update | 50:1.93 100:1.93 200:1.93 300:1.93 400:1.93 476:1.93 |
|  | Refit on all so far | 50:1.12 100:1.09 200:1.08 300:1.08 400:1.08 476:1.08 |
|  | Refit on the last 50 | 50:1.12 100:1.11 200:1.11 300:1.11 400:1.13 476:1.12 |
| M3->M1 | Never update | 50:1.95 100:1.95 200:1.95 300:1.95 400:1.95 476:1.95 |
|  | Refit on all so far | 50:1.11 100:1.09 200:1.08 300:1.08 400:1.08 476:1.07 |
|  | Refit on the last 50 | 50:1.11 100:1.12 200:1.12 300:1.12 400:1.12 476:1.11 |

### C3 (b): stream from M1 to M2, test-then-train

Batch median Q-error (batches of 10), geometric mean over 5 stream orders. Level before and final level: mean of the last 5 batches before the switch and of the last 5 of the stream; jump: first batch after the switch over the level before. Recovery: queries after the switch until every later batch is within 10% of the final level, for the aggregate curve and for each order. Never update is flat after the switch, so its count shows the noise of this statistic.

| Method | level before | first batch after | jump | final level | queries to recover | per order |
|---|---|---|---|---|---|---|
| Never update | 1.04 | 2.28 | 2.18 | 2.45 | 0 | 194, 194, 174, --, 144 |
| Refit on all so far | 1.05 | 2.27 | 2.17 | 1.69 | 64 | 174, 194, 174, --, 164 |
| Refit on the last 50 | 1.05 | 2.30 | 2.19 | 1.12 | 54 | 22, 54, 174, 44, 44 |

### C4: one family held out at a time (75 family groups)

Intervals from resampling the held-out units. First block, the primary statistic of Q2: sign test p = 0.818, Wilcoxon p = 0.561. Success (interval below 1): False.

| Held out | Model | median Q [95% CI] | P90 Q | ratio to cost line [95% CI] | wins |
|---|---|---|---|---|---|
| family (75) | Cost line | 2.13 [1.84, 2.65] | 11.15 | -- | -- |
|  | Correction model | 2.15 [1.88, 2.64] | 11.76 | 1.04 [0.90, 1.19] | 36/75 |
| template (93) | Cost line | 2.12 [1.86, 2.53] | 11.15 | -- | -- |
|  | Correction model | 1.96 [1.73, 2.24] | 11.76 | 0.93 [0.82, 1.06] | 49/93 |

