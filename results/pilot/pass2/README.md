# Pass 2 (reproducibility check)

Second collection pass over the same 420 queries in a different random order
(seed 1), run right after pass 1 on 2026-09-23 from 19:58 to 20:10. It is used
only to check how well the labels reproduce; every experiment uses pass 1.

- PostgreSQL chose the identical plan (same tree and estimated costs) for all
  420 queries. That was expected, since both passes ran on the same database
  with the same statistics, so it says nothing about the labels.
- The first 250 queries match pass 1 closely: median Q-error between the
  passes 1.03, 1.09 at the 90th percentile, 1.27 at most. Within a template,
  a query's deviation from its template mean correlates 0.47 between the
  passes, so about half of the small within-template spread is noise.
- From 20:03 on, every one of the remaining 170 queries ran slower than in
  pass 1, by a median factor of 1.70 (1.47 to 1.91 for the middle 80%). The
  ranking across templates was nearly kept (Spearman 0.987), but the
  within-template correlation fell to 0.10.

The laptop was on battery. Only the start and end of the pass were logged: the
charge fell from 26% to 17% (`runtime_dataset.power.log`). At pass 2's average
drain (0.8% per minute) it would have been about 22% at 20:03; at pass 1's
drain under the same load (1.2% per minute) about 20%, which would mean the
drain slowed after 20:03, as energy saver would cause. This laptop's power plan switches on Windows
energy saver at 20% charge with the "Aggressive" policy
(`powercfg /qh SCHEME_CURRENT SUB_ENERGYSAVER`: ESBATTTHRESHOLD = 0x14,
ESPOLICY on battery = 1, Aggressive), and energy saver was on when checked two
minutes after the pass ended. The slowdown starts between two consecutive
queries and reaches its new level within about five queries, which fits a
power-policy switch better than gradual heating. That it was energy saver is
an inference from these facts, not a logged event. Pass 1 ran between 37% and
26% and shows no drift over its execution order.

The data here is left exactly as collected. `scripts/compare_passes.py` finds
the shift without being told where it is and writes the numbers above to
`reproducibility.json` and `../pass2_numbers.tex`.
