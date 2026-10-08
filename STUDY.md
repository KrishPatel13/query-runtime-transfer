# Full study: how it is run

The design is in the course report (`final_report.pdf`) and, for the
extended study, `paper/PLAN.md`. This file is the run book: the steps in the order they run, the
command for each, what it writes, and the decisions taken after the proposal
was written.

## Machines

| Name | What | Database | Clock |
| --- | --- | --- | --- |
| M1 | Laptop (Core Ultra 7 155H, 22 threads, 16 GB), WSL2 Ubuntu 24.04, on AC power | `~/cs680-pilot/pgdata` | `hyperv_clocksource_tsc_page`, set by `scripts/wsl_clock.py` |
| M2 | M1 with the PostgreSQL server pinned to one virtual processor (`taskset -c 19`) | the same files | as M1 |
| M3 | UW CS server, 64 cores of an AMD EPYC 9754, 1 TB, shared, a KVM guest | a copy of M1's stopped cluster in `/var/tmp/pgstudy/pgdata` | `kvm-clock`, NTP-synced |

The server's login name reaches two different servers in turn, so every
server command named the server by its address.

## Steps

1. **Workload.** `python scripts/generate_widened_queries.py` wrote
   `queries/tpch_sf1_v2`: the pilot's 420 queries and 20 widened queries for
   each of the 13 templates with range parameters (680), `manifest.json` and
   `split.csv` (476 pool, 204 test, the same IDs on every machine).
   `scripts/check_workload.py` checked them on M1 (`results/workload_check`).
2. **Clock (M1, M2).** `wsl.exe -d Ubuntu-24.04 -u root -e python3
   scripts/wsl_clock.py` after every WSL start. See "Clock" below.
3. **Server build (M3).** `PILOT_HOME=/var/tmp/pgstudy PG_SECURE=1
   PGPORT=5468 PG_INITDB=0 bash scripts/setup_postgres.sh` built PostgreSQL
   18.6 with the same options as M1 (gcc 13.3.0 on both).
4. **Copy M1's cluster to M3.** `bash scripts/export_cluster.sh` on M1 (stops
   the cluster, archives it with a checksum, starts it again), `scp` of the
   archive, then `PILOT_HOME=/var/tmp/pgstudy bash
   scripts/import_cluster.sh <archive>` on M3: peer login only, no TCP
   socket, a private socket directory with mode 0700, every memory and
   planner setting as on M1.
5. **Plan check.** `python3 scripts/explain_all.py` on both machines and
   `--compare`: all 680 plans were identical (node types, costs, rows,
   widths, planned workers).
6. **M1 passes.** In WSL: `python3 src/run_passes.py --machine M1 --query-dir
   queries/tpch_sf1_v2`. Three passes in seeded orders into
   `results/runs/M1/pass<i>`, each checked against pass 1 for a level shift.
7. **M3 passes.** `scripts/run_m3.sh`, started detached, waits until after
   01:30 for a 5-minute load below 10 and runs the same command with
   `--machine M3` into `/var/tmp/pgstudy/runs`. It started at 03:10
   and finished at 04:52 on 1 October. The runs came back as two archives
   (the per-pass plans in `results/runs/M3/plans.tar.gz`), and
   `src/make_labels.py M3 --query-dir queries/tpch_sf1_v2` and
   `src/compare_machines.py M1 M3` ran on the laptop. The server's cluster
   was then stopped.
8. **TPC-DS (M1).** Regenerate with `scripts/tpcds_queries.py --force`,
   screen with `scripts/tpcds_screen.py` (writes `queries/tpcds_sf1_kept`:
   93 of 99 templates, 75 of 81 families, 482 queries), then three passes
   over the kept queries, as for TPC-H, with `src/run_passes.py --machine
   M1_tpcds --query-dir queries/tpcds_sf1_kept --database-url
   'postgresql:///tpcds?host=/tmp&port=5433'` (about 33 minutes a pass), and
   `src/make_labels.py M1_tpcds --query-dir queries/tpcds_sf1_kept`.
9. **M2 passes.** `bash scripts/start_m2.sh` stops the TPC-DS cluster and
   starts the TPC-H cluster with its postmaster pinned to one of WSL's 22
   virtual processors (`taskset -c 19 pg_ctl start`), so every backend and
   parallel worker it forks shares that processor; it checks the affinity
   of the server's processes and of a test query's parallel workers and
   records it in `results/runs/M2/affinity.txt`. Then `run_passes.py
   --machine M2`. After pass 1, `python
   scripts/shift_check.py M1 M2` runs the acceptance test: the share of the
   variance of the log slowdown between templates must exceed the 14% seen
   between two unchanged pilot passes. Afterwards the cluster is restarted
   without the pin. See "M2 by processor affinity" below for why this
   replaces limiting WSL to one processor. On 1 October pass 1's first
   attempt (12:31 to 13:30) was rejected because the laptop ran on battery
   from 12:45 to 13:09, and pass 3's first attempt (15:12 to 16:10) because
   it ran 10% faster after query 218, when heavy work by other programs on
   the laptop ended at 15:35. The kept passes took 50 to 52 minutes each and
   ended at 17:00. The acceptance test on pass 1 gave a template share of
   83%.
10. **Labels.** `src/make_labels.py` per machine writes `labels.csv` and
    `noise.json`; `src/compare_machines.py` checks plans and agreement
    between machines.
11. **Evaluation.** `python src/adapt_eval.py run --c4-labels
    results/runs/M1_tpcds/labels.csv` writes `results/final` (summary,
    tables, figures and `numbers.tex` for the report). It ran on the laptop
    (Windows, Python 3.14.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1)
    on 1 October in 163 s; `python src/adapt_eval.py report --outdir
    results/final` redraws the tables, macros and figures from its
    `summary.json` without refitting. The same code on the UW server (Linux,
    Python 3.12.3, the same package versions) does not reproduce every
    digit. C2 from M1 to M3, with the primary statistic, agrees to 4e-4
    (relative). Results that refit forests or choose ridge penalties anew
    for every held-out template differ more: C1's intervals by up to 7%
    and the correction model's wins on M3 (12 of 22 on Linux, 13 on
    Windows), the harder test's intervals, and C4 with one template held
    out (0.93 [0.82, 1.06] and 49 of 93 templates on Linux, 0.94
    [0.83, 1.06] and 48 of 93 on Windows; the family level is 1.04
    [0.90, 1.19] on both). No decision of the plan changes. On 7 October
    `results/final` was replaced by the run on the UW server
    (a second server of the same type, 174 s), the platform of the extended study in
    `results/paper/final`, so that the report quotes one run: its M1 to M2
    oracle (1.0557851930345266) equals the extended study's to the last
    digit. The laptop's run stays in git history (commit d44fd49).
12. **Report.** `python src/report_figures.py` writes `results/report`
    (the report's own figures, the environment, C1 and method tables, and
    macros drawn from `results/final`, `results/paper/final` and the labels,
    without refitting), and `latexmk -pdf final_report.tex` (MiKTeX) builds
    `final_report.pdf`; every number in it comes from the macro files
    above. Since 7 October the report also covers the extended study of
    `paper/` (six environments, both workloads, per-template offsets, query
    selection, the confirmatory tests), within the 8-page limit.

## Decisions taken after the proposal

- **Clock.** WSL's monotonic clock, which `EXPLAIN ANALYZE` uses, ran 1.2%
  to 2% fast against Windows; Windows agreed with the server's NTP time to
  0.01%. systemd-timesyncd in the guest fought Hyper-V's time sync and left
  the kernel tick at 10202 us. `scripts/wsl_clock.py` stops timesyncd,
  resets the tick and selects `hyperv_clocksource_tsc_page`; the clock then
  ran 1.00005 times as fast as Windows' clock (least-squares slope of guest
  time on host time over 3 minutes, on 1 October). An earlier check that
  read about 0.5% fast was biased by the delay of its first line. Each pass
  records its average clock rate (monotonic over wall time) in `run.json`:
  1.00004 to 1.00005 for M1's passes, 1.0 on M3. The pilot's laptop timings
  carry the old error of 1% to 2%.
- **Sleep, battery and other programs.** With Modern Standby, closing the
  lid suspends the laptop whatever the keep-awake helper asks (this happened
  on 30 September during the build). A pass whose power log has a gap of
  more than 90 s, or that ran partly on battery or with energy saver on, is
  measured again. The power log also records Windows CPU load every 30 s,
  which load1 inside WSL cannot see.
- **Q11 and Q18 kept.** Q11's widened queries barely widen its runtime (its
  only range parameter changes how many aggregated parts are sorted), and
  PostgreSQL gives every Q11 and Q18 query the same plan and cost, because
  its estimate of a HAVING threshold ignores the value. Q18's widened
  queries still vary 2.5 times in runtime and spill about 1 GB to temporary
  files per run. Both stay in the workload as the proposal describes; Q18
  sets a floor on its per-template error that no plan-feature model can
  cross, which the report states. `collect.py` records the temporary blocks
  each query writes.
- **M2 by processor affinity.** The proposal made M2 by restarting WSL with
  `processors=1`. Docker Desktop runs in the same WSL virtual machine, and
  on 1 October it had 27 containers of another project running (about half a
  runnable task on average). A restart
  would have stopped them, and with one processor they would have shared it
  with the database during every pass. Pinning the PostgreSQL server to one
  virtual processor instead gives the database exactly one processor, keeps
  the plans (the planner does not look at the processor count) and leaves
  the operating system, the collector and everything else on the laptop as
  they were for M1, so the only change between M1 and M2 is the database's
  processor count. The same containers ran during all laptop passes (M1,
  TPC-DS and M2).
- **TPC-DS pass 1.** Analysis work ran on the laptop during it: its host
  CPU median was 39%, against 15% and 18% in passes 2 and 3, and 46 of its
  482 queries ran at least 1.8 times slower than in their fastest pass. No
  rule rejects such a pass (no level shift, no power event), and the median
  of three passes removes those runs: one query was slow in two passes, and
  6 labels exceed 1.5 times their fastest pass. Since then nothing else runs
  on the laptop during a pass; checks run on the UW server instead.
- **M3 at quiet hours.** The server's load was 12 to 20 in the evening, so
  its passes start after 01:30 once the 5-minute load is below 10.
- **TPC-DS families.** Many TPC-DS templates are near copies of another (37
  and 82 differ only in catalog_sales against store_sales). Holding out one
  while its copy stays in training would bias C4 toward the correction
  model. `scripts/tpcds_families.py` links templates whose literal-masked
  first queries have a token similarity of at least 0.85: 99 templates form
  81 families, 15 of them with more than one template. C4 holds out
  families; the template-level result is reported beside it. Both were
  fixed before any TPC-DS label existed.
- **TPC-DS duplicates.** Template 14's first part depends only on the year,
  which has three values at scale factor 1; the generator now tests each
  part on its own and writes only new parts.
- **Evaluation.** Each source machine has one source model, trained on its
  own pool queries (not filtered by the target's timeouts). Queries whose
  plan features differ between two machines are left out of that pair and
  counted. C1 uses pool queries only. Stream batches have no short remainder
  batch. Pooled median and P90 carry template-resampling intervals.
