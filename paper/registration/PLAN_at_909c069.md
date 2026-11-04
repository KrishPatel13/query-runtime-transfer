# Paper plan: adapting query runtime models to new hardware

Branch `paper` (from `project`). The course report (`final_report.tex` on
`project`) is finished and is not changed here; this branch extends the study
into a paper.

## Question

A model that predicts a query's runtime from its PostgreSQL plan is trained
on machine A. The database moves to machine B: the same data, statistics and
settings, so the same plans, but different runtimes. We can afford to run k
queries on B. How should those k measurements be used, which queries should
they be, and how many are enough?

## Research questions

- RQ1 (shift). When plans stay fixed, how do runtimes change between
  environments? How much of the change is one speed factor, how much depends
  on the plan, and which plan properties carry it? Is the change caused by
  losing processors the same on different hardware?
- RQ2 (methods). With k target queries, which adaptation wins at which k, how
  many queries reach 90% of the achievable gain, and does the answer depend
  on the source model?
- RQ3 (selection). Which k queries should be measured: random, one per
  template in turn, spread over the feature space (k-center), or chosen for
  the linear correction (D-optimal)?
- RQ4 (coverage). Does the adapted model hold on templates that none of the k
  queries came from?
- RQ5 (streams). After a switch of machine in a query stream, how fast do
  online strategies recover?

## Environments (plans fixed in all of them)

| Name | What | TPC-H | TPC-DS |
| --- | --- | --- | --- |
| M1 | laptop, Intel Core Ultra 7 155H, WSL2 | done | done |
| M2 | M1, PostgreSQL pinned to one virtual processor | done | to measure (laptop) |
| M3 | UW server server-a, AMD EPYC 9754, KVM guest | done | to measure (server) |
| M4 | M3, PostgreSQL pinned to one processor | to measure | to measure |
| M3b (optional) | second server of the same type (server-b): a null-shift control | optional | optional |

M1 to M4 give 12 ordered pairs per workload: hardware alone (M1 to M3, M2 to
M4), processors alone (M1 to M2, M3 to M4), both, and the reverse
directions. Every pair keeps only queries with identical plan features on
both machines, as in the course study.

## Workloads

- TPC-H SF1: 680 queries, 22 templates, split 476 pool / 204 test
  (`queries/tpch_sf1_v2/split.csv`).
- TPC-DS SF1: 482 queries, 93 templates, split 294 pool / 188 test
  (`queries/tpcds_sf1_kept/split.csv`, written by
  `scripts/make_tpcds_split.py` on 2 October before any new TPC-DS label).

## Methods

Source models (trained on the source machine's pool): the course study's
cost line plus forest on its residuals; a random forest; gradient-boosted
trees; ridge.

Adaptation with k target queries (all predict log runtime):

- none; speed factor (one factor); target only; oracle (as in the course)
- two-factor: separate factors for plans with and without parallel workers,
  each shrunk toward the common factor
- ridge correction (as in the course)
- ridge correction plus template offsets with partial pooling
- forest on the residuals (fine-tuning a nonlinear correction)
- pooled refit: the source pipeline refitted on the source pool and the k
  queries, with the machine as an indicator feature (not Daume III's
  feature augmentation)

Selection of the k queries: random (20 seeded orders, as now),
template round robin, k-center greedy, D-optimal greedy for the ridge
design, and parallel/serial balanced.

## Measures

The course study's: per-template median Q-error averaged in log over test
templates, share of the gap to the oracle closed, paired ratios with
template-resampling intervals, pooled median and P90. Added: the smallest k
at which a method beats the speed factor, and the relation between the
shift's heterogeneity (template share, spread of template slowdowns) and the
gain of the ridge correction across all pairs.

## Measurement schedule

- Server (server-a): M3 TPC-DS, then M4 TPC-H, then M4 TPC-DS, three
  passes each, sequentially on the same host.
- Laptop: M2 TPC-DS (CPU 19, as M2), on AC power, nothing else running.
- Analysis runs on server-b while either machine measures.

## Confirmatory tests, fixed before the new target labels

Fixed on 2 October 2026, after the exploratory runs on TPC-H with M1, M2
and M3 (`results/paper/tpch_dev*` on the analysis server) and before any
label of TPC-DS on M2, M3 or M4, or of TPC-H on M4, was read. The methods
and settings are those of `src/paper_eval.py` in the commit that adds this
section. The tests below use only the new data: every ordered pair of
machines in TPC-DS, and every TPC-H pair that involves M4. Source model:
the course study's (cost line plus forest on its residuals). Intervals:
95%, from 2000 resamples of test templates; ratios are geometric means over
test templates of per-template median Q-errors (method over comparator).

Primary method: `lowdim+template` with `round-robin` selection. It fits the
affine+parallel line (the log error against the source model's log
prediction, separately for plans with parallel workers; ridge with an
unpenalized intercept, penalty by leave-one-out) and adds, for every
template among the k queries, the mean residual of its queries from that
line.

- H1 (the shift is template-specific). The share of the variance of the
  log slowdown between templates exceeds 0.5 in every pair.
- H2 (primary). At k = 100, `lowdim+template` with round-robin selection
  beats the speed factor with random selection in every pair (interval of
  the ratio below 1), and beats the ridge correction with random selection
  in at least two thirds of the pairs (interval below 1).
- H3 (small budgets). At k = 10, `affine+parallel` with random selection
  has a lower error than the speed factor with random selection (ratio
  below 1) in at least three quarters of the pairs whose change includes
  the processor count (M1-M2, M3-M4, M1-M4, M2-M3, both directions).
- H4 (unseen templates). In the harder test (k = 25, random selection,
  the template held out everywhere), the speed factor's error (geometric
  mean over templates) is at most 1.15 times the oracle's in every pair:
  on new templates the remaining error is the model's generalization
  error, not the machine change.
- H5 (no template labels). At k = 100, k-center selection with the
  `1-nn` correction has a lower error than random selection with the ridge
  correction (ratio below 1) in at least two thirds of the pairs.

Everything else (other methods, strategies, source models, streams,
transfer, per-operator timings) is exploratory and is reported as such.

Amendment, 2 October 2026, 04:10 (commit b2e59dc), after a code review of
`src/paper_eval.py` and still before any label file of the new data
existed (the passes were running; their progress lines had shown single
runtimes of about twenty queries, no label was read):

- TPC-DS has 15 families of near-copy templates (`labels.csv`'s family
  column, the unit of the course study's C4). In TPC-DS the intervals
  resample families instead of templates, so that near copies do not count
  as independent, and the harder test of H4 holds out the template's whole
  family (from the source model, the k queries, the oracle and any shift
  model), with the error still measured per template. TPC-H is unchanged:
  every template is its own unit.
- H1 is read beside the template share that noise alone gives on the same
  workload (the share of log(pass j / pass i) between the source machine's
  own passes), which is about 0.03 to 0.08 for TPC-H and about 0.18 for
  TPC-DS; the threshold 0.5 stays.
- Models of the shift itself (transfer, and the share of the shift that
  plan features carry to unseen templates) choose their ridge penalty by
  leaving out templates (families in TPC-DS), not single queries. These are
  exploratory.
- No gap closed is reported for the harder test, where the oracle is not
  an upper bound; stream recovery is counted per stream order (median).
- Bug fix in the primary method (and in affine+parallel): when the k
  queries contain only one plan class, the parallel terms are left out
  (their columns would be constant or duplicate the slope, and the fit
  invented a line for the other class); with one query in the smaller
  class only the separate intercept enters.

## Extension for the full paper, fixed before any new label

Fixed on 8 October 2026, in the commit that adds this section, before any
query of the new data below had run. The paper becomes a 12-page
experiments-and-analysis paper (ICDE 2027, round 2; see `VENUES.md`).
Everything already measured (the 60 pairs above) is development data from
now on; the tests below use only the new pairs.

New data:

- E1, TPC-H at scale factor 10 on the server. Queries:
  `queries/tpch_sf10_sub`, 220 queries, 7 pool and 3 test queries per
  template drawn with seed 0 from the pool and test sets of
  `queries/tpch_sf1_v2/split.csv`, with Q11's fraction set to 0.0001/SF
  (0.00001) as the TPC-H specification prescribes. A new cluster with the
  same build, initdb options and settings as M3 holds the data generated by
  dbgen at SF 10. Environments: `M3_sf10` (not pinned), `M4_sf10` (one
  processor) and `M6_sf10` (two processors), as M3, M4 and M6. Six ordered
  pairs.
- E2, the Join Order Benchmark (JOB) on the IMDB snapshot of its authors
  (event.cwi.nl/da/job/imdb.tgz), with the schema, the foreign-key indexes
  and the 113 queries of github.com/gregrahn/join-order-benchmark at commit
  a39603662e023e449cb2121997a5034df9e02ebf. A query's template is its
  number (33 templates; the letter is the variant). Split inside every
  template with seed 0: round(0.3 n) test queries, at least one, the rest
  pool. Before any label, every query runs once on `M3_job`, and a query
  over 60 s is dropped (the screening is reported). Environments:
  `M3_job`, `M4_job` and `M6_job` on the server, and `M1_job` and `M2_job`
  on the laptop if they are measured. All ordered pairs of measured
  environments.
- E3, a machine that is not virtualized (`M7`), if one is available:
  TPC-H and TPC-DS at scale factor 1 from copies of M1's clusters and the
  same build. Pairs of M7 with M1 to M6.

The protocol of the labels is unchanged (one warm-up and three timed runs
per pass, three passes in seeded orders, the median of medians, a pass with
a level shift over 10% is measured again). Source model: the course
study's (cost line plus forest on its residuals). Intervals and ratios as
for H1 to H5. The budget kH is 100 for E1 and E3, and the number of
templates with a pool query for E2 (one query per template).

- H1 to H5 hold on the new pairs of each experiment under their registered
  rules, with kH in place of k = 100 in H2 and H5.
- H6 (fine-tuning a neural model). A neural plan model is trained on the
  source machine's pool and fine-tuned on the kH target queries chosen by
  round robin, as learned cost models are moved in the literature. Its
  architecture follows DACE (a transformer over the plan's nodes with the
  optimizer's estimates as inputs), and it, its training and its
  fine-tuning are developed on the 60 existing pairs only and frozen in a
  commit, logged here, before their first evaluation on a new pair. The
  template correction with round-robin selection on the course study's
  source model has a lower error than the fine-tuned neural model on the
  same queries (interval of the ratio below 1) in at least two thirds of
  the new pairs.

Everything else on the new data is exploratory, among it the template
correction on the neural source model, and pairs that join a new
environment with an old one.

## Status

See the log at the end of this file.

## Log

- 2026-10-02: branch created; TPC-DS cluster archived and copied to the
  server; TPC-DS split written; literature and venue search started.
- 2026-10-02 03:37: M3 TPC-DS passes started on the server; 03:40: M2
  TPC-DS passes on the laptop (CPU 19). M4 (server, CPU 63) and the
  per-operator timing runs are queued behind them.
- 2026-10-02 03:56: confirmatory tests committed (17c328f), before any new
  TPC-DS or M4 label was read.
- 2026-10-02 04:00: added M5, the laptop with PostgreSQL's default buffer
  pool (shared_buffers 128MB instead of 2GB), a memory-configuration
  change that keeps the plans. It was added after the confirmatory tests
  were fixed, so its pairs are exploratory and do not enter H1 to H5.
  Queued on the laptop after the timing runs (`scripts/run_m5_chain.sh`).
- 2026-10-02 14:30: all labels of M2 to M5 in; the full evaluation
  (`results/paper/main`, then `results/paper/final` with the same numbers)
  confirms H1 to H5 under the registered rules. Counted by intervals, H3
  holds in 8 of 12 pairs and H5 in 16 of 18; the paper reports both.
- 2026-10-02 15:00: a paper review (five reviewers) led to: reporting the
  refit baselines (target only, pooled refit, forest on the errors), a
  model-free template lookup and a no-move control (a machine as its own
  source), the bias-corrected template share (epsilon squared), and exact
  wording of the registration timeline. All exploratory.
- 2026-10-02 15:24: M6 (two processors) first started on the laptop, and
  stopped after 96 queries because other programs were using the laptop's
  processors; its partial results were deleted. M6 was then defined as the
  server (M3) with PostgreSQL pinned to two processors (CPUs 62 and 63) and
  measured there from 15:56 (`scripts/run_m6_chain.sh`). Exploratory.
- 2026-10-02: per-operator timings (`scripts/node_timing.py`) on M1 to M5,
  both workloads; analysis in `src/node_analysis.py` and
  `src/paper_operators.py`. Exploratory.
- 2026-10-02 21:45: a final review of the paper (numbers, style, a
  program-committee read) asked how the k target measurements enter the
  corrections: they used the three-pass labels, while the cost in seconds
  counts one run. `src/paper_single_run.py` reruns the hybrid source model's
  C2 with the k queries measured by one run of one pass (a warm run, or the
  query's first run in the pass) and the test labels unchanged; its
  label variant reproduces `summary.json.gz` exactly. At one query per
  template the primary method errs by 1.12 and 1.22 (warm) and 1.14 and
  1.25 (first run) on TPC-H and TPC-DS, against 1.10 and 1.19 with the
  labels, and the no-move control rises about as much. Exploratory.
- 2026-10-08: the paper moves to a 12-page experiments-and-analysis
  version for ICDE 2027 round 2 (Nov 11), then PVLDB and ICPE if needed.
  New experiments E1 to E3 and hypothesis H6 fixed in the section
  "Extension for the full paper" before any of their queries ran.
- 2026-10-08: two details of E1 and E2, fixed before any of their queries
  ran. E1 divides every Q11 fraction by the scale factor, the widened
  queries' as well as the standard 0.0001 (PostgreSQL's estimates ignore
  the fraction). E2 screens first and splits the kept queries; a template
  with a single kept query is dropped, since it cannot be split. Tools:
  `scripts/make_sf10_queries.py`, `scripts/job_queries.py`,
  `scripts/load_job.sh`, `scripts/run_ext_chain.sh`.
- 2026-10-08: the neural model of H6 is frozen in the commit that adds
  this entry (`src/nn_model.py`, `src/paper_nn.py`, `requirements-nn.txt`)
  before any evaluation on a new pair: a DACE-style transformer (one layer,
  one head, width 128, a tree attention mask, MLP 128-64-1, Q-error loss on
  the root), dropout 0, learning rate 3e-3 and 800 epochs chosen by
  template-grouped 5-fold cross-validation inside the development source
  pools, and fine-tuning rules that depend on k only (chosen on the
  development pairs, scored on target pool queries outside the k). The
  development results are in `results/paper/nn_dev`. On them H6 does not
  hold: at k = 100 the ratio's interval lies below 1 in 0 of 30 pairs in
  each workload (geometric means 1.018 and 1.012; the network fine-tuned
  on the same queries is better by 1 to 2%), while at k = 5 and 10 the
  template correction is ahead (geometric means 0.95 to 0.97, point
  estimates below 1 in 19 to 29 of 30 pairs).
- 2026-10-08: amendment of H6, after its development results (previous
  entry) and before any evaluation on a new pair (no label file of a new
  environment existed; the first SF10 pass was running). H6 is still
  evaluated and reported as registered, and two tests are added in its
  place as the confirmatory ones. "Fine-tuned" means full fine-tuning (all
  weights) under the frozen rules; head-only fine-tuning is reported beside
  it. The ratio is the template correction with round-robin selection on
  the course study's source model over the network fine-tuned on the same
  queries.
  - H6a (small budgets). At k = 10 the ratio's point estimate is below 1 in
    more than half of the new pairs of each experiment, and its geometric
    mean over those pairs is below 1.
  - H6b (no large loss at kH). At kH the upper end of the ratio's 95%
    interval is below 1.15 in every new pair.
  On the development pairs H6a holds (point estimates below 1 in 29 and 19
  of 30 pairs, geometric means 0.958 and 0.969) and so does H6b (largest
  upper ends 1.082 and 1.112).
