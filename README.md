# Moving query runtime models to new hardware

Code, measurements and documents of a study on adapting learned query
runtime models when a PostgreSQL database moves to another machine. A model
that predicts a query's runtime from its plan is trained on one machine; on
new hardware the same plans change speed by different factors, and the model
drifts. The study asks how few queries measured on the new machine are enough
to adapt the model, which queries they should be, and whether anything helps
on query templates the new machine has never run.

Two documents are built from this repository:

| Document | File | Build |
| --- | --- | --- |
| Course project report (CS 680, University of Waterloo, Fall 2026): *Adapting Small Query Runtime Models to New Machines and Unseen Query Templates* | [`final_report.pdf`](final_report.pdf) | `latexmk -pdf final_report.tex` |
| Paper: *[Experiment, Analysis, and Benchmark] When Is One Query per Template Enough? Adapting Learned Cost Models to New Hardware* | [`paper/icde/paper.pdf`](paper/icde/paper.pdf) | `cd paper/icde && latexmk -pdf paper.tex` |

Every number, table and figure in both documents is a LaTeX macro or file
written by the scripts below from the saved measurements; nothing is typed in
by hand.

## Main results

The same 1,162 TPC-H and TPC-DS queries were timed, with identical plans, in
six environments, and models were moved between them. The error is the
per-template median Q-error, averaged geometrically over templates (1.00 is
perfect), at the budget that allows one measured query per template, averaged
over the 30 ordered pairs of environments of each benchmark:

| Method | TPC-H, 25 queries | TPC-DS, 100 queries |
| --- | --- | --- |
| No adaptation | 1.83 | 1.96 |
| One speed factor | 1.37 | 1.45 |
| Ridge correction of the model's errors | 1.24 | 1.38 |
| Template correction (one offset per measured template) | 1.10 | 1.19 |
| The same correction on a model trained on the new machine | 1.08 | 1.18 |

- Between 69% and 94% of the variance of the slowdown lies between query
  templates, so one speed factor stops improving after about five queries.
- Measuring one query per template and adding a per-template offset comes
  within about 2% of a model trained on the new machine. It beats the ridge
  correction in all 30 pairs of both benchmarks, and the speed factor with
  all four other source models tried (240 of 240 combinations).
- On templates without any measured query, no correction reliably beats the
  speed factor, and a model anchored to the optimizer's cost is no better
  than a line on that cost (ratio 1.04, 95% interval 0.90 to 1.19).
- Five hypotheses were fixed in [`paper/PLAN.md`](paper/PLAN.md) before 18 of
  the machine pairs were measured; all five hold on them.
- A second round, with its hypotheses fixed before any of its labels
  existed, repeated the tests on TPC-H at scale factor 10 (6 pairs on the
  server) and on the Join Order Benchmark (20 pairs on both machines). Most
  of its tests fail. At ten times the data the errors follow the first
  round (template correction 1.12 against 1.30 for the speed factor, with
  100 queries), but two of five tests miss. On JOB, whose queries differ
  in runtime by a factor of 2.3 within a template, the model errs by 2.36
  even when trained on the new machine, no correction gains much, and only
  the test on new templates holds; a neural model fits JOB better, and
  correcting it works as well as fine-tuning it.
- In all four workloads, once each template has a measured query, the
  moved model comes within 3% of the same correction on a model trained on
  the new machine.

## What is here

| Path | What |
| --- | --- |
| `queries/tpch_sf1_v2/` | the 680 TPC-H queries (22 templates), their manifest and the pool/test split |
| `queries/tpcds_sf1_kept/` | the 482 TPC-DS queries (93 templates, 75 families of near copies) and their split |
| `queries/tpch_sf1/`, `queries/tpcds_sf1/` | the pilot's 420 TPC-H queries and all generated TPC-DS queries before screening |
| `queries/tpch_sf10_sub/` | the second round's 220 TPC-H queries at scale factor 10: 10 SF1 queries per template (7 pool, 3 test), with Q11's fraction divided by the scale factor |
| `queries/job_all/`, `queries/job_kept/` | the 113 queries of the Join Order Benchmark (33 templates) as copied, and the kept set with its split (78 pool, 35 test); third-party files, see [`LICENSE`](LICENSE) |
| `results/runs/<env>/` | per environment: the three passes (`pass<i>/runtime_dataset.csv`), the labels (`labels.csv`), pass-to-pass noise, the plans of every pass (`plans.tar.gz`), per-operator timings (`node_times.jsonl.gz`, M1 to M5), the processor affinity of pinned environments, the laptop's power log, and passes that failed a check and were measured again (`rejected/`); `M1` to `M6` are TPC-H, `M1_tpcds` to `M6_tpcds` TPC-DS, `M3_sf10`, `M4_sf10` and `M6_sf10` TPC-H at scale factor 10, and `M1_job`, `M2_job`, `M3_job`, `M4_job` and `M6_job` JOB |
| `results/final/` | the course study's evaluation (`summary.json`, tables, figures, `numbers.tex`) |
| `results/paper/final/` | the paper's evaluation of the first round (`summary.json.gz`, tables, figures, macros, `hypotheses.json`) |
| `results/paper/nn_dev/` | the neural cost model on the first round's pairs, with the cross-validation and tuning runs that fixed its settings (`cv*/`, `tune/`) |
| `results/paper/ext/` | the second round's evaluation, its tests (`hypotheses_ext.json`), tables and macros |
| `results/paper/nn_ext/` | the neural cost model on the second round's pairs |
| `results/paper/extra/` | exploratory analyses that were not registered: a refit that knows the templates, shift transfer, the pool split into halves, the tails, and the first round's numbers on the second round's pairs |
| `results/paper/environments/` | the paper's table of the environments |
| `results/hardware.md` | what the machines report about their processors, memory and virtualization |
| `results/report/` | the course report's own figures, tables and macros |
| `results/job_screen/` | the JOB screen (every query once on the server, before any label) and its plans |
| `results/pilot/` | the pilot (one machine, 420 queries), which set the course study's thresholds |
| `results/tpcds_sf1/`, `results/workload_check/`, `results/eval_dryrun/` | the TPC-DS screening run, the check of the widened TPC-H queries, and the evaluation's dry run on synthetic targets |
| `src/` | measurement (`collect.py`, `run_passes.py`), labels (`make_labels.py`), evaluation and report code; `nn_model.py` is the neural cost model |
| `scripts/` | PostgreSQL build, data loading, query generation, environment setup and checks |
| `tests/` | unit tests (`python -m pytest tests`) |
| `STUDY.md` | the run book of the course study: every step, its command, and the decisions taken during the measurements |
| `paper/PLAN.md` | the paper's study plan, with both rounds' hypotheses and their timestamped amendments |

## The environments

| Name | Machine |
| --- | --- |
| M1 | laptop, Intel Core Ultra 7 155H (22 threads), 16 GB, PostgreSQL in WSL2 |
| M2 | M1 with PostgreSQL pinned to one processor |
| M3 | virtual machine with 64 virtual processors of an AMD EPYC 9754 |
| M4 | M3 with PostgreSQL pinned to one processor |
| M5 | M1 with `shared_buffers = 128MB` |
| M6 | M3 with PostgreSQL pinned to two processors |

All six run PostgreSQL 18.6 from one build, without JIT, on copies of the same
database files, so every plan and estimate is the same in all of them. A
query's label is the median over three passes, in random orders, of the
median of three timed runs after one warm-up run. The first round ran TPC-H
and TPC-DS at scale factor 1 in all six; the second ran TPC-H at scale
factor 10 in M3, M6 and M4, and JOB in M1, M2, M3, M6 and M4.

## Rebuilding the results from the labels

Python 3.12 with the packages in `requirements.txt` (the results were
computed with numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1 and
matplotlib 3.11.2, on Linux). The neural cost model needs PyTorch in a
second environment, `requirements-nn.txt` (CPU build).

Course study and report:

```sh
python src/adapt_eval.py run --c4-labels results/runs/M1_tpcds/labels.csv   # about 3 minutes
python src/report_figures.py
latexmk -pdf final_report.tex
```

The paper's first round (TPC-H and TPC-DS at scale factor 1):

```sh
python src/paper_eval.py run --out results/paper/final --jobs 16      # about 20 minutes on 16 cores
python src/paper_report.py --run results/paper/final                  # tables, figures, numbers.tex, hypotheses.json
python src/paper_sources.py --run results/paper/final                 # other source models
python src/paper_streams.py --run results/paper/final                 # streams
python src/paper_operators.py --out results/paper/final               # per-operator slowdowns
python src/paper_single_run.py --out results/paper/final --jobs 16    # k queries measured by one run each
```

The neural cost model on the first round's pairs (PyTorch environment). It
reads the plans of each run's first pass, which are packed in
`results/runs/<env>/plans.tar.gz`; unpack them once first:

```sh
for f in results/runs/*/plans.tar.gz; do tar -xzf "$f" -C "$(dirname "$f")"; done
python src/paper_nn.py run --out results/paper/nn_dev --jobs 16 --reference results/paper/final/summary.json.gz
python src/paper_nn_report.py --run results/paper/nn_dev --prefix NNDev
```

The exploratory analyses of the first round:

```sh
python src/paper_eval.py run --extra --sources hybrid --no-hard --no-transfer --no-stream --out results/paper/extra --jobs 16
python src/paper_extra_report.py --final results/paper/final --extra results/paper/extra
python src/paper_holdout.py --jobs 16
python src/paper_tails.py --jobs 16
```

The second round (TPC-H at scale factor 10 and JOB), the table of the
environments and the paper:

```sh
python src/paper_eval.py run --workloads tpch10 job --out results/paper/ext --jobs 16
python src/paper_nn.py run --workloads tpch10 job --out results/paper/nn_ext --jobs 16 \
    --reference results/paper/ext/summary.json                        # PyTorch environment
python src/paper_ext_report.py --eval results/paper/ext --nn results/paper/nn_ext
python src/paper_nn_report.py --run results/paper/nn_ext --prefix NNExt
python src/paper_ext_compare.py --final results/paper/final --ext results/paper/ext
python src/paper_ext_explain.py                                       # numbers that explain the second round
python src/paper_env_table.py
cd paper/icde && latexmk -pdf paper.tex
```

The paper needs a TeX installation with the IEEE conference class (TeX Live
works). Results that refit random forests or train the neural model can
differ in the last digits between operating systems; `STUDY.md` (step 11)
lists how much for the course study.

## Repeating the measurements

`scripts/setup_postgres.sh` builds PostgreSQL 18.6 in user space (no root
access needed), and `scripts/load_tpch.sh` (`SF=10` for scale factor 10) and
`scripts/tpcds_setup.sh` with `scripts/tpcds_load.sh` load the data with the
[tpch-kit](https://github.com/gregrahn/tpch-kit) and
[tpcds-kit](https://github.com/gregrahn/tpcds-kit) generators.
`scripts/make_sf10_queries.py` derives the scale factor 10 queries from the
SF1 queries. `scripts/job_queries.py` copies the JOB queries and splits them,
and `scripts/load_job.sh` downloads the IMDB snapshot of the benchmark's
authors and loads it. `src/run_passes.py` runs the three label passes of one
environment (see its `--help`), and `src/make_labels.py` turns them into
labels. The pinned environments start PostgreSQL with `scripts/start_m2.sh`,
`scripts/start_pinned.sh` or `scripts/run_m6_chain.sh`, and M5 with
`scripts/run_m5_chain.sh`; `scripts/run_ext_chain.sh` and
`scripts/run_job_laptop.sh` ran the second round on the server and the
laptop. `scripts/export_cluster.sh` and `scripts/import_cluster.sh` copy a
stopped cluster to another machine so that its statistics and plans stay
identical, and `scripts/explain_all.py --compare` checks the plans.
`STUDY.md` describes the precautions the measurements needed (the WSL clock,
power and sleep on the laptop, quiet hours on the shared server).

## Data format

`results/runs/<env>/labels.csv` has one row per query: `query_id`,
`template`, `variant` (standard or widened parameters), `status`, `label_ms`
(the label), `pass1_ms` to `pass3_ms`, and the 19 plan features of plain
`EXPLAIN`: estimated startup and total cost, rows and width; node count and
depth; the largest and summed log row estimates; counts of scans, joins,
sorts, aggregates, materializations and subplans by type; and planned
parallel workers. `pass<i>/runtime_dataset.csv` adds every timed run, the
warm-up, planning time, buffer reads, temporary blocks and the load at the
time of the run.

Host names, user names and directories in the run metadata were replaced
(for example by `laptop`, `server-a` and `/var/tmp/pgstudy`); no measured
value was changed.

## License and notices

The code is under the MIT license and the measurements and documents under
CC BY 4.0 (see [`LICENSE`](LICENSE)). The Join Order Benchmark's queries,
schema and indexes (`queries/job_all/`, `queries/job_kept/`,
`sql/job_schema.sql`, `sql/job_fkindexes.sql`) are copied from
[gregrahn/join-order-benchmark](https://github.com/gregrahn/join-order-benchmark),
which states no license, and keep their authors' terms; the IMDB data are not
included. `project.sty` and `watml.sty` keep their own licenses. TPC-H and
TPC-DS are trademarks of the Transaction Processing Performance Council; the
queries here were generated with the TPC tools, and the runtimes are not TPC
benchmark results and are not comparable with published ones.

## Citation

See [`CITATION.cff`](CITATION.cff).
