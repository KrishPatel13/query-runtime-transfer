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
| Paper: *One Query per Template: Moving Query Runtime Models to New Hardware* | [`paper/paper.pdf`](paper/paper.pdf) | `cd paper && latexmk -pdf paper.tex` |

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

## What is here

| Path | What |
| --- | --- |
| `queries/tpch_sf1_v2/` | the 680 TPC-H queries (22 templates), their manifest and the pool/test split |
| `queries/tpcds_sf1_kept/` | the 482 TPC-DS queries (93 templates, 75 families of near copies) and their split |
| `queries/tpch_sf1/`, `queries/tpcds_sf1/` | the pilot's 420 TPC-H queries and all generated TPC-DS queries before screening |
| `results/runs/<env>/` | per environment: the three passes (`pass<i>/runtime_dataset.csv`), the labels (`labels.csv`), pass-to-pass noise, the plans of every pass (`plans.tar.gz`), per-operator timings (`node_times.jsonl.gz`, M1 to M5), the processor affinity of pinned environments, the laptop's power log, and passes that failed a check and were measured again (`rejected/`); `M1` to `M6` are TPC-H, `M1_tpcds` to `M6_tpcds` TPC-DS |
| `results/final/` | the course study's evaluation (`summary.json`, tables, figures, `numbers.tex`) |
| `results/paper/final/` | the extended study's evaluation (`summary.json.gz`, tables, figures, macros, `hypotheses.json`) |
| `results/report/` | the course report's own figures, tables and macros |
| `results/pilot/` | the pilot (one machine, 420 queries), which set the course study's thresholds |
| `results/tpcds_sf1/`, `results/workload_check/`, `results/eval_dryrun/` | the TPC-DS screening run, the check of the widened TPC-H queries, and the evaluation's dry run on synthetic targets |
| `src/` | measurement (`collect.py`, `run_passes.py`), labels (`make_labels.py`), evaluation and report code |
| `scripts/` | PostgreSQL build, data loading, query generation, environment setup and checks |
| `tests/` | unit tests (`python -m pytest tests`) |
| `STUDY.md` | the run book of the course study: every step, its command, and the decisions taken during the measurements |
| `paper/PLAN.md` | the extended study's plan, with the hypotheses and their timestamped amendments |

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
median of three timed runs after one warm-up run.

## Rebuilding the results from the labels

Python 3.12 with the packages in `requirements.txt` (the results were
computed with numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1 and
matplotlib 3.11.2, on Linux).

Course study and report:

```sh
python src/adapt_eval.py run --c4-labels results/runs/M1_tpcds/labels.csv   # about 3 minutes
python src/report_figures.py
latexmk -pdf final_report.tex
```

Extended study and paper:

```sh
python src/paper_eval.py run --out results/paper/final --jobs 16      # about 20 minutes on 16 cores
python src/paper_report.py --run results/paper/final                  # tables, figures, numbers.tex, hypotheses.json
python src/paper_sources.py --run results/paper/final                 # other source models
python src/paper_streams.py --run results/paper/final                 # streams
python src/paper_operators.py --out results/paper/final               # per-operator slowdowns
python src/paper_single_run.py --out results/paper/final --jobs 16    # k queries measured by one run each
cd paper && latexmk -pdf paper.tex
```

The paper needs a TeX installation with the `acmart` class's packages
(TeX Live works). Results that refit random forests can differ in the last
digits between operating systems; `STUDY.md` (step 11) lists how much.

## Repeating the measurements

`scripts/setup_postgres.sh` builds PostgreSQL 18.6 in user space (no root
access needed), and `scripts/load_tpch.sh` and `scripts/tpcds_setup.sh` with
`scripts/tpcds_load.sh` load the data at scale factor 1 with the
[tpch-kit](https://github.com/gregrahn/tpch-kit) and
[tpcds-kit](https://github.com/gregrahn/tpcds-kit) generators.
`src/run_passes.py` runs the three label passes of one environment (see its
`--help`), and `src/make_labels.py` turns them into labels. The pinned
environments start PostgreSQL with `scripts/start_m2.sh`,
`scripts/start_pinned.sh` or `scripts/run_m6_chain.sh`, and M5 with
`scripts/run_m5_chain.sh`. `scripts/export_cluster.sh` and
`scripts/import_cluster.sh` copy a stopped cluster to another machine so that
its statistics and plans stay identical, and `scripts/explain_all.py
--compare` checks the plans. `STUDY.md` describes the precautions the
measurements needed (the WSL clock, power and sleep on the laptop, quiet
hours on the shared server).

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
CC BY 4.0 (see [`LICENSE`](LICENSE)). The LaTeX class and style files
`paper/acmart.cls`, `paper/ACM-Reference-Format.bst`, `project.sty` and
`watml.sty` keep their own licenses. TPC-H and TPC-DS are trademarks of the
Transaction Processing Performance Council; the queries here were generated
with the TPC tools, and the runtimes are not TPC benchmark results and are not
comparable with published ones.

## Citation

See [`CITATION.cff`](CITATION.cff).
