# Registration record

The paper (Section IV, "Exploratory and confirmatory analyses") cites six
commits as the record of what was fixed before which data existed. They are
commits of the study's development repository, which is private because it
also holds unrelated coursework. This public repository is an export of it
and has its own history, so the cited commits cannot be looked up here. This
folder shows what each of them contains: its full hash, its time, what it
fixed, the files it changed, and the study plan (`paper/PLAN.md`) exactly as
it stood at that commit. In this public copy, private host and user names are
replaced as everywhere else in the repository.

| Commit | Time (Eastern) | What it fixed | Plan at that commit |
| --- | --- | --- | --- |
| `17c328f1ec7ed2a31f35c28ccdef98c5d2b54211` | 2026-10-02 03:55:57 | The primary method (the template correction with round-robin selection) and hypotheses H1 to H5, after the first TPC-DS passes on M2 and M3 had started, before any label of TPC-DS on M2, M3 or M4 or of TPC-H on M4 existed. Files: `paper/PLAN.md`, `src/paper_eval.py`. | [PLAN_at_17c328f.md](PLAN_at_17c328f.md) |
| `b2e59dcde5bc48e69017b2c30b48a55886a92ef6` | 2026-10-02 04:10:16 | Amendment after a code review, still before any such label: TPC-DS intervals resample query families, and the test on new templates holds out families. Files: `paper/PLAN.md`, `src/paper_eval.py`, `src/node_analysis.py` and the draft paper. | [PLAN_at_b2e59dc.md](PLAN_at_b2e59dc.md) |
| `fc9dc9a1745b532742308a5d6116f27d0ca4277d` | 2026-10-02 04:13:26 | Amendment: affine+parallel and the primary method leave out their parallel terms when the k queries hold one plan class. Files: `paper/PLAN.md`, `src/paper_eval.py`. | [PLAN_at_fc9dc9a.md](PLAN_at_fc9dc9a.md) |
| `13f14ce61960ef88743225ef4be0f8b4277fd124` | 2026-10-08 14:37:56 | The second round: experiments E1 (TPC-H at scale factor 10), E2 (the Join Order Benchmark) and E3 (a machine that is not virtualized), H1 to H5 for their pairs and H6 on the neural baseline, before any of their queries ran. File: `paper/PLAN.md`. | [PLAN_at_13f14ce.md](PLAN_at_13f14ce.md) |
| `52cc5b4fe7078c450311e28ce33cc1ff8f0d65fe` | 2026-10-08 17:22:12 | The neural model and its fine-tuning rules frozen (`src/nn_model.py`, `src/paper_nn.py`), with its results on the development pairs (`results/paper/nn_dev`). | [PLAN_at_52cc5b4.md](PLAN_at_52cc5b4.md) |
| `909c069118de3e8046f79244b04b1aca00e20901` | 2026-10-08 18:18:58 | H6 amended into H6a and H6b after its development results, before any new pair was evaluated; the original H6 is still reported. File: `paper/PLAN.md`. | [PLAN_at_909c069.md](PLAN_at_909c069.md) |

## Changes after the freeze

`src/nn_model.py` has not changed since `52cc5b4`. `src/paper_nn.py` changed
once, in commit `609e56a` (2026-10-08 21:05:31 Eastern, before any
second-round evaluation): it loads the second round's workloads, takes their
k grids from `src/paper_eval.py` and reads the compressed summary. Nothing of
the model, its training or its fine-tuning changed.

## What this record shows

The times are those git recorded when each commit was made, so this folder
documents the order of the steps; the plan's log (`paper/PLAN.md`, section
"Log") gives the same order with the times data arrived. The second round was
evaluated once, on 2026-10-10 at 12:20, with the code as committed; the log
entry of that time lists every test's outcome as registered.
