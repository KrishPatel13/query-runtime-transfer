"""Single-run measurements of the k target queries: what measurement noise costs the corrections.

The study's labels are the median over three passes of the median of three
timed runs. Someone who moves a model would measure each of the k queries
once. This script repeats paper_eval's C2 for the hybrid source model with the
k queries' target runtimes taken from one run, while the test labels stay the
study's labels:

  label   the study's labels (reproduces paper_eval, as a check)
  warm    one warm run: the first timed run of a pass, after its warm-up run
  first   the warm-up run itself, the query's first execution in that pass

For the r-th order every one of the k queries comes from pass (r mod 3) + 1, as
if they were measured in one session. Methods: no adaptation, the speed
factor, template offsets, the primary method, 1-NN and the oracle, with random
and round-robin selection, on every ordered pair and on the no-move controls
(a machine as its own source). Exploratory: added after the hypotheses.

Writes <out>/single_run.json.gz and <out>/single_run.tex with, for the budget of
one query per template (k = 25 on TPC-H, 100 on TPC-DS) and round robin:

  Single<Workload><Variant><Method>   geometric mean over the 30 moved pairs
  Single<Workload><Variant>Control    the primary method on the no-move controls
                                      (geometric mean over the six machines)

Usage: python src/paper_single_run.py --out results/paper/final --jobs 16
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import adapt_eval as ae
import paper_eval as pe
from paper_report import GENERATED, fmt, geo, tex_name

VARIANTS = ["label", "warm", "first"]
METHODS = [pe.NONE, pe.SPEED, pe.TPL_RAW, pe.LOWDIM_TPL, pe.KNN, pe.ORACLE]
STRATEGIES = ["random", "round-robin"]
PAPER_K = {"tpch": 25, "tpcds": 100}


def single_runs(runs: Path, d: str) -> dict[str, pd.DataFrame]:
    """For one machine: (queries x passes 1 to 3) runtimes in ms of the first timed run (warm)
    and of the warm-up run (first)."""
    warm, first = {}, {}
    for i in (1, 2, 3):
        p = pd.read_csv(runs / d / f"pass{i}" / "runtime_dataset.csv").set_index("query_id")
        warm[i] = p["runs_ms"].map(lambda v: float(json.loads(v)[0]))
        first[i] = p["warmup_ms"].map(lambda v: float(json.loads(v)[0]))
    return {"warm": pd.DataFrame(warm), "first": pd.DataFrame(first)}


def pair_job(w: str, a: str, b: str, labels: dict, split: pd.DataFrame, single: dict, n_orders: int) -> dict:
    t0 = time.time()
    p = ae.Pair(labels, split, a, b, fit=pe.fit_hybrid)
    orders = ae.arrival_orders(split, n_orders)
    W = pe.unit_weights(p.templates, pe.unit_map(p))
    oracle = pe.fit_hybrid(p.pool, ae.SOURCE_SEED)(p.test)
    ks = [k for k in pe.KS if k <= p.n_pool]
    out = {}
    for v in VARIANTS:
        logs = {}
        for r, order in enumerate(orders):
            q = p
            if v != "label":
                # Only the corrections see the single runs: the source model, the oracle and the
                # test labels are unchanged.
                q = copy.copy(p)
                q.e_pool = np.log(single[v].loc[p.pool_ids, r % 3 + 1].to_numpy(float)) - p.base_pool
            pos = p.positions(order)
            for s in STRATEGIES:
                seq = pe.strategy_sequence(s, p, pos, r, max(ks))
                for k in ks:
                    for m, pred in pe.predictions(q, seq[:k], METHODS, pe.fit_hybrid, r, oracle, {}).items():
                        _, per_t, _, _ = ae.template_errors(p.y_test_ms, pred, p.groups)
                        logs.setdefault((s, m, k), np.empty((len(orders), len(p.templates))))[r] = per_t
        out[v] = pe.summarize_logs(logs, W)
    return {"workload": w, "pair": p.name, "a": a, "b": b, "c2": out, "seconds": round(time.time() - t0, 1)}


def macros(jobs: list[dict]) -> dict:
    m = {}
    for w, k in PAPER_K.items():
        moved = [j for j in jobs if j["workload"] == w and j["a"] != j["b"]]
        controls = [j for j in jobs if j["workload"] == w and j["a"] == j["b"]]
        for v in VARIANTS:
            for meth in (pe.SPEED, pe.TPL_RAW, pe.LOWDIM_TPL, pe.KNN):
                key = f"round-robin|{meth}|{k}"
                m[tex_name(f"Single{w.capitalize()}{v.capitalize()}", meth)] = fmt(geo([j["c2"][v][key]["q"] for j in moved]))
            key = f"round-robin|{pe.LOWDIM_TPL}|{k}"
            m[f"Single{w.capitalize()}{v.capitalize()}Control"] = fmt(geo([j["c2"][v][key]["q"] for j in controls]))
        m[f"Single{w.capitalize()}Pairs"] = str(len(moved))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(pe.HERE / "results" / "runs"))
    ap.add_argument("--out", default=str(pe.HERE / "results" / "paper" / "final"))
    ap.add_argument("--orders", type=int, default=pe.ORDERS)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    tasks = []
    for w in pe.WORKLOADS:
        labels, split = pe.load_workload(w, runs)
        single = {mach: single_runs(runs, d) for mach, d in pe.WORKLOADS[w]["runs"].items() if mach in labels}
        machines = sorted(labels)
        for a, b in list(permutations(machines, 2)) + [(x, x) for x in machines]:
            tasks.append((w, a, b, labels, split, single[b], args.orders))
    print(f"{len(tasks)} jobs on {args.jobs} processes", flush=True)
    t0 = time.time()
    jobs = Parallel(n_jobs=args.jobs, verbose=5)(delayed(pair_job)(*t) for t in tasks)
    text = json.dumps({"seconds": round(time.time() - t0, 1), "jobs": jobs}, indent=1, default=float) + "\n"
    (out / "single_run.json.gz").write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    m = macros(jobs)
    lines = [GENERATED] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (out / "single_run.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in sorted(m.items()):
        print(k, v)


if __name__ == "__main__":
    main()
