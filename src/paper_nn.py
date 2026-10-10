"""Neural baseline of the paper: a plan model after DACE, moved between machines (H6).

paper/PLAN.md (section "Extension for the full paper", H6) compares the
template correction on the course study's source model with a neural plan
model fine-tuned on the same k target queries. This module develops that
model (src/nn_model.py) on the development pairs, every ordered pair of M1
to M6 in TPC-H and TPC-DS at scale factor 1, and evaluates it there. cv and
tune read nothing else: the confirmatory data of H6 (run directories *_sf10,
*_job and M7*) stayed untouched until the model was frozen (commit 52cc5b4),
so the table of development environments (DEV) is kept here and not taken
from paper_eval.WORKLOADS. run also takes the second round's workloads
(EXT: tpch10 and job, with paper_eval's machines and run directories) and
evaluates the frozen model on them; nothing of the model, its training or
its fine-tuning depends on the workload. Their plans come from the first
machine whose pass 1 saved them (a workload has the same plans on every
machine), and their k grids are paper_eval's (ks_of), with the budget kH of
H6 (paper_eval.budget: 100 on tpch10, 33 on job) among them.

Steps
  cv    The source model's hyperparameters, by grouped cross-validation
        inside the source pool of every development machine: CV_FOLDS folds
        of GroupKFold over the units of paper_eval (templates; TPC-DS
        families of near-copy templates), every setting of CV_GRID trained
        to the largest of CV_EPOCHS and scored after each of them. The score
        is the paper's error (per-template median Q-error, averaged in log
        over the held-out templates), averaged over the machines of a
        workload and then over the two workloads, so that one setting serves
        every workload. A best setting at the edge of the grid is followed
        by a run beyond that edge (--extension n runs CV_EXTENSIONS[n - 1];
        --include adds the earlier runs' cv.csv), and the best over all
        runs is written into nn_model.CONFIG by hand.
  tune  The fine-tuning rule. On every development pair, for the first
        TUNE_ORDERS arrival orders, random and round-robin selection and
        every k, each (mode, learning rate) of TUNE_LRS is run to the largest
        of TUNE_STEPS and scored after each of them on the target pool
        queries outside the k (the test set is not read). For every mode
        and k, the (lr, steps) with the lowest mean over pairs, orders and
        strategies, the two workloads weighted equally, is written into
        nn_model.FINE_TUNE by hand.
  run   The evaluation with the frozen settings.

The evaluation of one pair builds two adapt_eval.Pair objects on the same
queries, one with the course study's source model (paper_eval.fit_hybrid)
and one with the neural model trained on the source pool, and takes the k
target queries as paper_eval.run_c2 does (paper_eval.strategy_sequence on
the 20 arrival orders of adapt_eval.arrival_orders, the r-th sequence of
every strategy from the r-th order), so every number is paired with
paper_eval's. Methods:
  hybrid:none, hybrid:speed, hybrid:ridge, hybrid:lowdim+template
              paper_eval.predictions on the hybrid source model, recomputed;
              with --reference, run checks them against paper_eval's
              summary.json.gz
  nn:none, nn:speed, nn:ridge, nn:lowdim+template
              the same corrections on the neural source model's predictions
  nn:full-ft  every weight fine-tuned on the k queries
  nn:head-ft  only the output head fine-tuned on the k queries
  nn:oracle   the neural model trained on the whole target pool
Errors and intervals are paper_eval's: the median Q-error of every test
template, its log averaged over the orders and then over test templates,
reported as exp (the geometric mean over templates), with 95% intervals
from 2000 resamples of units (paper_eval.unit_weights). The H6 statistic is
the paired ratio of hybrid:lowdim+template to nn:full-ft (and to
nn:head-ft), both with round-robin selection on the same queries.

Outputs (in --out): summary.json (settings, runtimes, geometric means over
pairs, the H6 counts), pairs.json.gz (every pair's cells and ratios) and
per_template.csv for run; cv.csv and cv.json for cv; tune.csv and tune.json
for tune.

Usage
  python src/paper_nn.py cv --out /var/tmp/pgstudy/nn_dev/cv --jobs 16
  python src/paper_nn.py cv --extension 1 --include /var/tmp/pgstudy/nn_dev/cv/cv.csv \
      --out /var/tmp/pgstudy/nn_dev/cv_ext --jobs 16
  python src/paper_nn.py cv --extension 2 --include /var/tmp/pgstudy/nn_dev/cv/cv.csv \
      /var/tmp/pgstudy/nn_dev/cv_ext/cv.csv --out /var/tmp/pgstudy/nn_dev/cv_ext2 --jobs 16
  python src/paper_nn.py tune --out /var/tmp/pgstudy/nn_dev/tune --jobs 16
  python src/paper_nn.py run --out /var/tmp/pgstudy/nn_dev --jobs 16 \
      --reference results/paper/final/summary.json.gz
  python src/paper_nn.py run --workloads tpch10 job --out results/paper/nn_ext --jobs 16 \
      --reference results/paper/ext/summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict, replace
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import GroupKFold

import adapt_eval as ae
import nn_model as nn
import paper_eval as pe
import train_eval as te

HERE = ae.HERE
LABEL = ae.LABEL

MACHINES = ["M1", "M2", "M3", "M4", "M5", "M6"]
DEV = {
    "tpch": {"split": "queries/tpch_sf1_v2/split.csv", "plans": "M1/pass1/plans",
             "runs": {m: m for m in MACHINES}},
    "tpcds": {"split": "queries/tpcds_sf1_kept/split.csv", "plans": "M1_tpcds/pass1/plans",
              "runs": {m: f"{m}_tpcds" for m in MACHINES}},
}
CONFIRMATORY = ("_sf10", "_job", "M7")
# The second round's workloads, for run only (paper/PLAN.md, E1 and E2).
EXT = ["tpch10", "job"]
MAX_JOBS = 16     # the analysis server is shared

KS = [5, 10, 25, 50, 100]
STRATEGIES = ["random", "round-robin"]
ORDERS = pe.ORDERS
HYBRID_METHODS = [pe.NONE, pe.SPEED, pe.RIDGE, pe.LOWDIM_TPL]
NN_CORRECTIONS = [pe.NONE, pe.SPEED, pe.RIDGE, pe.LOWDIM_TPL, pe.ORACLE]
FT_MODES = ["full", "head"]
PRIMARY = ("round-robin", "hybrid:" + pe.LOWDIM_TPL)

CV_GRID = {"dropout": [0.0, 0.1, 0.2], "lr": [3e-4, 1e-3, 3e-3]}
CV_EPOCHS = [25, 50, 100, 200, 400]
# The best setting of CV_GRID lay at its largest learning rate and length, so the grid was extended
# there (cv --extension 1), and the best of that at its largest length (--extension 2); the choice is
# made over all runs together.
CV_EXTENSIONS = [({"dropout": [0.0, 0.1], "lr": [3e-3, 1e-2]}, [25, 50, 100, 200, 400, 800]),
                 ({"dropout": [0.0], "lr": [3e-3]}, [25, 50, 100, 200, 400, 800, 1600])]
CV_FOLDS = 5
TUNE_LRS = [3e-4, 1e-3, 3e-3, 1e-2]
TUNE_STEPS = [10, 25, 50, 100, 200, 400]
TUNE_ORDERS = 3


# ------------------------------------------------------------------- data

def load_dev(name: str, runs: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Labels of the development machines, the split and the parsed plans of one workload."""
    cfg = DEV[name]
    dirs = list(cfg["runs"].values()) + [cfg["plans"]]
    if any(c in d for d in dirs for c in CONFIRMATORY):
        raise ValueError(f"{name}: a confirmatory run directory is listed among {dirs}")
    split = ae.load_split(HERE / cfg["split"])
    labels = {m: ae.load_labels(runs / d / "labels.csv") for m, d in cfg["runs"].items()
              if (runs / d / "labels.csv").exists()}
    ae.check_keys(labels, split)
    plans = nn.load_plans(runs / cfg["plans"], split["query_id"].astype(str))
    return labels, split, plans


def plan_dir(name: str, runs: Path) -> Path:
    """pass1/plans of the first machine of a second-round workload that has it."""
    for d in pe.WORKLOADS[name]["runs"].values():
        if (runs / d / "pass1" / "plans").is_dir():
            return runs / d / "pass1" / "plans"
    raise ValueError(f"{name}: no machine of {list(pe.WORKLOADS[name]['runs'].values())} has pass1/plans in {runs}")


def load_workload(name: str, runs: Path) -> tuple[dict, pd.DataFrame, dict]:
    """load_dev for a development workload; for a second-round one, paper_eval's labels and split
    (machines without labels left out) and the plans of plan_dir."""
    if name in DEV:
        return load_dev(name, runs)
    labels, split = pe.load_workload(name, runs)
    return labels, split, nn.load_plans(plan_dir(name, runs), split["query_id"].astype(str))


def ks_of(workload: str) -> list[int]:
    """The k grid of run: paper_eval's up to the largest k of KS. That is KS on TPC-H, TPC-DS and
    tpch10 (200 exceeds tpch10's pool), and 5, 10, 25, 33 and 50 on JOB, whose kH is 33."""
    return [k for k in pe.ks_of(workload) if k <= max(KS)]


def source_rows(labels: dict, split: pd.DataFrame, machine: str) -> pd.DataFrame:
    """The source machine's training rows, taken as adapt_eval.Pair takes source_train."""
    s = labels[machine].set_index("query_id")
    role = split.set_index("query_id")["set"]
    ok = s.index[s["status"] == "ok"].intersection(role.index)
    return s.loc[np.sort(ok[(role.loc[ok] == "pool").to_numpy()].to_numpy())]


def unit_of_rows(rows: pd.DataFrame) -> np.ndarray:
    """paper_eval's unit of every row: its TPC-DS family where there is one, else its template."""
    units = rows["family"] if "family" in rows.columns else rows["template"]
    return units.astype(str).to_numpy()


def template_log_errors(y_ms: np.ndarray, pred_log: np.ndarray, templates: np.ndarray) -> pd.Series:
    """log of the median Q-error of every template, as adapt_eval.template_errors computes it."""
    q = te.qerror(y_ms, te.to_ms(pred_log))
    return np.log(pd.Series(q).groupby(np.asarray(templates).astype(str)).median())


def fit_source(workload: str, machine: str, rows: pd.DataFrame, plans: dict) -> tuple:
    t0 = time.time()
    model = nn.PlanModel(plans).fit(rows.index, np.log(rows[LABEL].to_numpy(float)), seed=pe.SOURCE_SEED)
    return (workload, machine), model, round(time.time() - t0, 1)


def fit_sources(data: dict, jobs: int) -> tuple[dict, dict]:
    """The neural source model of every (workload, machine), trained on its source pool."""
    tasks = [(w, m, source_rows(labels, split, m), plans) for w, (labels, split, plans) in data.items()
             for m in sorted(labels)]
    fitted = parallel(jobs)(delayed(fit_source)(*t) for t in tasks)
    return {key: model for key, model, _ in fitted}, {f"{w}|{m}": s for (w, m), _, s in fitted}


def parallel(jobs: int, verbose: int = 0) -> Parallel:
    return Parallel(n_jobs=min(jobs, MAX_JOBS), verbose=verbose)


# --------------------------------------------------------------------- cv

def settings(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def cv_job(workload: str, machine: str, rows: pd.DataFrame, plans: dict, setting: dict, fold: int,
           epochs: list[int]) -> list[dict]:
    """Scores of one setting on one fold of one source pool, after each of the given epochs."""
    nn.set_determinism()
    units = unit_of_rows(rows)
    train, held = list(GroupKFold(n_splits=CV_FOLDS).split(rows, groups=units))[fold]
    cfg = replace(nn.CONFIG, **setting, epochs=max(epochs))
    ids = rows.index.to_numpy()
    y_ms = rows[LABEL].to_numpy(float)
    tpl = rows["template"].astype(str).to_numpy()
    out = []

    def score(epoch, model):
        e = template_log_errors(y_ms[held], model.predict(ids[held]), tpl[held])
        out.extend({"workload": workload, "machine": machine, **setting, "fold": fold, "epochs": epoch,
                    "template": t, "log_q": float(v)} for t, v in e.items())
    nn.PlanModel(plans, cfg).fit(ids[train], np.log(y_ms[train]), seed=pe.SOURCE_SEED,
                                 checkpoints=epochs, callback=score)
    return out


def run_cv(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {w: load_dev(w, Path(args.runs)) for w in args.workloads}
    grid, epochs = CV_EXTENSIONS[args.extension - 1] if args.extension else (CV_GRID, CV_EPOCHS)
    tasks = [(w, m, source_rows(labels, split, m), plans, s, f, epochs)
             for w, (labels, split, plans) in data.items() for m in sorted(labels)
             for s in settings(grid) for f in range(CV_FOLDS)]
    print(f"{len(tasks)} cv jobs on {min(args.jobs, MAX_JOBS)} processes", flush=True)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=nn.THREADS):
        rows = parallel(args.jobs, verbose=5)(delayed(cv_job)(*t) for t in tasks)
    df = pd.DataFrame([r for part in rows for r in part])
    df.to_csv(out / "cv.csv", index=False)
    keys = list(CV_GRID) + ["epochs"]
    # Runs of the same setting agree on their common epochs (training is deterministic), so pooling
    # earlier runs only adds settings.
    df = pd.concat([df] + [pd.read_csv(f) for f in args.include], ignore_index=True)
    df = df.drop_duplicates(["workload", "machine"] + keys + ["fold", "template"])
    # Every template is held out once per machine: the mean over a machine's templates, then over
    # machines, then over workloads.
    per_machine = df.groupby(["workload", "machine"] + keys)["log_q"].mean().reset_index()
    per_workload = per_machine.groupby(["workload"] + keys)["log_q"].mean().unstack("workload")
    per_workload["both"] = per_workload.mean(axis=1)
    table = np.exp(per_workload).sort_values("both")
    best = dict(zip(keys, table.index[0]))
    result = {"best": {k: (int(v) if k == "epochs" else float(v)) for k, v in best.items()},
              "best_q": table.iloc[0].to_dict(), "grid": grid, "epochs": epochs, "folds": CV_FOLDS,
              "included": [ae.shown_path(Path(f)) for f in args.include],
              "fixed": asdict(nn.CONFIG), "seconds": round(time.time() - t0, 1),
              "table": [{**dict(zip(keys, idx)), **row.to_dict()} for idx, row in table.iterrows()]}
    (out / "cv.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(table.head(10).to_string(), flush=True)
    print(f"best {result['best']}; wrote {out} in {result['seconds']} s", flush=True)


# ------------------------------------------------------------------- tune

def tune_job(workload: str, a: str, b: str, labels: dict, split: pd.DataFrame, model: nn.PlanModel,
             mode: str, lr: float) -> list[dict]:
    """Validation errors of fine-tuning (mode, lr) on one pair, on the pool queries outside the k."""
    nn.set_determinism()
    p = ae.Pair(labels, split, a, b, fit=lambda d, seed: model)
    if list(p.source_train.index) != model.train_ids:
        raise ValueError(f"{p.name}: the neural source model was trained on other rows")
    orders = ae.arrival_orders(split, TUNE_ORDERS)
    ids = p.pool.index.to_numpy()
    y_ms = p.pool[LABEL].to_numpy(float)
    tpl = p.pool["template"].astype(str).to_numpy()
    rows = []
    for strategy in STRATEGIES:
        for r, order in enumerate(orders):
            seq = pe.strategy_sequence(strategy, p, p.positions(order), r, max(KS))
            for k in KS:
                sel = seq[:k]
                rest = np.setdiff1d(np.arange(p.n_pool), sel)
                key = {"workload": workload, "pair": p.name, "strategy": strategy, "order": r, "k": k}

                def score(step, m):
                    e = template_log_errors(y_ms[rest], m.predict(ids[rest]), tpl[rest])
                    rows.append({**key, "mode": mode, "lr": lr, "steps": step, "log_q": float(e.mean())})
                model.fine_tuned(ids[sel], p.y_pool[sel], mode, seed=r, lr=lr, steps=max(TUNE_STEPS),
                                 checkpoints=TUNE_STEPS, callback=score)
                if mode == FT_MODES[0] and lr == TUNE_LRS[0]:
                    # The unadapted source model and its speed factor on the same queries, for reference.
                    shift = p.e_pool[sel].mean()
                    for name, pred in (("none", p.base_pool[rest]), ("speed", p.base_pool[rest] + shift)):
                        e = template_log_errors(y_ms[rest], pred, tpl[rest])
                        rows.append({**key, "mode": name, "lr": 0.0, "steps": 0, "log_q": float(e.mean())})
    return rows


def run_tune(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {w: load_dev(w, Path(args.runs)) for w in args.workloads}
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=nn.THREADS):
        models, source_s = fit_sources(data, args.jobs)
        tasks = [(w, a, b, labels, split, models[(w, a)], mode, lr)
                 for w, (labels, split, _) in data.items() for a, b in permutations(sorted(labels), 2)
                 for mode in FT_MODES for lr in TUNE_LRS]
        print(f"{len(tasks)} tune jobs on {min(args.jobs, MAX_JOBS)} processes", flush=True)
        rows = parallel(args.jobs, verbose=5)(delayed(tune_job)(*t) for t in tasks)
    df = pd.DataFrame([r for part in rows for r in part])
    df.to_csv(out / "tune.csv", index=False)
    per_workload = df.groupby(["mode", "k", "lr", "steps", "workload"])["log_q"].mean().unstack("workload")
    both = per_workload.mean(axis=1)
    rule, table = {}, {}
    for (mode, k), g in both.groupby(level=["mode", "k"]):
        g = g.droplevel(["mode", "k"])
        table.setdefault(mode, {})[int(k)] = {f"{lr:g}|{s}": float(np.exp(v)) for (lr, s), v in g.items()}
        if mode in FT_MODES:
            lr, steps = g.idxmin()
            rule.setdefault(mode, {})[int(k)] = [float(lr), int(steps), float(np.exp(g.min()))]
    result = {"rule": rule, "lrs": TUNE_LRS, "steps": TUNE_STEPS, "orders": TUNE_ORDERS, "config": asdict(nn.CONFIG),
              "source_seconds": source_s, "seconds": round(time.time() - t0, 1),
              "per_workload": {f"{m}|{k}|{lr:g}|{s}": {w: float(np.exp(v)) for w, v in row.items()}
                               for (m, k, lr, s), row in per_workload.iterrows()},
              "table": table}
    (out / "tune.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(rule, indent=1), flush=True)
    print(f"wrote {out} in {result['seconds']} s", flush=True)


# -------------------------------------------------------------------- run

def pair_job(workload: str, a: str, b: str, labels: dict, split: pd.DataFrame, plans: dict,
             model: nn.PlanModel, ks: list[int], n_orders: int) -> dict:
    """C2 of one pair for the hybrid and the neural source model on the same selections."""
    t0 = time.time()
    nn.set_determinism()
    p_h = ae.Pair(labels, split, a, b, fit=pe.fit_hybrid)
    p_n = ae.Pair(labels, split, a, b, fit=lambda d, seed: model)
    if list(p_n.source_train.index) != model.train_ids:
        raise ValueError(f"{p_n.name}: the neural source model was trained on other rows")
    ks = [k for k in ks if k <= p_h.n_pool]      # as paper_eval.run_c2: no k above the pair's pool
    oracle = nn.PlanModel(plans, model.config).fit(p_n.pool.index, p_n.y_pool, seed=pe.SOURCE_SEED)(p_n.test)
    orders = ae.arrival_orders(split, n_orders)
    unit_of = pe.unit_map(p_h)
    W = pe.unit_weights(p_h.templates, unit_of)
    pool_ids = p_n.pool.index.to_numpy()
    logs = {}
    for strategy in STRATEGIES:
        for r, order in enumerate(orders):
            seq = pe.strategy_sequence(strategy, p_h, p_h.positions(order), r, max(ks))
            for k in ks:
                sel = seq[:k]
                preds = {f"hybrid:{m}": v for m, v in
                         pe.predictions(p_h, sel, HYBRID_METHODS, pe.fit_hybrid, r, None, {}).items()}
                preds.update({f"nn:{m}": v for m, v in
                              pe.predictions(p_n, sel, NN_CORRECTIONS, None, r, oracle, {}).items()})
                for mode in FT_MODES:
                    preds[f"nn:{mode}-ft"] = model.fine_tuned(pool_ids[sel], p_n.y_pool[sel], mode, seed=r)(p_n.test)
                for name, pred in preds.items():
                    _, per_t, _, _ = ae.template_errors(p_h.y_test_ms, pred, p_h.groups)
                    logs.setdefault((strategy, name, k), np.empty((len(orders), len(p_h.templates))))[r] = per_t
    L = {key: v.mean(axis=0) for key, v in logs.items()}
    cells = {}
    for (strategy, name, k), v in L.items():
        cell = pe.error_cell(v, W)
        for ref_name, ref in (("vs_hybrid_speed", ("random", "hybrid:" + pe.SPEED, k)), ("vs_primary", (*PRIMARY, k))):
            if ref != (strategy, name, k):
                cell[ref_name] = pe.paired_ratio(v, L[ref], W)
        cells[f"{strategy}|{name}|{k}"] = cell
    h6 = {str(k): {mode: pe.paired_ratio(L[(*PRIMARY, k)], L[("round-robin", f"nn:{mode}-ft", k)], W)
                   for mode in FT_MODES} for k in ks}
    frame = pd.DataFrame([{"workload": workload, "pair": p_h.name, "strategy": s, "method": m, "k": k,
                           "template": str(t), "log_q": float(x)}
                          for (s, m, k), v in L.items() for t, x in zip(p_h.templates, v)])
    return {"workload": workload, "pair": p_h.name, "a": a, "b": b, "pool": p_h.n_pool, "test": p_h.n_test,
            "templates": [str(t) for t in p_h.templates],
            "units": len(set(unit_of[str(t)] for t in p_h.templates)),
            "cells": cells, "h6": h6, "seconds": round(time.time() - t0, 1), "frame": frame}


def over_pairs(results: list[dict]) -> dict:
    """Geometric mean over the pairs of a workload of every cell's error (over the pairs that have
    the cell: a k above a pair's pool is left out)."""
    out = {}
    for w in dict.fromkeys(r["workload"] for r in results):
        rs = [r for r in results if r["workload"] == w]
        for key in dict.fromkeys(key for r in rs for key in r["cells"]):
            strategy, m, k = key.split("|")
            q = np.exp(np.mean([np.log(r["cells"][key]["q"]) for r in rs if key in r["cells"]]))
            out.setdefault(w, {}).setdefault(strategy, {}).setdefault(m, {})[k] = float(q)
    return out


def h6_counts(results: list[dict]) -> dict:
    """Per workload, k and fine-tuning mode: pairs whose ratio's interval lies below 1, and the
    geometric mean of the ratio over pairs."""
    out = {}
    for w in dict.fromkeys(r["workload"] for r in results):
        rs = [r for r in results if r["workload"] == w]
        for k in dict.fromkeys(k for r in rs for k in r["h6"]):
            for mode in FT_MODES:
                cells = [r["h6"][k][mode] for r in rs if k in r["h6"]]
                out.setdefault(w, {}).setdefault(k, {})[mode] = {
                    "below_1": int(sum(c["success"] for c in cells)), "pairs": len(cells),
                    "point_below_1": int(sum(c["gm"] < 1 for c in cells)),
                    "gm": float(np.exp(np.mean([np.log(c["gm"]) for c in cells]))),
                    "max_hi": float(max(c["gm_hi"] for c in cells))}
    return out


def check_reference(results: list[dict], path: Path) -> dict:
    """The recomputed hybrid cells against paper_eval's summary: the largest difference in log, and
    paper_eval's own speed factor (random) and primary method (round robin) over the same pairs.
    The summary may be gzipped (as committed) or not (as paper_eval writes it)."""
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as f:
        ref = json.load(f)
    c2 = {(j["workload"], j["pair"]): j["c2"] for j in ref["jobs"] if j["source"] == "hybrid"}
    diffs, missing = [], 0
    for r in results:
        cells = c2.get((r["workload"], r["pair"]), {})
        for key, cell in r["cells"].items():
            strategy, m, k = key.split("|")
            if not m.startswith("hybrid:"):
                continue
            other = cells.get(f"{strategy}|{m.removeprefix('hybrid:')}|{k}")
            if other is None:
                missing += 1
            else:
                diffs.append(abs(np.log(cell["q"]) - np.log(other["q"])))
    paper = {}
    for w in dict.fromkeys(r["workload"] for r in results):
        pairs = [c2[(w, r["pair"])] for r in results if r["workload"] == w and (w, r["pair"]) in c2]
        for key in ("random|" + pe.SPEED, "round-robin|" + pe.LOWDIM_TPL):
            paper.setdefault(w, {})[key] = {str(k): float(np.exp(np.mean([np.log(c[f"{key}|{k}"]["q"]) for c in pairs
                                                                          if f"{key}|{k}" in c])))
                                            for k in ks_of(w)}
            paper[w]["pairs"] = len(pairs)
    return {"reference": ae.shown_path(path), "compared": len(diffs), "missing": missing,
            "max_abs_log_diff": float(max(diffs)) if diffs else None, "paper_eval_gm_over_pairs": paper}


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {w: load_workload(w, Path(args.runs)) for w in args.workloads}
    for w in data:
        if pe.budget(w) not in ks_of(w):
            raise ValueError(f"{w}: kH = {pe.budget(w)} is not in the k grid {ks_of(w)}")
    # The second round's workloads are listed with their grids and kH; a development run's summary
    # stays as it was.
    own = {w: {"ks": ks_of(w), "kh": pe.budget(w)} for w in args.workloads if w not in DEV}
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=nn.THREADS):
        models, source_s = fit_sources(data, args.jobs)
        t_src = time.time() - t0
        # TPC-DS first: its pairs take longest.
        tasks = [(w, a, b, *data[w], models[(w, a)], ks_of(w), args.orders)
                 for w in sorted(data, key=lambda x: x != "tpcds") for a, b in permutations(sorted(data[w][0]), 2)]
        print(f"{len(tasks)} pair jobs on {min(args.jobs, MAX_JOBS)} processes", flush=True)
        results = parallel(args.jobs, verbose=5)(delayed(pair_job)(*t) for t in tasks)
    frames = [r.pop("frame") for r in results]
    pd.concat(frames, ignore_index=True).to_csv(out / "per_template.csv", index=False)
    with gzip.open(out / "pairs.json.gz", "wt", encoding="utf-8") as f:
        json.dump(results, f, default=float)
    first = next(iter(models.values()))
    summary = {"config": {**first.describe(), "fine_tune": nn.FINE_TUNE, "ks": KS, "orders": args.orders,
                          "strategies": STRATEGIES, "workloads": args.workloads, "threads": nn.THREADS,
                          **({"by_workload": own} if own else {})},
               "seconds": round(time.time() - t0, 1), "source_seconds_total": round(t_src, 1),
               "source_seconds": source_s, "pair_seconds": {f"{r['workload']}|{r['pair']}": r["seconds"] for r in results},
               "parameters": {f"{w}|{m}": model.n_parameters() for (w, m), model in models.items()},
               "gm_over_pairs": over_pairs(results), "h6": h6_counts(results)}
    if args.reference:
        summary["reference_check"] = check_reference(results, Path(args.reference))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {out} in {summary['seconds']} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cv", "tune", "run"):
        s = sub.add_parser(name)
        s.add_argument("--runs", default=str(HERE / "results" / "runs"))
        s.add_argument("--out", required=True)
        # The default stays the development workloads; only run takes the second round's.
        s.add_argument("--workloads", nargs="+", default=list(DEV),
                       choices=list(DEV) + (EXT if name == "run" else []))
        s.add_argument("--jobs", type=int, default=MAX_JOBS)
        if name == "cv":
            s.add_argument("--extension", type=int, default=0, choices=range(len(CV_EXTENSIONS) + 1),
                           help="run CV_EXTENSIONS[n - 1] instead of CV_GRID")
            s.add_argument("--include", nargs="*", default=[], help="cv.csv files of earlier runs to choose among too")
        if name == "run":
            s.add_argument("--orders", type=int, default=ORDERS)
            s.add_argument("--reference", default=None, help="paper_eval's summary.json.gz to check the hybrid cells")
    args = ap.parse_args()
    nn.set_determinism()
    {"cv": run_cv, "tune": run_tune, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
