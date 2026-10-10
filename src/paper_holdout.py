"""Do the k queries need to lie outside the source model's training set? (exploratory)

Added on 8 October 2026 after an internal review; not a registered test.

In the main protocol (src/paper_eval.py) the source model is trained on A's
whole pool and the k queries are drawn from B's pool, so the model has seen
every measured query on A, and its error on them is an in-sample residual.
Here each template's pool queries of a pair are split in two halves with a
fixed seed (the first half gets the odd query). The source model, the
forest model of paper_eval (fit_hybrid), is trained on A's first half, and
the k queries are drawn by round robin, over the same 20 random orders,
either from the first half (in-sample, as in the main protocol) or from the
second (out-of-sample). Both arms use only templates with a query in each
half, so they cover the same templates, and both are scored on B's whole
test set. k is that number of templates, one query each, and 10.

The no-move control repeats this with B as its own source (the forest model
trained on B's first half), so the cost of the move can be read under both
arms.

Writes into --out (default results/paper/extra):
  holdout.json          per workload and pair: k, and the error (geometric
                        mean over test templates of the median Q-error,
                        averaged in log over the orders) of no adaptation,
                        the speed factor and the primary method in each arm
  holdout_numbers.tex   geometric means over the 30 moved pairs of each
                        workload: Xho<W><Arm><Method> with Arm In or Out and
                        Method Speed, Primary or Control (the primary method
                        on the no-move control); Xho<W>K, the budget; and
                        Xho<W>OutLower, the pairs where the primary method
                        errs less out-of-sample than in-sample

Usage: python src/paper_holdout.py [--runs results/runs] [--out results/paper/extra] [--jobs 16]
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import adapt_eval as ae
import paper_eval as pe

HERE = Path(__file__).resolve().parents[1]
WORKLOAD = {"tpch": "Tpch", "tpcds": "Tpcds"}
HALF_SEED = 6000
METHODS = [pe.NONE, pe.SPEED, pe.LOWDIM_TPL]


class HalfView:
    """A Pair whose source model was trained on part of the pool, adapting from a subset of the pool."""

    def __init__(self, p: ae.Pair, f_half, keep: np.ndarray, source_train):
        self.pool = p.pool[keep]
        self.source_pool = p.source_pool[keep]
        self.n_pool = int(keep.sum())
        self.test = p.test
        self.source_train = source_train
        self.X_pool, self.X_test = p.X_pool[keep], p.X_test
        self.base_pool, self.base_test = f_half(self.pool), f_half(self.test)
        self.e_pool = p.y_pool[keep] - self.base_pool
        self.y_test_ms = p.y_test_ms
        self.groups, self.templates = p.groups, p.templates
        self._position = {q: i for i, q in enumerate(np.asarray(p.pool_ids)[keep])}

    def positions(self, order: np.ndarray) -> np.ndarray:
        return np.array([self._position[q] for q in order if q in self._position], dtype=int)


def halves(p: ae.Pair) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """first, second: the pool split in halves inside each template; both: templates in both halves."""
    tpl = p.pool["template"].astype(str).to_numpy()
    rng = np.random.default_rng(HALF_SEED)
    first = np.zeros(p.n_pool, dtype=bool)
    for t in sorted(set(tpl)):
        idx = np.flatnonzero(tpl == t)
        idx = idx[rng.permutation(len(idx))]
        first[idx[: (len(idx) + 1) // 2]] = True
    second = ~first
    both_t = set(tpl[first]) & set(tpl[second])
    in_both = np.isin(tpl, list(both_t))
    return first, second, in_both


def arms(p: ae.Pair, orders: list[np.ndarray]) -> dict:
    """Errors of METHODS with round robin in both arms, at k = 10 and k = templates in both halves."""
    first, second, in_both = halves(p)
    train_ids = set(np.asarray(p.pool_ids)[first])
    source_train = p.source_train[p.source_train.index.isin(train_ids)]
    f_half = pe.fit_hybrid(source_train, ae.SOURCE_SEED)
    k_all = int(len(set(p.pool["template"].astype(str).to_numpy()[in_both])))
    out = {"k": k_all}
    for arm, keep in (("in", first & in_both), ("out", second)):
        view = HalfView(p, f_half, keep, source_train)
        logs, _ = pe.run_c2(view, pe.fit_hybrid, orders, sorted({10, k_all}), {"round-robin": METHODS}, {})
        for (strategy, m, k), arr in logs.items():
            # Geometric mean over test templates of the median Q-error, averaged in log over the orders.
            out[f"{arm}|{m}|{k}"] = float(np.exp(arr.mean()))
    return out


def job(w: str, a: str, b: str, labels: dict, split) -> dict:
    p = ae.Pair(labels, split, a, b, fit=pe.fit_hybrid)
    orders = ae.arrival_orders(split, pe.ORDERS)
    return {"workload": w, "pair": ae.pair_name(a, b), "a": a, "b": b} | arms(p, orders)


def gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=HERE / "results" / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "paper" / "extra")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for w in WORKLOAD:
        labels, split = pe.load_workload(w, a.runs)
        machines = sorted(labels)
        jobs += [(w, x, y, labels, split) for x, y in permutations(machines, 2)]
        jobs += [(w, x, x, labels, split) for x in machines]
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(job)(*j) for j in jobs)
    (a.out / "holdout.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")

    m = {}
    for w, W in WORKLOAD.items():
        moved = [r for r in rows if r["workload"] == w and r["a"] != r["b"]]
        control = {r["b"]: r for r in rows if r["workload"] == w and r["a"] == r["b"]}
        ks = {r["k"] for r in moved}
        if len(ks) != 1:
            raise SystemExit(f"{w}: the budget differs between pairs: {sorted(ks)}")
        k = ks.pop()
        m[f"Xho{W}K"] = k
        for arm, A in (("in", "In"), ("out", "Out")):
            m[f"Xho{W}{A}Speed"] = f"{gm([r[f'{arm}|speed|{k}'] for r in moved]):.2f}"
            m[f"Xho{W}{A}Primary"] = f"{gm([r[f'{arm}|lowdim+template|{k}'] for r in moved]):.2f}"
            # The control at its own budget (its pool can differ from a pair's by queries that failed on A).
            m[f"Xho{W}{A}Control"] = f"{gm([control[r['b']][f'{arm}|lowdim+template|' + str(control[r['b']]['k'])] for r in moved]):.2f}"
        m[f"Xho{W}OutLower"] = sum(r[f"out|lowdim+template|{k}"] < r[f"in|lowdim+template|{k}"] for r in moved)
        m[f"Xho{W}Pairs"] = len(moved)
    (a.out / "holdout_numbers.tex").write_text(
        "% Generated by src/paper_holdout.py. Do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8", newline="\n")
    for k, v in m.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
