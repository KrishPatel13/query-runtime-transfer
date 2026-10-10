"""Paper evaluation: adapting a query runtime model to a new machine.

The course study (src/adapt_eval.py) measured how a runtime model moves from
one machine to another on TPC-H. This module extends its C2 (adaptation with
k target queries), its harder test (templates absent from the k queries) and
its C3 stream to

  1. two workloads (TPC-H and TPC-DS) on every machine, and in a second round TPC-H at
     scale factor 10 and the Join Order Benchmark (WORKLOADS),
  2. several source models,
  3. more adaptation methods,
  4. strategies for choosing which k target queries to measure,
  5. a description of the shift itself, and
  6. transfer of a learned shift between pairs and workloads.

It reuses adapt_eval.Pair (the same query selection, plan check and split),
its error measure (the median Q-error of each test template, averaged in log
over the test templates and reported as exp, the geometric mean over
templates) and stats' template-resampling intervals.

Methods (all predict log runtime; e = log y_target - f_s(x) on the k queries)
  none, speed, ridge, target, oracle
                  as in adapt_eval (target and oracle refit the source
                  model's own pipeline on target queries)
  two-factor      one factor for plans with parallel workers and one for the
                  rest: the mean e of each class shrunk toward the mean of all
                  k by empirical Bayes (pooled_offsets)
  template        the speed factor plus a shrunken offset for every template
                  among the k queries (pooled_offsets); other templates get
                  the speed factor alone
  ridge+template  the ridge correction plus shrunken template offsets of its
                  residuals
  forest-resid    a random forest fitted to e: a nonlinear correction
  pooled          the source pipeline refitted on the source pool and the k
                  target queries together, the machine as a feature of both
                  the cost line and the forest
  transfer-*      a ridge model of the shift log(t_B / t_A), learned on the
                  pool of another pair or workload, plus the mean remaining
                  error of the k queries; transfer-*+ridge also fits the
                  ridge correction to what remains (a correction regularized
                  toward the transferred shift instead of toward zero)
Selection of the k target queries (plan features only, before measuring)
  random          the course study's 20 seeded arrival orders
  round-robin     one query of each template in turn; templates and the
                  queries inside them in a seeded order
  k-center        farthest first in the standardized feature space, starting
                  from the first query of the random order
  d-optimal       each next query maximizes x' M^-1 x, M = I + sum of x x'
                  over the chosen queries, x standardized with an intercept:
                  greedy Bayesian D-optimal design for the linear correction,
                  starting from the first query of the random order
  plan-class      alternately a plan with and without parallel workers, each
                  class in the random order
Every strategy derives its r-th sequence from the r-th random order, so all
strategies and methods are compared on paired sequences.

Usage
  python src/paper_eval.py run --out results/paper/main --jobs 16
  python src/paper_eval.py run --workloads tpch --sources hybrid --out results/paper/dev --jobs 8
  python src/paper_eval.py run --workloads tpch10 job --out results/paper/ext --jobs 16
The default workloads are the first round's (tpch and tpcds); the second round's are named.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import adapt_eval as ae
import stats
import train_eval as te

HERE = ae.HERE
LABEL = ae.LABEL

# What changes between two machines: M2 and M4 are M1 and M3 with one processor, M5 is M1 with
# PostgreSQL's default buffer pool (128MB instead of 2GB). Pairs of M5 with M2 to M4 change two
# things at once ("mixed").
CHANGE = {frozenset({"M1", "M2"}): "processors", frozenset({"M3", "M4"}): "processors",
          frozenset({"M1", "M3"}): "hardware", frozenset({"M2", "M4"}): "hardware",
          frozenset({"M1", "M4"}): "both", frozenset({"M2", "M3"}): "both",
          frozenset({"M1", "M5"}): "memory",
          frozenset({"M3", "M6"}): "processors (2)"}   # M6: M3 pinned to two processors (exploratory)
# The other pairs of M6, named in the second round's workloads only: the first round calls them
# "mixed", and its labels stay so that results/paper/final is reproduced.
CHANGE_EXT = {frozenset({"M4", "M6"}): "processors (1 vs 2)",     # one against two processors
              frozenset({"M1", "M6"}): "both (2)",                # laptop against the server on two
              frozenset({"M2", "M6"}): "both (1 vs 2)"}           # laptop on one against the server on two
# Machine names per workload: the paper's machine M1 to M6 and the run directory of its labels; the
# workload whose shifts the transfer methods learn from ("other"); the extra change labels; the grid of
# budgets k (KS unless a workload has its own) and the registered budget kH of H2, H5 and H6 (100
# unless set). tpch10 and job are the second round of confirmatory tests (paper/PLAN.md, "Extension
# for the full paper"): E1, TPC-H at scale factor 10, and E2, the Join Order Benchmark. JOB's pool
# has 78 queries in 33 templates, so its grid stops at 50 (run_c2 would drop 100 and 200 anyway) and
# holds kH = 33, one pool query per template.
WORKLOADS = {
    "tpch": {"split": "queries/tpch_sf1_v2/split.csv",
             "runs": {"M1": "M1", "M2": "M2", "M3": "M3", "M4": "M4", "M5": "M5", "M6": "M6"},
             "other": "tpcds"},
    "tpcds": {"split": "queries/tpcds_sf1_kept/split.csv",
              "runs": {"M1": "M1_tpcds", "M2": "M2_tpcds", "M3": "M3_tpcds", "M4": "M4_tpcds",
                       "M5": "M5_tpcds", "M6": "M6_tpcds"},
              "other": "tpch"},
    # The second round learns its (exploratory) transfers from TPC-DS at scale factor 1. Not from TPC-H
    # at scale factor 1: it has the same templates as tpch10, and run_hard keeps the other workload's
    # shift models whole, so the held-out template's shift would be seen there.
    "tpch10": {"split": "queries/tpch_sf10_sub/split.csv",
               "runs": {"M3": "M3_sf10", "M4": "M4_sf10", "M6": "M6_sf10"},
               "other": "tpcds", "change": CHANGE_EXT, "kh": 100},
    "job": {"split": "queries/job_kept/split.csv",
            "runs": {"M1": "M1_job", "M2": "M2_job", "M3": "M3_job", "M4": "M4_job", "M6": "M6_job"},
            "other": "tpcds", "change": CHANGE_EXT, "ks": [5, 10, 25, 33, 50], "kh": 33},
}
# The default of run: the first round, so that the README's command reproduces results/paper/final.
DEFAULT_WORKLOADS = ["tpch", "tpcds"]
# The pair whose shift is the same kind of change on other hardware.
ANALOG = {("M1", "M2"): ("M3", "M4"), ("M3", "M4"): ("M1", "M2"),
          ("M2", "M1"): ("M4", "M3"), ("M4", "M3"): ("M2", "M1"),
          ("M1", "M3"): ("M2", "M4"), ("M2", "M4"): ("M1", "M3"),
          ("M3", "M1"): ("M4", "M2"), ("M4", "M2"): ("M3", "M1")}

KS = [5, 10, 25, 50, 100, 200]
ORDERS = 20
HARD_K = 25
HARD_ORDERS = 5
SOURCE_SEED = ae.SOURCE_SEED

NONE, SPEED, TWO, TPL, TPL_RAW = "none", "speed", "two-factor", "template", "template-raw"
RIDGE, RIDGE_TPL, FOREST_RESID, POOLED, KNN = "ridge", "ridge+template", "forest-resid", "pooled", "1-nn"
RIDGE_TPL_RAW = "ridge+template-raw"
AFFINE, AFFINE_PAR = "affine", "affine+parallel"
LOWDIM_TPL = "lowdim+template"   # the primary method fixed before the TPC-DS target labels (paper/PLAN.md)
LOOKUP = "lookup"   # no model: the mean measured log runtime of the template, else source model + speed factor
TARGET, ORACLE = "target", "oracle"
# Shift transfer from: the analog pair of this workload, this pair of the other workload,
# and the analog pair of the other workload.
TRANSFER_FROM = ["analog", "workload", "analog-workload"]
STRATEGIES = ["random", "round-robin", "rr-cheap", "k-center", "d-optimal", "plan-class"]
CHEAP = [NONE, SPEED, TWO, AFFINE, AFFINE_PAR, LOWDIM_TPL, TPL, TPL_RAW, KNN, LOOKUP, RIDGE, RIDGE_TPL, RIDGE_TPL_RAW,
         ORACLE]
COSTLY = [FOREST_RESID, POOLED, TARGET]
# Exploratory, added on 8 October 2026 after an internal review and run only with --extra: the forest on
# the errors given one indicator per template, a refit that knows the templates as the primary method does.
FOREST_RESID_TPL = "forest-resid+template"
EXTRA = [FOREST_RESID_TPL]


def ks_of(workload: str) -> list[int]:
    """The budgets k of a workload's C2. run_c2 leaves out every k above a pair's pool."""
    return list(WORKLOADS[workload].get("ks", KS))


def budget(workload: str) -> int:
    """kH, the registered budget of H2, H5 and H6: 100, and for JOB one query per template."""
    return int(WORKLOADS[workload].get("kh", 100))


def change_of(workload: str, a: str, b: str) -> str:
    """What changes from machine a to b in a workload; 'none' for the no-move control."""
    if a == b:
        return "none"
    return {**CHANGE, **WORKLOADS[workload].get("change", {})}.get(frozenset({a, b}), "mixed")


# ------------------------------------------------------------- source models

def _log_label(df: pd.DataFrame) -> np.ndarray:
    return np.log(df[LABEL].to_numpy(float))


def fit_hybrid(df: pd.DataFrame, seed: int):
    """The course study's source model: cost line plus a forest on its residuals."""
    return ae.fit_source(df, seed)


def fit_forest(df: pd.DataFrame, seed: int):
    """train_eval's random forest on the 19 plan features (cost and row features on log scale)."""
    m = ae.make_forest(seed).fit(te.design(df, te.HYBRID), _log_label(df))
    return lambda d: m.predict(te.design(d, te.HYBRID))


def fit_boosting(df: pd.DataFrame, seed: int):
    """Gradient-boosted trees on the same features (fixed settings, nothing tuned)."""
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=5,
                                      l2_regularization=1.0, early_stopping=False,
                                      random_state=seed).fit(te.design(df, te.HYBRID), _log_label(df))
    return lambda d: m.predict(te.design(d, te.HYBRID))


def fit_ridge(df: pd.DataFrame, seed: int):
    """train_eval's Ridge: standardized features, penalty by leave-one-out."""
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=te.ALPHAS)).fit(te.design(df, "Ridge"), _log_label(df))
    return lambda d: m.predict(te.design(d, "Ridge"))


def fit_cost(df: pd.DataFrame, seed: int):
    """The cost line: least squares on log(1 + total cost), PostgreSQL's estimate recalibrated."""
    m = LinearRegression().fit(te.design(df, "Cost-only"), _log_label(df))
    return lambda d: m.predict(te.design(d, "Cost-only"))


SOURCES = {"hybrid": fit_hybrid, "forest": fit_forest, "boosting": fit_boosting, "ridge": fit_ridge,
           "cost": fit_cost}


# ---------------------------------------------------------------- adaptation

def pooled_offsets(values: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    """Empirical-Bayes partial pooling of group means.

    Returns the mean of all values and, for every group, its shrunken offset
    from that mean. With group means m_g over n_g values, the within-group
    variance s2 (pooled over the groups with at least two values; the
    variance of all values when no group has two) and the between-group
    variance tau2 = max(0, var(m_g) - mean(s2 / n_g)) (method of moments),
    a group's offset is (m_g - mean) * tau2 / (tau2 + s2 / n_g). One group,
    or no spread between groups, gives zero offsets: the speed factor.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups).astype(str)
    grand = float(v.mean())
    labels, inv = np.unique(g, return_inverse=True)
    if len(labels) < 2:
        return grand, {labels[0]: 0.0} if len(labels) else {}
    n = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=v) / n
    df_within = float((n - 1).sum())
    if df_within > 0:
        s2 = float(((v - m[inv]) ** 2).sum() / df_within)
    else:
        s2 = float(v.var(ddof=1)) if len(v) > 1 else 0.0
    tau2 = max(0.0, float(np.var(m, ddof=1)) - float(np.mean(s2 / n)))
    if tau2 <= 0.0:
        return grand, {lab: 0.0 for lab in labels}
    w = tau2 / (tau2 + s2 / n)
    return grand, {lab: float(w[i] * (m[i] - grand)) for i, lab in enumerate(labels)}


def apply_offsets(grand: float, offsets: dict, groups: np.ndarray) -> np.ndarray:
    return grand + np.array([offsets.get(str(x), 0.0) for x in groups], dtype=float)


def fit_shift_model(X: np.ndarray, r: np.ndarray, groups: np.ndarray | None = None):
    """Ridge model of the shift r = log(t_B / t_A) on standardized plan features.

    The model is meant for queries of other templates, so the penalty is
    chosen by leaving out whole units (templates, or TPC-DS families of
    near-copy templates) in 5 folds. Leaving out single queries, as RidgeCV
    does, picks almost no penalty and fits structure inside templates that
    does not carry over to new ones. Without groups, RidgeCV is used.
    """
    if groups is None or len(np.unique(groups)) < 3:
        return make_pipeline(StandardScaler(), RidgeCV(alphas=te.ALPHAS)).fit(X, r)
    folds = min(5, len(np.unique(groups)))
    search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": te.ALPHAS},
                          cv=GroupKFold(n_splits=folds), scoring="neg_mean_squared_error")
    return search.fit(X, r, groups=groups).best_estimator_


def fit_pooled(source_train: pd.DataFrame, target_rows: pd.DataFrame, seed: int):
    """The hybrid pipeline refitted on source and target rows, the machine as a feature.

    The cost line gets the machine indicator as a second regressor, so a
    plain speed factor is a shift of the line, and the forest sees it as a
    feature, so it can learn plan-dependent differences.
    """
    df = pd.concat([source_train, target_rows])
    flag = np.r_[np.zeros(len(source_train)), np.ones(len(target_rows))][:, None]
    y = _log_label(df)
    lin_x = np.hstack([te.design(df, "Cost-only"), flag])
    line = LinearRegression().fit(lin_x, y)
    rf = ae.make_forest(seed).fit(np.hstack([te.design(df, te.HYBRID), flag]), y - line.predict(lin_x))

    def predict(d: pd.DataFrame) -> np.ndarray:
        one = np.ones((len(d), 1))
        return (line.predict(np.hstack([te.design(d, "Cost-only"), one]))
                + rf.predict(np.hstack([te.design(d, te.HYBRID), one])))
    return predict


# ----------------------------------------------------------------- selection

def strategy_sequence(name: str, p: ae.Pair, order_positions: np.ndarray, r: int, n: int) -> np.ndarray:
    """The first n pool positions chosen by a strategy, derived from the r-th random order."""
    n = min(n, p.n_pool)
    if name == "random":
        return order_positions[:n]
    if name in ("round-robin", "rr-cheap"):
        rng = np.random.default_rng(5000 + r)
        tpl = p.pool["template"].astype(str).to_numpy()
        names = list(pd.unique(tpl[order_positions]))      # templates in the order they first arrive
        rng.shuffle(names)
        queues = {t: [i for i in order_positions if tpl[i] == t] for t in names}
        if name == "rr-cheap":
            # Inside each template the query that ran fastest on the source machine goes first:
            # its runtime is known before anything runs on the target, and a cheap query costs
            # little time to measure there.
            src_ms = p.source_pool[LABEL].to_numpy(float)
            queues = {t: sorted(q, key=lambda i: (src_ms[i], i)) for t, q in queues.items()}
        out = []
        while len(out) < n and any(queues.values()):
            for t in names:
                if queues[t]:
                    out.append(queues[t].pop(0))
                    if len(out) == n:
                        break
        return np.array(out, dtype=int)
    if name == "plan-class":
        par = p.pool["workers_planned"].to_numpy(float) > 0
        first = par[order_positions[0]]
        queues = {c: [i for i in order_positions if par[i] == c] for c in (True, False)}
        out, c = [], first
        while len(out) < n and (queues[True] or queues[False]):
            if queues[c]:
                out.append(queues[c].pop(0))
            elif queues[not c]:
                out.append(queues[not c].pop(0))
            c = not c
        return np.array(out, dtype=int)
    Z = StandardScaler().fit_transform(p.X_pool)
    if name == "k-center":
        chosen = [int(order_positions[0])]
        dist = np.linalg.norm(Z - Z[chosen[0]], axis=1)
        rank = np.empty(p.n_pool, dtype=int)
        rank[order_positions] = np.arange(len(order_positions))
        while len(chosen) < n:
            dist[chosen] = -np.inf
            best = np.flatnonzero(dist == dist.max())
            nxt = int(best[np.argmin(rank[best])])          # ties go to the earlier query of the random order
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(Z - Z[nxt], axis=1))
        return np.array(chosen, dtype=int)
    if name == "d-optimal":
        A = np.hstack([np.ones((p.n_pool, 1)), Z])
        Minv = np.eye(A.shape[1])
        rank = np.empty(p.n_pool, dtype=int)
        rank[order_positions] = np.arange(len(order_positions))
        chosen = [int(order_positions[0])]
        taken = np.zeros(p.n_pool, dtype=bool)
        while True:
            x = A[chosen[-1]]
            taken[chosen[-1]] = True
            Mx = Minv @ x
            Minv -= np.outer(Mx, Mx) / (1.0 + x @ Mx)       # Sherman-Morrison
            if len(chosen) == n:
                break
            lev = np.einsum("ij,jk,ik->i", A, Minv, A)
            lev[taken] = -np.inf
            best = np.flatnonzero(lev >= lev.max() - 1e-12)
            chosen.append(int(best[np.argmin(rank[best])]))
        return np.array(chosen, dtype=int)
    raise ValueError(f"unknown strategy {name}")


# -------------------------------------------------------------- one pair, C2

def predictions(p: ae.Pair, sel: np.ndarray, methods: list[str], fit, seed: int, oracle: np.ndarray,
                shift_models: dict) -> dict:
    """Test-set predictions (log ms) of every method from the k target queries at positions sel."""
    e = p.e_pool[sel]
    out = {}
    need_ridge = any(m in methods for m in (RIDGE, RIDGE_TPL, RIDGE_TPL_RAW))
    ridge = ae.fit_correction(p.X_pool[sel], e) if need_ridge else None
    for m in methods:
        if m == NONE:
            out[m] = p.base_test
        elif m == SPEED:
            out[m] = p.base_test + e.mean()
        elif m == TWO:
            par_sel = p.pool["workers_planned"].to_numpy(float)[sel] > 0
            grand, off = pooled_offsets(e, par_sel)
            par_test = p.test["workers_planned"].to_numpy(float) > 0
            out[m] = p.base_test + apply_offsets(grand, off, par_test)
        elif m in (AFFINE, AFFINE_PAR, LOWDIM_TPL):
            # The log error as a line in the source model's own log prediction (a power law in
            # runtime), with a separate line for plans with parallel workers in affine+parallel;
            # ridge with an unpenalized intercept, so a large penalty gives back the speed factor.
            # The parallel terms enter only when the k queries contain both plan classes: with one
            # class the indicator is constant and par * base duplicates base, and the fit would make
            # up a line for the other class. A separate slope needs at least two queries per class.
            par_sel = (p.pool["workers_planned"].to_numpy(float)[sel] > 0).astype(float)
            par_test = (p.test["workers_planned"].to_numpy(float) > 0).astype(float)
            fewest = int(min(par_sel.sum(), len(sel) - par_sel.sum()))

            def design(base, par):
                if m == AFFINE or fewest == 0:
                    return base[:, None]
                if fewest == 1:
                    return np.c_[base, par]
                return np.c_[base, par, par * base]
            line = ae.fit_correction(design(p.base_pool[sel], par_sel), e)
            low_test = line.predict(design(p.base_test, par_test))
            if m == LOWDIM_TPL:
                # lowdim+template: affine+parallel, plus for every template among the k the mean of
                # its queries' residuals from that line (unshrunk); other templates get the line alone.
                resid = pd.Series(e - line.predict(design(p.base_pool[sel], par_sel)))
                off = resid.groupby(p.pool["template"].astype(str).to_numpy()[sel]).mean().to_dict()
                low_test = low_test + apply_offsets(0.0, off, p.test["template"].to_numpy())
            out[m] = p.base_test + low_test
        elif m == TPL:
            grand, off = pooled_offsets(e, p.pool["template"].to_numpy()[sel])
            out[m] = p.base_test + apply_offsets(grand, off, p.test["template"].to_numpy())
        elif m == LOOKUP:
            # A table of measured runtimes: a test query gets the mean log runtime that its template's
            # measured queries had on B, whatever its own plan; a template without a measured query
            # falls back to the source model with the speed factor.
            y_sel = pd.Series(np.log(p.pool[LABEL].to_numpy(float)[sel]))
            means = y_sel.groupby(p.pool["template"].astype(str).to_numpy()[sel]).mean().to_dict()
            tt = p.test["template"].astype(str).to_numpy()
            out[m] = np.array([means[t] if t in means else b + e.mean() for t, b in zip(tt, p.base_test)])
        elif m == TPL_RAW:
            # Unshrunk: each template seen among the k gets its own mean error, the others the speed factor.
            tpl = pd.Series(e).groupby(p.pool["template"].astype(str).to_numpy()[sel]).mean()
            off = {t: float(v - e.mean()) for t, v in tpl.items()}
            out[m] = p.base_test + apply_offsets(float(e.mean()), off, p.test["template"].to_numpy())
        elif m == KNN:
            # The error of the nearest of the k queries in the standardized feature space (no template labels).
            sc = StandardScaler().fit(p.X_pool)
            A, B = sc.transform(p.X_pool[sel]), sc.transform(p.X_test)
            nearest = np.argmin(((B[:, None, :] - A[None, :, :]) ** 2).sum(axis=2), axis=1)
            out[m] = p.base_test + e[nearest]
        elif m == RIDGE:
            out[m] = p.base_test + ridge.predict(p.X_test)
        elif m == RIDGE_TPL:
            resid = e - ridge.predict(p.X_pool[sel])
            _, off = pooled_offsets(resid, p.pool["template"].to_numpy()[sel])
            out[m] = p.base_test + ridge.predict(p.X_test) + apply_offsets(0.0, off, p.test["template"].to_numpy())
        elif m == RIDGE_TPL_RAW:
            resid = pd.Series(e - ridge.predict(p.X_pool[sel]))
            off = resid.groupby(p.pool["template"].astype(str).to_numpy()[sel]).mean().to_dict()
            out[m] = p.base_test + ridge.predict(p.X_test) + apply_offsets(0.0, off, p.test["template"].to_numpy())
        elif m == FOREST_RESID:
            rf = ae.make_forest(seed).fit(p.X_pool[sel], e)
            out[m] = p.base_test + rf.predict(p.X_test)
        elif m == FOREST_RESID_TPL:
            # A test query of a template without a measured query has every indicator 0.
            names = np.array([str(t) for t in p.templates])
            def onehot(tt):
                return (np.asarray(tt).astype(str)[:, None] == names[None, :]).astype(float)
            rf = ae.make_forest(seed).fit(np.c_[p.X_pool[sel], onehot(p.pool["template"].to_numpy()[sel])], e)
            out[m] = p.base_test + rf.predict(np.c_[p.X_test, onehot(p.test["template"].to_numpy())])
        elif m == POOLED:
            out[m] = fit_pooled(p.source_train, p.pool.iloc[sel], seed)(p.test)
        elif m == TARGET:
            out[m] = fit(p.pool.iloc[sel], seed)(p.test)
        elif m == ORACLE:
            out[m] = oracle
        elif m.startswith("transfer-"):
            src, _, plus = m[len("transfer-"):].partition("+")
            sm = shift_models.get(src)
            if sm is None:
                continue
            s_sel, s_test = sm.predict(p.X_pool[sel]), sm.predict(p.X_test)
            rest = e - s_sel
            if plus == "ridge":
                out[m] = p.base_test + s_test + ae.fit_correction(p.X_pool[sel], rest).predict(p.X_test)
            else:
                out[m] = p.base_test + s_test + rest.mean()
        else:
            raise ValueError(f"unknown method {m}")
    return out


def run_c2(p: ae.Pair, fit, orders: list[np.ndarray], ks: list[int], plan: dict,
           shift_models: dict) -> tuple[dict, dict]:
    """logs[(strategy, method, k)]: (orders x test templates) log median Q-errors, and
    cost[(strategy, k)]: per order, the target runtime in seconds of the k chosen queries
    for one run each (the study's labels take 12 runs: three passes of a warm-up and three
    timed runs).

    plan maps each strategy to the methods it runs. The oracle is fitted once.
    """
    ks = [k for k in ks if k <= p.n_pool]
    oracle = fit(p.pool, SOURCE_SEED)(p.test)
    target_ms = p.pool[LABEL].to_numpy(float)
    logs, cost = {}, {}
    for strategy, methods in plan.items():
        for r, order in enumerate(orders):
            pos = p.positions(order)
            seq = strategy_sequence(strategy, p, pos, r, max(ks))
            for k in ks:
                cost.setdefault((strategy, k), np.empty(len(orders)))[r] = target_ms[seq[:k]].sum() / 1000.0
                preds = predictions(p, seq[:k], methods, fit, r, oracle, shift_models)
                for m, pred in preds.items():
                    _, per_t, _, _ = ae.template_errors(p.y_test_ms, pred, p.groups)
                    logs.setdefault((strategy, m, k), np.empty((len(orders), len(p.templates))))[r] = per_t
    return logs, cost


def run_hard(p: ae.Pair, fit, orders: list[np.ndarray], strategies: list[str], methods: list[str],
             unit_of: dict, k: int = HARD_K, n_orders: int = HARD_ORDERS, shift_data: dict | None = None) -> dict:
    """Adaptation to templates absent from the k queries and from the source model's training.

    As adapt_eval.c2_hard, but a whole unit U is held out at a time: a
    template, or in TPC-DS a family of near-copy templates (unit_map), so
    that no near copy of a held-out template stays anywhere. f_s is trained
    on the source training queries outside U, the k target queries come from
    the pool outside U (chosen by each strategy among them), the oracle is
    trained on the pool outside U, the analog shift model is refitted
    outside U, and every method is scored on the target test queries of each
    template in U. logs[(strategy, method)]: (orders x templates).
    """
    logs = {}
    n_orders = min(n_orders, len(orders))
    src_u = np.array([unit_of.get(str(t), str(t)) for t in p.source_train["template"]])
    pool_u = np.array([unit_of[str(t)] for t in p.pool["template"]])
    t_units = [unit_of[str(t)] for t in p.templates]
    for u in dict.fromkeys(t_units):
        js = [j for j, x in enumerate(t_units) if x == u]
        rows = np.concatenate([p.groups[j] for j in js])
        local, start = [], 0
        for j in js:
            local.append(np.arange(start, start + len(p.groups[j])))
            start += len(p.groups[j])
        f_u = fit(p.source_train[src_u != u], SOURCE_SEED)
        keep = pool_u != u
        sub = HardView(p, f_u, keep, rows, src_u != u)
        oracle = fit(p.pool[keep], SOURCE_SEED)(sub.test)
        # A shift learned on the same workload's analog pair must not have seen U either; the other
        # workload's queries are different queries, so its shift models stay as they are.
        shift_models = {}
        for name, (X, rr, units) in (shift_data or {}).items():
            ok = units != u if name == "analog" else np.ones(len(rr), dtype=bool)
            shift_models[name] = fit_shift_model(X[ok], rr[ok], units[ok])
        for r in range(n_orders):
            pos = sub.positions(orders[r])
            for strategy in strategies:
                seq = strategy_sequence(strategy, sub, pos, r, k)
                preds = predictions(sub, seq[:k], methods, fit, r, oracle, shift_models)
                for m, pred in preds.items():
                    q = te.qerror(sub.y_test_ms, te.to_ms(pred))
                    arr = logs.setdefault((strategy, m), np.full((n_orders, len(p.templates)), np.nan))
                    for j, loc in zip(js, local):
                        arr[r, j] = np.log(np.median(q[loc]))
    return logs


class HardView:
    """A Pair restricted to the pool outside one held-out unit, scored on that unit's test queries."""

    def __init__(self, p: ae.Pair, f_u, keep: np.ndarray, rows: np.ndarray, source_keep: np.ndarray):
        self.pool = p.pool[keep]
        self.source_pool = p.source_pool[keep]
        self.n_pool = int(keep.sum())
        self.test = p.test.iloc[rows]
        self.source_train = p.source_train[source_keep]
        self.X_pool = p.X_pool[keep]
        self.X_test = p.X_test[rows]
        self.base_test = f_u(self.test)
        self.base_pool = f_u(self.pool)
        self.e_pool = p.y_pool[keep] - self.base_pool
        self.y_test_ms = p.y_test_ms[rows]
        self._position = {q: i for i, q in enumerate(np.asarray(p.pool_ids)[keep])}

    def positions(self, order: np.ndarray) -> np.ndarray:
        return np.array([self._position[q] for q in order if q in self._position], dtype=int)


# ------------------------------------------------------------------- streams

S_NEVER, S_RIDGE_ALL, S_RIDGE_LAST = "never", "ridge-all", "ridge-last50"
S_SPEED_LAST, S_RT_LAST = "speed-last50", "ridge+template-last50"
S_LT_LAST = "lowdim+template-last50"
STREAM = [S_NEVER, S_RIDGE_ALL, S_RIDGE_LAST, S_SPEED_LAST, S_RT_LAST, S_LT_LAST]


def fit_lowdim_template(base: np.ndarray, par: np.ndarray, e: np.ndarray, tpl: np.ndarray):
    """The primary method's correction fitted on given queries, as a function of (base, par, tpl).

    The same model as predictions()'s lowdim+template: the affine+parallel
    line (parallel terms only when both plan classes are present, the
    separate slope only with two of each), plus the unshrunk mean residual
    of every template present.
    """
    fewest = int(min(par.sum(), len(par) - par.sum()))

    def design(b, pr):
        if fewest == 0:
            return b[:, None]
        if fewest == 1:
            return np.c_[b, pr]
        return np.c_[b, pr, pr * b]
    line = ae.fit_correction(design(base, par), e)
    off = pd.Series(e - line.predict(design(base, par))).groupby(np.asarray(tpl).astype(str)).mean().to_dict()
    return lambda b, pr, t: line.predict(design(b, pr)) + apply_offsets(0.0, off, t)


def run_stream(p: ae.Pair, split: pd.DataFrame, n_orders: int = ae.STREAM_ORDERS,
               batch: int = ae.BATCH) -> pd.DataFrame:
    """adapt_eval.c3_stream with more update rules.

    The same stream (the pair's test queries on the source machine, then on
    the target, each part in a seeded random order, batches of about 10,
    test then train) and the same rows; methods: never update, the ridge
    correction on everything so far or on the last 50 queries (as in the
    course study), the speed factor of the last 50, and the ridge correction
    plus shrunken template offsets of the last 50.
    """
    first, second = p.source, p.target
    all_test = np.sort(split.loc[split["set"] == "test", "query_id"].to_numpy())
    n = p.n_test
    frames = [p.source_test, p.test]
    X = np.vstack([te.design(f, "Ridge") for f in frames])
    base = np.concatenate([p.f_s(f) for f in frames])
    y_ms = np.concatenate([f[LABEL].to_numpy(float) for f in frames])
    tpl = np.concatenate([f["template"].astype(str).to_numpy() for f in frames])
    par = np.concatenate([(f["workers_planned"].to_numpy(float) > 0).astype(float) for f in frames])
    e = np.log(y_ms) - base
    pos = {q: i for i, q in enumerate(p.test_ids)}
    cuts = [(s, s * n + int(c[0]), len(c)) for s in range(2) for c in ae.stream_batches(n, batch)]
    rows = []
    for j in range(n_orders):
        rng = np.random.default_rng(ae.STREAM_SEED + j)
        stream = np.concatenate([np.array([pos[q] for q in rng.permutation(all_test) if q in pos], dtype=int) + s * n
                                 for s in range(2)])
        for b_i, (s, start, size) in enumerate(cuts):
            now = stream[start:start + size]
            seen = stream[:start]
            recent = seen[-ae.WINDOW:]
            preds = {S_NEVER: base[now]}
            if len(seen):
                preds[S_RIDGE_ALL] = base[now] + ae.fit_correction(X[seen], e[seen]).predict(X[now])
                ridge = ae.fit_correction(X[recent], e[recent])
                preds[S_RIDGE_LAST] = base[now] + ridge.predict(X[now])
                preds[S_SPEED_LAST] = base[now] + e[recent].mean()
                _, off = pooled_offsets(e[recent] - ridge.predict(X[recent]), tpl[recent])
                preds[S_RT_LAST] = base[now] + ridge.predict(X[now]) + apply_offsets(0.0, off, tpl[now])
                lt = fit_lowdim_template(base[recent], par[recent], e[recent], tpl[recent])
                preds[S_LT_LAST] = base[now] + lt(base[now], par[now], tpl[now])
            else:
                for m in STREAM[1:]:
                    preds[m] = base[now]
            for m, pred in preds.items():
                rows.append({"order": j, "batch": b_i, "machine": (first, second)[s], "start": start,
                             "after_switch": start - n, "size": len(now), "method": m,
                             "median_q": float(np.median(te.qerror(y_ms[now], te.to_ms(pred))))})
    return pd.DataFrame(rows)


def summarize_stream(rows: pd.DataFrame, second: str) -> dict:
    """Batch curves (mean in log over orders) and adapt_eval.stream_levels for every method."""
    meta = rows[(rows["order"] == 0) & (rows["method"] == S_NEVER)].set_index("batch")
    post = (meta["machine"] == second).to_numpy()
    after = meta["after_switch"].to_numpy()
    lq = rows.assign(lq=np.log(rows["median_q"])).pivot_table(index="batch", columns="method", values="lq",
                                                                aggfunc="mean")
    out = {"after_switch": after.astype(int).tolist(), "curves": {}, "levels": {}}
    for m in STREAM:
        curve = np.exp(lq[m].to_numpy())
        out["curves"][m] = curve.tolist()
        lev = ae.stream_levels(curve, post, after)
        # Recovery read off the order-averaged curve is too optimistic for flat curves (the
        # average smooths the batch noise away), so it is counted in every order and summarized
        # by the median; a stream that never stays within 10% of its final level counts as never.
        rec = [ae.stream_levels(g.sort_values("batch")["median_q"].to_numpy(), post, after)["recover_queries"]
               for _, g in rows[rows["method"] == m].groupby("order")]
        med = float(np.median([np.inf if x is None else x for x in rec]))
        lev.pop("recover_queries")
        lev["recover_by_order"] = rec
        lev["recover_median"] = None if not np.isfinite(med) else med
        out["levels"][m] = lev
    return out


# ---------------------------------------------------------------- the shift

def shift_vector(p: ae.Pair) -> pd.DataFrame:
    """log(target label / source label) of every query of the pair, with its template, set and plan class."""
    src = pd.concat([p.source_pool, p.source_test])
    tgt = pd.concat([p.pool, p.test])
    return pd.DataFrame({"query_id": list(p.pool_ids) + list(p.test_ids),
                         "template": tgt["template"].astype(str).to_numpy(),
                         "set": ["pool"] * p.n_pool + ["test"] * p.n_test,
                         "parallel": tgt["workers_planned"].to_numpy(float) > 0,
                         "r": np.log(tgt[LABEL].to_numpy(float) / src[LABEL].to_numpy(float))})


def describe_shift(p: ae.Pair, unit_of: dict) -> dict:
    """adapt_eval's shift summary plus how much of the shift plan features explain.

    r2_in: R^2 of the ridge model of r on the plan features over all queries
    (in sample). r2_lot: R^2 when each unit's r (a template, or a TPC-DS
    family) is predicted by a model fitted without that unit, its penalty
    chosen by leaving out units inside the training part: the part of the
    shift that features carry to unseen templates. sd_template: standard
    deviation of the template means of r (how far one factor is from right).
    share_noise: the template share of log(pass j / pass i) between the
    source machine's own passes, averaged over the three pairs of passes:
    the share that noise alone gives on this workload (it grows with the
    number of templates per query), the reference for template_share.
    """
    out = ae.shift_summary(p, None)
    out.pop("threshold", None)
    out.pop("accept", None)
    sv = shift_vector(p)
    X = np.vstack([p.X_pool, p.X_test])
    r = sv["r"].to_numpy()
    units = np.array([unit_of[t] for t in sv["template"]])
    out["r2_in"] = float(fit_shift_model(X, r, units).score(X, r))
    pred = np.empty_like(r)
    for u in np.unique(units):
        m = units == u
        pred[m] = fit_shift_model(X[~m], r[~m], units[~m]).predict(X[m])
    out["r2_lot"] = float(1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum())
    src = pd.concat([p.source_pool, p.source_test])
    shares = []
    for i, j in ((1, 2), (1, 3), (2, 3)):
        a_col, b_col = f"pass{i}_ms", f"pass{j}_ms"
        if a_col in src.columns and b_col in src.columns:
            v = np.log(src[b_col].to_numpy(float) / src[a_col].to_numpy(float))
            shares.append(stats.template_share(v, src["template"].astype(str).to_numpy())[0])
    out["share_noise"] = float(np.mean(shares)) if shares else None
    # The share of variance between templates is biased upward when templates hold few queries;
    # omega squared corrects for it: (SS_between - (G - 1) MS_within) / (SS_total + MS_within). The key
    # stays "epsilon2", the name the first round's summary was written with.
    tpl = sv["template"].to_numpy()
    names, inv = np.unique(tpl, return_inverse=True)
    n = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=r) / n
    ss_t = float(((r - r.mean()) ** 2).sum())
    ss_b = float((n * (means - r.mean()) ** 2).sum())
    ms_w = (ss_t - ss_b) / max(len(r) - len(names), 1)
    out["epsilon2"] = float((ss_b - (len(names) - 1) * ms_w) / (ss_t + ms_w)) if ss_t > 0 else None
    tm = sv.groupby("template")["r"].mean()
    out["sd_template"] = float(tm.std(ddof=1))
    out["mean_abs_dev_template"] = float(np.mean(np.abs(tm - np.median(r))))
    return out


# ------------------------------------------------------------- summarizing

def unit_map(p: ae.Pair) -> dict:
    """The unit of every template: its TPC-DS family (labels.csv's family column, which joins
    near-copy templates) where there is one, else the template itself."""
    out = {}
    for f in (p.pool, p.test):
        units = f["family"] if "family" in f.columns else f["template"]
        out.update({str(t): str(u) for t, u in zip(f["template"], units)})
    return out


def unit_weights(templates: list, unit_of: dict, draws: int = stats.BOOTSTRAP,
                 seed: int = ae.BOOT_SEED) -> np.ndarray:
    """(draws x templates) weights of a bootstrap that resamples whole units.

    Each draw picks as many units as there are, with replacement, by
    stats.resamples with the pair's seed, and a template's weight is the
    number of times its unit was picked; units are numbered in the order
    of the pair's templates. With one template per unit (TPC-H) the
    weighted means are exactly stats.mean_interval's means over the same
    resamples. Near-copy TPC-DS templates are not independent, so resampling
    them one by one would make intervals too narrow.
    """
    units = list(dict.fromkeys(unit_of[str(t)] for t in templates))
    index = {u: i for i, u in enumerate(units)}
    tu = np.array([index[unit_of[str(t)]] for t in templates])
    picks = stats.resamples(len(units), np.random.default_rng(seed), draws)
    counts = np.zeros((draws, len(units)))
    np.add.at(counts, (np.repeat(np.arange(draws), picks.shape[1]), picks.ravel()), 1.0)
    return counts[:, tu]


def w_interval(values: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    draws = (W @ np.asarray(values, float)) / W.sum(axis=1)
    tail = (1 - stats.LEVEL) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def error_cell(values: np.ndarray, W: np.ndarray) -> dict:
    """exp of the mean over templates, with its 95% unit-resampling interval."""
    lo, hi = w_interval(values, W)
    return {"q": float(np.exp(np.mean(values))), "lo": float(np.exp(lo)), "hi": float(np.exp(hi))}


def paired_ratio(log_a: np.ndarray, log_b: np.ndarray, W: np.ndarray) -> dict:
    """stats.paired_ratio with its interval from the unit resamples (no Wilcoxon test)."""
    d = np.asarray(log_a, float) - np.asarray(log_b, float)
    lo, hi = w_interval(d, W)
    wins, n = int((d < 0).sum()), int((d != 0).sum())
    return {"gm": float(np.exp(d.mean())), "gm_lo": float(np.exp(lo)), "gm_hi": float(np.exp(hi)),
            "success": bool(np.exp(hi) < 1.0), "wins": wins, "n": n, "templates": int(len(d)),
            "sign_p": float(binomtest(wins, n, 0.5).pvalue) if n else 1.0}


def gap_closed(log_none: np.ndarray, log_method: np.ndarray, log_oracle: np.ndarray, W: np.ndarray) -> dict:
    """stats.gap_closed with the unit resamples; no interval when the point itself is undefined."""
    a, m, o = (np.asarray(v, float) for v in (log_none, log_method, log_oracle))

    def g(num, den):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    point = float(g(a.mean() - m.mean(), a.mean() - o.mean()))
    if not np.isfinite(point):
        return {"g": point, "g_lo": float("nan"), "g_hi": float("nan")}
    wa, wm, wo = ((W @ v) / W.sum(axis=1) for v in (a, m, o))
    draws = g(wa - wm, wa - wo)
    draws = draws[np.isfinite(draws)]
    tail = (1 - stats.LEVEL) / 2
    return {"g": point, "g_lo": float(np.quantile(draws, tail)), "g_hi": float(np.quantile(draws, 1 - tail))}


def summarize_logs(logs: dict, W: np.ndarray) -> dict:
    """Error, gap closed and paired ratios for every (strategy, method, k) of one pair.

    vs_speed and vs_ridge compare with the speed factor and the ridge
    correction under random selection at the same k (method and selection
    together, against the common baseline); vs_random compares with the same
    method under random selection (the effect of the selection alone).
    Intervals come from one set of unit resamples per pair, so every
    comparison inside a pair is paired.
    """
    L = {key: v.mean(axis=0) for key, v in logs.items()}
    out = {}
    for (strategy, m, k), v in L.items():
        none, oracle = L.get(("random", NONE, k)), L.get(("random", ORACLE, k))
        speed, ridge = L.get(("random", SPEED, k)), L.get(("random", RIDGE, k))
        same = L.get(("random", m, k))
        cell = error_cell(v, W)
        if none is not None and oracle is not None:
            cell["gap"] = gap_closed(none, v, oracle, W)
        if speed is not None:
            cell["vs_speed"] = paired_ratio(v, speed, W)
        if ridge is not None:
            cell["vs_ridge"] = paired_ratio(v, ridge, W)
        if same is not None and strategy != "random":
            cell["vs_random"] = paired_ratio(v, same, W)
        out[f"{strategy}|{m}|{k}"] = cell
    return out


def summarize_hard(logs: dict, W: np.ndarray) -> dict:
    """Errors and paired ratios on held-out units. The oracle trained without the unit is not an
    upper bound there (it can be worse than the speed factor), so no gap closed is reported."""
    L = {key: v.mean(axis=0) for key, v in logs.items()}
    out = {}
    for (strategy, m), v in L.items():
        cell = error_cell(v, W)
        for name, ref in (("vs_speed", ("random", SPEED)), ("vs_ridge", ("random", RIDGE)),
                          ("vs_oracle", ("random", ORACLE)), ("vs_random", ("random", m))):
            if ref in L and not (name == "vs_random" and strategy == "random"):
                cell[name] = paired_ratio(v, L[ref], W)
        out[f"{strategy}|{m}"] = cell
    return out


def long_frame(workload: str, pair: str, source: str, templates: list, logs: dict, hard: bool = False) -> pd.DataFrame:
    rows = []
    for key, v in logs.items():
        strategy, m = key[0], key[1]
        k = HARD_K if hard else key[2]
        mean = np.nanmean(v, axis=0)
        for t, val in zip(templates, mean):
            rows.append({"workload": workload, "pair": pair, "source": source, "test": "hard" if hard else "c2",
                         "strategy": strategy, "method": m, "k": k, "template": str(t), "log_q": float(val)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- driver

def load_workload(name: str, runs: Path) -> tuple[dict, pd.DataFrame]:
    cfg = WORKLOADS[name]
    split = ae.load_split(HERE / cfg["split"])
    labels = {}
    for machine, d in cfg["runs"].items():
        path = runs / d / "labels.csv"
        if path.exists():
            labels[machine] = ae.load_labels(path)
    ae.check_keys(labels, split)
    return labels, split


def shift_training(labels: dict, split: pd.DataFrame, a: str, b: str):
    """Pool rows of the pair a -> b for learning its shift: (X, r, unit of each query)."""
    p = ae.Pair(labels, split, a, b, fit=fit_cost)       # the source model is not used here
    sv = shift_vector(p)
    pool = (sv["set"] == "pool").to_numpy()
    unit_of = unit_map(p)
    units = np.array([unit_of[t] for t in sv["template"]])
    return p.X_pool, sv["r"].to_numpy()[pool], units[pool]


def pair_job(workload: str, source_name: str, a: str, b: str, labels: dict, split: pd.DataFrame,
             other: tuple | None, cfg: dict) -> dict:
    """Everything for one (workload, source model, pair): C2 with all strategies, the hard test, the shift."""
    t0 = time.time()
    fit = SOURCES[source_name]
    p = ae.Pair(labels, split, a, b, fit=fit)
    orders = ae.arrival_orders(split, cfg["orders"])
    ks = cfg["ks"]
    unit_of = unit_map(p)
    W = unit_weights(p.templates, unit_of)

    # Shift models for transfer, learned on pool queries of other pairs or the other workload.
    shift_data = {}
    if source_name == "hybrid" and cfg["transfer"]:
        if (a, b) in ANALOG and all(m in labels for m in ANALOG[(a, b)]):
            shift_data["analog"] = shift_training(labels, split, *ANALOG[(a, b)])
        if other is not None:
            o_labels, o_split = other
            if a in o_labels and b in o_labels:
                shift_data["workload"] = shift_training(o_labels, o_split, a, b)
            if (a, b) in ANALOG and all(m in o_labels for m in ANALOG[(a, b)]):
                shift_data["analog-workload"] = shift_training(o_labels, o_split, *ANALOG[(a, b)])
    # The workload transfers learn from the other workload's queries, so the test templates are unseen there.
    shift_models = {name: fit_shift_model(X, r, u) for name, (X, r, u) in shift_data.items()}
    transfer = [f"transfer-{n}{plus}" for n in shift_models for plus in ("", "+ridge")]

    plan = {}
    if source_name == "hybrid":
        plan["random"] = CHEAP + (COSTLY if cfg["costly"] else []) + transfer
        for s in STRATEGIES[1:]:
            plan[s] = [NONE, SPEED, TWO, AFFINE, AFFINE_PAR, LOWDIM_TPL, TPL, TPL_RAW, KNN, LOOKUP, RIDGE,
                       RIDGE_TPL, RIDGE_TPL_RAW, ORACLE] + transfer
        if cfg.get("extra"):
            plan["random"] = plan["random"] + EXTRA
            plan["round-robin"] = plan["round-robin"] + EXTRA
    else:
        plan["random"] = [NONE, SPEED, AFFINE_PAR, RIDGE, RIDGE_TPL, LOWDIM_TPL, ORACLE] + \
            ([TARGET] if cfg["costly"] else [])
        plan["round-robin"] = [NONE, SPEED, LOWDIM_TPL, ORACLE]
    if a == b:
        # The no-move control: the target machine is its own source, so the source model is the
        # oracle's pipeline trained on the target pool, and the corrections use the same k queries
        # (which that model was trained on, as the moved model was trained on the same queries on A).
        plan = {s: [NONE, SPEED, TPL_RAW, LOWDIM_TPL, KNN, LOOKUP, ORACLE] for s in ("random", "round-robin")}
    logs, cost = run_c2(p, fit, orders, ks, plan, shift_models)
    out = {"workload": workload, "source": source_name, "pair": ae.pair_name(a, b), "a": a, "b": b,
           "change": change_of(workload, a, b),
           "pool": p.n_pool, "test": p.n_test,
           "templates": [str(t) for t in p.templates], "features_differ": len(p.features_differ),
           "units": len(set(unit_of[str(t)] for t in p.templates)),
           "c2": summarize_logs(logs, W),
           "calibration_s": {f"{s}|{k}": float(v.mean()) for (s, k), v in cost.items()},
           "frames": [long_frame(workload, p.name, source_name, p.templates, logs)]}
    if source_name == "hybrid" and a != b:
        out["shift"] = describe_shift(p, unit_of)
        out["shift_vector"] = shift_vector(p).assign(workload=workload, pair=p.name)
        if cfg["stream"]:
            out["stream"] = summarize_stream(run_stream(p, split), b)
        if cfg["hard"]:
            hard_methods = [NONE, SPEED, TWO, AFFINE, AFFINE_PAR, LOWDIM_TPL, RIDGE, KNN, ORACLE] + [m for m in transfer]
            hard = run_hard(p, fit, orders, ["random", "round-robin", "rr-cheap", "k-center", "d-optimal"],
                            hard_methods, unit_of, shift_data=shift_data)
            out["hard"] = summarize_hard(hard, W)
            out["frames"].append(long_frame(workload, p.name, source_name, p.templates, hard, hard=True))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)
    data = {w: load_workload(w, runs) for w in args.workloads}
    cfg = {"orders": args.orders, "ks": KS, "costly": not args.no_costly, "hard": not args.no_hard,
           "transfer": not args.no_transfer, "stream": not args.no_stream, "extra": args.extra}
    # A workload with its own grid or budget (the second round's) is listed in the summary; its pairs
    # run with its own grid. A machine without labels is left out (load_workload).
    own = {w: {"ks": ks_of(w), "kh": budget(w)} for w in args.workloads
           if "ks" in WORKLOADS[w] or "kh" in WORKLOADS[w]}
    jobs = []
    for w, (labels, split) in data.items():
        o = WORKLOADS[w].get("other")
        other = None
        if o is not None:
            other = data[o] if o in data else (load_workload(o, runs) if (HERE / WORKLOADS[o]["split"]).exists() else None)
        w_cfg = {**cfg, "ks": ks_of(w)}
        machines = sorted(labels)
        for a, b in permutations(machines, 2):
            for s in args.sources:
                jobs.append((w, s, a, b, labels, split, other, w_cfg))
        if "hybrid" in args.sources and not args.no_control:
            for m in machines:
                jobs.append((w, "hybrid", m, m, labels, split, other, w_cfg))
    print(f"{len(jobs)} jobs on {args.jobs} processes", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(pair_job)(*j) for j in jobs)
    frames = [f for r in results for f in r.pop("frames")]
    shifts = [r.pop("shift_vector") for r in results if "shift_vector" in r]
    pd.concat(frames, ignore_index=True).to_csv(out / "per_template.csv", index=False)
    if shifts:
        pd.concat(shifts, ignore_index=True).to_csv(out / "shift_vectors.csv", index=False)
    summary = {"config": {**cfg, "workloads": args.workloads, "sources": args.sources,
                          "strategies": STRATEGIES, "hard_k": HARD_K, "hard_orders": HARD_ORDERS,
                          **({"by_workload": own} if own else {})},
               "seconds": round(time.time() - t0, 1), "jobs": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {out} in {summary['seconds']} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", default=str(HERE / "results" / "runs"))
    r.add_argument("--out", default=str(HERE / "results" / "paper" / "main"))
    r.add_argument("--workloads", nargs="+", default=list(DEFAULT_WORKLOADS), choices=list(WORKLOADS))
    r.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    r.add_argument("--orders", type=int, default=ORDERS)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--no-costly", action="store_true", help="skip forest-resid, pooled and target")
    r.add_argument("--no-hard", action="store_true")
    r.add_argument("--no-transfer", action="store_true")
    r.add_argument("--no-stream", action="store_true")
    r.add_argument("--no-control", action="store_true", help="skip the no-move controls (a machine as its own source)")
    r.add_argument("--extra", action="store_true", help="also run the exploratory methods of EXTRA (hybrid source)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
