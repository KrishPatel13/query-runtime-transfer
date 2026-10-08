"""Evaluate how a runtime model adapts to new machines (components C1 to C4).

The final proposal (final_proposal.tex, Table 4, and FINAL_PROPOSAL_PLAN.md,
section 4) asks two questions. Q1: a model trained on one machine is moved to
another; how few queries measured there are enough to adapt it, and does it
keep learning as more arrive? Q2: for templates never seen in training, does
the correction model beat PostgreSQL's cost line? This module runs the
evaluation of both on the labels the collection writes.

Inputs (the contract with the collection code)
  <runs>/<machine>/labels.csv  one row per query: query_id, template, variant,
                               status, label_ms and the 19 plan features of
                               src/features.py (FEATURE_NAMES)
  <split>                      query_id, template, variant, set: each query ID
                               is in the pool (70%) or the test set (30%),
                               split once inside every template and variant,
                               the same IDs on every machine
Models are trained only on pools and scored only on test sets. The source
model of a machine is trained once, on its pool queries with status ok on
that machine, so every pair with that source starts from the same model and
no target's failures shape it. A pair adapts and scores only on the queries
with status ok on both machines and with the same plan features on both:
FINAL_PROPOSAL_PLAN.md (section 3) requires identical plans everywhere, so a
query whose plan differs between the machines is left out of the pair and
counted in the reports.

Methods (all predict log runtime)
  source model f_s   the pilot's correction model ("Cost + RF residual" in
                     train_eval.py): a least-squares line a + b log(1 + total
                     cost) plus a random forest fitted to its residuals, with
                     train_eval's forest settings, trained on the source pool.
                     It is fixed in advance (final_proposal.tex, Goal); C1
                     reports how it compares with the other models
  No adaptation      f_s unchanged
  Speed factor       f_s plus the mean log error e = log y_t - f_s(x) of the k
                     target queries, one factor for every plan
  Ridge correction   f_s plus a ridge regression of e on the standardized plan
                     features (train_eval's Ridge design). The intercept is
                     not penalized and the penalty is chosen from
                     train_eval.ALPHAS by leave-one-out on the k queries
                     (RidgeCV), so a large penalty gives back the speed factor
                     and a small one a full linear correction
  Target only        the source pipeline trained on the k target queries alone
  Oracle             target only, trained on the whole target pool

Components
  C1  Per machine, the pilot's two experiments, run by train_eval.predict_all
      on the machine's pool queries (the test set stays unseen, so C1 can
      inform the choice of a source model): a 70/30 split inside every
      template (5 seeds) and leave one template out, for the cost line,
      Ridge, the forest, the correction model and the template mean (the
      global mean when the template is held out).
  C2  For each pair (source to target), f_s is the source machine's model.
      The target pool arrives in 20 seeded orders shared by all methods and
      pairs, every method is refitted on the first k = 5, 10, 25, 50 and 100
      queries of each order, and it is scored on the target test set. The
      error of a test template is the median Q-error of its test queries, and
      its log is averaged over the orders. E, the mean of that over test
      templates, gives (a) the error against k, reported as exp(E), the
      geometric mean over templates, (b) the share of the gap to the oracle
      closed, G(k) = (E_none - E_method) / (E_none - E_oracle), with the
      smallest k at which G reaches 0.9, and (c) the primary statistic of Q1:
      ridge correction against speed factor for one pair at one k (M1 to M3 at
      k = 25), the geometric mean over test templates of the ratio of their
      errors, with a 95% interval from 2000 paired resamples of templates
      (src/stats.py). The pooled median and P90 Q-error over all test queries
      are reported too, with intervals from the same resamples. A harder test
      holds out one template T at a time: f_s is trained without T, the 25
      target queries are drawn without T, and T's target test queries are
      scored (5 orders). Each pair also describes the shift itself: the
      median slowdown, the share of the variance of log(target / source
      label) between templates with its ANOVA p-value (the acceptance test
      for a target, FINAL_PROPOSAL_PLAN.md section 3), and how much of it
      PostgreSQL's cost explains.
  C3  Continued learning. (a) The C2 orders go on to the whole pool, and the
      correction is refitted every 50 queries. (b) A stream of the source
      machine's test queries in random order, then the other machine's test
      queries in random order, each part cut into batches of about 10 (10 or
      11 queries, so that no short remainder batch is left at the end of a
      part), scored test-then-train: each batch is predicted before its
      labels are used for an update (Gama et al., 2013). Methods: never
      update (f_s), the ridge correction refitted on all stream queries so
      far, and refitted on the last 50. Reported: the batch median Q-error
      (mean in log over 5 stream orders), the jump at the switch, and how
      many queries after the switch it takes until the batch error stays
      within 10% of its final level.
  C4  Unseen templates within one benchmark: leave one TPC-DS family out
      (templates that are near copies of each other, from
      scripts/tpcds_families.py, in labels.csv's family column), and compare
      the correction model with the cost line; the same with one template
      group held out is reported beside it, where the two parts of templates
      14, 23, 24 and 39 count as one group. The settings are frozen
      (train_eval's forest, nothing tuned on the benchmark), and the
      statistic is the one of C2 (c).

Outputs (in --outdir): summary.json, CSV files with the per-run numbers,
LaTeX tables (table_*.tex), a LaTeX macro for every number a document may
quote (numbers.tex), results.md, and figures as PDF and PNG (fig_*).
The run subcommand writes to results/final by default, where the final
report reads numbers.tex and its figures. Numbers that a decision rule
compares with a level (G with 0.9, the ends of a ratio's interval with 1)
get more decimals when two would put them on or across the level (0.8978
is shown as 0.898, not 0.90), so a shown value never contradicts the
decision made from the unrounded one.

Usage
  python src/adapt_eval.py run --runs results/runs --split queries/tpch_sf1_v2/split.csv \
      --c4-labels results/runs/M1_tpcds/labels.csv
  python src/adapt_eval.py c4 --labels results/runs/M1_tpcds/labels.csv --outdir results/eval_tpcds
  python src/adapt_eval.py dryrun --outdir results/eval_dryrun
  python src/adapt_eval.py report --outdir results/eval_dryrun
The dry run builds a synthetic three-machine dataset from the pilot's
measurements (see synthetic_runs) and runs everything on it. report redraws
the tables, macros and figures from a saved summary.json without refitting.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stats
import train_eval as te
from features import FEATURE_NAMES

HERE = Path(__file__).resolve().parents[1]
LABEL = "label_ms"
KEYS = ["query_id", "template", "variant"]
LABEL_COLUMNS = KEYS + ["status", LABEL] + FEATURE_NAMES
SPLIT_COLUMNS = KEYS + ["set"]

NONE, SPEED, RIDGE, TARGET, ORACLE = "No adaptation", "Speed factor", "Ridge correction", "Target only", "Oracle"
METHODS = [NONE, SPEED, RIDGE, TARGET, ORACLE]
ADAPTED = [SPEED, RIDGE, TARGET]
NEVER, ALL_SEEN, LAST = "Never update", "Refit on all so far", "Refit on the last 50"
STREAM_METHODS = [NEVER, ALL_SEEN, LAST]

KS = [5, 10, 25, 50, 100]
ORDERS = 20
HARD_K = 25
HARD_ORDERS = 5
REFIT_EVERY = 50
STREAM_ORDERS = 5
BATCH = 10
WINDOW = 50              # queries kept by "Refit on the last 50"
GAP_LEVEL = 0.9
RECOVER_TOLERANCE = 1.10
FINAL_BATCHES = 5        # the final level of a stream curve is its mean over the last 5 batches
SOURCE_SEED = 0
ORDER_SEED = 1000        # arrival order r is drawn with seed ORDER_SEED + r
STREAM_SEED = 2000       # stream order j is drawn with seed STREAM_SEED + j
BOOT_SEED = 4000         # the seed train_eval uses for its paired comparisons
POOLED_BOOT_SEED = 3000  # the seed train_eval uses for pooled-median intervals
MULTI_PART = (14, 23, 24, 39)
P_FLOOR = 1e-300         # smaller p-values, and those that underflow to 0, are shown as "<1e-300"

CODE = {NONE: "None", SPEED: "Speed", RIDGE: "Ridge", TARGET: "Target", ORACLE: "Oracle",
        NEVER: "Never", ALL_SEEN: "All", LAST: "Last"}
MODEL_CODE = {"Cost-only": "Cost", "Ridge": "Ridge", "Random Forest": "RF", te.HYBRID: "Hyb",
              te.REFERENCE: "TM", te.GLOBAL: "GM"}
MODEL_LABEL = {"Cost-only": "Cost line", "Ridge": "Ridge", "Random Forest": "Random forest",
               te.HYBRID: "Correction model", te.REFERENCE: "Template mean", te.GLOBAL: "Global mean"}
GENERATED = "% Generated by src/adapt_eval.py. Do not edit by hand."


# ------------------------------------------------------------------- loading

def load_labels(path: Path) -> pd.DataFrame:
    """One machine's labels.csv, checked against the contract. Row order is kept."""
    df = pd.read_csv(path)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["query_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate query_id values")
    ok = df["status"] == "ok"
    bad = ok & ~(pd.to_numeric(df[LABEL], errors="coerce") > 0)
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} rows with status ok have no positive {LABEL}")
    if df.loc[ok, FEATURE_NAMES].isna().any().any():
        raise ValueError(f"{path}: rows with status ok have missing plan features")
    return df


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["query_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate query_id values")
    other = set(df["set"]) - {"pool", "test"}
    if other:
        raise ValueError(f"{path}: set must be 'pool' or 'test', found {sorted(other)}")
    return df


def check_keys(labels: dict[str, pd.DataFrame], split: pd.DataFrame) -> None:
    """Template and variant of every query must agree between the split and each machine."""
    ref = split.set_index("query_id")[["template", "variant"]].astype(str)
    for machine, df in labels.items():
        d = df.set_index("query_id")[["template", "variant"]].astype(str)
        common = d.index.intersection(ref.index)
        differ = (d.loc[common] != ref.loc[common]).any(axis=1)
        if differ.any():
            raise ValueError(f"{machine}: template or variant differs from the split for "
                             f"{list(common[differ.to_numpy()][:5])}")


def ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with status ok in file order, the label also under train_eval's column name runtime_ms."""
    d = df[df["status"] == "ok"].reset_index(drop=True).copy()
    d["runtime_ms"] = d[LABEL].astype(float)
    return d


def sort_templates(values) -> list:
    """Templates in numeric order, by their leading number when they are text ('14a')."""
    def key(t):
        m = re.match(r"\D*(\d+)", str(t))
        return (int(m.group(1)) if m else math.inf, str(t))
    return sorted(set(values), key=key)


def template_seed(template, position: int) -> int:
    """Forest seed for a held-out template: its number, as in train_eval's E2, else its position."""
    return int(template) if str(template).isdigit() else position


def pair_name(source: str, target: str) -> str:
    return f"{source}->{target}"


def shown_path(path: Path) -> str:
    """A path for the reports: relative to the project when it lies inside it, so no local directory is recorded."""
    try:
        return Path(path).resolve().relative_to(HERE).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ----------------------------------------------------------------- modelling

def make_forest(seed: int):
    """train_eval's forest on one thread.

    The trees do not depend on n_jobs. On a few hundred rows threads cost
    more than they save, and with one thread the trees' predictions are
    summed in a fixed order, so repeated runs agree to the last bit.
    """
    return te.forest(seed).set_params(n_jobs=1)


def fit_source(df: pd.DataFrame, seed: int):
    """Fit the source pipeline on df and return a function giving log runtime for a frame.

    It is the pilot's correction model: the cost line fitted by least squares
    on log(1 + total cost), plus train_eval's random forest fitted to the
    line's residuals on the 19 plan features.
    """
    y = np.log(df[LABEL].to_numpy(float))
    line = LinearRegression().fit(te.design(df, "Cost-only"), y)
    rf = make_forest(seed).fit(te.design(df, te.HYBRID), y - line.predict(te.design(df, "Cost-only")))
    return lambda d: line.predict(te.design(d, "Cost-only")) + rf.predict(te.design(d, te.HYBRID))


def fit_correction(X: np.ndarray, e: np.ndarray):
    """Ridge regression of the source model's log error e on the plan features X.

    StandardScaler then RidgeCV over train_eval.ALPHAS. RidgeCV fits the
    intercept on centred data, so the intercept is not penalized: as the
    penalty grows the correction tends to the mean of e, the speed factor.
    With cv=None the penalty is chosen by efficient leave-one-out.
    """
    return make_pipeline(StandardScaler(), RidgeCV(alphas=te.ALPHAS)).fit(X, e)


def template_errors(y_ms: np.ndarray, pred_log: np.ndarray, groups: list[np.ndarray]):
    """Q-errors of the test queries, the log median Q-error of every test template, and the pooled
    median and P90 Q-error.

    Predictions are clipped in log space exactly as train_eval clips them.
    """
    q = te.qerror(y_ms, te.to_ms(pred_log))
    return q, np.log([np.median(q[g]) for g in groups]), float(np.median(q)), float(np.quantile(q, 0.9))


def pooled_cells(pred: pd.DataFrame) -> dict:
    """Intervals for the pooled median and P90 Q-error of one model's held-out predictions.

    pred has one row per test query with its template, y and yhat. The
    templates are resampled exactly as train_eval.template_bootstrap does
    (seed 3000, templates in sorted order), so ci_lo and ci_hi are the
    pilot's interval for the pooled median, and p90_lo and p90_hi come from
    the same resamples.
    """
    t = pred["template"].to_numpy()
    templates = sorted(pd.unique(t))
    picks = stats.resamples(len(templates), np.random.default_rng(POOLED_BOOT_SEED))
    columns = stats.resample_columns([np.flatnonzero(t == x) for x in templates], picks)
    r = stats.pooled_quantiles(te.qerror(pred["y"], pred["yhat"]), columns)
    return {"ci_lo": r[0.5][1], "ci_hi": r[0.5][2], "p90_lo": r[0.9][1], "p90_hi": r[0.9][2]}


def arrival_orders(split: pd.DataFrame, n: int, seed: int = ORDER_SEED, which: str = "pool") -> list[np.ndarray]:
    """n seeded random orders of all query IDs in one part of the split.

    Every method and every pair uses the same orders. A pair keeps the IDs it
    can use (status ok on both machines) in the same relative order, so two
    pairs with the same usable IDs see identical orders.
    """
    ids = np.sort(split.loc[split["set"] == which, "query_id"].to_numpy())
    return [np.random.default_rng(seed + r).permutation(ids) for r in range(n)]


class Pair:
    """The data of one source-target pair and the source model, built once.

    The source model f_s is the source machine's model: it is trained on
    every source-pool query with status ok on the source (source_train),
    whatever happens to those queries on the target. Every pair with the
    same source, and the C3 stream, therefore starts from the same model
    (the fit is seeded, so building it again gives the identical forest),
    and a target's timeouts never shape it.

    The pair adapts and scores on the split's queries with status ok on both
    machines and with the same plan features on both. The method assumes
    that only the runtime given x changes between machines, and
    FINAL_PROPOSAL_PLAN.md (section 3) requires identical plans, so a query
    whose features differ (a changed plan) would count a plan change as a
    machine slowdown; it is left out and listed in features_differ.
    source_pool and source_test are the source rows of the pair's queries,
    pool and test the target rows, each with its machine's own label and
    features. f_s's predictions, the target log errors and the Ridge design
    are computed once for the target pool and the target test set.
    """

    def __init__(self, labels: dict[str, pd.DataFrame], split: pd.DataFrame, source: str, target: str,
                 fit=None):
        # fit(frame, seed) -> predictor of log runtime; the course study always uses fit_source.
        fit = fit or fit_source
        self.source, self.target, self.name = source, target, pair_name(source, target)
        s = labels[source].set_index("query_id")
        t = labels[target].set_index("query_id")
        role = split.set_index("query_id")["set"]
        source_ok = s.index[s["status"] == "ok"].intersection(role.index)
        self.source_train = s.loc[np.sort(source_ok[(role.loc[source_ok] == "pool").to_numpy()].to_numpy())]
        ids = np.sort(source_ok.intersection(t.index[t["status"] == "ok"]).to_numpy())
        a, b = s.loc[ids, FEATURE_NAMES].to_numpy(float), t.loc[ids, FEATURE_NAMES].to_numpy(float)
        differ = (~np.isclose(a, b, rtol=1e-9, atol=0.0)).any(axis=1)
        self.features_differ = [str(q) for q in ids[differ]]
        ids = ids[~differ]
        self.pool_ids = [q for q in ids if role[q] == "pool"]
        self.test_ids = [q for q in ids if role[q] == "test"]
        self.source_pool, self.source_test = s.loc[self.pool_ids], s.loc[self.test_ids]
        self.pool = t.loc[self.pool_ids]
        self.test = t.loc[self.test_ids]
        self.n_pool, self.n_test = len(self.pool_ids), len(self.test_ids)

        self.f_s = fit(self.source_train, SOURCE_SEED)
        self.base_pool, self.base_test = self.f_s(self.pool), self.f_s(self.test)
        self.y_pool = np.log(self.pool[LABEL].to_numpy(float))
        self.y_test_ms = self.test[LABEL].to_numpy(float)
        self.e_pool = self.y_pool - self.base_pool
        self.X_pool, self.X_test = te.design(self.pool, "Ridge"), te.design(self.test, "Ridge")
        tt = self.test["template"].to_numpy()
        self.templates = sort_templates(tt)
        self.groups = [np.flatnonzero(tt == t) for t in self.templates]
        self._position = {q: i for i, q in enumerate(self.pool_ids)}

    def positions(self, order: np.ndarray) -> np.ndarray:
        """Positions in this pair's pool of the usable IDs of an arrival order, in that order."""
        return np.array([self._position[q] for q in order if q in self._position], dtype=int)


# ------------------------------------------------------------------------ C1

def c1_machine(labels: pd.DataFrame) -> dict:
    """The pilot's seen-template split and leave-one-template-out on one machine.

    Runs train_eval.predict_all exactly as train_eval's E1 and E2 do (same
    splits, seeds and Ridge tuning) on the given rows with status ok, in
    the order of its labels.csv, and summarizes them as train_eval.summarize
    does: seen templates as the mean and std over the 5 splits, held-out
    templates pooled once with 95% intervals from resampling templates (for
    the median as the pilot reports it, and for P90 from the same
    resamples), and every model paired with the cost line on the same
    held-out templates. evaluate passes the machine's pool queries only, so
    no C1 fit or score touches the test set that C2 and C3 report on, and
    C1 can inform the choice of a source model without seeing it.
    """
    d = ok_rows(labels)
    first_alpha = len(te.CHOSEN_ALPHAS)
    recs = []
    for seed in te.SEEDS:
        tr, ts = te.within_split(d, seed)
        recs += te.records("E1", f"seed{seed}", ts, te.predict_all(tr, ts, seed, reference=True, tag="C1 seen"))
    templates = sort_templates(d["template"])
    for i, t in enumerate(templates):
        tr, ts = d[d["template"] != t], d[d["template"] == t]
        recs += te.records("E2", str(t), ts, te.predict_all(tr, ts, template_seed(t, i), reference=False,
                                                            unseen=True, tag="C1 unseen"))
    pred = pd.concat(recs, ignore_index=True)

    by_fold = pred.groupby(["experiment", "fold", "model"]).apply(te.metrics, include_groups=False).reset_index()
    seen = by_fold[by_fold.experiment == "E1"].groupby("model")[["median_q", "p90_q", "log_mae"]].agg(["mean", "std"])
    e2 = pred[pred.experiment == "E2"]
    pooled = e2.groupby("model").apply(te.metrics, include_groups=False)
    ci = {m: pooled_cells(g) for m, g in e2.groupby("model")}
    per_t = by_fold[by_fold.experiment == "E2"].pivot(index="fold", columns="model", values="median_q")
    per_t = per_t.loc[[str(t) for t in templates]]
    alphas = pd.DataFrame(te.CHOSEN_ALPHAS[first_alpha:], columns=["tag", "rule", "alpha"])
    alphas["at_end"] = np.isclose(alphas["alpha"], te.ALPHAS[0]) | np.isclose(alphas["alpha"], te.ALPHAS[-1])
    return {
        "queries": int(len(d)), "templates": int(len(templates)),
        "seen": {m: {k: float(seen.loc[m, (k, "mean")]) for k in ("median_q", "p90_q", "log_mae")}
                 | {k + "_std": float(seen.loc[m, (k, "std")]) for k in ("median_q", "p90_q", "log_mae")}
                 for m in te.ALL_MODELS + [te.REFERENCE]},
        "unseen": {m: {k: float(v) for k, v in pooled.loc[m].items()} | ci[m] for m in te.ALL_MODELS + [te.GLOBAL]},
        "unseen_paired_vs_cost": te.paired_vs_cost(e2, np.random.default_rng(BOOT_SEED)),
        "unseen_wins_vs_cost": {m: int((per_t[m] < per_t["Cost-only"]).sum())
                                for m in te.ALL_MODELS + [te.GLOBAL] if m != "Cost-only"},
        "unseen_per_template": {t: {m: float(v) for m, v in row.items()} for t, row in per_t.iterrows()},
        "ridge_alpha": {f"{tag} ({rule})": {"min": float(g["alpha"].min()), "median": float(g["alpha"].median()),
                                            "max": float(g["alpha"].max()), "n": int(len(g)),
                                            "at_grid_end": int(g["at_end"].sum())}
                        for (tag, rule), g in alphas.groupby(["tag", "rule"], sort=False)},
    }


# ------------------------------------------------------------------------ C2

def pilot_share_before() -> float | None:
    """The level a target's template share must exceed: the share of the
    variance of log(pass 2 / pass 1) between templates over the pilot queries
    measured before pass 2 slowed down, when nothing had changed between the
    passes. Read from results/pilot/adaptation/summary.json (written by
    scripts/pilot_adaptation.py); None when that file is missing."""
    path = HERE / "results/pilot/adaptation/summary.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["shift"]["template_share_before"])


def shift_summary(p: Pair, threshold: float | None) -> dict:
    """How the target's runtimes differ from the source's, over all queries of the pair.

    With r = log(target label / source label) on the pair's pool and test
    queries: the median slowdown exp(median r); the smallest and largest
    template slowdown, exp of the mean r of a template with at least 3
    queries (as scripts/pilot_adaptation.py reports the pilot's); the share of
    the variance of r between templates, with its one-way ANOVA p-value; and
    the R^2 of r on log(1 + total cost), the part of the shift that re-fitting
    the cost line could follow. The template share is the acceptance test for
    a target in FINAL_PROPOSAL_PLAN.md (section 3): it must exceed the share
    between two unchanged pilot passes (threshold). A shift that is one speed
    factor has a share near 0, and then the ridge correction has nothing to
    add to the speed factor. The test labels are only described here; nothing
    is fitted on them.
    """
    src = pd.concat([p.source_pool, p.source_test])
    tgt = pd.concat([p.pool, p.test])
    r = np.log(tgt[LABEL].to_numpy(float) / src[LABEL].to_numpy(float))
    t = tgt["template"].to_numpy()
    share, anova_p = stats.template_share(r, t)
    x = np.log1p(tgt[["total_cost"]].to_numpy(float))
    per_t = pd.Series(r).groupby(t).agg(["mean", "size"])
    per_t = per_t[per_t["size"] >= 3]
    # Plans with parallel workers against plans without: one processor slows only the first kind.
    par = tgt["workers_planned"].to_numpy(float) > 0
    by_plan = {name: {"median": float(np.exp(np.median(r[sel]))) if sel.any() else float("nan"),
                      "queries": int(sel.sum())} for name, sel in (("serial", ~par), ("parallel", par))}
    return {"queries": int(len(r)), "median": float(np.exp(np.median(r))), "by_plan": by_plan,
            "template_min": float(np.exp(per_t["mean"].min())), "template_max": float(np.exp(per_t["mean"].max())),
            "template_share": share, "anova_p": anova_p,
            "cost_r2": float(LinearRegression().fit(x, r).score(x, r)), "threshold": threshold,
            "accept": None if threshold is None or not np.isfinite(share) else bool(share > threshold)}


def c2_pair(p: Pair, orders: list[np.ndarray], ks: list[int]) -> tuple[dict, pd.DataFrame, dict]:
    """Refit every method on the first k queries of each arrival order; score on the target test set.

    Returns logs[(method, k)], an (orders x test templates) array of log
    per-template median Q-errors; a frame with one row per order, k and
    method: the pooled median and P90 Q-error, the mean over templates of
    the log per-template median, and the penalty the ridge correction chose;
    and qerr[(method, k)], the (orders x test queries) Q-errors behind them,
    for the intervals of the pooled median and P90. No adaptation and the
    oracle depend on neither the order nor k; they are stored at every
    (order, k) so that all methods line up. The oracle's training set is the
    same for every order, so it is fitted once.
    """
    ks = [k for k in ks if k <= p.n_pool]
    oracle = fit_source(p.pool, SOURCE_SEED)(p.test)
    logs = {(m, k): np.empty((len(orders), len(p.templates))) for m in METHODS for k in ks}
    qerr = {(m, k): np.empty((len(orders), p.n_test)) for m in METHODS for k in ks}
    rows = []
    for r, order in enumerate(orders):
        idx = p.positions(order)
        for k in ks:
            sel = idx[:k]
            e = p.e_pool[sel]
            ridge = fit_correction(p.X_pool[sel], e)
            preds = {NONE: p.base_test, SPEED: p.base_test + e.mean(),
                     RIDGE: p.base_test + ridge.predict(p.X_test),
                     TARGET: fit_source(p.pool.iloc[sel], seed=r)(p.test), ORACLE: oracle}
            for m, pred in preds.items():
                q, per_t, med, p90 = template_errors(p.y_test_ms, pred, p.groups)
                logs[(m, k)][r], qerr[(m, k)][r] = per_t, q
                rows.append({"pair": p.name, "order": r, "k": k, "method": m, "median_q": med, "p90_q": p90,
                             "mean_log_template_q": float(per_t.mean()),
                             "ridge_alpha": float(ridge[-1].alpha_) if m == RIDGE else np.nan})
    return logs, pd.DataFrame(rows), qerr


def error_cell(values: np.ndarray, picks: np.ndarray) -> dict:
    """exp of the mean over templates, with its 95% template-resampling interval."""
    lo, hi = stats.mean_interval(values, picks)
    return {"q": float(np.exp(np.mean(values))), "lo": float(np.exp(lo)), "hi": float(np.exp(hi))}


def pooled_summary(qerr: dict, columns: list[np.ndarray], ks: list[int]) -> dict:
    """Pooled median and P90 Q-error of every method and k with their template-resampling intervals.

    For each order the median (P90) is taken over all test queries, and the
    orders are averaged in log, as in c2_runs.csv; the intervals come from
    the pair's template resamples (stats.pooled_quantiles). No adaptation
    and the oracle are the same at every k, so they are computed once.
    """
    out: dict = {}
    for m in METHODS:
        cells = {}
        for k in ks:
            if m in (NONE, ORACLE) and cells:
                cells[str(k)] = dict(next(iter(cells.values())))
                continue
            r = stats.pooled_quantiles(qerr[(m, k)], columns)
            cells[str(k)] = {"median_q": r[0.5][0], "median_lo": r[0.5][1], "median_hi": r[0.5][2],
                             "p90_q": r[0.9][0], "p90_lo": r[0.9][1], "p90_hi": r[0.9][2]}
        out[m] = cells
    return out


def c2_summary(p: Pair, logs: dict, runs: pd.DataFrame, qerr: dict) -> dict:
    """Error against k, gap closed and the ridge-against-speed comparison for one pair.

    The per-template log errors are first averaged over the orders. All
    intervals, including those of the pooled median and P90, use one set of
    template resamples, so every comparison within the pair is paired.
    """
    ks = sorted({k for _, k in logs})
    picks = stats.resamples(len(p.templates), np.random.default_rng(BOOT_SEED))
    L = {key: v.mean(axis=0) for key, v in logs.items()}
    gap = {m: {str(k): stats.gap_closed(L[(NONE, k)], L[(m, k)], L[(ORACLE, k)], picks) for k in ks}
           for m in ADAPTED}
    alpha = runs[runs["method"] == RIDGE].groupby("k")["ridge_alpha"]
    return {
        "source": p.source, "target": p.target, "pool": p.n_pool, "test": p.n_test,
        "templates": [str(t) for t in p.templates], "source_train": int(len(p.source_train)),
        "features_differ": len(p.features_differ), "features_differ_ids": p.features_differ,
        "error": {m: {str(k): error_cell(L[(m, k)], picks) for k in ks} for m in METHODS},
        "pooled": pooled_summary(qerr, stats.resample_columns(p.groups, picks), ks),
        "gap": gap,
        "k_gap": {m: stats.first_reaching(ks, [gap[m][str(k)]["g"] for k in ks], GAP_LEVEL) for m in ADAPTED},
        "ridge_vs_speed": {str(k): stats.paired_ratio(L[(RIDGE, k)], L[(SPEED, k)], picks) for k in ks},
        # How often leave-one-out chose an end of the penalty grid: the smallest penalty fits the k errors
        # almost exactly, the largest gives back the speed factor (Proposition 1 of the report).
        "ridge_alpha": {str(k): {"median": float(g.median()), "min": float(g.min()), "max": float(g.max()),
                                 "at_grid_end": int((np.isclose(g, te.ALPHAS[0]) | np.isclose(g, te.ALPHAS[-1])).sum()),
                                 "at_smallest": int(np.isclose(g, te.ALPHAS[0]).sum()),
                                 "at_largest": int(np.isclose(g, te.ALPHAS[-1]).sum()), "orders": int(len(g))}
                        for k, g in alpha},
    }


def per_template_frame(name: str, p: Pair, logs: dict, key_name: str) -> pd.DataFrame:
    """Long table of the order-averaged log per-template median Q-errors; logs is keyed (method, k or n)."""
    rows = []
    for (m, x), v in logs.items():
        for t, val in zip(p.templates, v.mean(axis=0)):
            rows.append({"pair": name, "method": m, key_name: x, "template": t, "log_q": float(val),
                         "q": float(np.exp(val))})
    return pd.DataFrame(rows)


def c2_hard(p: Pair, orders: list[np.ndarray], k: int = HARD_K, n_orders: int = HARD_ORDERS) -> dict:
    """Adaptation to a template that neither machine's training data contains.

    For every test template T: f_s is trained on the source machine's
    training queries (Pair.source_train) without T, the first k target
    queries of each of the first n_orders arrival orders are taken from the
    target pool without T, and every method is scored on T's target test
    queries. The oracle is trained on the target pool without T. Returns
    logs[method], an (orders x templates) array of log median Q-errors.
    """
    logs = {m: np.empty((n_orders, len(p.templates))) for m in METHODS}
    src_t = p.source_train["template"].to_numpy()
    pool_t = p.pool["template"].to_numpy()
    for j, (t, rows) in enumerate(zip(p.templates, p.groups)):
        f_t = fit_source(p.source_train[src_t != t], SOURCE_SEED)
        test_t = p.test.iloc[rows]
        base, X, y = f_t(test_t), p.X_test[rows], p.y_test_ms[rows]
        keep = pool_t != t
        e_pool = p.y_pool - f_t(p.pool)
        oracle = fit_source(p.pool[keep], SOURCE_SEED)(test_t)
        for r in range(n_orders):
            idx = p.positions(orders[r])
            sel = idx[keep[idx]][:k]
            e = e_pool[sel]
            preds = {NONE: base, SPEED: base + e.mean(),
                     RIDGE: base + fit_correction(p.X_pool[sel], e).predict(X),
                     TARGET: fit_source(p.pool.iloc[sel], seed=r)(test_t), ORACLE: oracle}
            for m, pred in preds.items():
                logs[m][r, j] = np.log(np.median(te.qerror(y, te.to_ms(pred))))
    return logs


def c2_hard_summary(p: Pair, logs: dict, k: int, n_orders: int) -> dict:
    picks = stats.resamples(len(p.templates), np.random.default_rng(BOOT_SEED))
    L = {m: v.mean(axis=0) for m, v in logs.items()}
    return {"k": k, "orders": n_orders, "templates": len(p.templates),
            "error": {m: error_cell(L[m], picks) for m in METHODS},
            "gap": {m: stats.gap_closed(L[NONE], L[m], L[ORACLE], picks) for m in ADAPTED},
            "ridge_vs_speed": stats.paired_ratio(L[RIDGE], L[SPEED], picks)}


# ------------------------------------------------------------------------ C3

def refit_steps(n_pool: int, every: int | None = None) -> list[int]:
    """Every 50 queries (REFIT_EVERY), and the whole pool as the last step."""
    every = every or REFIT_EVERY
    steps = list(range(every, n_pool + 1, every))
    return steps + [n_pool] if not steps or steps[-1] != n_pool else steps


def c3_continued(p: Pair, orders: list[np.ndarray]) -> tuple[list[int], dict]:
    """The C2 orders continued to the whole pool, the correction refitted every 50 queries.

    At every step n: never update (f_s), the ridge correction on the first n
    queries of the order, and on queries n - 50 to n. Returns the steps and
    logs[(method, n)] as in c2_pair.
    """
    steps = refit_steps(p.n_pool)
    logs = {(m, n): np.empty((len(orders), len(p.templates))) for m in STREAM_METHODS for n in steps}
    for r, order in enumerate(orders):
        idx = p.positions(order)
        for n in steps:
            seen, recent = idx[:n], idx[max(0, n - WINDOW):n]
            preds = {NEVER: p.base_test,
                     ALL_SEEN: p.base_test + fit_correction(p.X_pool[seen], p.e_pool[seen]).predict(p.X_test),
                     LAST: p.base_test + fit_correction(p.X_pool[recent], p.e_pool[recent]).predict(p.X_test)}
            for m, pred in preds.items():
                logs[(m, n)][r] = template_errors(p.y_test_ms, pred, p.groups)[1]
    return steps, logs


def c3_continued_summary(p: Pair, steps: list[int], logs: dict) -> dict:
    picks = stats.resamples(len(p.templates), np.random.default_rng(BOOT_SEED))
    cells = {m: [error_cell(logs[(m, n)].mean(axis=0), picks) for n in steps] for m in STREAM_METHODS}
    return {"steps": steps, "error": {m: [c["q"] for c in v] for m, v in cells.items()},
            "lo": {m: [c["lo"] for c in v] for m, v in cells.items()},
            "hi": {m: [c["hi"] for c in v] for m, v in cells.items()}}


def stream_batches(n: int, size: int) -> list[np.ndarray]:
    """Positions 0 to n - 1 cut into n // size consecutive batches whose sizes differ by at most one.

    With n = 204 and size 10 that is 4 batches of 11 and 16 of 10, so no
    batch is shorter than size unless n itself is. Fixed batches of size
    would leave a remainder batch (204 = 20 x 10 + 4) at the end of each
    machine's part, exactly where the level before the switch, the final
    level and the recovery are read, and the median of 4 queries is much
    noisier than that of 10.
    """
    if n < 1:
        raise ValueError("a stream part needs at least one query")
    return np.array_split(np.arange(n), max(1, n // size))


def c3_stream(p: Pair, split: pd.DataFrame, n_orders: int = STREAM_ORDERS, batch: int = BATCH) -> pd.DataFrame:
    """Test-then-train on a stream that switches machines without warning.

    The stream is the pair's test queries on its source machine in random
    order, followed by the same queries on its target machine (now with the
    target's labels and plan features) in another random order. Each
    machine's part is cut into batches of about `batch` queries
    (stream_batches). Each batch is first predicted by every method, using
    only the queries before it, and then joins the history. Never update
    predicts with f_s, the source machine's model; the ridge correction is
    refitted before every batch on the whole history, or on its last 50
    queries. A stream order permutes all test IDs of the split with a seed,
    and the pair keeps its own queries in that order, as arrival_orders does
    for the pool. Returns one row per stream order, batch and method with
    the batch median Q-error and the number of queries the correction was
    fitted on.
    """
    first, second = p.source, p.target
    all_test = np.sort(split.loc[split["set"] == "test", "query_id"].to_numpy())
    n = p.n_test
    frames = [p.source_test, p.test]
    X = np.vstack([te.design(f, "Ridge") for f in frames])
    base = np.concatenate([p.f_s(f) for f in frames])
    y_ms = np.concatenate([f[LABEL].to_numpy(float) for f in frames])
    e = np.log(y_ms) - base
    pos = {q: i for i, q in enumerate(p.test_ids)}
    cuts = [(s, s * n + int(c[0]), len(c)) for s in range(2) for c in stream_batches(n, batch)]
    rows = []
    for j in range(n_orders):
        rng = np.random.default_rng(STREAM_SEED + j)
        stream = np.concatenate([np.array([pos[q] for q in rng.permutation(all_test) if q in pos], dtype=int) + s * n
                                 for s in range(2)])
        for b_i, (s, start, size) in enumerate(cuts):
            now = stream[start:start + size]
            seen = stream[:start]
            recent = seen[-WINDOW:]
            preds = {NEVER: (base[now], 0),
                     ALL_SEEN: (base[now] + fit_correction(X[seen], e[seen]).predict(X[now]), len(seen))
                     if len(seen) else (base[now], 0),
                     LAST: (base[now] + fit_correction(X[recent], e[recent]).predict(X[now]), len(recent))
                     if len(recent) else (base[now], 0)}
            for m, (pred, n_train) in preds.items():
                rows.append({"order": j, "batch": b_i, "machine": (first, second)[s], "start": start,
                             "after_switch": start - n, "size": len(now), "method": m,
                             "median_q": float(np.median(te.qerror(y_ms[now], te.to_ms(pred)))),
                             "n_train": n_train})
    return pd.DataFrame(rows)


def stream_levels(curve: np.ndarray, post: np.ndarray, after: np.ndarray) -> dict:
    """Jump at the switch, final level, and queries until the curve stays within 10% of it.

    curve holds the batch errors in stream order and post marks the batches
    after the switch. The level before the switch is the geometric mean of
    the last 5 batches before it, and the final level that of the last 5
    batches of the stream, because single batches are noisy. The jump is the
    first batch after the switch divided by the level before it. Recovery
    counts the second machine's queries seen before the first batch from
    which every later batch is at most 1.1 times the final level (being
    below it counts as recovered). A method that never updates is flat after
    the switch, so its count only shows how far batch noise alone moves this
    statistic.
    """
    pre_c, post_c = curve[~post], curve[post]
    before = float(np.exp(np.log(pre_c[-FINAL_BATCHES:]).mean()))
    final = float(np.exp(np.log(post_c[-FINAL_BATCHES:]).mean()))
    i = stats.recovery_index(post_c, final, RECOVER_TOLERANCE)
    return {"before": before, "first_after": float(post_c[0]), "jump": float(post_c[0] / before),
            "final": final, "recover_queries": None if i is None else int(after[post][i])}


def c3_stream_summary(rows: pd.DataFrame, first: str, second: str, batch: int = BATCH) -> dict:
    """The stream's batch curves (mean in log over the stream orders) and stream_levels for each method.

    batch is the nominal batch size; batch_sizes lists the sizes that occur
    and curve.size the size of every batch.
    """
    meta = rows[(rows["order"] == 0) & (rows["method"] == NEVER)].set_index("batch")
    post = (meta["machine"] == second).to_numpy()
    after = meta["after_switch"].to_numpy()
    lq = rows.assign(lq=np.log(rows["median_q"])).pivot_table(index="batch", columns="method", values="lq",
                                                                aggfunc="mean")
    out = {"first": first, "second": second, "batch": int(batch),
           "batch_sizes": sorted({int(x) for x in meta["size"]}), "window": WINDOW,
           "orders": int(rows["order"].nunique()), "queries_each": int(meta.loc[~post, "size"].sum()),
           "curve": {"start": meta["start"].astype(int).tolist(), "size": meta["size"].astype(int).tolist(),
                     "machine": meta["machine"].tolist(), "after_switch": after.astype(int).tolist()}
           | {m: np.exp(lq[m].to_numpy()).tolist() for m in STREAM_METHODS},
           "methods": {}}
    for m in STREAM_METHODS:
        res = stream_levels(np.exp(lq[m].to_numpy()), post, after)
        by_order = []
        for _, g in rows[rows["method"] == m].groupby("order"):
            by_order.append(stream_levels(g.sort_values("batch")["median_q"].to_numpy(), post, after)["recover_queries"])
        out["methods"][m] = res | {"recover_by_order": by_order}
    return out


# ------------------------------------------------------------------------ C4

def template_group(template, multi_part=MULTI_PART) -> str:
    """The unit C4 holds out: TPC-DS templates 14, 23, 24 and 39 have two parts,
    which share the template number and are held out together. Every other
    template is its own unit. The TPC-DS manifest gives both parts the same
    template (14) and marks the part in the query ID (q14_001a, q14_001b), so
    they already share a unit; a template written with its part ('14a') is
    mapped to its number too."""
    s = str(template).strip()
    m = re.match(r"\D*(\d+)", s)
    return str(int(m.group(1))) if m and int(m.group(1)) in multi_part else s


def query_part(query_id) -> str:
    """The part letter that ends a TPC-DS multi-part query ID (q14_001a -> 'a'), else ''."""
    m = re.search(r"\d([a-z])$", str(query_id))
    return m.group(1) if m else ""


def c4_families(labels: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict | None]:
    """C4 with the template family as the held-out unit, and with the template beside it.

    Many TPC-DS templates are near copies of another one (37 and 82 differ
    only in the sales channel). scripts/tpcds_families.py groups them into
    families before any label exists, and labels.csv carries each query's
    family. Holding out a template while its near copy stays in training
    would let the forest see it, so the family is the primary unit; the
    template-level result is returned too, for comparison. Labels without a
    family column (TPC-H) give the template-level result only.
    """
    if "family" not in labels.columns:
        res, pred = c4_unseen(labels)
        return res | {"unit": "template"}, pred, None
    fam = {int(t): int(f) for t, f in ok_rows(labels)[["template", "family"]].drop_duplicates().itertuples(index=False)}
    if len(fam) != ok_rows(labels)["template"].nunique():
        raise ValueError("a template has more than one family in labels.csv")
    res, pred = c4_unseen(labels, group_of=lambda t: str(fam[int(t)]))
    by_template, _ = c4_unseen(labels)
    return res | {"unit": "family"}, pred, by_template | {"unit": "template"}


def c4_unseen(labels: pd.DataFrame, group_of=template_group) -> tuple[dict, pd.DataFrame]:
    """Leave one template group out; the correction model against the cost line.

    Both models are fitted on all other groups with the settings frozen on
    TPC-H (the cost line and train_eval's forest; nothing is tuned here) and
    predict the held-out group. The forest seed is the group's number, as in
    train_eval's E2, so on TPC-H this reproduces the pilot's leave-one-template-
    out comparison. Reported: the pooled median and P90 Q-error with intervals
    from resampling groups, and the primary statistic of Q2, the geometric
    mean over groups of the ratio of the two models' per-group median
    Q-errors (correction model over cost line) with its 95% interval, as in
    stats.paired_ratio. multi_part_groups lists the groups that hold more
    than one part, counted from the template and the part letter of the
    query ID, so it records that both parts were held out together whichever
    way the labels write the part.
    """
    d = ok_rows(labels)
    group = d["template"].map(group_of).astype(str)
    units = sort_templates(group.unique())
    numeric = all(u.isdigit() for u in units)
    recs = []
    for i, g in enumerate(units):
        held = (group == g).to_numpy()
        tr, ts = d[~held], d[held]
        y = np.log(tr[LABEL].to_numpy(float))
        line = LinearRegression().fit(te.design(tr, "Cost-only"), y)
        cost = line.predict(te.design(ts, "Cost-only"))
        rf = make_forest(template_seed(g, i)).fit(te.design(tr, te.HYBRID), y - line.predict(te.design(tr, "Cost-only")))
        for model, pred in (("Cost-only", cost), (te.HYBRID, cost + rf.predict(te.design(ts, te.HYBRID)))):
            recs.append(pd.DataFrame({"model": model, "template": int(g) if numeric else g,
                                      "source_template": ts["template"].to_numpy(),
                                      "query_id": ts["query_id"].to_numpy(),
                                      "y": ts[LABEL].to_numpy(float), "yhat": te.to_ms(pred)}))
    pred = pd.concat(recs, ignore_index=True)
    q = pred.assign(q=te.qerror(pred["y"], pred["yhat"]))
    per = q.groupby(["template", "model"])["q"].median().unstack("model")
    per = per.loc[[int(u) if numeric else u for u in units]]
    picks = stats.resamples(len(units), np.random.default_rng(BOOT_SEED))
    paired = stats.paired_ratio(np.log(per[te.HYBRID].to_numpy()), np.log(per["Cost-only"].to_numpy()), picks)
    pooled = {m: {k: float(v) for k, v in te.metrics(g).items()} | pooled_cells(g) for m, g in pred.groupby("model")}
    parts = (d["template"].astype(str) + "|" + d["query_id"].map(query_part)).groupby(group).nunique()
    return {"queries": int(len(d)), "groups": int(len(units)),
            "multi_part_groups": [str(g) for g in sort_templates(g for g, n in parts.items() if n > 1)],
            "pooled": pooled, "paired": paired,
            "per_group": {str(g): {m: float(v) for m, v in row.items()} for g, row in per.iterrows()}}, pred


# ------------------------------------------------------------------ pipeline

def data_summary(labels: dict[str, pd.DataFrame], split: pd.DataFrame) -> dict:
    machines = {}
    for m, df in labels.items():
        ok = df[df["status"] == "ok"]
        machines[m] = {"rows": int(len(df)), "ok": int(len(ok)),
                       "status": {str(k): int(v) for k, v in df["status"].value_counts().items()},
                       "templates": int(ok["template"].nunique()),
                       "variants": {str(k): int(v) for k, v in ok["variant"].value_counts().items()},
                       "not_in_split": int((~df["query_id"].isin(split["query_id"])).sum()),
                       "label_ms": {"min": float(ok[LABEL].min()), "median": float(ok[LABEL].median()),
                                    "max": float(ok[LABEL].max())}}
    return {"machines": machines,
            "split": {"pool": int((split["set"] == "pool").sum()), "test": int((split["set"] == "test").sum()),
                      "templates": int(split["template"].nunique()),
                      "variants": {str(k): int(v) for k, v in split["variant"].value_counts().items()}}}


def c1_task(machine: str, labels: pd.DataFrame) -> tuple[str, dict, float]:
    clock = time.perf_counter()
    return machine, c1_machine(labels), time.perf_counter() - clock


def pair_task(labels: dict[str, pd.DataFrame], split: pd.DataFrame, source: str, target: str,
              orders: list[np.ndarray], cfg: dict) -> dict:
    """C2, its harder test and C3 (a) for one pair; one task when the pairs run in parallel."""
    seconds, frames = {}, {}
    clock = time.perf_counter()
    p = Pair(labels, split, source, target)
    name = p.name
    logs, runs, qerr = c2_pair(p, orders, cfg["ks"])
    out = {"name": name, "c2": c2_summary(p, logs, runs, qerr) | {"descriptive": name in cfg["descriptive"],
                                                                  "shift": shift_summary(p, cfg["shift_threshold"])}}
    frames["c2_runs"], frames["c2_per_template"] = runs, per_template_frame(name, p, logs, "k")
    seconds["c2"] = time.perf_counter() - clock
    if "hard" not in cfg["skip"]:
        clock = time.perf_counter()
        n_hard = min(cfg["hard_orders"], len(orders))
        logs = c2_hard(p, orders, cfg["hard_k"], n_hard)
        out["c2_hard"] = c2_hard_summary(p, logs, cfg["hard_k"], n_hard)
        frames["c2_hard_per_template"] = per_template_frame(name, p, {(m, cfg["hard_k"]): v for m, v in logs.items()},
                                                            "k")
        seconds["hard"] = time.perf_counter() - clock
    if "c3" not in cfg["skip"]:
        clock = time.perf_counter()
        steps, logs = c3_continued(p, orders)
        out["c3_continued"] = c3_continued_summary(p, steps, logs)
        frames["c3_continued_per_template"] = per_template_frame(name, p, logs, "n")
        seconds["c3_continued"] = time.perf_counter() - clock
    return out | {"frames": frames, "seconds": seconds}


def evaluate(labels: dict[str, pd.DataFrame], split: pd.DataFrame, cfg: dict,
             c4_labels: pd.DataFrame | None = None, jobs: int = 1, log=print) -> tuple[dict, dict]:
    """Run every component in cfg and return the summary and the per-run frames.

    C1 sees each machine's pool queries only. The machines of C1 and the
    pairs of C2 and C3 (a) are independent, so with jobs > 1 they run in
    separate processes (joblib). Every fit is seeded, so the results do not
    depend on jobs, and each process that needs a machine's source model
    fits the identical one. (C1 runs train_eval's own code, whose forests
    sum their trees on several threads, so its numbers can differ between
    runs in the last bit, around 1e-16.)
    """
    from joblib import Parallel, delayed

    clock = time.perf_counter()
    seconds: dict = {}
    check_keys(labels, split)
    rf = te.forest(0)
    summary = {"config": cfg | {"forest": {"n_estimators": rf.n_estimators, "min_samples_leaf": rf.min_samples_leaf,
                                           "max_features": rf.max_features},
                                "alphas": {"min": float(te.ALPHAS[0]), "max": float(te.ALPHAS[-1]),
                                           "n": int(len(te.ALPHAS))},
                                "seeds": {"source": SOURCE_SEED, "orders": ORDER_SEED, "stream": STREAM_SEED,
                                          "bootstrap": BOOT_SEED}, "bootstrap": stats.BOOTSTRAP},
               "data": data_summary(labels, split)}
    frames: dict[str, list] = {}
    skip = set(cfg["skip"])
    parallel = Parallel(n_jobs=jobs)

    if "c1" not in skip:
        t0 = time.perf_counter()
        summary["c1"] = {}
        pool = set(split.loc[split["set"] == "pool", "query_id"])
        for m, res, sec in parallel(delayed(c1_task)(m, labels[m][labels[m]["query_id"].isin(pool)])
                                    for m in cfg["machines"]):
            summary["c1"][m] = res
            log(f"C1 {m}: {sec:.0f} s")
        seconds["c1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    orders = arrival_orders(split, cfg["orders"])
    seconds["pair_tasks"] = {}
    for res in parallel(delayed(pair_task)(labels, split, s, t, orders, cfg) for s, t in cfg["pairs"]):
        name = res["name"]
        for part in ("c2", "c2_hard", "c3_continued"):
            if part in res:
                summary.setdefault(part, {})[name] = res[part]
        for key, f in res["frames"].items():
            frames.setdefault(key, []).append(f)
        seconds["pair_tasks"][name] = res["seconds"]
        log(f"pair {name}: " + ", ".join(f"{k} {v:.0f} s" for k, v in res["seconds"].items()))
        if res["c2"]["features_differ"]:
            log(f"pair {name}: left out {res['c2']['features_differ']} queries whose plan features differ "
                f"between the machines, e.g. {res['c2']['features_differ_ids'][:5]}")
    seconds["pairs"] = time.perf_counter() - t0
    primary, k = cfg["primary"], str(cfg["primary_k"])
    if primary in summary["c2"] and k in summary["c2"][primary]["ridge_vs_speed"]:
        summary["c2_primary"] = {"pair": primary, "k": cfg["primary_k"]} | summary["c2"][primary]["ridge_vs_speed"][k]

    if "c3" not in skip:
        t0 = time.perf_counter()
        first, second = cfg["stream"]
        p = Pair(labels, split, first, second)
        rows = c3_stream(p, split, cfg["stream_orders"], cfg["batch"])
        summary["c3_stream"] = c3_stream_summary(rows, first, second, cfg["batch"]) | {
            "source_train": int(len(p.source_train)), "features_differ": len(p.features_differ)}
        frames["c3_stream"] = [rows]
        seconds["c3_stream"] = time.perf_counter() - t0
        log(f"C3 stream: {seconds['c3_stream']:.0f} s")

    if c4_labels is not None:
        t0 = time.perf_counter()
        summary["c4"], pred, by_template = c4_families(c4_labels)
        if by_template is not None:
            summary["c4_by_template"] = by_template
        frames["c4_predictions"] = [pred]
        seconds["c4"] = time.perf_counter() - t0
        log(f"C4: {seconds['c4']:.0f} s")

    seconds["total"] = time.perf_counter() - clock
    summary["seconds"] = seconds
    return summary, {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


# ------------------------------------------------------------------- reports

def fmt(x, digits: int = 2) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def fmt_level(x, level: float, digits: int = 2, max_digits: int = 4) -> str:
    """fmt, with more decimals when rounding would put x on or across a decision level.

    G is compared with 0.9 and the ends of a ratio's interval with 1, so
    0.8978 is shown as 0.898 rather than 0.90, and 0.99996 as 1.0000 only
    after four decimals; a shown value then never contradicts the decision
    made from the unrounded one.
    """
    if x is None or not np.isfinite(x):
        return "--"
    d = digits
    while d < max_digits:
        r = round(float(x), d)
        if (r == level and x != level) or ((r < level) != (x < level)):
            d += 1
            continue
        break
    return f"{x:.{d}f}"


def fmt_p(p: float) -> str:
    """p-values with three decimals; below 0.001 as a power of ten that works in text and math mode.

    A p-value below P_FLOOR, or one that underflowed to 0, is shown as a
    bound, since 0 x 10^0 would read as an exact zero.
    """
    if p is None or not np.isfinite(p):
        return "--"
    if p < P_FLOOR:
        return f"\\ensuremath{{<10^{{{int(np.log10(P_FLOOR))}}}}}"
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.0e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def pct(x) -> str:
    """A share as a whole percentage, as the pilot's template shares are quoted."""
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def pct_threshold(x) -> str:
    """The acceptance test's threshold, with one decimal (13.6%), so that a share shown beside it can be
    compared with the level the test uses."""
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def pct_level(x, level) -> str:
    """A share as a whole percentage, with decimals where a whole number would put it on or across the
    acceptance threshold (pct_threshold), so the shown share never contradicts the decision."""
    if x is None or not np.isfinite(x):
        return "--"
    if level is None or not np.isfinite(level):
        return pct(x)
    return fmt_level(100 * x, 100 * level, digits=0, max_digits=3) + "%"


def fmt_ci(c: dict, key: str = "q", lo: str = "lo", hi: str = "hi") -> str:
    return f"{fmt(c[key])} [{fmt(c[lo])}, {fmt(c[hi])}]"


def fmt_ratio(c: dict, key: str = "gm", lo: str = "gm_lo", hi: str = "gm_hi") -> str:
    """A ratio of two methods' errors with its interval, each number formatted by fmt_level against 1,
    as in the macros, so a table never shows an interval end of 0.998 as 1.00."""
    return f"{fmt_level(c[key], 1.0)} [{fmt_level(c[lo], 1.0)}, {fmt_level(c[hi], 1.0)}]"


def k_gap_text(summary_pair: dict, m: str) -> str:
    """Smallest k with G >= 0.9, '>K' for the largest k tried when G is defined but never gets there,
    '--' when G is undefined (the oracle is not better than no adaptation)."""
    k = summary_pair["k_gap"][m]
    if k is not None:
        return str(k)
    gaps = summary_pair["gap"][m]
    finite = [v["g"] for v in gaps.values() if v["g"] is not None and np.isfinite(v["g"])]
    return f">{max(int(k) for k in gaps)}" if finite else "--"


def tex(cell) -> str:
    """Plain table text to LaTeX (cells from fmt_p are LaTeX already and pass through unchanged)."""
    s = re.sub(r"([%&#_])", r"\\\1", str(cell))
    s = re.sub(r"\bk\s*=\s*(\d+)", r"$k=\1$", s)
    s = re.sub(r"^G >= ([\d.]+) from k$", r"$G \\ge \1$ from $k$", s)
    s = s.replace("->", "$\\to$").replace(">=", "$\\ge$").replace("±", "$\\pm$").replace("R^2", "$R^2$")
    s = s.replace("ANOVA p", "ANOVA $p$").replace("after n queries (n:", "after $n$ queries ($n$:")
    s = re.sub(r">(?=\d)", r"$>$", s)
    s = re.sub(r"(?<![\w.{-])-(?=\d)", r"$-$", s)  # a minus sign, but not in an exponent or a '--'
    return re.sub(r"\b(median|P90) Q\b", r"\1 $Q$", s)


def md(cell) -> str:
    """Plain table text to Markdown (p-values arrive as LaTeX when very small)."""
    s = str(cell)
    m = re.fullmatch(r"\\ensuremath\{(.*)\\times 10\^\{(-?\d+)\}\}", s)
    return f"{m.group(1)}e{m.group(2)}" if m else s


def write_table(out: Path, name: str, colspec: str, header: list[str], rows: list, md_lines: list[str],
                title: str, note: str = "", super_header: str | None = None) -> None:
    """Write table_<name>.tex and add the same table to the Markdown report.

    rows holds lists of plain-text cells; None draws a rule between groups.
    """
    lines = [GENERATED, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    if super_header:
        lines.append(super_header)
    lines += [" & ".join(tex(h) for h in header) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if r is None else " & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / f"table_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    md_lines += [f"### {title}", ""] + ([note, ""] if note else [])
    md_lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md_lines += ["| " + " | ".join(md(c) for c in r) + " |" for r in rows if r is not None]
    md_lines.append("")


def tables(s: dict, out: Path) -> list[str]:
    """Write the LaTeX table of every component present in the summary; return them as Markdown."""
    md_lines: list[str] = []
    if "c1" in s:
        rows = []
        for m, c in s["c1"].items():
            if rows:
                rows.append(None)
            for i, model in enumerate(te.ALL_MODELS + [te.REFERENCE, te.GLOBAL]):
                seen, unseen = c["seen"].get(model), c["unseen"].get(model)
                pair = c["unseen_paired_vs_cost"].get(model)
                rows.append([m if i == 0 else "", MODEL_LABEL[model],
                             f"{fmt(seen['median_q'])} ± {fmt(seen['median_q_std'])}" if seen else "--",
                             fmt(seen["p90_q"]) if seen else "--",
                             fmt_ci(unseen, "median_q", "ci_lo", "ci_hi") if unseen else "--",
                             fmt(unseen["p90_q"]) if unseen else "--",
                             fmt_ratio(pair) if pair else "--",
                             f"{c['unseen_wins_vs_cost'][model]}/{c['templates']}" if pair else "--"])
        write_table(out, "c1", "@{}llcccccc@{}",
                    ["Machine", "Model", "median Q", "P90 Q", "median Q [95% CI]", "P90 Q",
                     "ratio [95% CI]", "wins"], rows, md_lines,
                    "C1: source models on each machine",
                    "Seen: 70/30 split inside every template, mean ± std over 5 splits. Unseen: one template held "
                    "out at a time, pooled, 95% intervals from resampling templates. Ratio: to the cost line, the "
                    "geometric mean over held-out templates of the per-template median Q-error over the cost line's.",
                    super_header=" & & \\multicolumn{2}{c}{Seen templates} & \\multicolumn{4}{c}{Unseen templates}"
                                 " \\\\\n\\cmidrule(lr){3-4}\\cmidrule(l){5-8}")

    shifts = {name: c["shift"] for name, c in s.get("c2", {}).items() if "shift" in c}
    if shifts:
        threshold = s["config"].get("shift_threshold")
        rows = [[name, str(h["queries"]), fmt(h["median"]),
                 (f"{fmt(h['by_plan']['serial']['median'])} / {fmt(h['by_plan']['parallel']['median'])}"
                  if "by_plan" in h else "--"),
                 f"{fmt(h['template_min'])} to {fmt(h['template_max'])}",
                 pct_level(h["template_share"], threshold), fmt_p(h["anova_p"]), fmt(h["cost_r2"], 3),
                 "--" if h["accept"] is None else ("yes" if h["accept"] else "no")] for name, h in shifts.items()]
        write_table(out, "shift", "@{}lcccccccc@{}",
                    ["Pair", "queries", "median slowdown", "serial / parallel plans", "template slowdowns",
                     "template share", "ANOVA p", "cost R^2", "accepted"], rows, md_lines,
                    "The shift between the machines of each pair",
                    "Over all queries of the pair (pool and test), r = log(target label / source label). Serial / "
                    "parallel plans: median slowdown of the plans without and with parallel workers. Template "
                    "slowdowns: smallest and largest exp(mean r) of a template with at least 3 queries. Template "
                    "share: share of the variance of r between templates (one-way ANOVA). Cost R^2: of r on "
                    "log(1 + total cost). Accepted: template share above the "
                    + ("--" if threshold is None else pct_threshold(threshold))
                    + " seen between two unchanged pilot passes.")

    if s.get("c2"):
        ks = s["config"]["ks"]
        head = ["Pair", "Method"] + [f"k={k}" for k in ks]
        rows, gap_rows, pooled_rows = [], [], []
        for name, c in s["c2"].items():
            label = name + (" (descr.)" if c["descriptive"] else "")
            for block in (rows, gap_rows, pooled_rows):
                if block:
                    block.append(None)
            for i, m in enumerate(METHODS):
                first = label if i == 0 else ""
                rows.append([first, m] + [fmt(c["error"][m][str(k)]["q"]) if str(k) in c["error"][m] else "--"
                                          for k in ks] + [k_gap_text(c, m) if m in ADAPTED else "--"])
                pooled_rows.append([first, m] + [f"{fmt(c['pooled'][m][str(k)]['median_q'])} / "
                                                 f"{fmt(c['pooled'][m][str(k)]['p90_q'])}"
                                                 if str(k) in c["pooled"][m] else "--" for k in ks])
            for i, m in enumerate(ADAPTED):
                gap_rows.append([label if i == 0 else "", m]
                                + [fmt_level(c["gap"][m][str(k)]["g"], GAP_LEVEL) if str(k) in c["gap"][m] else "--"
                                   for k in ks]
                                + [k_gap_text(c, m)])
        # The ratio with its interval is too wide for a row per pair, so it gets a row per k.
        pair_labels = [name + (" (descr.)" if c["descriptive"] else "") for name, c in s["c2"].items()]
        ratio_rows = [[f"k={k}"] + [fmt_ratio(c["ridge_vs_speed"][str(k)]) if str(k) in c["ridge_vs_speed"] else "--"
                                    for c in s["c2"].values()] for k in ks]
        ncol = "c" * len(ks)
        write_table(out, "c2", f"@{{}}ll{ncol}c@{{}}", head + [f"G >= {GAP_LEVEL} from k"], rows, md_lines,
                    "C2: error against k",
                    f"Geometric mean over test templates of the per-template median Q-error, averaged in log over "
                    f"{s['config']['orders']} arrival orders. Last column: smallest k at which the share of the gap "
                    f"to the oracle closed reaches {GAP_LEVEL}.")
        write_table(out, "c2_gap", f"@{{}}ll{ncol}c@{{}}", head + [f"G >= {GAP_LEVEL} from k"], gap_rows,
                    md_lines, "C2: share of the gap to the oracle closed, G(k)",
                    "G = (E_none - E_method) / (E_none - E_oracle).")
        write_table(out, "c2_ratio", "@{}l" + "c" * len(pair_labels) + "@{}", [""] + pair_labels, ratio_rows,
                    md_lines, "C2: ridge correction over speed factor",
                    f"The ridge correction's error over the speed factor's: geometric mean over test templates of "
                    f"the ratio of their per-template errors, 95% interval from {s['config']['bootstrap']} paired "
                    f"resamples of templates.")
        write_table(out, "c2_pooled", f"@{{}}ll{ncol}@{{}}", head, pooled_rows, md_lines,
                    "C2: pooled median / P90 Q-error over all test queries",
                    "Geometric mean over the arrival orders.")

    if "c2_hard" in s:
        names = list(s["c2_hard"])
        k = s["config"]["hard_k"]
        rows = [[m] + [fmt_ci(s["c2_hard"][n]["error"][m]) for n in names] for m in METHODS]
        rows.append(["Ridge / speed ratio"] + [fmt_ratio(s["c2_hard"][n]["ridge_vs_speed"]) for n in names])
        write_table(out, "c2_hard", "@{}l" + "c" * len(names) + "@{}", ["Method"] + names, rows, md_lines,
                    f"C2 harder test: a template held out on both machines (k={k})",
                    f"Median Q-error of the held-out template's target test queries, geometric mean over templates "
                    f"[95% interval], averaged in log over {s['config']['hard_orders']} orders.")

    if "c3_continued" in s:
        rows = []
        for name, c in s["c3_continued"].items():
            if rows:
                rows.append(None)
            show = sorted({c["steps"][0], *[n for n in c["steps"] if n % 100 == 0], c["steps"][-1]})
            for i, m in enumerate(STREAM_METHODS):
                vals = dict(zip(c["steps"], c["error"][m]))
                rows.append([name if i == 0 else "", m, " ".join(f"{n}:{fmt(vals[n])}" for n in show)])
        write_table(out, "c3_continued", "@{}lll@{}", ["Pair", "Method", "error after n queries (n:error)"], rows,
                    md_lines, "C3 (a): the orders continued to the whole pool",
                    "Correction refitted every 50 queries; error as in C2.")
    if "c3_stream" in s:
        c = s["c3_stream"]
        rows = [[m, fmt(r["before"]), fmt(r["first_after"]), fmt(r["jump"]), fmt(r["final"]),
                 "--" if r["recover_queries"] is None else str(r["recover_queries"]),
                 ", ".join("--" if v is None else str(v) for v in r["recover_by_order"])]
                for m, r in c["methods"].items()]
        write_table(out, "c3_stream", "@{}lcccccl@{}",
                    ["Method", "level before", "first batch after", "jump", "final level",
                     "queries to recover", "per order"], rows, md_lines,
                    f"C3 (b): stream from {c['first']} to {c['second']}, test-then-train",
                    f"Batch median Q-error (batches of {c['batch']}), geometric mean over {c['orders']} stream "
                    f"orders. Level before and final level: mean of the last {FINAL_BATCHES} batches before the "
                    f"switch and of the last {FINAL_BATCHES} of the stream; jump: first batch after the switch "
                    f"over the level before. Recovery: queries after the switch until every later batch is within "
                    f"{100 * (RECOVER_TOLERANCE - 1):.0f}% of the final level, for the aggregate curve and for each "
                    f"order. Never update is flat after the switch, so its count shows the noise of this "
                    f"statistic.")
    if "c4" in s:
        c = s["c4"]
        blocks = [(c, c.get("unit", "template"))]
        if "c4_by_template" in s:
            blocks.append((s["c4_by_template"], "template"))
        rows = []
        for b, unit in blocks:
            if rows:
                rows.append(None)
            for i, model in enumerate(("Cost-only", te.HYBRID)):
                r = b["pooled"][model]
                ratio = fmt_ratio(b["paired"]) if model == te.HYBRID else "--"
                wins = f"{b['paired']['wins']}/{b['paired']['n']}" if model == te.HYBRID else "--"
                rows.append([f"{unit} ({b['groups']})" if i == 0 else "", MODEL_LABEL[model],
                             fmt_ci(r, "median_q", "ci_lo", "ci_hi"), fmt(r["p90_q"]), ratio, wins])
        unit = c.get("unit", "template")
        write_table(out, "c4", "@{}llcccc@{}", ["Held out", "Model", "median Q [95% CI]", "P90 Q",
                                                 "ratio to cost line [95% CI]", "wins"], rows, md_lines,
                    f"C4: one {unit} held out at a time ({c['groups']} {unit} groups)",
                    f"Intervals from resampling the held-out units. First block, the primary statistic of Q2: sign "
                    f"test p = {md(fmt_p(c['paired']['sign_p']))}, Wilcoxon p = {md(fmt_p(c['paired']['wilcoxon_p']))}."
                    f" Success (interval below 1): {c['paired']['success']}.")
    return md_lines


def number_word(n: int) -> str:
    small = ("Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen "
             "Sixteen Seventeen Eighteen Nineteen").split()
    tens = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")
    if n < 20:
        return small[n]
    if n < 100:
        return tens[n // 10] + (small[n % 10] if n % 10 else "")
    if n < 1000:
        return small[n // 100] + "Hundred" + (number_word(n % 100) if n % 100 else "")
    if n < 1_000_000:
        return number_word(n // 1000) + "Thousand" + (number_word(n % 1000) if n % 1000 else "")
    raise ValueError(n)


def tex_name(*parts) -> str:
    """A TeX macro name from parts: digits become words, everything but letters is dropped."""
    s = re.sub(r"\d+", lambda m: number_word(int(m.group())), "".join(str(p) for p in parts))
    return re.sub(r"[^A-Za-z]", "", s)


def macros(s: dict, prefix: str = "") -> dict[str, str]:
    """Every number a document may quote, keyed by macro name (letters only)."""
    m: dict[str, str] = {}

    def put(value, *parts):
        name = tex_name(prefix, *parts)
        if name in m:
            raise ValueError(f"duplicate macro name {name}")
        v = str(value)
        # A hyphen before a number is a minus sign in a document.
        m[name] = "\\ensuremath{-}" + v[1:] if re.match(r"-\d", v) else v

    if "config" in s:
        cfg = s["config"]
        put(cfg["orders"], "EvalOrders")
        put(cfg["hard_orders"], "EvalHardOrders")
        put(cfg["stream_orders"], "EvalStreamOrders")
        put(cfg["bootstrap"], "EvalBootstrap")
        if cfg.get("shift_threshold") is not None:
            put(pct_threshold(cfg["shift_threshold"]).rstrip("%"), "ShiftShareThreshold")
    if "data" in s:
        for machine, d in s["data"]["machines"].items():
            put(d["ok"], "Data", machine, "Queries")
            put(d["templates"], "Data", machine, "Templates")
        put(s["data"]["split"]["pool"], "DataPool")
        put(s["data"]["split"]["test"], "DataTest")
    for machine, c in s.get("c1", {}).items():
        for model, code in MODEL_CODE.items():
            if model in c["seen"]:
                put(fmt(c["seen"][model]["median_q"]), "COne", machine, code, "SeenMed")
                put(fmt(c["seen"][model]["p90_q"]), "COne", machine, code, "SeenPninety")
            if model in c["unseen"]:
                u = c["unseen"][model]
                put(fmt(u["median_q"]), "COne", machine, code, "UnseenMed")
                put(fmt(u["ci_lo"]), "COne", machine, code, "UnseenLo")
                put(fmt(u["ci_hi"]), "COne", machine, code, "UnseenHi")
                put(fmt(u["p90_q"]), "COne", machine, code, "UnseenPninety")
                if "p90_lo" in u:
                    put(fmt(u["p90_lo"]), "COne", machine, code, "UnseenPninetyLo")
                    put(fmt(u["p90_hi"]), "COne", machine, code, "UnseenPninetyHi")
            p = c["unseen_paired_vs_cost"].get(model)
            if p:
                put(fmt_level(p["gm"], 1.0), "COne", machine, code, "Geo")
                put(fmt_level(p["gm_lo"], 1.0), "COne", machine, code, "GeoLo")
                put(fmt_level(p["gm_hi"], 1.0), "COne", machine, code, "GeoHi")
                put(c["unseen_wins_vs_cost"][model], "COne", machine, code, "Wins")
                put(fmt_p(p["sign_p"]), "COne", machine, code, "SignP")
                put(fmt_p(p["wilcoxon_p"]), "COne", machine, code, "WilcoxP")
    for name, c in s.get("c2", {}).items():
        put(c["pool"], "CTwo", name, "Pool")
        put(c["test"], "CTwo", name, "Test")
        put(len(c["templates"]), "CTwo", name, "Templates")
        if "shift" in c:
            h = c["shift"]
            put(fmt(h["median"]), "CTwo", name, "ShiftMed")
            for kind, code in (("serial", "Serial"), ("parallel", "Parallel")):
                if kind in h.get("by_plan", {}):
                    put(fmt(h["by_plan"][kind]["median"]), "CTwo", name, "Shift", code)
                    put(h["by_plan"][kind]["queries"], "CTwo", name, "Shift", code, "N")
            put(fmt(h["template_min"]), "CTwo", name, "ShiftMin")
            put(fmt(h["template_max"]), "CTwo", name, "ShiftMax")
            put(pct_level(h["template_share"], s.get("config", {}).get("shift_threshold")).rstrip("%"),
                "CTwo", name, "ShiftShare")
            put(fmt_p(h["anova_p"]), "CTwo", name, "ShiftAnovaP")
            put(fmt(h["cost_r2"], 3), "CTwo", name, "ShiftCostRsq")
            put("--" if h["accept"] is None else ("yes" if h["accept"] else "no"), "CTwo", name, "ShiftAccept")
        ks = sorted(int(k) for k in c["error"][NONE])
        put(fmt(c["error"][NONE][str(ks[0])]["q"]), "CTwo", name, CODE[NONE])
        put(fmt(c["error"][ORACLE][str(ks[0])]["q"]), "CTwo", name, CODE[ORACLE])
        for meth in ADAPTED:
            put(k_gap_text(c, meth).replace(">", "\\ensuremath{>}"), "CTwo", name, CODE[meth], "KNinety")
            for k in ks:
                e = c["error"][meth][str(k)]
                put(fmt(e["q"]), "CTwo", name, CODE[meth], "K", k)
                put(fmt(e["lo"]), "CTwo", name, CODE[meth], "K", k, "Lo")
                put(fmt(e["hi"]), "CTwo", name, CODE[meth], "K", k, "Hi")
                put(fmt_level(c["gap"][meth][str(k)]["g"], GAP_LEVEL), "CTwo", name, CODE[meth], "GapK", k)
        for meth in METHODS:
            for k in ks:
                cell = c["pooled"][meth][str(k)]
                for key, code in (("median_q", "Med"), ("median_lo", "MedLo"), ("median_hi", "MedHi"),
                                  ("p90_q", "Pninety"), ("p90_lo", "PninetyLo"), ("p90_hi", "PninetyHi")):
                    if key in cell:
                        put(fmt(cell[key]), "CTwo", name, CODE[meth], "K", k, code)
        for k in ks:
            a = c.get("ridge_alpha", {}).get(str(k), {})
            if "at_smallest" in a:
                put(a["at_smallest"], "CTwo", name, "AlphaSmallestK", k)
                put(a["at_largest"], "CTwo", name, "AlphaLargestK", k)
            r = c["ridge_vs_speed"][str(k)]
            put(fmt_level(r["gm"], 1.0), "CTwo", name, "RvsSK", k)
            put(fmt_level(r["gm_lo"], 1.0), "CTwo", name, "RvsSK", k, "Lo")
            put(fmt_level(r["gm_hi"], 1.0), "CTwo", name, "RvsSK", k, "Hi")
    if "c2_primary" in s:
        p = s["c2_primary"]
        src, dst = p["pair"].split("->")
        put(f"{src}$\\to${dst}", "PrimaryPair")
        put(p["k"], "PrimaryK")
        for key, code in (("gm", "Geo"), ("gm_lo", "GeoLo"), ("gm_hi", "GeoHi")):
            put(fmt_level(p[key], 1.0), "Primary", code)
        put(p["wins"], "PrimaryWins")
        put(p["n"], "PrimaryN")
        put(p["templates"], "PrimaryTemplates")
        put(fmt_p(p["sign_p"]), "PrimarySignP")
        put(fmt_p(p["wilcoxon_p"]), "PrimaryWilcoxP")
        put("yes" if p["success"] else "no", "PrimarySuccess")
    for name, c in s.get("c2_hard", {}).items():
        for meth in METHODS:
            put(fmt(c["error"][meth]["q"]), "CTwoHard", name, CODE[meth])
        r = c["ridge_vs_speed"]
        put(fmt_level(r["gm"], 1.0), "CTwoHard", name, "RvsS")
        put(fmt_level(r["gm_lo"], 1.0), "CTwoHard", name, "RvsSLo")
        put(fmt_level(r["gm_hi"], 1.0), "CTwoHard", name, "RvsSHi")
    for name, c in s.get("c3_continued", {}).items():
        for meth in STREAM_METHODS:
            put(fmt(c["error"][meth][0]), "CThree", name, CODE[meth], "First")
            put(fmt(c["error"][meth][-1]), "CThree", name, CODE[meth], "Pool")
        put(c["steps"][0], "CThree", name, "FirstN")
    if "c3_stream" in s:
        c = s["c3_stream"]
        put(c["queries_each"], "CThreeStreamQueries")
        for meth, r in c["methods"].items():
            for key, code in (("before", "Before"), ("first_after", "After"), ("jump", "Jump"), ("final", "Final")):
                put(fmt(r[key]), "CThreeStream", CODE[meth], code)
            put("--" if r["recover_queries"] is None else r["recover_queries"], "CThreeStream", CODE[meth], "Recover")
    if "c4" in s:
        c = s["c4"]
        put(c["groups"], "CFourGroups")
        put(c["queries"], "CFourQueries")
        for model in ("Cost-only", te.HYBRID):
            r = c["pooled"][model]
            code = MODEL_CODE[model]
            put(fmt(r["median_q"]), "CFour", code, "Med")
            put(fmt(r["ci_lo"]), "CFour", code, "Lo")
            put(fmt(r["ci_hi"]), "CFour", code, "Hi")
            put(fmt(r["p90_q"]), "CFour", code, "Pninety")
            if "p90_lo" in r:
                put(fmt(r["p90_lo"]), "CFour", code, "PninetyLo")
                put(fmt(r["p90_hi"]), "CFour", code, "PninetyHi")
        put(c.get("unit", "template"), "CFourUnit")
        p = c["paired"]
        for key, code in (("gm", "Geo"), ("gm_lo", "GeoLo"), ("gm_hi", "GeoHi")):
            put(fmt_level(p[key], 1.0), "CFour", code)
        put(p["wins"], "CFourWins")
        put(p["n"], "CFourN")
        put(fmt_p(p["sign_p"]), "CFourSignP")
        put(fmt_p(p["wilcoxon_p"]), "CFourWilcoxP")
        put("yes" if p["success"] else "no", "CFourSuccess")
    if "c4_by_template" in s:
        c = s["c4_by_template"]
        put(c["groups"], "CFourTplGroups")
        for model in ("Cost-only", te.HYBRID):
            r = c["pooled"][model]
            code = MODEL_CODE[model]
            put(fmt(r["median_q"]), "CFourTpl", code, "Med")
            put(fmt(r["ci_lo"]), "CFourTpl", code, "Lo")
            put(fmt(r["ci_hi"]), "CFourTpl", code, "Hi")
            put(fmt(r["p90_q"]), "CFourTpl", code, "Pninety")
        p = c["paired"]
        for key, code in (("gm", "Geo"), ("gm_lo", "GeoLo"), ("gm_hi", "GeoHi")):
            put(fmt_level(p[key], 1.0), "CFourTpl", code)
        put(p["wins"], "CFourTplWins")
        put(p["n"], "CFourTplN")
        put(fmt_p(p["sign_p"]), "CFourTplSignP")
        put(fmt_p(p["wilcoxon_p"]), "CFourTplWilcoxP")
        put("yes" if p["success"] else "no", "CFourTplSuccess")
    return m


def write_macros(s: dict, out: Path, prefix: str = "") -> None:
    lines = [GENERATED] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros(s, prefix).items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_markdown(s: dict, table_md: list[str], out: Path) -> None:
    lines = [f"# Adaptation evaluation: {s.get('label', '')}", ""]
    for m, d in s.get("data", {}).get("machines", {}).items():
        lines.append(f"- {m}: {d['ok']} of {d['rows']} queries ok ({d['templates']} templates), status "
                     + ", ".join(f"{k} {v}" for k, v in d["status"].items()) + ".")
    if "data" in s:
        sp = s["data"]["split"]
        lines.append(f"- Split: {sp['pool']} pool, {sp['test']} test queries in {sp['templates']} templates.")
    for name, c in s.get("c2", {}).items():
        lines.append(f"- {name}: {c['pool']} pool and {c['test']} test queries ok on both machines, "
                     f"{len(c['templates'])} test templates, {c['features_differ']} queries whose plan features "
                     f"differ between the machines.")
    if "c2_primary" in s:
        p = s["c2_primary"]
        lines += ["", f"**Primary statistic (Q1)**: ridge correction over speed factor, {p['pair']}, k = {p['k']}: "
                      f"{fmt_ratio(p)} over {p['templates']} test templates; "
                      f"ridge lower on {p['wins']} of {p['n']} (sign test p = {md(fmt_p(p['sign_p']))}, Wilcoxon "
                      f"p = {md(fmt_p(p['wilcoxon_p']))}). Success (interval below 1): {p['success']}."]
    if "c4" in s:
        p = s["c4"]["paired"]
        unit = s["c4"].get("unit", "template")
        lines += ["", f"**Primary statistic (Q2, C4)**: correction model over cost line: {fmt_ratio(p)} over "
                      f"{p['templates']} held-out {unit} groups. Success: {p['success']}."]
        if "c4_by_template" in s:
            t = s["c4_by_template"]["paired"]
            lines.append(f"With one template group held out instead: {fmt_ratio(t)} over {t['templates']} groups.")
    lines += ["", "Run time (wall clock, s): " + ", ".join(f"{k} {v:.0f}" for k, v in s["seconds"].items()
                                                           if not isinstance(v, dict))]
    for name, sec in s["seconds"].get("pair_tasks", {}).items():
        lines.append(f"- task {name}: " + ", ".join(f"{k} {v:.0f}" for k, v in sec.items()))
    lines.append("")
    lines += table_md
    if "dryrun" in s:
        lines += ["## Dry-run checks", "", "| Check | Value | Pass |", "|---|---|---|"]
        lines += [f"| {c['check']} | {c['value']} | {'yes' if c['pass'] else 'NO'} |" for c in s["dryrun"]["checks"]]
        lines.append("")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------- figures

# Colours are the first slots of train_eval's colour-blind-checked palette, in
# fixed order; the "last 50" variant takes the palette's violet, which stays
# distinct from orange. References (no adaptation, the oracle) stay neutral.
COLORS = {NONE: te.MUTED, SPEED: "#2a78d6", RIDGE: "#eb6834", TARGET: "#1baf7a", ORACLE: te.INK2,
          NEVER: te.MUTED, ALL_SEEN: "#eb6834", LAST: "#4a3aa7"}
MARKERS = {SPEED: "o", RIDGE: "s", TARGET: "^", NEVER: "", ALL_SEEN: "s", LAST: "D"}
Q_TICKS = (1, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000)


def q_ticks(lo: float, hi: float, min_gap: float = 0.08) -> list[float]:
    """Q-error ticks inside [lo, hi], each at least min_gap of the log range above the previous one."""
    span = math.log(hi / lo)
    ticks: list[float] = []
    for t in Q_TICKS:
        if lo <= t <= hi and (not ticks or math.log(t / ticks[-1]) >= min_gap * span):
            ticks.append(t)
    return ticks


def arrow(name: str) -> str:
    return name.replace("->", r" $\rightarrow$ ")


def fig_c2(s: dict, out: Path) -> None:
    names = list(s["c2"])
    fig, axes = plt.subplots(1, len(names), figsize=(te.FIG_W, 2.05), sharey=True, squeeze=False)
    lo, hi = np.inf, 0.0
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        c = s["c2"][name]
        ks = sorted(int(k) for k in c["error"][SPEED])
        te.style_axes(ax, logx=True, logy=True)
        for m, style in ((NONE, "--"), (ORACLE, ":")):
            v = c["error"][m][str(ks[0])]["q"]
            ax.axhline(v, color=COLORS[m], linewidth=1.1, linestyle=style, label=m)
            lo, hi = min(lo, v), max(hi, v)
        for m in ADAPTED:
            e = [c["error"][m][str(k)] for k in ks]
            ax.fill_between(ks, [x["lo"] for x in e], [x["hi"] for x in e], color=COLORS[m], alpha=0.12,
                            linewidth=0)
            ax.plot(ks, [x["q"] for x in e], color=COLORS[m], marker=MARKERS[m], markersize=3.5, linewidth=1.3,
                    label=m)
            lo, hi = min(lo, min(x["lo"] for x in e)), max(hi, max(x["hi"] for x in e))
        ax.set_xticks(ks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(k) for k in ks], fontsize=te.FS)
        ax.set_xlabel("labeled target queries $k$", color=te.INK, fontsize=te.FS_LABEL)
        title = f"({chr(97 + i)}) {arrow(name)}" + (" (descriptive)" if c["descriptive"] else "")
        ax.set_title(title, color=te.INK, fontsize=te.FS_LABEL, loc="left")
    axes[0][0].set_ylabel("Q-error (geometric mean over\ntest templates of the median)", color=te.INK,
                          fontsize=te.FS_LABEL)
    for ax in axes[0]:
        ax.set_ylim(max(1.0, lo / 1.05), hi * 1.1)
        te.qerror_axis(ax, ticks=q_ticks(max(1.0, lo / 1.05), hi * 1.1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(m) for m in (SPEED, RIDGE, TARGET, NONE, ORACLE)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=5, frameon=False,
               fontsize=te.FS, bbox_to_anchor=(0.5, 1.08), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    te.save(fig, out / "fig_c2_error_vs_k")


def fig_c3_continued(s: dict, out: Path) -> None:
    names = list(s["c3_continued"])
    fig, axes = plt.subplots(1, len(names), figsize=(te.FIG_W, 2.05), sharey=True, squeeze=False)
    lo, hi = np.inf, 0.0
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        c = s["c3_continued"][name]
        te.style_axes(ax, logy=True)
        oracle = s["c2"][name]["error"][ORACLE]
        v = next(iter(oracle.values()))["q"]
        ax.axhline(v, color=COLORS[ORACLE], linewidth=1.1, linestyle=":", label="Oracle")
        lo, hi = min(lo, v), max(hi, v)
        for m in STREAM_METHODS:
            dashed = {"linestyle": "--"} if m == NEVER else {"marker": MARKERS[m], "markersize": 3}
            ax.plot(c["steps"], c["error"][m], color=COLORS[m], linewidth=1.3, label=m, **dashed)
            if m != NEVER:
                ax.fill_between(c["steps"], c["lo"][m], c["hi"][m], color=COLORS[m], alpha=0.12, linewidth=0)
            lo, hi = min(lo, min(c["lo"][m])), max(hi, max(c["hi"][m]))
        ax.set_xlabel("target queries seen", color=te.INK, fontsize=te.FS_LABEL)
        title = f"({chr(97 + i)}) {arrow(name)}" + (" (descriptive)" if s["c2"][name]["descriptive"] else "")
        ax.set_title(title, color=te.INK, fontsize=te.FS_LABEL, loc="left")
    axes[0][0].set_ylabel("Q-error (geometric mean over\ntest templates of the median)", color=te.INK,
                          fontsize=te.FS_LABEL)
    for ax in axes[0]:
        ax.set_ylim(max(1.0, lo / 1.05), hi * 1.1)
        te.qerror_axis(ax, ticks=q_ticks(max(1.0, lo / 1.05), hi * 1.1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(m) for m in (ALL_SEEN, LAST, NEVER, "Oracle")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4, frameon=False,
               fontsize=te.FS, bbox_to_anchor=(0.5, 1.08), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    te.save(fig, out / "fig_c3_continued")


def fig_c3_stream(s: dict, out: Path) -> None:
    c = s["c3_stream"]
    x = np.array(c["curve"]["start"])
    switch = x[np.array(c["curve"]["machine"]) == c["second"]].min()
    fig, ax = plt.subplots(figsize=(te.FIG_W, 2.1))
    te.style_axes(ax, logy=True)
    for m in STREAM_METHODS:
        style = {"linestyle": "--"} if m == NEVER else {"marker": MARKERS[m], "markersize": 2.5}
        ax.plot(x, c["curve"][m], color=COLORS[m], linewidth=1.2, label=m, **style)
    ax.axvline(switch, color=te.INK2, linewidth=0.8)
    vals = np.concatenate([c["curve"][m] for m in STREAM_METHODS])
    top = vals.max() * 1.15
    ax.text(switch - 4, top / 1.04, f"{c['first']} test queries", ha="right", va="top", fontsize=te.FS, color=te.INK2)
    ax.text(switch + 4, top / 1.04, f"switch to {c['second']}", ha="left", va="top", fontsize=te.FS, color=te.INK2)
    ax.set_ylim(max(1.0, vals.min() / 1.05), top)
    te.qerror_axis(ax, ticks=q_ticks(max(1.0, vals.min() / 1.05), top))
    ax.set_xlabel(f"stream queries seen before the batch (batches of {c['batch']})", color=te.INK,
                  fontsize=te.FS_LABEL)
    ax.set_ylabel("batch median Q-error", color=te.INK, fontsize=te.FS_LABEL)
    ax.legend(frameon=False, fontsize=te.FS, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.tight_layout()
    te.save(fig, out / "fig_c3_stream")


def write_reports(s: dict, out: Path, prefix: str = "") -> None:
    """Tables, macros, results.md and figures, all computed from the summary alone."""
    out.mkdir(parents=True, exist_ok=True)
    table_md = tables(s, out)
    write_macros(s, out, prefix)
    write_markdown(s, table_md, out)
    if s.get("c2"):
        fig_c2(s, out)
    if "c3_continued" in s:
        fig_c3_continued(s, out)
    if "c3_stream" in s:
        fig_c3_stream(s, out)


def save_run(s: dict, frames: dict, out: Path, prefix: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_csv(out / f"{name}.csv", index=False, float_format="%.6g", lineterminator="\n")
    num = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=num) + "\n", encoding="utf-8", newline="\n")
    write_reports(s, out, prefix)


# ------------------------------------------------------------------- dry run

# Per synthetic target: the range the template slowdown factors are drawn from
# (log-uniformly), and the change of log runtime per planned parallel worker.
# M2 imitates one core (parallel plans slow down most), M3 a slower server
# whose slowdown depends mostly on the template.
SYNTHETIC = {"M2": {"slowdown": (1.0, 1.3), "per_worker": 0.25},
             "M3": {"slowdown": (1.2, 2.2), "per_worker": -0.05}}
NOISE_SD = 0.05


def make_split(queries: pd.DataFrame, seed: int = 0, test_frac: float = 0.3) -> pd.DataFrame:
    """Split query IDs once into pool and test inside every template and variant.

    A group of n >= 2 queries puts round(test_frac * n) of them (at least one)
    in the test set; a single query goes to the pool.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for (t, v), g in queries.groupby(["template", "variant"], sort=True):
        ids = rng.permutation(np.sort(g["query_id"].to_numpy()))
        n_test = 0 if len(ids) < 2 else max(1, round(test_frac * len(ids)))
        parts.append(pd.DataFrame({"query_id": ids, "template": t, "variant": v,
                                   "set": ["test"] * n_test + ["pool"] * (len(ids) - n_test)}))
    return pd.concat(parts).sort_values("query_id").reset_index(drop=True)


def synthetic_runs(pilot_csv: Path, out: Path, seed: int = 0) -> dict:
    """Write a synthetic three-machine dataset in the collection's format.

    M1 is the pilot's pass-1 labels. M2 and M3 multiply every M1 label by a
    factor for its template, drawn log-uniformly from the machine's range in
    SYNTHETIC, by exp(per_worker * workers_planned), and by multiplicative
    noise with log standard deviation 0.05. The template factors are
    slowdowns that one speed factor cannot follow; workers_planned is a plan
    feature the ridge correction sees. A few queries fail on the targets (the
    two slowest time out on M3 and one fails on M2), so the rule that a pair
    uses only queries with status ok on both machines is exercised. The split
    puts 30% of every template and variant in the test set. Writes
    <out>/runs/<machine>/labels.csv, <out>/split.csv and <out>/truth.json and
    returns the truth.
    """
    base = pd.read_csv(pilot_csv)
    base = base[base["status"] == "ok"].reset_index(drop=True)
    common = base[["query_id", "template"]].assign(variant="std")
    feats = base[FEATURE_NAMES]
    rng = np.random.default_rng(seed)
    templates = sort_templates(base["template"])
    truth = {"seed": seed, "noise_sd": NOISE_SD, "machines": {}}
    machines = {"M1": base["runtime_ms"].to_numpy(float)}
    for m, spec in SYNTHETIC.items():
        lo, hi = np.log(spec["slowdown"])
        factor = dict(zip(templates, rng.uniform(lo, hi, len(templates))))
        shift = base["template"].map(factor).to_numpy() + spec["per_worker"] * base["workers_planned"].to_numpy(float)
        machines[m] = machines["M1"] * np.exp(shift + rng.normal(0.0, NOISE_SD, len(base)))
        truth["machines"][m] = {"per_worker": spec["per_worker"], "slowdown_range": list(spec["slowdown"]),
                                "log_template_factor": {str(t): float(v) for t, v in factor.items()}}
    failed = {"M2": {base.index[int(rng.integers(len(base)))]: "error: synthetic failure"},
              "M3": {i: "timeout" for i in np.argsort(machines["M3"])[-2:]}}
    for m, label in machines.items():
        status = pd.Series("ok", index=base.index)
        for i, st in failed.get(m, {}).items():
            status[i] = st
        df = pd.concat([common, pd.DataFrame({"status": status, LABEL: np.where(status == "ok", label, np.nan)}),
                        feats], axis=1)
        (out / "runs" / m).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "runs" / m / "labels.csv", index=False, lineterminator="\n")
        truth["machines"].setdefault(m, {})["failed"] = [str(base.loc[i, "query_id"]) for i in failed.get(m, {})]
    make_split(common, seed).to_csv(out / "split.csv", index=False, lineterminator="\n")
    (out / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8", newline="\n")
    return truth


def shift_recovery(p: Pair, truth: dict, orders: list[np.ndarray], k: int) -> dict:
    """How well each correction learned from k queries matches the true synthetic shift.

    The true shift of a test query is its template's log factor plus
    per_worker * workers_planned (the noise left out). Both corrections are
    averaged over the orders and compared per test template (mean over its
    test queries): the RMSE against the true shift, and the correlation with
    it across templates. A speed factor is one number for every template, so
    its correlation is undefined.
    """
    spec = truth["machines"][p.target]
    true = (p.test["template"].astype(str).map(spec["log_template_factor"]).to_numpy(float)
            + spec["per_worker"] * p.test["workers_planned"].to_numpy(float))
    ridge, speed = np.zeros(p.n_test), 0.0
    for order in orders:
        sel = p.positions(order)[:k]
        ridge += fit_correction(p.X_pool[sel], p.e_pool[sel]).predict(p.X_test) / len(orders)
        speed += p.e_pool[sel].mean() / len(orders)
    per = pd.DataFrame({"t": p.test["template"].to_numpy(), "true": true, "ridge": ridge}).groupby("t").mean()
    return {"k": k, "ridge_rmse": float(np.sqrt(((per["ridge"] - per["true"]) ** 2).mean())),
            "speed_rmse": float(np.sqrt(((speed - per["true"]) ** 2).mean())),
            "ridge_corr": float(np.corrcoef(per["ridge"], per["true"])[0, 1]),
            "true_sd": float(per["true"].std(ddof=0))}


def dryrun_checks(s: dict, truth: dict, labels: dict[str, pd.DataFrame], split: pd.DataFrame,
                  orders: list[np.ndarray]) -> list[dict]:
    """Checks that the dry-run numbers make sense, each with its value and whether it passed.

    M1 is the pilot's pass 1, so C1's code run on all of M1's queries, and
    C4 on M1, must give the pilot's numbers (results/pilot/all22/summary.json
    from train_eval.py); the C1 that the study reports sees the pool queries
    only, so it is run once more here on every query for this check. The
    synthetic targets have known slowdowns, so they must pass the acceptance
    test, the adaptation methods must rank as the design implies, and the
    ridge correction must find the per-template slowdowns that one speed
    factor cannot.
    """
    checks = []

    def check(name: str, value, passed: bool):
        checks.append({"check": name, "value": value, "pass": bool(passed)})

    # Same computation means agreement far below the two decimals reported.
    # It is not exact: on seen templates Ridge picks a penalty near 1e-6, so
    # it is almost least squares on collinear features, and differences in
    # the order of floating-point sums grow to about 1e-10.
    tol = 1e-6
    pilot_path = HERE / "results/pilot/all22/summary.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else None
    if pilot and "c1" in s:
        c1 = c1_machine(labels["M1"])
        diffs = [abs(c1["seen"][m][k] - pilot["e1"][m][k]) for m in pilot["e1"] for k in ("median_q", "p90_q")]
        diffs += [abs(c1["unseen"][m][k] - pilot["e2_pooled"][m][k]) for m in pilot["e2_pooled"]
                  for k in ("median_q", "p90_q", "ci_lo", "ci_hi")]
        diffs += [abs(c1["unseen_paired_vs_cost"][m][k] - pilot["e2_paired_vs_cost"][m][k])
                  for m in pilot["e2_paired_vs_cost"] for k in ("gm", "gm_lo", "gm_hi", "sign_p", "wilcoxon_p")]
        check("C1's code on all of M1's queries reproduces the pilot's E1 and E2 (largest absolute difference)",
              f"{max(diffs):.1e}",
              max(diffs) < tol)
    if pilot and "c4" in s:
        pp, cp = pilot["e2_paired_vs_cost"][te.HYBRID], s["c4"]["paired"]
        diff = max(abs(pp[k] - cp[k]) for k in ("gm", "gm_lo", "gm_hi", "sign_p", "wilcoxon_p"))
        check("C4 on M1 reproduces the pilot's correction-model ratio to the cost line "
              f"({fmt(cp['gm'])} [{fmt(cp['gm_lo'])}, {fmt(cp['gm_hi'])}], largest absolute difference)",
              f"{diff:.1e}", diff < tol)
    for name, c in s["c2"].items():
        if c["descriptive"]:
            continue
        h = c["shift"]
        check(f"{name}: the shift depends on the template (template share above the pilot's between-pass share)",
              f"{pct(h['template_share'])} vs {pct(h['threshold'])}", h["accept"] is True)
        ks = sorted(int(k) for k in c["error"][NONE])
        err = {m: [c["error"][m][str(k)]["q"] for k in ks] for m in METHODS}
        check(f"{name}: speed factor beats no adaptation at every k",
              f"{fmt(max(err[SPEED]))} < {fmt(err[NONE][0])}", max(err[SPEED]) < err[NONE][0])
        big = [i for i, k in enumerate(ks) if k >= 25]
        check(f"{name}: ridge correction beats the speed factor for k >= 25",
              ", ".join(f"{fmt(err[RIDGE][i])} vs {fmt(err[SPEED][i])}" for i in big),
              all(err[RIDGE][i] < err[SPEED][i] for i in big))
        check(f"{name}: target only improves from the smallest to the largest k",
              f"{fmt(err[TARGET][0])} -> {fmt(err[TARGET][-1])}", err[TARGET][-1] < err[TARGET][0])
        check(f"{name}: oracle beats no adaptation", f"{fmt(err[ORACLE][0])} < {fmt(err[NONE][0])}",
              err[ORACLE][0] < err[NONE][0])
        r = shift_recovery(Pair(labels, split, c["source"], c["target"]), truth, orders, max(ks))
        check(f"{name}: ridge correction recovers the per-template slowdown (RMSE ridge vs speed factor, k = {r['k']})",
              f"{r['ridge_rmse']:.3f} vs {r['speed_rmse']:.3f}, correlation {r['ridge_corr']:.2f}",
              r["ridge_rmse"] < 0.5 * r["speed_rmse"] and r["ridge_corr"] > 0.8)
    if "c3_continued" in s:
        for name, c in s["c3_continued"].items():
            if s["c2"][name]["descriptive"]:
                continue
            a = c["error"][ALL_SEEN]
            check(f"{name}: refitting on all queries so far keeps improving to the whole pool",
                  f"{fmt(a[0])} -> {fmt(a[-1])}", a[-1] < a[0])
    if "c3_stream" in s:
        r = s["c3_stream"]["methods"]
        check("stream: the switch raises the batch error by more than 20%", fmt(r[NEVER]["jump"]),
              r[NEVER]["jump"] > 1.2)
        check("stream: refitting on the last 50 recovers within 60 queries of the switch",
              r[LAST]["recover_queries"], r[LAST]["recover_queries"] is not None and r[LAST]["recover_queries"] <= 60)
        check("stream: refitting on all so far settles higher than on the last 50 (the old machine's queries stay)",
              f"{fmt(r[ALL_SEEN]['final'])} vs {fmt(r[LAST]['final'])}", r[ALL_SEEN]["final"] > r[LAST]["final"])
    return checks


# ----------------------------------------------------------------------- CLI

def config(args) -> dict:
    parse = lambda text: [tuple(x.split(":")) for x in text.split(",") if x]  # noqa: E731
    pairs = parse(args.pairs)
    return {"machines": sorted({m for p in pairs for m in p} | set(args.stream.split(":"))),
            "pairs": [list(p) for p in pairs],
            "descriptive": [pair_name(*p) for p in parse(args.descriptive)],
            "primary": pair_name(*args.primary.split(":")), "primary_k": args.primary_k,
            "stream": args.stream.split(":"), "ks": [int(k) for k in args.ks.split(",")],
            "orders": args.orders, "hard_k": HARD_K, "hard_orders": args.hard_orders,
            "refit_every": REFIT_EVERY, "stream_orders": args.stream_orders, "batch": BATCH, "window": WINDOW,
            "gap_level": GAP_LEVEL, "recover_tolerance": RECOVER_TOLERANCE, "final_batches": FINAL_BATCHES,
            "shift_threshold": pilot_share_before(), "skip": sorted(x for x in args.skip.split(",") if x)}


def add_run_options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pairs", default="M1:M2,M1:M3,M3:M1", help="source:target pairs for C2 and C3 (a)")
    ap.add_argument("--descriptive", default="M3:M1", help="pairs reported as descriptive only")
    ap.add_argument("--primary", default="M1:M3", help="pair of the primary statistic")
    ap.add_argument("--primary-k", type=int, default=25)
    ap.add_argument("--stream", default="M1:M2", help="C3 (b): the stream's first and second machine")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--orders", type=int, default=ORDERS)
    ap.add_argument("--hard-orders", type=int, default=HARD_ORDERS)
    ap.add_argument("--stream-orders", type=int, default=STREAM_ORDERS)
    ap.add_argument("--skip", default="", help="comma list of components to skip: c1, hard, c3")
    ap.add_argument("--jobs", type=int, default=3, help="processes for the machines of C1 and the pairs of C2")
    ap.add_argument("--macro-prefix", default="", help="letters put in front of every macro name")
    ap.add_argument("--label", default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="evaluate the collected labels")
    run.add_argument("--runs", default=str(HERE / "results/runs"))
    run.add_argument("--split", default=str(HERE / "queries/tpch_sf1_v2/split.csv"))
    run.add_argument("--outdir", default=str(HERE / "results/final"))
    run.add_argument("--c4-labels", default=None, help="labels.csv of the C4 benchmark (TPC-DS on M1), if collected")
    add_run_options(run)
    dry = sub.add_parser("dryrun", help="build the synthetic dataset from the pilot and evaluate it")
    dry.add_argument("--pilot", default=str(HERE / "results/pilot/runtime_dataset.csv"))
    dry.add_argument("--outdir", default=str(HERE / "results/eval_dryrun"))
    dry.add_argument("--seed", type=int, default=0)
    add_run_options(dry)
    c4 = sub.add_parser("c4", help="C4 alone: leave one template group out on one benchmark")
    c4.add_argument("--labels", required=True)
    c4.add_argument("--outdir", required=True)
    c4.add_argument("--macro-prefix", default="")
    rep = sub.add_parser("report", help="rewrite tables, macros and figures from summary.json")
    rep.add_argument("--outdir", required=True)
    rep.add_argument("--macro-prefix", default="")
    args = ap.parse_args()
    out = Path(args.outdir)

    if args.command == "report":
        write_reports(json.loads((out / "summary.json").read_text(encoding="utf-8")), out, args.macro_prefix)
        return
    if args.command == "c4":
        clock = time.perf_counter()
        res, pred, by_template = c4_families(load_labels(Path(args.labels)))
        s = {"label": f"C4 on {shown_path(Path(args.labels))}", "c4": res,
             "seconds": {"total": time.perf_counter() - clock}}
        if by_template is not None:
            s["c4_by_template"] = by_template
        save_run(s, {"c4_predictions": pred}, out, args.macro_prefix)
        print((out / "results.md").read_text(encoding="utf-8"))
        return

    clock = time.perf_counter()
    cfg = config(args)
    truth = None
    if args.command == "dryrun":
        data = out / "data"
        truth = synthetic_runs(Path(args.pilot), data, args.seed)
        runs_dir, split_path, c4_path = data / "runs", data / "split.csv", data / "runs" / "M1" / "labels.csv"
    else:
        runs_dir, split_path = Path(args.runs), Path(args.split)
        c4_path = Path(args.c4_labels) if args.c4_labels else None
    labels = {m: load_labels(runs_dir / m / "labels.csv") for m in cfg["machines"]}
    split = load_split(split_path)
    s, frames = evaluate(labels, split, cfg, load_labels(c4_path) if c4_path else None, jobs=args.jobs)
    s = {"label": args.label or ("dry run on synthetic machines built from the pilot" if truth
                                 else f"labels in {shown_path(runs_dir)}, split {shown_path(split_path)}")} | s
    if truth is not None:
        t0 = time.perf_counter()
        orders = arrival_orders(split, cfg["orders"])
        s["dryrun"] = {"truth": truth, "checks": dryrun_checks(s, truth, labels, split, orders)}
        s["seconds"]["checks"] = time.perf_counter() - t0
    # Everything up to the reports: building the data, loading, evaluating and checking.
    s["seconds"]["evaluate"] = s["seconds"].pop("total")
    s["seconds"]["total"] = time.perf_counter() - clock
    save_run(s, frames, out, args.macro_prefix)
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
