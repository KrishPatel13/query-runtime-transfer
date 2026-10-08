"""Train and evaluate runtime predictors on a collected dataset.

Experiments
  E1  within-template: each template's queries are split 70/30, 5 seeds.
  E2  held-out template: leave one whole template out, one fold per template.
  E3  learning curve over training queries, on the E1 splits.
  E4  learning curve over distinct training templates: train on k templates,
      test on all the others; every possible set for k = 1 and k = T - 1,
      20 different random sets for other k.

Predictors (all fit log runtime; errors are measured in milliseconds)
  Cost-only      least-squares line from log(1 + PostgreSQL total cost)
  Ridge          standardized plan features (log1p on the heavy-tailed ones),
                 alpha chosen on the training set by cross-validation that
                 mirrors the test: leave one query out when every template is
                 seen (E1, E3), leave one training template out when the test
                 template is unseen (E2, E4; with a single training template,
                 E4 at k = 1, it falls back to leaving one query out)
  Random Forest  300 trees on the same features
Exploratory (not one of the three proposed models)
  Cost + RF residual  the Cost-only line plus a Random Forest fitted to its
                      residuals, a first try at keeping the cost model's
                      behaviour on new templates while learning corrections
References (constants that ignore the plan)
  Template mean  E1: geometric-mean runtime of the training queries from the
                 same template, i.e. what knowing the template alone is worth
  Global mean    E2 and E4: geometric-mean runtime of all training queries,
                 i.e. what a model gets without looking at the plan at all

Usage
  python src/train_eval.py --data results/pilot/runtime_dataset.csv --outdir results/pilot/all22
  python src/train_eval.py --data results/pilot/runtime_dataset.csv --templates 1,3,5,6,14 --outdir results/pilot/proposed5
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, LOG_FEATURES, extract_features

# On Python 3.14, scikit-learn 1.9 warns about its own joblib wrapper on every
# forest fit. The warning is raised inside worker threads, where the main
# thread's filters do not apply, so drop it at display time instead.
_show_warning = warnings.showwarning


def _quiet(message, category, filename, lineno, file=None, line=None):
    if "sklearn.utils.parallel.delayed" not in str(message):
        _show_warning(message, category, filename, lineno, file, line)


warnings.showwarning = _quiet

MODELS = ["Cost-only", "Ridge", "Random Forest"]
HYBRID = "Cost + RF residual"
ALL_MODELS = MODELS + [HYBRID]
REFERENCE = "Template mean"
GLOBAL = "Global mean"
SEEDS = range(5)
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
DRAWS = 20
BOOTSTRAP = 2000
# Chosen to contain both tuning rules' choices; summary.json records the range
# actually picked ("ridge_alpha"), so a choice at an end point is visible.
ALPHAS = np.logspace(-6, 5, 45)
# (experiment, tuning rule, chosen alpha) for every Ridge fit
CHOSEN_ALPHAS: list[tuple[str, str, float]] = []
# Predictions are clipped to 1 microsecond .. 1e8 ms (about 28 hours) in log
# space, because a linear model extrapolating to an unseen template can
# otherwise overflow exp(). results.md reports how many predictions hit a bound.
LOG_MIN, LOG_MAX = math.log(1e-3), math.log(1e8)

# The models take the first slots of a colour-blind-checked categorical palette
# in fixed order; the reference and all chart chrome stay neutral.
COLORS = {"Cost-only": "#2a78d6", "Ridge": "#eb6834", "Random Forest": "#1baf7a", HYBRID: "#eda100",
          REFERENCE: "#898781", GLOBAL: "#898781"}
MARKERS = {"Cost-only": "o", "Ridge": "s", "Random Forest": "^", HYBRID: "v", REFERENCE: "D", GLOBAL: "D"}
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"


# ----------------------------------------------------------------- modelling

def design(df: pd.DataFrame, model: str) -> np.ndarray:
    if model == "Cost-only":
        return np.log1p(df[["total_cost"]].to_numpy(float))
    X = df[FEATURE_NAMES].to_numpy(float).copy()
    for j, name in enumerate(FEATURE_NAMES):
        if name in LOG_FEATURES:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
    return X


def forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                 random_state=seed, n_jobs=-1)


def make_model(name: str, seed: int, group_cv: bool = False):
    if name == "Cost-only":
        return LinearRegression()
    if name == "Ridge":
        if group_cv:
            return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                                cv=LeaveOneGroupOut(), scoring="neg_mean_squared_error")
        return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return forest(seed)


def to_ms(log_pred: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(log_pred, LOG_MIN, LOG_MAX))


def predict_all(train: pd.DataFrame, test: pd.DataFrame, seed: int, reference: bool,
                unseen: bool = False, tag: str = "") -> dict:
    """Fit every model on train and predict test.

    unseen=True means the test templates are absent from train (E2, E4); Ridge
    then tunes its penalty by leaving out one training template at a time.
    """
    y = np.log(train["runtime_ms"].to_numpy(float))
    group_cv = unseen and train["template"].nunique() >= 2
    logs = {}
    for m in MODELS:
        model = make_model(m, seed, group_cv=group_cv)
        if m == "Ridge" and group_cv:
            model.fit(design(train, m), y, groups=train["template"].to_numpy())
            CHOSEN_ALPHAS.append((tag, "leave one template out", model.best_params_["ridge__alpha"]))
        else:
            model.fit(design(train, m), y)
            if m == "Ridge":
                CHOSEN_ALPHAS.append((tag, "leave one query out", model[-1].alpha_))
        logs[m] = model.predict(design(test, m))
    cost = make_model("Cost-only", seed).fit(design(train, "Cost-only"), y)
    resid = y - cost.predict(design(train, "Cost-only"))
    rf = forest(seed).fit(design(train, HYBRID), resid)
    logs[HYBRID] = logs["Cost-only"] + rf.predict(design(test, HYBRID))
    out = {m: to_ms(v) for m, v in logs.items()}
    if reference:
        gm = np.exp(np.log(train["runtime_ms"]).groupby(train["template"]).mean())
        out[REFERENCE] = test["template"].map(gm).to_numpy(float)
    if unseen:
        out[GLOBAL] = np.full(len(test), np.exp(y.mean()))
    return out


def records(experiment: str, fold: str, test: pd.DataFrame, preds: dict, **extra) -> list[pd.DataFrame]:
    return [pd.DataFrame({
        "experiment": experiment, "fold": fold, "model": m,
        "query_id": test["query_id"].to_numpy(), "template": test["template"].to_numpy(),
        "y": test["runtime_ms"].to_numpy(float), "yhat": p, **extra,
    }) for m, p in preds.items()]


def qerror(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.maximum(np.asarray(p, float), 1e-9)
    return np.maximum(y / p, p / y)


def metrics(g: pd.DataFrame) -> pd.Series:
    q = qerror(g["y"], g["yhat"])
    return pd.Series({
        "median_q": float(np.median(q)),
        "p90_q": float(np.quantile(q, 0.9)),
        "log_mae": float(np.mean(np.log(q))),   # mean |log y - log yhat|
        "within_2x": float(np.mean(q <= 2.0)),
        "n": float(len(q)),
    })


def within_split(df: pd.DataFrame, seed: int, frac: float = 0.7):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for _, g in df.groupby("template", sort=True):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        # A template with a single query goes to training; it cannot be tested
        # as a seen template.
        n = len(idx) if len(idx) == 1 else min(len(idx) - 1, max(1, round(frac * len(idx))))
        tr.extend(idx[:n])
        te.extend(idx[n:])
    return df.loc[tr], df.loc[te]


def template_bootstrap(g: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for the pooled median Q-error, resampling whole templates.

    Queries from one template are strongly correlated, so resampling
    individual queries would give intervals that are far too narrow.
    """
    q = qerror(g["y"], g["yhat"])
    by_t = [q[(g["template"] == t).to_numpy()] for t in sorted(g["template"].unique())]
    stats = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(by_t), len(by_t))
        stats.append(np.median(np.concatenate([by_t[i] for i in pick])))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired_vs_cost(e2: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Each model against the cost line on the same held-out templates (E2).

    Both models are scored on identical templates, so the comparison resamples
    templates once for both. Two summaries, below 1 meaning better than the
    cost line:
      gm     geometric mean over templates of the ratio of the two models'
             per-template median Q-errors, with a percentile interval from
             resampling templates. This is the primary statistic: a mean is
             smooth, so its percentile interval is well behaved.
      ratio  ratio of the pooled median Q-errors. A ratio of medians is not
             smooth, and its percentile interval can sit far from the point
             estimate (the global mean's does), so it is secondary.
    The sign test counts held-out templates where the model's median Q-error
    is lower; the Wilcoxon signed-rank test on the per-template log ratios also
    weighs how large each difference is. Both treat templates as independent,
    although leave-one-template-out training sets overlap in all but one
    template, so their p-values are optimistic. The intervals resample fixed
    predictions and do not refit, so they also leave out training variability.
    """
    q = e2.assign(q=qerror(e2["y"], e2["yhat"]))
    templates = sorted(q["template"].unique())
    by = {m: [g.loc[g["template"] == t, "q"].to_numpy() for t in templates] for m, g in q.groupby("model")}
    picks = [rng.integers(0, len(templates), len(templates)) for _ in range(BOOTSTRAP)]
    base = by["Cost-only"]
    pooled = lambda parts, idx: np.median(np.concatenate([parts[i] for i in idx]))  # noqa: E731
    out = {}
    for m in ALL_MODELS + [GLOBAL]:
        if m == "Cost-only" or m not in by:
            continue
        ratios = [pooled(by[m], p) / pooled(base, p) for p in picks]
        per_m, per_c = np.array([np.median(x) for x in by[m]]), np.array([np.median(x) for x in base])
        wins, n = int((per_m < per_c).sum()), int((per_m != per_c).sum())
        log_ratio = np.log(per_m / per_c)
        gm_draws = [log_ratio[p].mean() for p in picks]
        out[m] = {"gm": float(np.exp(log_ratio.mean())), "gm_lo": float(np.exp(np.quantile(gm_draws, 0.025))),
                  "gm_hi": float(np.exp(np.quantile(gm_draws, 0.975))),
                  "ratio": float(pooled(by[m], range(len(templates))) / pooled(base, range(len(templates)))),
                  "ratio_lo": float(np.quantile(ratios, 0.025)), "ratio_hi": float(np.quantile(ratios, 0.975)),
                  "wins": wins, "n": n, "sign_p": float(binomtest(wins, n, 0.5).pvalue),
                  "wilcoxon_p": float(wilcoxon(log_ratio).pvalue) if np.any(log_ratio != 0) else 1.0}
    return out


def template_subsets(templates: list, k: int, draws: int, seed: int) -> list[set]:
    """Distinct training-template sets of size k: all of them when there are at
    most `draws` (always the case for k = 1 and k = T - 1), otherwise `draws`
    different ones chosen at random."""
    n_sets = math.comb(len(templates), k)
    if n_sets <= max(draws, len(templates)):
        return [set(c) for c in combinations(templates, k)]
    rng = np.random.default_rng(seed)
    seen: list[set] = []
    while len(seen) < draws:
        s = set(rng.choice(templates, size=k, replace=False).tolist())
        if s not in seen:
            seen.append(s)
    return seen


# --------------------------------------------------------------- experiments

def run_experiments(df: pd.DataFrame) -> pd.DataFrame:
    templates = sorted(df["template"].unique())
    recs: list[pd.DataFrame] = []

    for seed in SEEDS:
        tr, te = within_split(df, seed)
        recs += records("E1", f"seed{seed}", te, predict_all(tr, te, seed, reference=True, tag="E1"))

    for t in templates:
        tr, te = df[df["template"] != t], df[df["template"] == t]
        recs += records("E2", f"Q{t}", te, predict_all(tr, te, int(t), reference=False, unseen=True, tag="E2"))

    for seed in SEEDS:
        tr, te = within_split(df, seed)
        rng = np.random.default_rng(1000 + seed)
        # Shuffle each template once so smaller training sets are subsets of
        # larger ones; this keeps the curve from zig-zagging on sampling noise.
        order = {t: rng.permutation(g.index.to_numpy()) for t, g in tr.groupby("template")}
        for frac in FRACS:
            idx = np.concatenate([o[: max(1, math.ceil(frac * len(o)))] for o in order.values()])
            sub = tr.loc[idx]
            recs += records("E3", f"seed{seed}", te, predict_all(sub, te, seed, reference=False, tag="E3"),
                            frac=frac, n_train=len(sub))

    T = len(templates)
    ks = sorted({k for k in (1, 2, 3, 4, 6, 8, 12, 16, T - 1) if 1 <= k <= T - 1})
    for k in ks:
        for r, chosen in enumerate(template_subsets(templates, k, DRAWS, seed=2000 + k)):
            tr, te = df[df["template"].isin(chosen)], df[~df["template"].isin(chosen)]
            # With one template held out, use E2's forest seed (the held-out
            # template), so that E4 at k = T - 1 reproduces E2 exactly.
            seed = int(next(iter(set(templates) - set(chosen)))) if k == T - 1 else r
            recs += records("E4", f"k{k}_draw{r}", te,
                            predict_all(tr, te, seed, reference=False, unseen=True, tag=f"E4 k={k}"),
                            k=k, n_train=len(tr))

    return pd.concat(recs, ignore_index=True)


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def summarize(pred: pd.DataFrame) -> dict:
    keys = ["experiment", "fold", "model"] + [c for c in ("frac", "k") if c in pred]
    by_fold = pred.groupby(keys, dropna=False).apply(metrics, include_groups=False).reset_index()
    stats = ["median_q", "p90_q", "log_mae", "within_2x"]

    e1 = by_fold[by_fold.experiment == "E1"].groupby("model")[stats].agg(["mean", "std"])

    e2 = pred[pred.experiment == "E2"]
    e2_pooled = e2.groupby("model").apply(metrics, include_groups=False)
    # The same seed for every model: all models are scored on the same
    # resampled template sets, and adding a model does not move the others.
    ci = {m: template_bootstrap(g, np.random.default_rng(3000)) for m, g in e2.groupby("model")}
    e2_pooled["ci_lo"] = [ci[m][0] for m in e2_pooled.index]
    e2_pooled["ci_hi"] = [ci[m][1] for m in e2_pooled.index]
    e2_folds = by_fold[by_fold.experiment == "E2"]
    per_template = e2_folds.pivot(index="fold", columns="model", values="median_q")
    per_template.index = per_template.index.str.lstrip("Q").astype(int)
    per_template = per_template.sort_index()
    wins = {m: int((per_template[m] < per_template["Cost-only"]).sum())
            for m in ALL_MODELS + [GLOBAL] if m != "Cost-only" and m in per_template}
    worst = per_template.max()
    paired = paired_vs_cost(e2, np.random.default_rng(4000))

    e3 = by_fold[by_fold.experiment == "E3"].groupby(["frac", "model"])[["median_q", "p90_q"]].agg(["mean", "std"])
    n_train_e3 = pred[pred.experiment == "E3"].groupby("frac")["n_train"].mean()
    # E4 scores each unseen template separately and summarizes over all (draw,
    # unseen template) pairs. Pooling the queries of a draw instead would mix
    # 21 test templates at k=1 with a single one at k=T-1, so the statistic
    # would change with k.
    e4p = pred[pred.experiment == "E4"].copy()
    e4p["q"] = qerror(e4p["y"], e4p["yhat"])
    per_pair = e4p.groupby(["k", "fold", "template", "model"])["q"].median().rename("median_q").reset_index()
    e4 = per_pair.groupby(["k", "model"])["median_q"].agg(["median", q25, q75])

    logp = np.log(pred["yhat"].clip(lower=1e-300))
    at_bound = pred[np.isclose(logp, LOG_MIN) | np.isclose(logp, LOG_MAX)]
    clipped = at_bound.groupby(["experiment", "model"]).size().to_dict()

    return {"clipped": clipped, "by_fold": by_fold, "e1": e1, "e2_pooled": e2_pooled, "per_template": per_template,
            "wins_vs_cost": wins, "worst_template": worst, "paired": paired, "e3": e3, "n_train_e3": n_train_e3,
            "e4": e4}


def data_summary(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    runs = df["runs_ms"].apply(json.loads)
    # Largest factor between any timed run and the median label. Back-to-back
    # runs agree more closely than independent passes do, so this understates
    # the noise in a label; compare_passes.py measures the pass-to-pass part.
    spread = np.array([max(max(r / m, m / r) for r in rs) for rs, m in zip(runs, df["runtime_ms"])])
    cv = (df["runtime_std_ms"] / df["runtime_mean_ms"]).to_numpy()
    per_t = df.groupby("template")["runtime_ms"].agg(["count", "min", "median", "max"])
    per_t["max_over_min"] = per_t["max"] / per_t["min"]
    rel = np.log(df["runtime_ms"] / df.groupby("template")["runtime_ms"].transform("median"))
    return {
        "n_rows": int(len(raw)),
        "n_ok": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in raw["status"].value_counts().items()},
        "n_templates": int(df["template"].nunique()),
        "runtime_ms": {"min": float(df.runtime_ms.min()), "median": float(df.runtime_ms.median()),
                       "max": float(df.runtime_ms.max())},
        "repeat_cv": {"median": float(np.median(cv)), "p90": float(np.quantile(cv, 0.9)),
                      "max": float(np.max(cv))},
        "repeat_qerror_vs_median": {"median": float(np.median(spread)), "p90": float(np.quantile(spread, 0.9))},
        "within_template_max_over_min": {"median": float(per_t["max_over_min"].median()),
                                         "min": float(per_t["max_over_min"].min()),
                                         "max": float(per_t["max_over_min"].max())},
        "drift_corr_order_vs_log_relative_runtime": float(np.corrcoef(df["order"], rel)[0, 1]),
        "per_template": per_t.reset_index().to_dict(orient="records"),
        "total_timed_seconds": float(df["runs_ms"].apply(lambda s: sum(json.loads(s))).sum() / 1000.0),
    }


# ------------------------------------------------------------------- figures

def style_axes(ax, logx=False, logy=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, color=AXIS)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def qerror_axis(ax, axis="y", ticks=(1, 1.5, 2, 3, 5, 10, 20, 50, 100)):
    """Log-scaled Q-error axis with plain tick labels (1, 2, 5, 10, ...)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# The figures are drawn at the width they are printed (5.5 in, the text width
# of the course template), so these sizes are the printed point sizes.
FIG_W = 5.5
FS, FS_LABEL = 6.5, 7.5
DISPLAY = {"Cost-only": "Cost-only", "Ridge": "Ridge", "Random Forest": "Random forest", HYBRID: HYBRID,
           GLOBAL: "Global mean"}
TICK_LABELS = {"Cost-only": "Cost-only", "Ridge": "Ridge", "Random Forest": "Random\nforest",
               HYBRID: "Cost+RF\nresidual", REFERENCE: "Template\nmean", GLOBAL: "Global\nmean"}


def fig_data(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.1), gridspec_kw={"width_ratios": [1, 1.8]})
    ax = axes[0]
    style_axes(ax, logx=True, logy=True)
    ax.scatter(df["total_cost"], df["runtime_ms"], s=8, color=COLORS["Cost-only"], alpha=0.75,
               edgecolors="white", linewidths=0.3)
    ax.set_xlabel("PostgreSQL estimated total cost", color=INK, fontsize=FS_LABEL)
    ax.set_ylabel("measured runtime (ms)", color=INK, fontsize=FS_LABEL)
    ax.set_title("(a) Cost vs. runtime", color=INK, fontsize=FS_LABEL, loc="left")

    ax = axes[1]
    style_axes(ax, logy=True)
    templates = sorted(df["template"].unique())
    jitter = np.random.default_rng(0)
    for i, t in enumerate(templates):
        y = df.loc[df.template == t, "runtime_ms"].to_numpy()
        ax.scatter(i + jitter.uniform(-0.18, 0.18, len(y)), y, s=6, color=COLORS["Cost-only"], alpha=0.75,
                   edgecolors="white", linewidths=0.3)
    ax.set_xticks(range(len(templates)))
    ax.set_xticklabels([f"{t}" for t in templates], fontsize=5.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.set_xlabel("TPC-H template", color=INK, fontsize=FS_LABEL)
    ax.set_title("(b) Runtime within each template", color=INK, fontsize=FS_LABEL, loc="left")
    fig.tight_layout(w_pad=1.0)
    save(fig, out / "fig_data")


def fig_pred_vs_actual(pred: pd.DataFrame, out: Path):
    panels = [("E1", "seed0", "Seen templates (E1)"), ("E2", None, "Unseen template (E2)")]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 3.7), sharex=True, sharey=True)
    sub = pred[pred.experiment.isin(["E1", "E2"]) & pred.model.isin(MODELS)]
    lo = sub[["y", "yhat"]].min().min() / 1.5
    hi = sub[["y", "yhat"]].max().max() * 1.5
    grid = np.array([lo, hi])
    for r, (exp, fold, label) in enumerate(panels):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            style_axes(ax, logx=True, logy=True)
            g = sub[(sub.experiment == exp) & (sub.model == m)]
            if fold:
                g = g[g.fold == fold]
            ax.fill_between(grid, grid / 2, grid * 2, color=BAND, linewidth=0, zorder=0)
            ax.plot(grid, grid, color=MUTED, linewidth=0.8, zorder=1)
            ax.scatter(g["y"], g["yhat"], s=7, color=COLORS[m], marker=MARKERS[m], alpha=0.8,
                       edgecolors="white", linewidths=0.3, zorder=2)
            q = qerror(g["y"], g["yhat"])
            ax.text(0.04, 0.96, f"median Q {np.median(q):.2f}\nP90 Q {np.quantile(q, 0.9):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=FS, color=INK)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(DISPLAY[m], color=INK, fontsize=FS_LABEL)
            if c == 0:
                ax.set_ylabel(f"{label}\npredicted (ms)", color=INK, fontsize=FS_LABEL)
            if r == 1:
                ax.set_xlabel("measured (ms)", color=INK, fontsize=FS_LABEL)
    fig.tight_layout()
    save(fig, out / "fig_pred_vs_actual")


def _bars(ax, rows: list[tuple[str, float, float, float, float]]):
    """rows: (model, median, whisker low, whisker high, p90)."""
    for i, (m, med, lo, hi, p90) in enumerate(rows):
        ax.bar(i, med - 1, bottom=1, width=0.5, color=COLORS[m], linewidth=0)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color=INK2, capsize=2, linewidth=0.8)
        ax.scatter(i, p90, marker="_", s=70, color=INK, linewidths=1.2, zorder=3)
        ax.text(i + 0.28, med, f"{med:.2f}", ha="left", va="center", fontsize=FS, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([TICK_LABELS[r[0]] for r in rows], fontsize=FS)
    ax.set_xlim(-0.5, len(rows) - 0.2)


def fig_errors(s: dict, out: Path):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.95), sharey=True)
    e1 = s["e1"]
    rows = []
    for m in ALL_MODELS + [REFERENCE]:
        med, sd = e1.loc[m, ("median_q", "mean")], e1.loc[m, ("median_q", "std")]
        rows.append((m, med, med - min(sd, med - 1), med + sd, e1.loc[m, ("p90_q", "mean")]))
    style_axes(axes[0], logy=True)
    _bars(axes[0], rows)
    axes[0].set_title("(a) Seen templates (E1)", color=INK, fontsize=FS_LABEL, loc="left")
    axes[0].set_ylabel("Q-error (log scale)", color=INK, fontsize=FS_LABEL)

    e2 = s["e2_pooled"]
    rows = [(m, e2.loc[m, "median_q"], e2.loc[m, "ci_lo"], e2.loc[m, "ci_hi"], e2.loc[m, "p90_q"])
            for m in ALL_MODELS + [GLOBAL] if m in e2.index]
    style_axes(axes[1], logy=True)
    _bars(axes[1], rows)
    axes[1].set_title("(b) Unseen template (E2)", color=INK, fontsize=FS_LABEL, loc="left")
    axes[1].legend(handles=[Line2D([], [], marker="_", linestyle="none", color=INK, markersize=8,
                                   markeredgewidth=1.2, label="90th percentile")],
                   frameon=False, fontsize=FS, loc="upper left")
    top = max(e1[("p90_q", "mean")].max(), e2["p90_q"].max()) * 1.35
    for a in axes:
        a.set_ylim(1, top)
        qerror_axis(a, ticks=(1, 1.5, 2, 3, 5, 10, 20))
    fig.tight_layout(w_pad=0.8)
    save(fig, out / "fig_errors")


def fig_learning(s: dict, out: Path):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.95))
    ax = axes[0]
    style_axes(ax, logy=True)
    ntr = s["n_train_e3"]
    for m in ALL_MODELS:
        tab = s["e3"].xs(m, level="model")
        mean, sd = tab[("median_q", "mean")], tab[("median_q", "std")]
        x = ntr.loc[mean.index].to_numpy()
        ax.plot(x, mean.to_numpy(), color=COLORS[m], marker=MARKERS[m], markersize=3.5, linewidth=1.3,
                label=DISPLAY[m])
        ax.fill_between(x, (mean - sd).to_numpy(), (mean + sd).to_numpy(), color=COLORS[m], alpha=0.12,
                        linewidth=0)
    ax.set_xlabel("training queries (all templates seen)", color=INK, fontsize=FS_LABEL)
    ax.set_ylabel("median Q-error", color=INK, fontsize=FS_LABEL)
    ax.set_title("(a) Seen templates (E3)", color=INK, fontsize=FS_LABEL, loc="left")
    ax.set_ylim(1, None)
    qerror_axis(ax, ticks=(1, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5))

    ax = axes[1]
    style_axes(ax, logx=True, logy=True)
    for m in ALL_MODELS + [GLOBAL]:
        tab = s["e4"].xs(m, level="model")
        x = tab.index.to_numpy(float)
        dashed = {"linestyle": "--", "linewidth": 1.1, "markersize": 3} if m == GLOBAL else {}
        ax.plot(x, tab["median"], **({"color": COLORS[m], "marker": MARKERS[m], "markersize": 3.5,
                                      "linewidth": 1.3, "label": DISPLAY[m]} | dashed))
        if m == "Cost-only":
            ax.fill_between(x, tab["q25"], tab["q75"], color=COLORS[m], alpha=0.12, linewidth=0)
    ks = s["e4"].index.get_level_values("k").unique().to_numpy(float)
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("distinct training templates", color=INK, fontsize=FS_LABEL)
    ax.set_title("(b) Unseen templates (E4)", color=INK, fontsize=FS_LABEL, loc="left")
    ax.set_ylim(1.2, 12)
    qerror_axis(ax, ticks=(1.5, 2, 3, 5, 10))
    handles, labels = axes[0].get_legend_handles_labels()
    for h, lab in zip(*axes[1].get_legend_handles_labels()):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=FS,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, out / "fig_learning")


def fig_per_template(s: dict, out: Path):
    pt = s["per_template"]
    order = pt.sort_values("Cost-only").index
    fig, ax = plt.subplots(figsize=(3.3, 0.16 * len(order) + 0.9))
    style_axes(ax, logx=True)
    y = np.arange(len(order))
    for m in MODELS:
        ax.scatter(pt.loc[order, m], y, color=COLORS[m], marker=MARKERS[m], s=16, label=m,
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Q{t}" for t in order], fontsize=6.5)
    ax.set_xlabel("median Q-error on the held-out template", color=INK, fontsize=7.5)
    hi = max(10, float(pt[MODELS].max().max()) * 1.2)
    ticks = [t for t in (1, 2, 3, 5, 10, 30, 100, 300, 1000) if t <= hi]
    qerror_axis(ax, axis="x", ticks=ticks)
    ax.set_xlim(0.95, hi)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, out / "fig_per_template")


# ------------------------------------------------------------------- reports

def write_tables(s: dict, ds: dict, out: Path, label: str):
    e1, e2 = s["e1"], s["e2_pooled"]
    lines = [f"# Results: {label}", "",
             f"{ds['n_ok']} queries from {ds['n_templates']} templates. Runtime {ds['runtime_ms']['min']:.0f} to "
             f"{ds['runtime_ms']['max']:.0f} ms. Within a template the slowest query is at most "
             f"{ds['within_template_max_over_min']['max']:.2f}x the fastest (median "
             f"{ds['within_template_max_over_min']['median']:.2f}x). Repeat spread: the three back-to-back timed "
             f"runs lie within {ds['repeat_qerror_vs_median']['median']:.3f}x of their median (median over "
             f"queries), {ds['repeat_qerror_vs_median']['p90']:.3f}x at the 90th percentile. This is a lower bound "
             f"on label noise; scripts/compare_passes.py measures agreement between independent passes.", "",
             "Predictions clipped at a bound (see LOG_MIN, LOG_MAX): " +
             (", ".join(f"{e}/{m} {n}" for (e, m), n in sorted(s["clipped"].items())) or "none") + ".", "",
             "E1: mean ± std over 5 random 70/30 splits. E2: every query's held-out prediction pooled once; "
             "the 95% interval resamples whole templates (2000 draws).", "",
             "| Model | E1 median Q | E1 P90 Q | E1 log-MAE | E2 median Q [95% CI] | E2 P90 Q | E2 log-MAE | E2 within 2x |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ALL_MODELS + [REFERENCE, GLOBAL]:
        if m in e1.index:
            cells = [f"{e1.loc[m, (k, 'mean')]:.2f} ± {e1.loc[m, (k, 'std')]:.2f}"
                     for k in ("median_q", "p90_q", "log_mae")]
        else:
            cells = ["n/a"] * 3
        if m in e2.index:
            r = e2.loc[m]
            cells += [f"{r['median_q']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", f"{r['p90_q']:.2f}",
                      f"{r['log_mae']:.2f}", f"{r['within_2x']:.0%}"]
        else:
            cells += ["n/a"] * 4
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    n_t = len(s["per_template"])
    e2_models = [m for m in ALL_MODELS + [GLOBAL] if m in s["per_template"]]
    lines += ["", "E2 median over held-out templates of each template's median Q-error: " +
              ", ".join(f"{m} {s['per_template'][m].median():.2f}" for m in e2_models) + "."]
    lines += ["", "Held-out templates where the model's median Q-error beats Cost-only: " +
              ", ".join(f"{m} {v}/{n_t}" for m, v in s["wins_vs_cost"].items()) + ". Worst held-out template: " +
              ", ".join(f"{m} {s['worst_template'][m]:.2f}" for m in e2_models) + ".", ""]
    lines += ["Paired with Cost-only on the same held-out templates. Primary: geometric mean over templates of the "
              "ratio of per-template median Q-errors, with a 95% interval from resampling templates. Secondary: "
              "ratio of pooled median Q-errors (same resamples). Then the two-sided sign test and the Wilcoxon "
              "signed-rank test over templates (both treat the overlapping folds as independent): " +
              "; ".join(f"{m} GM {r['gm']:.2f} [{r['gm_lo']:.2f}, {r['gm_hi']:.2f}], pooled {r['ratio']:.2f} "
                        f"[{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}], {r['wins']}/{r['n']} lower, "
                        f"sign p = {r['sign_p']:.3f}, Wilcoxon p = {r['wilcoxon_p']:.3f}"
                        for m, r in s["paired"].items()) + ".", ""]
    if s.get("ridge_alpha"):
        lines += ["Ridge penalty chosen (min, median, max over fits; grid 1e-6 to 1e5): " +
                  "; ".join(f"{k} {v['min']:.2g}, {v['median']:.2g}, {v['max']:.2g} ({v['n']} fits, "
                            f"{v['at_grid_end']} at a grid end)"
                            for k, v in s["ridge_alpha"].items()) + ".", ""]
    lines += ["## E2 median Q-error per held-out template", "",
              "| Template | " + " | ".join(e2_models) + " |", "|---|" + "---|" * len(e2_models)]
    for t, row in s["per_template"].iterrows():
        lines.append(f"| Q{t} | " + " | ".join(f"{row[m]:.2f}" for m in e2_models) + " |")
    lines += ["", "## E3 learning curve (median Q-error, mean ± std over 5 splits)", "",
              "| Training queries | " + " | ".join(ALL_MODELS) + " |", "|---|" + "---|" * len(ALL_MODELS)]
    for frac in FRACS:
        cells = [f"{s['e3'].loc[(frac, m), ('median_q', 'mean')]:.2f} ± {s['e3'].loc[(frac, m), ('median_q', 'std')]:.2f}"
                 for m in ALL_MODELS]
        lines.append(f"| {s['n_train_e3'].loc[frac]:.0f} ({frac:.0%}) | " + " | ".join(cells) + " |")
    lines += ["", "## E4 template-diversity curve", "",
              "Each unseen template's median Q-error, summarized as median [IQR] over all (training set, unseen "
              "template) pairs. Every possible training set for k = 1 and k = T - 1, 20 random ones otherwise.",
              "",
              "| Training templates | " + " | ".join(ALL_MODELS + [GLOBAL]) + " |",
              "|---|" + "---|" * (len(ALL_MODELS) + 1)]
    for k in sorted(s["e4"].index.get_level_values("k").unique()):
        cells = []
        for m in ALL_MODELS + [GLOBAL]:
            r = s["e4"].loc[(k, m)]
            cells.append(f"{r['median']:.2f} [{r['q25']:.2f}, {r['q75']:.2f}]")
        lines.append(f"| {int(k)} | " + " | ".join(cells) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/pilot/runtime_dataset.csv")
    ap.add_argument("--outdir", default="results/pilot/all22")
    ap.add_argument("--templates", default=None, help="comma list to restrict to, e.g. 1,3,5,6,14")
    ap.add_argument("--label", default=None)
    ap.add_argument("--plans", default=None, help="saved EXPLAIN JSON (default: <data dir>/plans)")
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="summarize <outdir>/predictions.csv again instead of refitting every model")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    if args.templates:
        keep = {int(t) for t in args.templates.split(",")}
        raw = raw[raw["template"].isin(keep)]
    df = raw[raw["status"] == "ok"].reset_index(drop=True)
    df["template"] = df["template"].astype(int)
    plan_dir = Path(args.plans) if args.plans else Path(args.data).parent / "plans"
    if plan_dir.is_dir():
        # Recompute the features from the saved plans, so a change to
        # features.py never needs a new measurement run.
        feats = [extract_features(json.loads((plan_dir / f"{q}.json").read_text(encoding="utf-8")))
                 for q in df["query_id"]]
        df = df.drop(columns=[c for c in df.columns if c in FEATURE_NAMES or c in ("hash_count", "join_count")])
        df = pd.concat([df, pd.DataFrame(feats, index=df.index)[FEATURE_NAMES]], axis=1)
    label = args.label or f"{df['template'].nunique()} templates, {len(df)} queries"

    if args.reuse_predictions:
        pred = pd.read_csv(out / "predictions.csv")
    else:
        pred = run_experiments(df)
        pred.to_csv(out / "predictions.csv", index=False)
    s = summarize(pred)
    alpha_path = out / "ridge_alpha.json"
    if not args.reuse_predictions:
        ca = pd.DataFrame(CHOSEN_ALPHAS, columns=["tag", "rule", "alpha"])
        ca.to_csv(out / "ridge_alpha_all.csv", index=False)
        ca["at_end"] = np.isclose(ca["alpha"], ALPHAS[0]) | np.isclose(ca["alpha"], ALPHAS[-1])
        alphas = {f"{tag} ({rule})": {"min": float(g["alpha"].min()), "median": float(g["alpha"].median()),
                                      "max": float(g["alpha"].max()), "n": int(len(g)),
                                      "at_grid_end": int(g["at_end"].sum())}
                  for (tag, rule), g in ca.groupby(["tag", "rule"], sort=False)}
        alpha_path.write_text(json.dumps(alphas, indent=1) + "\n", encoding="utf-8")
    s["ridge_alpha"] = json.loads(alpha_path.read_text(encoding="utf-8")) if alpha_path.exists() else {}
    s["by_fold"].to_csv(out / "metrics_by_fold.csv", index=False)
    s["per_template"].to_csv(out / "e2_per_template.csv")
    ds = data_summary(raw, df)

    num = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    summary = {
        "label": label,
        "data": ds,
        "e1": {m: {k: float(s["e1"].loc[m, (k, "mean")]) for k in ("median_q", "p90_q", "log_mae", "within_2x")}
               | {k + "_std": float(s["e1"].loc[m, (k, "std")]) for k in ("median_q", "p90_q", "log_mae")}
               for m in ALL_MODELS + [REFERENCE]},
        "e2_pooled": {m: {k: float(v) for k, v in s["e2_pooled"].loc[m].items()}
                      for m in ALL_MODELS + [GLOBAL] if m in s["e2_pooled"].index},
        "e2_wins_vs_cost": s["wins_vs_cost"],
        "e2_worst_template": {m: float(s["worst_template"][m]) for m in ALL_MODELS + [GLOBAL]
                              if m in s["worst_template"]},
        "e2_paired_vs_cost": s["paired"],
        "ridge_alpha": s["ridge_alpha"],
        "e2_per_template": {int(t): {m: float(v) for m, v in row.items()} for t, row in s["per_template"].iterrows()},
        "e3": {f"{frac}": {m: float(s["e3"].loc[(frac, m), ("median_q", "mean")]) for m in ALL_MODELS}
               | {"n_train": float(s["n_train_e3"].loc[frac])} for frac in FRACS},
        "e4": {str(int(k)): {m: {c: float(s["e4"].loc[(k, m), c]) for c in ("median", "q25", "q75")}
                             for m in ALL_MODELS + [GLOBAL]}
               for k in sorted(s["e4"].index.get_level_values("k").unique())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=num) + "\n", encoding="utf-8")
    write_tables(s, ds, out, label)

    fig_data(df, out)
    fig_pred_vs_actual(pred, out)
    fig_errors(s, out)
    fig_learning(s, out)
    fig_per_template(s, out)
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
