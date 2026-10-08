"""Statistics over templates for the adaptation and unseen-template comparisons.

Queries of one template are strongly correlated, so every comparison treats
the template as the unit. A method is summarized per test template by the
median Q-error of that template's test queries, in log. Two methods are
compared through the per-template log ratio, and intervals come from
resampling whole templates with replacement (a percentile bootstrap). Both
methods are always scored on the same resamples, so the comparison is paired.

The resamples are drawn exactly as train_eval.paired_vs_cost draws them, one
resample at a time from a seeded generator. Given the same per-template
errors and the same seed, paired_ratio therefore reproduces the pilot's
statistic, so the pilot and the final study use one computation.

Functions
  resamples         indices of the template bootstrap resamples
  resample_columns  the test queries each resample pools
  mean_interval     percentile interval for a mean over templates
  pooled_quantiles  pooled median and P90 Q-error over all test queries, with
                    percentile intervals from the same template resamples
  paired_ratio      geometric-mean ratio of two methods with its interval,
                    sign test and Wilcoxon signed-rank test (the primary
                    statistic of both questions)
  gap_closed        share of the gap between no adaptation and the oracle that
                    a method closes, with its interval
  first_reaching    smallest k at which a curve reaches a level
  recovery_index    first batch from which an error curve stays near its final
                    level
  template_share    share of a quantity's variance that lies between templates,
                    with the one-way ANOVA p-value (the acceptance test for a
                    target machine)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, f_oneway, wilcoxon

BOOTSTRAP = 2000
LEVEL = 0.95


def resamples(n: int, rng: np.random.Generator, draws: int = BOOTSTRAP) -> np.ndarray:
    """Indices of `draws` resamples of n templates with replacement, one per row.

    Drawn one resample at a time, as train_eval.paired_vs_cost does: a single
    call for all of them would consume the generator differently and give
    other resamples for the same seed.
    """
    return np.stack([rng.integers(0, n, n) for _ in range(draws)])


def resample_columns(groups: list[np.ndarray], picks: np.ndarray) -> list[np.ndarray]:
    """For every resample, the positions of the test queries it pools.

    groups[i] holds the positions of template i's test queries. A template
    drawn twice contributes its queries twice, as in
    train_eval.template_bootstrap.
    """
    return [np.concatenate([groups[i] for i in pick]) for pick in picks]


def mean_interval(values: np.ndarray, picks: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """Percentile interval for the mean of per-template values over the resamples."""
    draws = np.asarray(values, float)[picks].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def pooled_quantiles(q: np.ndarray, columns: list[np.ndarray], quantiles=(0.5, 0.9),
                     level: float = LEVEL) -> dict:
    """Pooled quantiles of the Q-error over all test queries, with template-resampling intervals.

    q holds the Q-errors of the same test queries under several arrival
    orders, one row per order (or a single row), and columns comes from
    resample_columns. For one quantile (0.5 for the median, 0.9 for P90),
    the statistic is the quantile over all test queries in each order,
    averaged in log over the orders, that is their geometric mean. Each
    resample recomputes it on the queries of the drawn templates, and the
    interval is the 2.5th and 97.5th percentile over the resamples. With a
    single row and train_eval's resamples this is
    train_eval.template_bootstrap for the median. Returns
    {quantile: (value, lo, hi)}.
    """
    q = np.atleast_2d(np.asarray(q, float))
    qs = list(quantiles)

    def stat(x):
        return np.exp(np.log(np.quantile(x, qs, axis=1)).mean(axis=1))

    point = stat(q)
    draws = np.array([stat(q[:, cols]) for cols in columns])
    tail = (1 - level) / 2
    lo, hi = np.quantile(draws, tail, axis=0), np.quantile(draws, 1 - tail, axis=0)
    return {x: (float(point[i]), float(lo[i]), float(hi[i])) for i, x in enumerate(qs)}


def paired_ratio(log_a: np.ndarray, log_b: np.ndarray, picks: np.ndarray) -> dict:
    """Compare method a with method b on the same templates.

    log_a and log_b hold one value per template in the same order: the log of
    the template's median Q-error (averaged over arrival orders where there
    are several). With d = log_a - log_b:
      gm           exp(mean d), the geometric mean over templates of the ratio
                   of the two per-template median Q-errors; below 1 means a is
                   more accurate
      gm_lo, gm_hi 95% percentile interval of exp(mean d) over the template
                   resamples
      success      the whole interval lies below 1, the decision rule of the
                   primary comparisons
      wins, n      templates where a is lower, and templates where the two
                   differ; sign_p is the two-sided sign test on them
      wilcoxon_p   Wilcoxon signed-rank test on d, which also weighs how large
                   each difference is
    The two tests treat templates as independent and are descriptive only.
    """
    d = np.asarray(log_a, float) - np.asarray(log_b, float)
    lo, hi = mean_interval(d, picks)
    wins, n = int((d < 0).sum()), int((d != 0).sum())
    return {"gm": float(np.exp(d.mean())), "gm_lo": float(np.exp(lo)), "gm_hi": float(np.exp(hi)),
            "success": bool(np.exp(hi) < 1.0), "wins": wins, "n": n, "templates": int(len(d)),
            "sign_p": float(binomtest(wins, n, 0.5).pvalue) if n else 1.0,
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def gap_closed(log_none: np.ndarray, log_method: np.ndarray, log_oracle: np.ndarray,
               picks: np.ndarray) -> dict:
    """Share of the gap to the oracle that a method closes.

    With E the mean over test templates of the log per-template median
    Q-error, G = (E_none - E_method) / (E_none - E_oracle): 0 means no better
    than leaving the source model unchanged, 1 means as accurate as the oracle
    trained on the whole target pool, and values above 1 beat the oracle. The
    interval recomputes G on the same template resamples. G is undefined (NaN)
    when the oracle is not better than no adaptation.
    """
    a, m, o = (np.asarray(v, float) for v in (log_none, log_method, log_oracle))

    def g(num, den):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    point = float(g(a.mean() - m.mean(), a.mean() - o.mean()))
    draws = g(a[picks].mean(axis=1) - m[picks].mean(axis=1), a[picks].mean(axis=1) - o[picks].mean(axis=1))
    draws = draws[np.isfinite(draws)]
    tail = (1 - LEVEL) / 2
    lo, hi = (float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))) if draws.size else (np.nan, np.nan)
    return {"g": point, "g_lo": lo, "g_hi": hi}


def first_reaching(ks: list[int], values: list[float], level: float) -> int | None:
    """Smallest k whose value is at least `level`, or None if no k reaches it."""
    for k, v in zip(ks, values):
        if np.isfinite(v) and v >= level:
            return int(k)
    return None


def recovery_index(errors: np.ndarray, final: float, tolerance: float = 1.10) -> int | None:
    """First position from which every later error is at most tolerance x final.

    errors are the batch errors after a change, in stream order, and final is
    the level the curve settles at. Returns 0 when the curve is within the
    band from the first batch on, and None when its last batch is still
    outside it.
    """
    outside = np.flatnonzero(np.asarray(errors, float) > tolerance * final)
    if outside.size == 0:
        return 0
    first = int(outside[-1]) + 1
    return first if first < len(errors) else None


def template_share(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Share of the variance of values between groups, and the one-way ANOVA p-value.

    The share is the between-group sum of squares over the total sum of
    squares. The ANOVA uses the groups with at least two values. Applied to
    log(target label / source label) grouped by template, this is the
    acceptance test for a target machine in FINAL_PROPOSAL_PLAN.md (section
    3), computed as scripts/pilot_adaptation.py computes it for the pilot's
    slowdown. Both are NaN when the values do not vary. The p-value
    underflows to exactly 0 for a strong template effect on many queries
    (on 680 queries in 22 templates, a share above about 0.9); the reports
    then show it as a bound.
    """
    v = np.asarray(values, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=v) / counts
    total = float(((v - v.mean()) ** 2).sum())
    if not total > 1e-12 * len(v):
        return float("nan"), float("nan")
    between = float(((means[inv] - v.mean()) ** 2).sum())
    parts = [v[inv == i] for i in np.flatnonzero(counts > 1)]
    return between / total, float(f_oneway(*parts).pvalue)
