"""Checks that two collection passes over the same queries agree.

A pass is one run of src/collect.py over a query set: runtime_dataset.csv
with one row per query (its label is the median of the timed runs), the meta
file, and the EXPLAIN plans in plans/<query_id>.json. The checks compare a
new pass with a reference pass (pass 1 in the full study):

  shape         the plan tree without run-specific fields: node types,
                relations, indexes, join types, estimated costs, rows, widths
                and planned workers. Two passes chose the same plan when the
                shapes are equal.
  change_point  the level-shift test. A pass can shift level partway through,
                for example when the laptop sleeps or Windows switches on its
                energy saver (in the pilot, every query after the 250th of
                pass 2 ran about 1.7 times slower). The test takes the log
                ratio of the new pass's labels to the reference labels in the
                new pass's execution order, and finds the single split that
                minimizes the squared error around the two segment means, with
                at least MIN_SEGMENT queries on each side. The split is a shift
                when the medians of the two segments differ by at least
                MIN_SHIFT (10%).
  agreement     how closely two passes' labels agree: the Q-error between
                them, the correlation of log runtimes, the rank correlation,
                and the correlation within templates.

compare() runs all three the way scripts/compare_passes.py reports them.
check_pass() runs shape and change_point with the standard library alone,
in the execution orders of both passes, to tell which of the two shifted.
The collection machines (WSL on the laptop, the UW server) have no numpy or
pandas, so src/run_passes.py checks each pass with check_pass(), and
agreement() and compare() import numpy, pandas and scipy only when called.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

MIN_SEGMENT = 30            # queries on each side of a reported shift
MIN_SHIFT = math.log(1.10)  # smallest level change reported as a shift


def shape(node: dict):
    """Plan tree without run-specific fields: node types, relations, costs, rows."""
    keep = ("Node Type", "Relation Name", "Index Name", "Join Type", "Strategy", "Parent Relationship",
            "Startup Cost", "Total Cost", "Plan Rows", "Plan Width", "Workers Planned")
    return ({k: node.get(k) for k in keep}, [shape(c) for c in node.get("Plans", []) or []])


def same_shape(a: dict | None, b: dict | None) -> bool:
    """Whether two plan roots have the same shape; a missing plan (None)
    never matches."""
    return a is not None and b is not None and shape(a) == shape(b)


def _sse(x: list[float]) -> float:
    m = sum(x) / len(x)
    return sum((v - m) ** 2 for v in x)


def shift_size(log_ratio, cut: int) -> float:
    """Median log ratio after the split minus the median before it."""
    x = [float(v) for v in log_ratio]
    return statistics.median(x[cut:]) - statistics.median(x[:cut])


def best_split(log_ratio) -> tuple[int | None, float]:
    """Split index minimizing the within-segment squared error (at least
    MIN_SEGMENT values on each side), and the share of the total squared
    error around the overall mean that the split removes (0 to 1). The
    index is None when the sequence is too short to split."""
    x = [float(v) for v in log_ratio]
    n = len(x)
    best, cut = math.inf, None
    for c in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        sse = _sse(x[:c]) + _sse(x[c:])
        if sse < best:
            best, cut = sse, c
    if cut is None:
        return None, 0.0
    total = _sse(x)
    return cut, 1 - best / total if total > 0 else 0.0


def change_point(log_ratio) -> int | None:
    """Split index minimizing the within-segment squared error, if the level
    change there is large enough to report.

    log_ratio is in execution order. The result c splits it into [:c] and
    [c:], so c is also the number of queries before the shift. This is the
    test compare_passes.py has always used, computed on plain floats so that
    it runs without numpy; tests/test_passcheck.py checks that it picks the
    same split as the numpy version it replaces.
    """
    x = [float(v) for v in log_ratio]
    cut, _ = best_split(x)
    if cut is None or abs(shift_size(x, cut)) < MIN_SHIFT:
        return None
    return cut


def agreement(g) -> dict:
    """Agreement of the two passes' labels on the queries in g.

    g holds two passes merged on query_id (columns runtime_ms_1, runtime_ms_2,
    template_1). log_corr is dominated by the differences between templates,
    which span two orders of magnitude. within_template_corr removes each
    template's mean log runtime first (separately per pass), so it measures
    whether the small differences between parameterizations of one template
    repeat.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    r1, r2 = g.runtime_ms_1.to_numpy(), g.runtime_ms_2.to_numpy()
    ratio = r2 / r1
    q = np.maximum(ratio, 1 / ratio)
    l1, l2 = np.log(r1), np.log(r2)
    d1 = l1 - pd.Series(l1).groupby(g.template_1.to_numpy()).transform("mean").to_numpy()
    d2 = l2 - pd.Series(l2).groupby(g.template_1.to_numpy()).transform("mean").to_numpy()
    return {"n": int(len(q)), "median_q": float(np.median(q)), "p90_q": float(np.quantile(q, 0.9)),
            "max_q": float(q.max()), "median_ratio": float(np.median(ratio)),
            "ratio_p10": float(np.quantile(ratio, 0.1)), "ratio_p90": float(np.quantile(ratio, 0.9)),
            "log_corr": float(np.corrcoef(l1, l2)[0, 1]),
            "spearman": float(spearmanr(r1, r2).statistic),
            "within_template_corr": float(np.corrcoef(d1, d2)[0, 1])}


def compare(a, b, plans_a: Path, plans_b: Path) -> dict:
    """Compare pass b with the reference pass a, as compare_passes.py reports it.

    a and b are the two runtime_dataset.csv files as data frames, plans_a and
    plans_b their plan directories. Only queries with status ok in both passes
    count. Returns how many were compared and how many chose the same plan,
    the agreement over all of them, the template with the worst median
    Q-error, the longest gap between two consecutive queries of b, and the
    shift in b's execution order with the agreement before and after it
    (None when there is no shift).
    """
    import numpy as np
    import pandas as pd

    m = a.merge(b, on="query_id", suffixes=("_1", "_2"))
    m = m[(m.status_1 == "ok") & (m.status_2 == "ok")].sort_values("order_2").reset_index(drop=True)
    same_plan = sum(same_shape(read_plan(plans_a, qid), read_plan(plans_b, qid)) for qid in m.query_id)

    t2 = pd.to_datetime(m.started_at_2)
    gaps = t2.diff().dt.total_seconds().fillna(0)
    log_ratio = np.log(m.runtime_ms_2 / m.runtime_ms_1).to_numpy()
    cut = change_point(log_ratio)
    shift = None
    if cut is not None:
        before, after = m.iloc[:cut], m.iloc[cut:]
        shift = {"after_query_number": int(cut), "at": str(m.started_at_2.iloc[cut]),
                 "before": agreement(before), "after": agreement(after),
                 "share_slower_after": float((after.runtime_ms_2 > after.runtime_ms_1).mean())}

    q = np.exp(np.abs(log_ratio))
    per_t = m.assign(q=q).groupby("template_1").q.median()
    return {
        "n_compared": int(len(m)),
        "same_plan": int(same_plan),
        "all": agreement(m),
        "worst_template_median_q": {"template": int(per_t.idxmax()), "q": float(per_t.max())},
        "longest_gap_s": float(gaps.max()),
        "shift": shift,
    }


# ------------------------------------------------ standard library only

def read_meta(d: Path) -> list[dict]:
    """The runs recorded in a pass's meta file (collect.py appends one per
    run, so a resumed pass has several); empty if there is no meta file."""
    p = Path(d) / "runtime_dataset.meta.json"
    if not p.exists():
        return []
    meta = json.loads(p.read_text(encoding="utf-8"))
    return meta if isinstance(meta, list) else [meta]


def read_pass(d: Path) -> dict[str, dict]:
    """Rows of a pass's runtime_dataset.csv by query_id, as strings."""
    with open(Path(d) / "runtime_dataset.csv", newline="", encoding="utf-8") as f:
        return {r["query_id"]: r for r in csv.DictReader(f)}


def read_plan(plan_dir: Path, qid: str) -> dict | None:
    """Root node of a saved EXPLAIN (FORMAT JSON) plan, or None if not saved."""
    p = Path(plan_dir) / f"{qid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))[0]["Plan"]


def macro_prefix(machine: str) -> str:
    """Prefix for the LaTeX macros of a machine. Macro names cannot hold
    digits, so M1 becomes MOne."""
    digits = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    name = "".join(digits[int(c)] if c.isdigit() else c for c in machine if c.isascii() and c.isalnum())
    return name[:1].upper() + name[1:]


def quantile(x: list[float], p: float) -> float:
    """Quantile with linear interpolation (numpy's default method)."""
    s = sorted(x)
    h = (len(s) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def _shift_in(ids: list[str], rows: dict[str, dict], log_ratio: list[float]) -> tuple[dict | None, float]:
    """change_point in one execution order: the shift (None if there is
    none) and the share of the squared error its best split explains."""
    _, explained = best_split(log_ratio)
    cut = change_point(log_ratio)
    if cut is None:
        return None, explained
    return {"after_query_number": cut, "of": len(ids), "at": rows[ids[cut]]["started_at"],
            "level_change": math.exp(shift_size(log_ratio, cut)),
            "median_ratio_before": math.exp(statistics.median(log_ratio[:cut])),
            "median_ratio_after": math.exp(statistics.median(log_ratio[cut:])),
            "explained": explained}, explained


def check_pass(ref_dir: Path, new_dir: Path) -> dict:
    """The level-shift test for one new pass against the reference pass.

    Queries with status ok in both passes are put in the new pass's execution
    order (its order column), and change_point is applied to
    log(new label / reference label). The result has the number of queries
    compared, how many chose the same plan, the median and 90th-percentile
    Q-error between the passes, and shift: None, or the number of queries
    before the shift, the start time of the first query after it, and
    level_change, the factor between the median ratios after and before it.

    shift_in_reference_order is the same test in the reference pass's
    execution order: the new pass runs in an unrelated order, so a shift
    inside the reference pass shows up only there. A real shift in one
    pass's order appears in the other pass's order as a random mix of two
    levels, and when the mix is near half and half, the medians of two
    segments of it can differ by more than 10% as well. So when both orders
    report a shift, only the one whose best split explains the larger share
    of the squared error (split_explained) is kept: a real step explains
    most of it, a split of a random mix very little.
    """
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    ref, new = read_pass(ref_dir), read_pass(new_dir)

    def usable(r: dict | None) -> bool:
        return r is not None and r["status"] == "ok" and float(r["runtime_ms"]) > 0

    ids = sorted((q for q, r in new.items() if usable(r) and usable(ref.get(q))), key=lambda q: int(new[q]["order"]))
    log_ratio = [math.log(float(new[q]["runtime_ms"]) / float(ref[q]["runtime_ms"])) for q in ids]
    same_plan = sum(same_shape(read_plan(ref_dir / "plans", q), read_plan(new_dir / "plans", q)) for q in ids)
    qerr = [math.exp(abs(v)) for v in log_ratio]
    shift, explained = _shift_in(ids, new, log_ratio)
    by_ref = sorted(range(len(ids)), key=lambda k: int(ref[ids[k]]["order"]))
    ref_shift, ref_explained = _shift_in([ids[k] for k in by_ref], ref, [log_ratio[k] for k in by_ref])
    if shift and ref_shift:
        if ref_explained > explained:
            shift = None
        else:
            ref_shift = None
    return {"reference": ref_dir.name, "pass": new_dir.name, "n_compared": len(ids), "same_plan": same_plan,
            "median_q": statistics.median(qerr) if qerr else None,
            "p90_q": quantile(qerr, 0.9) if qerr else None,
            "shift": shift, "shift_in_reference_order": ref_shift,
            "split_explained": {"pass": explained, "reference": ref_explained}}
