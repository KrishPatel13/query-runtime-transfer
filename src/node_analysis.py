"""Which operators slow down between two machines: per-node times from scripts/node_timing.py.

Plans are identical on every machine, so the i-th node of a query's plan
(in pre-order) is the same operator on both. For every node this computes
its own time, the node's actual total time times its loops minus the same
for its children, both on machine A and on B, and reports per node type the
median ratio B / A, weighted by A's time, over the nodes that take at least
MIN_MS on A. Times inside a parallel part are summed over the leader and the
workers (time times loops), so they measure work, not elapsed time; the
Gather or Gather Merge node above them shows the elapsed time of the
parallel part, and its own time is its total time minus that of its child
divided by the processes that ran it.

Usage: python src/node_analysis.py results/runs/M1/node_times.jsonl results/runs/M3/node_times.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_MS = 1.0


def nodes(plan: dict) -> list[dict]:
    """Pre-order list of (node type, own time in ms, parallel flag) of one EXPLAIN ANALYZE plan."""
    out = []

    def walk(n: dict, parallel: bool) -> None:
        loops = float(n.get("Actual Loops", 1) or 1)
        total = float(n.get("Actual Total Time", 0.0) or 0.0) * loops
        kids = n.get("Plans", []) or []
        child = sum(float(c.get("Actual Total Time", 0.0) or 0.0) * float(c.get("Actual Loops", 1) or 1)
                    for c in kids)
        ntype = n.get("Node Type", "")
        if ntype in ("Gather", "Gather Merge"):
            # The child's time is summed over the processes that ran it; per process it is
            # child / processes, and the Gather's own time is what remains of its elapsed time.
            procs = float(n.get("Workers Launched", 0) or 0) + 1.0
            own = total - child / procs
        else:
            own = total - child
        name = ntype
        if n.get("Strategy") and ntype == "Aggregate":
            name = f"Aggregate ({n['Strategy']})"
        out.append({"type": name, "own_ms": max(own, 0.0), "parallel": parallel or bool(n.get("Parallel Aware"))})
        for c in kids:
            walk(c, parallel or ntype in ("Gather", "Gather Merge") or bool(n.get("Parallel Aware")))

    walk(plan[0]["Plan"], False)
    return out


def load(path: Path) -> dict:
    """query_id -> plan of a node_times.jsonl file (gzip-compressed when its name ends in .gz)."""
    out = {}
    if path.suffix == ".gz":
        import gzip
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok":
            out[rec["query_id"]] = rec["plan"]
    return out


def compare(a: dict, b: dict) -> pd.DataFrame:
    rows = []
    for q in sorted(set(a) & set(b)):
        na, nb = nodes(a[q]), nodes(b[q])
        if [x["type"] for x in na] != [x["type"] for x in nb]:
            continue
        ta = float(a[q][0]["Execution Time"])
        tb = float(b[q][0]["Execution Time"])
        for i, (x, y) in enumerate(zip(na, nb)):
            rows.append({"query_id": q, "template": q.split("_")[0], "node": i, "type": x["type"],
                         "parallel": x["parallel"], "a_ms": x["own_ms"], "b_ms": y["own_ms"],
                         "query_a_ms": ta, "query_b_ms": tb})
    return pd.DataFrame(rows)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["a_ms"] >= MIN_MS) & (df["b_ms"] > 0)].copy()
    d["ratio"] = d["b_ms"] / d["a_ms"]
    rows = []
    for (t, par), g in d.groupby(["type", "parallel"]):
        if len(g) < 5:
            continue
        rows.append({"type": t, "parallel": par, "nodes": len(g), "queries": g["query_id"].nunique(),
                     "share_of_a_time": g["a_ms"].sum() / d["a_ms"].sum(),
                     "median_ratio": weighted_median(g["ratio"].to_numpy(), g["a_ms"].to_numpy()),
                     "q25": float(g["ratio"].quantile(0.25)), "q75": float(g["ratio"].quantile(0.75))})
    return pd.DataFrame(rows).sort_values("share_of_a_time", ascending=False)


def main() -> None:
    a, b = (load(Path(p)) for p in sys.argv[1:3])
    df = compare(a, b)
    s = summarize(df)
    total = (df.groupby("query_id")[["query_a_ms", "query_b_ms"]].first())
    print(f"{len(df['query_id'].unique())} queries, {len(df)} nodes; query-level median ratio "
          f"{np.median(total['query_b_ms'] / total['query_a_ms']):.2f}")
    with pd.option_context("display.width", 160, "display.max_rows", 60):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
