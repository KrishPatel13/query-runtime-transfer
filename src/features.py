from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, Iterable

# Two counts from the original list are left out because PostgreSQL makes them
# exact copies of others: every Hash Join has exactly one Hash child, and the
# join count is the sum of the three join types. Exact duplicates add nothing
# and make the linear model's problem worse conditioned.
FEATURE_NAMES = [
    "startup_cost", "total_cost", "plan_rows", "plan_width",
    "node_count", "max_depth", "max_plan_rows", "sum_log1p_plan_rows",
    "scan_count", "seq_scan_count", "index_scan_count",
    "nested_loop_count", "hash_join_count", "merge_join_count",
    "sort_count", "aggregate_count", "materialize_count",
    "workers_planned", "subplan_count",
]

# Features spanning several orders of magnitude. Linear models see them on a
# log1p scale; tree models are unaffected by the transform.
LOG_FEATURES = ["startup_cost", "total_cost", "plan_rows", "plan_width", "max_plan_rows"]

def _walk(node: Dict[str, Any], depth: int = 0) -> Iterable[tuple[Dict[str, Any], int]]:
    yield node, depth
    for child in node.get("Plans", []) or []:
        yield from _walk(child, depth + 1)

def extract_features(explain_json: Any) -> Dict[str, float]:
    """Extract a small fixed feature vector from PostgreSQL EXPLAIN (FORMAT JSON).

    Accepts the decoded JSON returned by PostgreSQL. No ANALYZE-only fields are used.
    """
    if isinstance(explain_json, list):
        top = explain_json[0]
    else:
        top = explain_json
    root = top["Plan"]
    nodes = list(_walk(root))
    types = Counter(str(n.get("Node Type", "")) for n, _ in nodes)

    # Counts are of plan nodes. Every node type whose name ends in "Scan" is a
    # scan, including CTE Scan (Q15) and both halves of a bitmap scan, and a
    # parallel aggregate contributes its Partial and its Finalize node.
    sort_types = {"Sort", "Incremental Sort"}
    agg_types = {"Aggregate", "Group", "GroupAggregate", "HashAggregate"}

    plan_rows = [float(n.get("Plan Rows", 0.0) or 0.0) for n, _ in nodes]
    node_types = [str(n.get("Node Type", "")) for n, _ in nodes]

    feats = {
        "startup_cost": float(root.get("Startup Cost", 0.0) or 0.0),
        "total_cost": float(root.get("Total Cost", 0.0) or 0.0),
        "plan_rows": float(root.get("Plan Rows", 0.0) or 0.0),
        "plan_width": float(root.get("Plan Width", 0.0) or 0.0),
        "node_count": float(len(nodes)),
        "max_depth": float(max((d for _, d in nodes), default=0)),
        "max_plan_rows": float(max(plan_rows, default=0.0)),
        "sum_log1p_plan_rows": float(sum(math.log1p(max(0.0, x)) for x in plan_rows)),
        "scan_count": float(sum(t.endswith("Scan") for t in node_types)),
        "seq_scan_count": float(types["Seq Scan"]),
        "index_scan_count": float(types["Index Scan"] + types["Index Only Scan"] + types["Bitmap Heap Scan"] + types["Bitmap Index Scan"]),
        "nested_loop_count": float(types["Nested Loop"]),
        "hash_join_count": float(types["Hash Join"]),
        "merge_join_count": float(types["Merge Join"]),
        "sort_count": float(sum(t in sort_types for t in node_types)),
        "aggregate_count": float(sum(t in agg_types for t in node_types)),
        "materialize_count": float(types["Materialize"] + types["Memoize"]),
        "workers_planned": float(sum(float(n.get("Workers Planned", 0) or 0) for n, _ in nodes
                                     if n.get("Node Type") in ("Gather", "Gather Merge"))),
        "subplan_count": float(sum(n.get("Parent Relationship") in ("SubPlan", "InitPlan") for n, _ in nodes)),
    }
    return feats
