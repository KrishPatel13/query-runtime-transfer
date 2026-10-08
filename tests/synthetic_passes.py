"""Synthetic collection passes for the tests: the files src/collect.py writes.

A pass directory holds runtime_dataset.csv (collect.py's columns), the meta
file and plans/<query_id>.json. write_pass() makes one from a dict of labels,
so tests can build shifted passes, failed queries and changed plans without a
database.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path

from collect import BASE_FIELDS
from features import FEATURE_NAMES, extract_features

HOST = {"hostname": "laptop", "cpu_model": "Test CPU", "cpu_count": 22, "kernel": "6.0-test"}
SETTINGS = {"shared_buffers": "262144", "work_mem": "65536", "port": "5432", "listen_addresses": "localhost"}


def make_plan(qid: str, cost: float | None = None, node: str = "Hash Join") -> dict:
    """A small plan tree; its costs depend on the query, as real ones do."""
    base = cost if cost is not None else 100.0 + 7 * sum(map(ord, qid))
    return {"Node Type": "Aggregate", "Strategy": "Plain", "Startup Cost": base, "Total Cost": base + 1,
            "Plan Rows": 1, "Plan Width": 8,
            "Plans": [{"Node Type": node, "Parent Relationship": "Outer", "Join Type": "Inner",
                       "Startup Cost": 1.0, "Total Cost": base, "Plan Rows": 500, "Plan Width": 8,
                       "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer",
                                  "Relation Name": "lineitem", "Startup Cost": 0.0, "Total Cost": base / 2,
                                  "Plan Rows": 6000, "Plan Width": 8},
                                 {"Node Type": "Seq Scan", "Parent Relationship": "Inner",
                                  "Relation Name": "orders", "Startup Cost": 0.0, "Total Cost": base / 4,
                                  "Plan Rows": 1500, "Plan Width": 8}]}]}


def template_of(qid: str) -> int:
    return int(re.match(r"q(\d+)", qid).group(1))


def write_pass(d: Path, labels: dict[str, float], order: list[str] | None = None,
               status: dict[str, str] | None = None, plans: dict[str, dict | None] | None = None,
               finished: bool = True, started_at: str = "2026-10-01T10:00:00", seed: int = 0,
               query_dir: str | None = None, host: dict | None = None, settings: dict | None = None,
               machine: str = "M1") -> Path:
    """Write one pass into d.

    labels: runtime_ms of every query with status ok. order: the execution
    order (default: sorted query IDs). status: queries that did not finish,
    e.g. {"q03_001": "timeout"}. plans: plan roots replacing make_plan(qid);
    None means no plan file (EXPLAIN failed).
    """
    d = Path(d)
    (d / "plans").mkdir(parents=True, exist_ok=True)
    status, plans = status or {}, plans or {}
    ids = order or sorted(set(labels) | set(status))
    t0 = dt.datetime.fromisoformat(started_at)
    rows = []
    for k, qid in enumerate(ids, 1):
        plan = plans.get(qid, make_plan(qid))
        row = {"query_id": qid, "template": template_of(qid), "status": status.get(qid, "ok"), "order": k,
               "started_at": (t0 + dt.timedelta(seconds=2 * k)).isoformat(timespec="seconds"), "stalls": 0,
               "load1": 0.5}
        if plan is not None:
            (d / "plans" / f"{qid}.json").write_text(json.dumps([{"Plan": plan}], indent=1), encoding="utf-8")
            row |= extract_features([{"Plan": plan}])
        if row["status"] == "ok":
            v = labels[qid]
            row |= {"runtime_ms": v, "runtime_mean_ms": v, "runtime_std_ms": 0.0, "runtime_min_ms": v,
                    "runtime_max_ms": v, "repeats": 3, "runs_ms": json.dumps([v, v, v]),
                    "warmup_ms": json.dumps([v]), "planning_ms": 0.2, "shared_read_blocks": 0, "wall_s": 0.1}
        rows.append(row)
    with open(d / "runtime_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BASE_FIELDS + FEATURE_NAMES)
        w.writeheader()
        w.writerows(rows)
    meta = [{"started_at": started_at, "finished_at": "2026-10-01T11:00:00" if finished else None,
             "machine": machine, "args": {"query_dir": query_dir, "seed": seed, "machine": machine},
             "server": {"version": "PostgreSQL 18.6 on x86_64-pc-linux-gnu", "settings": settings or SETTINGS},
             "host": host or HOST, "n_queries": len(ids)}]
    (d / "runtime_dataset.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return d
