"""Sanity check of the widened TPC-H workload (queries/tpch_sf1_v2) on M1.

This is not a label collection: the widened queries are timed only to show
that the workload is usable before the real passes.

1. EXPLAIN (FORMAT JSON) of all 680 queries: each must be one statement that
   PostgreSQL plans. The SHA-1 of each query's text is stored with its plan,
   so tests/test_widened.py can tell whether these results still belong to
   the query files. A standard query's plan is also compared with the plan
   the pilot saved (results/pilot/plans) on the fields that
   scripts/compare_passes.py compares (node tree, relations, estimated costs
   and rows). Identical plans show that the database and its statistics are
   unchanged since the pilot.
2. Every widened query runs once untimed and then once timed, with a
   statement timeout of 120 s, in a seeded random order. The timing is
   src/collect.py's: EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON) and
   the server-side execution time. The untimed run comes first because the
   first execution after the cluster starts reads the tables from disk, which
   would hide the spread that the check is about. Both times are kept, and
   the 1-minute load average is logged with every query, since other work
   can share the laptop.
3. How far Q11's parameters can move its runtime. Q11 has only a nation and
   a fraction, and the fraction only decides how many of the aggregated
   parts are kept and sorted, so it is the one template that may not allow
   a factor of 2. Q11 runs with the fraction at 0 (every part kept) and at 1
   (none kept) for each of the 25 nations in qgen's list, and every widened
   Q11 query runs too, each once untimed and then Q11_REPEATS times timed
   (median). Keeping more parts only adds rows to sort, so every Q11 query
   runs between the two ends of its nation, and the spread over the 50 end
   runs is the most any Q11 query can reach on M1, up to noise. The rows
   returned by the plan's root are the parts kept.
4. Summary per template: the runtime range of the standard queries in pilot
   pass 1 (results/pilot/runtime_dataset.csv), the runtime range of the
   widened queries from step 2, the range of PostgreSQL's estimated total
   cost and the number of distinct plan trees for both, and the range of
   every parameter in the manifest.

The workload passes when every query plans, every widened query finishes
within 60 s (both runs), and the widened queries of a template span a wider
runtime range than its standard ones, by at least a factor of 2 between the
fastest and the slowest where the template allows it (for Q11, step 3 shows
what it allows).

Writes results/workload_check/explain.csv, widened_runtimes.csv,
q11_bounds.csv, run.meta.json, README.md (generated, with the tables) and
workload_numbers.tex (LaTeX macros, so documents never quote these numbers by
hand).

Usage, inside WSL from project_proposal after "source ~/cs680-pilot/pg.env":
  python3 scripts/check_workload.py              # steps 1 to 3, then the summary
  python3 scripts/check_workload.py --summary    # only the summary, from the saved files
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from collect import QueryError, execute_timed, explain, host_info, run_psql, server_info  # noqa: E402
from generate_widened_queries import NATIONS, WIDENED_TEMPLATES, kind, render, write_value  # noqa: E402

QUERY_DIR = HERE / "queries/tpch_sf1_v2"
PILOT = HERE / "results/pilot"
OUT = HERE / "results/workload_check"
TIMEOUT_S = 120
LIMIT_S = 60
MIN_SPREAD = 2.0
SEED = 0
Q11_REPEATS = 3
Q11_ENDS = {"fraction 0": 0.0, "fraction 1": 1.0}   # every part kept, no part kept
EXPLAIN_FIELDS = ["query_id", "template", "variant", "sql_sha1", "status", "total_cost", "plan_rows",
                  "node_count", "plan_tree", "same_plan_as_pilot"]
RUN_FIELDS = ["query_id", "template", "order", "started_at", "load_1min", "status", "runtime_ms", "warmup_ms",
              "planning_ms", "wall_s"]
Q11_FIELDS = ["case", "query_id", "nation", "fraction", "parts_kept", "runtime_ms", "runs_ms", "load_1min"]


def plan_shape(node: dict):
    """Plan tree without run-specific fields, on the fields compare_passes.shape uses."""
    keep = ("Node Type", "Relation Name", "Index Name", "Join Type", "Strategy", "Parent Relationship",
            "Startup Cost", "Total Cost", "Plan Rows", "Plan Width", "Workers Planned")
    return ({k: node.get(k) for k in keep}, [plan_shape(c) for c in node.get("Plans", []) or []])


def plan_tree(node: dict) -> str:
    """Node types and relations only, so that plans differing only in their
    estimates count as the same tree."""
    kids = ",".join(plan_tree(c) for c in node.get("Plans", []) or [])
    return f"{node['Node Type']}:{node.get('Relation Name', '')}({kids})"


def count_nodes(node: dict) -> int:
    return 1 + sum(count_nodes(c) for c in node.get("Plans", []) or [])


def load_1min() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return float("nan")


def power_state() -> str:
    """Battery status as run_collection.sh logs it (1 = on battery, 2 = on AC), in WSL only."""
    cmd = ["powershell.exe", "-NoProfile", "-Command",
           '$b = Get-CimInstance Win32_Battery; "$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)%"']
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=60).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def read_sql(qid: str) -> str:
    return (QUERY_DIR / f"{qid}.sql").read_text(encoding="utf-8").strip()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ passes

def sql_sha1(sql: str) -> str:
    """Hash of a query's text as the check runs it. tests/test_widened.py
    compares it with the query files, so results that no longer belong to the
    committed queries fail the tests instead of going unnoticed."""
    return hashlib.sha1(sql.encode("utf-8")).hexdigest()


def explain_pass(url: str, manifest: list[dict]) -> list[dict]:
    rows = []
    for i, e in enumerate(manifest, 1):
        qid = e["query_id"]
        sql = read_sql(qid)
        row = {"query_id": qid, "template": e["template"], "variant": e["variant"], "sql_sha1": sql_sha1(sql)}
        try:
            plan = explain(url, sql, TIMEOUT_S * 1000)[0]["Plan"]
        except QueryError as err:
            row["status"] = "error: " + str(err).splitlines()[0]
            rows.append(row)
            print(f"[{i}/{len(manifest)}] {qid}: {row['status']}", file=sys.stderr)
            continue
        row |= {"status": "ok", "total_cost": plan["Total Cost"], "plan_rows": plan["Plan Rows"],
                "node_count": count_nodes(plan),
                "plan_tree": hashlib.sha1(plan_tree(plan).encode()).hexdigest()[:10]}
        pilot_plan = PILOT / "plans" / f"{qid}.json"
        if e["variant"] == "std" and pilot_plan.exists():
            old = json.loads(pilot_plan.read_text(encoding="utf-8"))[0]["Plan"]
            row["same_plan_as_pilot"] = int(plan_shape(plan) == plan_shape(old))
        rows.append(row)
    return rows


def timing_pass(url: str, manifest: list[dict]) -> list[dict]:
    wide = [e for e in manifest if e["variant"] == "wide"]
    random.Random(SEED).shuffle(wide)
    rows = []
    for i, e in enumerate(wide, 1):
        qid = e["query_id"]
        row = {"query_id": qid, "template": e["template"], "order": i, "load_1min": load_1min(),
               "started_at": dt.datetime.now().isoformat(timespec="seconds")}
        t0 = time.time()
        try:
            warm = execute_timed(url, read_sql(qid), TIMEOUT_S * 1000)
            run = execute_timed(url, read_sql(qid), TIMEOUT_S * 1000)
        except QueryError as err:
            msg = str(err).splitlines()[0]
            row["status"] = "timeout" if "statement timeout" in msg else f"error: {msg}"
        else:
            row |= {"status": "ok", "runtime_ms": round(run["execution_ms"], 3),
                    "warmup_ms": round(warm["execution_ms"], 3), "planning_ms": run["planning_ms"]}
        row["wall_s"] = round(time.time() - t0, 3)
        rows.append(row)
        print(f"[{i}/{len(wide)}] {qid}: {row.get('runtime_ms', row['status'])}", file=sys.stderr)
    return rows


def run_counted(url: str, sql: str) -> dict:
    """One execution with the EXPLAIN options of collect.execute_timed: its
    execution time and the number of rows the plan's root returned."""
    obj = run_psql(url, f"EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON) {sql}\n",
                   TIMEOUT_S * 1000)[0]
    return {"execution_ms": float(obj["Execution Time"]), "rows": round(float(obj["Plan"]["Actual Rows"]))}


def q11_bounds_pass(url: str, manifest: list[dict]) -> list[dict]:
    """Step 3: Q11 at both ends of its fraction for every nation, then every
    widened Q11 query, each run once untimed and then Q11_REPEATS times timed.
    A query error stops the check, since Q11 already planned and ran in steps
    1 and 2."""
    cases = [{"case": case, "query_id": "", "nation": nation, "fraction": fraction}
             for nation in NATIONS for case, fraction in Q11_ENDS.items()]
    cases += [{"case": "widened", "query_id": e["query_id"], **e["parameters"]}
              for e in manifest if e["template"] == 11 and e["variant"] == "wide"]
    rows = []
    for i, c in enumerate(cases, 1):
        sql = read_sql(c["query_id"]) if c["query_id"] else render(11, c).strip()
        load = load_1min()
        run_counted(url, sql)
        runs = [run_counted(url, sql) for _ in range(Q11_REPEATS)]
        ms = [r["execution_ms"] for r in runs]
        rows.append({"case": c["case"], "query_id": c["query_id"], "nation": c["nation"],
                     "fraction": write_value("fraction", c["fraction"]), "parts_kept": runs[0]["rows"],
                     "runtime_ms": round(statistics.median(ms), 3),
                     "runs_ms": json.dumps([round(v, 3) for v in ms]), "load_1min": load})
        print(f"[{i}/{len(cases)}] Q11 {c['case']} {c['nation']}: {rows[-1]['runtime_ms']} ms, "
              f"{rows[-1]['parts_kept']} parts kept", file=sys.stderr)
    return rows


# ------------------------------------------------------------------ summary

def spread(values: list[float]) -> float:
    return max(values) / min(values) if values else math.nan


def fmt_ms(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def fmt_range(values: list, name: str) -> str:
    """Range of one parameter's values, written the way the SQL writes them."""
    if not values:
        return "--"
    k = kind(name)
    if k == "text":
        n = len(set(values))
        if name == "prefix":
            lengths = sorted(len(v) for v in values)
            return f"{n} different, {lengths[0]} to {lengths[-1]} characters"
        return f"{n} different"
    lo, hi = min(values), max(values)
    lo_s, hi_s = write_value(name, lo), write_value(name, hi)
    return lo_s if lo == hi else f"{lo_s} to {hi_s}"


def parameter_rows(manifest: list[dict], t: int) -> list[tuple[str, str, str]]:
    """(parameter, standard range, widened range) for one template; a window
    is shown with its unit."""
    by: dict[str, dict[str, list]] = {"std": defaultdict(list), "wide": defaultdict(list)}
    for e in manifest:
        if e["template"] == t:
            for k, v in e["parameters"].items():
                by[e["variant"]][k].append(v)
    names = [k for k in by["wide"] if k != "unit"]
    out = []
    for name in names:
        cells = []
        for variant in ("std", "wide"):
            cell = fmt_range(by[variant][name], name)
            if name == "window":
                cell += " " + "/".join(sorted(set(by[variant]["unit"])))
            cells.append(cell)
        out.append((name, *cells))
    return out


def summarize(manifest: list[dict]) -> None:
    explain_rows = read_csv(OUT / "explain.csv")
    run_rows = read_csv(OUT / "widened_runtimes.csv")
    meta = json.loads((OUT / "run.meta.json").read_text(encoding="utf-8"))
    pilot = {r["query_id"]: float(r["runtime_ms"]) for r in read_csv(PILOT / "runtime_dataset.csv")
             if r["status"] == "ok"}
    variant = {e["query_id"]: e["variant"] for e in manifest}
    split = {r["query_id"]: r["set"] for r in read_csv(QUERY_DIR / "split.csv")}

    n_plan_ok = sum(r["status"] == "ok" for r in explain_rows)
    std_compared = [r for r in explain_rows if r.get("same_plan_as_pilot") not in (None, "")]
    n_same = sum(r["same_plan_as_pilot"] == "1" for r in std_compared)
    ok_runs = [r for r in run_rows if r["status"] == "ok"]
    slowest = max(max(float(r["runtime_ms"]), float(r["warmup_ms"])) for r in ok_runs)
    over_limit = [r["query_id"] for r in ok_runs
                  if max(float(r["runtime_ms"]), float(r["warmup_ms"])) > LIMIT_S * 1000]

    per_t = []
    for t in WIDENED_TEMPLATES:
        std_ids = [e["query_id"] for e in manifest if e["template"] == t and e["variant"] == "std"]
        std_ms = [pilot[q] for q in std_ids if q in pilot]
        wide_ms = [float(r["runtime_ms"]) for r in ok_runs if int(r["template"]) == t]
        cost = {v: [float(r["total_cost"]) for r in explain_rows
                    if int(r["template"]) == t and r["variant"] == v and r["status"] == "ok"] for v in ("std", "wide")}
        trees = {v: len({r["plan_tree"] for r in explain_rows
                         if int(r["template"]) == t and r["variant"] == v and r["status"] == "ok"})
                 for v in ("std", "wide")}
        per_t.append({"template": t, "std_n": len(std_ms), "std_min": min(std_ms), "std_max": max(std_ms),
                      "std_spread": spread(std_ms), "wide_n": len(wide_ms), "wide_min": min(wide_ms),
                      "wide_max": max(wide_ms), "wide_spread": spread(wide_ms),
                      "std_cost_spread": spread(cost["std"]), "wide_cost_spread": spread(cost["wide"]),
                      "std_trees": trees["std"], "wide_trees": trees["wide"]})
    below = [f"Q{p['template']}" for p in per_t if p["wide_spread"] < MIN_SPREAD]
    same_cost = [f"Q{p['template']}" for p in per_t if p["wide_cost_spread"] < 1.0001]
    not_wider = [f"Q{p['template']}" for p in per_t if p["wide_spread"] <= p["std_spread"]]
    loads = [float(r["load_1min"]) for r in run_rows]

    q11 = read_csv(OUT / "q11_bounds.csv")
    every, none = Q11_ENDS
    ends: dict[str, dict[str, float]] = defaultdict(dict)
    for r in q11:
        if r["case"] in Q11_ENDS:
            ends[r["nation"]][r["case"]] = float(r["runtime_ms"])
    q11_effect = [e[every] / e[none] for e in ends.values()]
    q11_end_ms = [ms for e in ends.values() for ms in e.values()]
    q11_bound = spread(q11_end_ms)
    kept = {case: [int(r["parts_kept"]) for r in q11 if r["case"] == case] for case in (every, none, "widened")}
    q11_wide_ms = [float(r["runtime_ms"]) for r in q11 if r["case"] == "widened"]
    q11_noise = [spread(json.loads(r["runs_ms"])) for r in q11]
    q11_loads = [float(r["load_1min"]) for r in q11]

    def names(items: list[str]) -> str:
        return ", ".join(items) if items else "none"

    n_wide = sum(v == "wide" for v in variant.values())
    lines = [
        "# Widened workload check on M1",
        "",
        f"Generated by `scripts/check_workload.py` from the run of {meta['started_at'][:10]}. "
        "Do not edit by hand; run the script again to update it.",
        "",
        "This is a sanity check of `queries/tpch_sf1_v2`, not a label collection. Every query was planned "
        "with `EXPLAIN (FORMAT JSON)`, and every widened query ran once untimed and then once timed "
        f"(statement timeout {TIMEOUT_S} s), timed as `src/collect.py` times a run. Q11 also ran at both ends "
        "of its fraction, to measure how much spread its template allows. Method and criteria are in "
        "the docstring of `scripts/check_workload.py`; the widened ranges and why they were chosen are in the "
        "docstring of `scripts/generate_widened_queries.py`.",
        "",
        "## Result",
        "",
        f"- Plans: {n_plan_ok} of {len(explain_rows)} queries plan without error. {n_same} of "
        f"{len(std_compared)} standard queries have the same plan as in the pilot (node tree, estimated "
        "costs and rows), so the database and its statistics are unchanged.",
        f"- Widened queries: {len(ok_runs)} of {n_wide} finished. The slowest execution, untimed run "
        f"included, took {slowest / 1000:.1f} s (limit {LIMIT_S} s); over the limit: {names(over_limit)}.",
        f"- Spread, slowest over fastest widened query of a template: at least {MIN_SPREAD:.0f} in "
        f"{len(per_t) - len(below)} of {len(per_t)} templates; below {MIN_SPREAD:.0f}: {names(below)}. "
        f"Not wider than the standard queries' spread: {names(not_wider)}. Q11's template allows a spread "
        f"of about {q11_bound:.2f} on M1 (section Q11 below).",
        f"- Templates whose widened queries all have the same estimated cost, because PostgreSQL's "
        f"estimates do not depend on the widened value: {names(same_cost)}.",
        "",
        "## Runtime per template",
        "",
        "Standard: pilot pass 1 labels (median of 3 runs). Widened: the timed run of this check. Spread is "
        "slowest over fastest. Cost is PostgreSQL's estimated total cost; plan trees counts different "
        "trees of node types and relations.",
        "",
        "| Template | Standard runtime, ms | Spread | Widened runtime, ms | Spread | Cost spread, std / wide "
        "| Plan trees, std / wide |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in per_t:
        lines.append(
            f"| Q{p['template']} | {fmt_ms(p['std_min'])} to {fmt_ms(p['std_max'])} (n = {p['std_n']}) "
            f"| {p['std_spread']:.2f} | {fmt_ms(p['wide_min'])} to {fmt_ms(p['wide_max'])} (n = {p['wide_n']}) "
            f"| {p['wide_spread']:.2f} | {p['std_cost_spread']:.2f} / {p['wide_cost_spread']:.2f} "
            f"| {p['std_trees']} / {p['wide_trees']} |")
    lines += [
        "", "## Q11", "",
        "Q11's parameters are a nation and a fraction, and the fraction only decides how many of the "
        f"aggregated parts are kept and sorted. Fraction 0 keeps every part ({min(kept[every]):,} to "
        f"{max(kept[every]):,}, depending on the nation) and fraction 1 keeps "
        f"{'none' if max(kept[none]) == 0 else f'up to {max(kept[none]):,}'}. The widened fractions keep from "
        f"{min(kept['widened']):,} to {max(kept['widened']):,} parts, so they reach both ends.",
        "",
        f"Timed at both ends for each of the {len(ends)} nations (median of {Q11_REPEATS} runs after an untimed "
        f"one), keeping every part instead of none changed the runtime by a median factor of "
        f"{statistics.median(q11_effect):.2f} ({min(q11_effect):.2f} to {max(q11_effect):.2f} over the "
        f"nations). All {len(q11_end_ms)} of these runs lay between {fmt_ms(min(q11_end_ms))} and "
        f"{fmt_ms(max(q11_end_ms))} ms, a spread of {q11_bound:.2f}. Any Q11 query runs between the two ends "
        "of its nation, so up to measurement noise this is the widest spread the template allows on M1. The "
        f"{len(q11_wide_ms)} widened Q11 queries, timed the same way, ran from {fmt_ms(min(q11_wide_ms))} to "
        f"{fmt_ms(max(q11_wide_ms))} ms (spread {spread(q11_wide_ms):.2f}). For comparison, the "
        f"{Q11_REPEATS} timed runs of one query differed by a median factor of "
        f"{statistics.median(q11_noise):.2f} (up to {max(q11_noise):.2f}).",
    ]
    lines += ["", "## Parameter ranges", "",
              "Values as they appear in the SQL, over the queries of each variant (standard: qgen's draws; "
              "widened: this workload's draws). A text parameter shows how many different values occur.", "",
              "| Template | Parameter | Standard | Widened |", "|---|---|---|---|"]
    for t in WIDENED_TEMPLATES:
        for name, std_cell, wide_cell in parameter_rows(manifest, t):
            lines.append(f"| Q{t} | {name} | {std_cell} | {wide_cell} |")
    pilot_start = json.loads((PILOT / "runtime_dataset.meta.json").read_text(encoding="utf-8"))[-1]["started_at"]
    lines += [
        "", "## Conditions", "",
        f"Standard runtimes come from pilot pass 1 ({pilot_start[:10]}, laptop on battery). This check ran from "
        f"{meta['started_at']} to {meta['finished_at']} with battery status {meta['power_start']} at the "
        f"start and {meta['power_end']} at the end (1 = on battery, 2 = on AC), on "
        f"{meta['server']['version'].split(' on ')[0]}. The 1-minute load average in WSL had median "
        f"{statistics.median(loads):.2f} and maximum {max(loads):.2f} during the timing pass, and median "
        f"{statistics.median(q11_loads):.2f} and maximum {max(q11_loads):.2f} during the Q11 step. It counts "
        "the check's own query and its parallel workers, and it does not see work running in Windows outside "
        "WSL.",
        "",
        "Files: `explain.csv` (one row per query), `widened_runtimes.csv` (one row per widened query, "
        "both runs), `q11_bounds.csv` (the Q11 step: both ends for every nation, and the widened Q11 "
        "queries), `run.meta.json` (settings and host), `workload_numbers.tex` (LaTeX macros).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    wide_spreads = [p["wide_spread"] for p in per_t]
    std_spreads = [p["std_spread"] for p in per_t]
    narrowest = min(per_t, key=lambda p: p["wide_spread"])
    macros = {
        "WorkloadN": len(manifest),
        "WorkloadNStd": sum(v == "std" for v in variant.values()),
        "WorkloadNWide": n_wide,
        "WorkloadNWideTemplates": len(WIDENED_TEMPLATES),
        "WorkloadNPool": sum(s == "pool" for s in split.values()),
        "WorkloadNTest": sum(s == "test" for s in split.values()),
        "CheckPlansOk": n_plan_ok,
        "CheckSamePlan": n_same,
        "CheckWideOk": len(ok_runs),
        "CheckSlowestS": f"{slowest / 1000:.1f}",
        "CheckStdSpreadMedian": f"{statistics.median(std_spreads):.2f}",
        "CheckWideSpreadMedian": f"{statistics.median(wide_spreads):.1f}",
        "CheckWideSpreadMin": f"{narrowest['wide_spread']:.2f}",
        "CheckWideSpreadMinT": f"Q{narrowest['template']}",
        "CheckWideSpreadMax": f"{max(wide_spreads):.0f}",
        "CheckTwoFold": len(per_t) - len(below),
        "CheckQElevenEffect": f"{statistics.median(q11_effect):.2f}",
        "CheckQElevenEffectMax": f"{max(q11_effect):.2f}",
        "CheckQElevenBound": f"{q11_bound:.2f}",
    }
    (OUT / "workload_numbers.tex").write_text(
        "% Generated by scripts/check_workload.py. Do not edit by hand.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8", newline="\n")
    print((OUT / "README.md").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql:///tpch"))
    ap.add_argument("--summary", action="store_true", help="only rewrite the summary from the saved files")
    args = ap.parse_args()
    manifest = json.loads((QUERY_DIR / "manifest.json").read_text(encoding="utf-8"))

    if not args.summary:
        OUT.mkdir(parents=True, exist_ok=True)
        meta = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "power_start": power_state(),
                "statement_timeout_s": TIMEOUT_S, "order_seed": SEED, "q11_repeats": Q11_REPEATS,
                "server": server_info(args.database_url, TIMEOUT_S * 1000), "host": host_info()}
        write_csv(OUT / "explain.csv", EXPLAIN_FIELDS, explain_pass(args.database_url, manifest))
        write_csv(OUT / "widened_runtimes.csv", RUN_FIELDS, timing_pass(args.database_url, manifest))
        write_csv(OUT / "q11_bounds.csv", Q11_FIELDS, q11_bounds_pass(args.database_url, manifest))
        meta |= {"finished_at": dt.datetime.now().isoformat(timespec="seconds"), "power_end": power_state()}
        (OUT / "run.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    summarize(manifest)


if __name__ == "__main__":
    main()
