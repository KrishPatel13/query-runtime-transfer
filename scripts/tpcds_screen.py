"""Screen the TPC-DS templates: time the first query of each template once.

FINAL_PROPOSAL_PLAN.md (section 3) measures TPC-DS on M1 in one pass, with
only the templates that run fast enough. The rule is fixed here, before any
label is collected:
  * the first query of each template (q<t>_001, or q<t>_001a and q<t>_001b
    for the two-statement templates 14, 23, 24 and 39) runs once, with a
    60 s statement timeout (--timeout-s);
  * a query is kept when its execution time is at most 30 s (--keep-s); a
    query that times out or fails is dropped;
  * a template group is kept when at least one of its parts is kept, and
    only the kept parts are collected, all of their queries.
The pilot's power analysis (the PowerTemplates macro in
results/pilot/numbers.tex, about 76 held-out units for 80% power) is the
default --min-templates, and the summary says whether the kept template
groups and the kept template families (scripts/tpcds_families.py; C4 holds
out families) reach it.

A query is timed the way src/collect.py times its runs: EXPLAIN (ANALYZE,
TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON), with the server-side
execution time as the runtime and no result rows sent to the client. There
is no warm-up run, since the screen only decides on which side of 30 s a
template falls, but every file of the tpcds database is read once first, so
that no query pays for reading its tables from disk (the labels are warm
medians); the CSV keeps each query's shared hit, shared read and temporary
blocks to show it. A query whose wall time exceeds its execution time by
more than collect.STALL_SLACK_S stalled (the laptop slept) and runs once
more. Queries run in a seeded random order, as in collect.py, so that drift
in the machine's state is not lined up with template numbers.

On the laptop the screen uses src/run_passes.py's LaptopGuard: it refuses
to start on battery, with energy saver on or with the WSL clock not set up
(scripts/wsl_clock.py), keeps Windows awake, and logs the power state and
Windows CPU load every 30 s. It also refuses to start when the 1-minute
load in WSL is 1 or more. A screen that ran partly on battery or with
energy saver on, or whose power log has a gap (the laptop slept), is marked
invalid in the JSON file and the script exits with an error.

Writes a CSV with one row per screened query (status, keep, execution and
planning time, the planner's total cost and estimated rows, the actual rows,
the wall-clock time, the 1-minute load average before the query) and a JSON
file next to it with the rule, the summary (kept and dropped templates, and
the IDs of the queries the kept parts give, collect_query_ids), the server
settings, the host, and the power state at the start and the end (WSL only).
With --out-tex it also writes the rule and the counts as LaTeX macros
(prefix DSScreen).

src/collect.py runs every .sql file in its --query-dir, so a screen of all
templates also writes the kept workload as its own query directory
(--kept-dir, default queries/tpcds_sf1_kept): the kept .sql files unchanged,
manifest.json and families.json filtered to them, and kept_from.json naming
the screen. The collection pass and C4 read only that directory. Query IDs
need no change: collect.py reads the template group from the part before
the first underscore.

Usage (inside WSL, with the TPC-DS cluster running; on AC power):
  python3 scripts/tpcds_screen.py                          # every template
  python3 scripts/tpcds_screen.py --templates 3,14,36,77,96 --out /tmp/screen_check.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from collect import STALL_SLACK_S, QueryError, host_info, run_psql, server_info  # noqa: E402
from generate_qgen_queries import parse_templates  # noqa: E402
from run_passes import LaptopGuard, NoGuard, disruption, on_laptop  # noqa: E402

# The TPC-DS cluster made by scripts/tpcds_setup.sh.
DEFAULT_URL = "postgresql:///tpcds?host=/tmp&port=5433"
FIELDS = ["query_id", "template", "part", "status", "keep", "execution_ms", "planning_ms",
          "total_cost", "plan_rows", "actual_rows", "shared_hit_blocks", "shared_read_blocks",
          "temp_written_blocks", "wall_s", "stalls", "loadavg_1m", "started_at", "order"]
PILOT_NUMBERS = HERE / "results" / "pilot" / "numbers.tex"
MAX_START_LOAD = 1.0


def power_target(path: Path = PILOT_NUMBERS) -> int | None:
    """\\PowerTemplates from the pilot's generated macros: held-out units for 80% power."""
    try:
        m = re.search(r"\\newcommand\{\\PowerTemplates\}\{(\d+)\}", path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(m.group(1)) if m else None


def first_queries(manifest: list[dict], templates: set[int] | None = None) -> list[dict]:
    """The first query (lowest query number) of every template part."""
    chosen: dict[tuple[int, str], dict] = {}
    for e in manifest:
        if templates and e["template"] not in templates:
            continue
        key = (e["template"], e.get("part") or "")
        if key not in chosen or e["query_id"] < chosen[key]["query_id"]:
            chosen[key] = e
    return [chosen[k] for k in sorted(chosen)]


def screen_query(database_url: str, sql: str, timeout_s: int) -> dict:
    """Run one query once under EXPLAIN ANALYZE; return its status and times.

    The wall-clock time comes from a monotonic clock. In the first check run
    time.time() gave a wall time 0.75 s shorter than the server's execution
    time for one query, most likely because the WSL2 system clock jumped
    when it was resynchronized with Windows.
    """
    t0 = time.monotonic()
    try:
        obj = run_psql(database_url,
                       f"EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON) {sql}\n",
                       timeout_s * 1000)[0]
    except QueryError as e:
        msg = str(e).splitlines()[0]
        return {"status": "timeout" if "statement timeout" in msg else f"error: {msg}",
                "wall_s": round(time.monotonic() - t0, 3)}
    plan = obj["Plan"]
    return {"status": "ok", "execution_ms": float(obj["Execution Time"]),
            "planning_ms": float(obj.get("Planning Time", float("nan"))),
            "total_cost": float(plan["Total Cost"]), "plan_rows": float(plan["Plan Rows"]),
            "actual_rows": float(plan.get("Actual Rows", float("nan"))),
            "shared_hit_blocks": int(plan.get("Shared Hit Blocks", 0)),
            "shared_read_blocks": int(plan.get("Shared Read Blocks", 0)),
            "temp_written_blocks": int(plan.get("Temp Written Blocks", 0)),
            "wall_s": round(time.monotonic() - t0, 3)}


def stalled(row: dict) -> bool:
    """The wall time is far above the execution time: the machine stopped (slept) meanwhile."""
    return row["status"] == "ok" and row["wall_s"] - row["execution_ms"] / 1000 > STALL_SLACK_S


def warm_cache(database_url: str, timeout_s: int) -> dict:
    """Read every file of the database once, so the OS cache holds it.

    pg_prewarm is not built (setup_postgres.sh builds no contrib modules),
    and the screen's single cold run would otherwise read tables from disk.
    """
    rows = run_psql(database_url, "select json_build_object('dir', current_setting('data_directory'), "
                                  "'oid', (select oid from pg_database where datname = current_database()));\n",
                    timeout_s * 1000)
    base = Path(rows["dir"]) / "base" / str(rows["oid"])
    t0, size = time.monotonic(), 0
    for root, _, names in os.walk(base):
        for name in names:
            with open(Path(root) / name, "rb") as f:
                while chunk := f.read(1 << 20):
                    size += len(chunk)
    return {"dir": str(base), "bytes": size, "seconds": round(time.monotonic() - t0, 1)}


def keeps(row: dict, keep_s: float) -> bool:
    """The keep rule: finished without error in at most keep_s seconds."""
    return row["status"] == "ok" and row["execution_ms"] <= keep_s * 1000


def summarize(rows: list[dict], manifest: list[dict], min_templates: int) -> dict:
    """Kept and dropped template parts, groups and families, and the queries to collect."""
    kept_parts = {(r["template"], r["part"] or "") for r in rows if r["keep"]}
    kept_groups = sorted({t for t, _ in kept_parts})
    screened_groups = sorted({r["template"] for r in rows})
    to_collect = [e["query_id"] for e in manifest if (e["template"], e.get("part") or "") in kept_parts]
    family = {e["template"]: e.get("family", e["template"]) for e in manifest}
    kept_families = sorted({family[t] for t in kept_groups})
    draws = {e["query_id"].rstrip("abcdefgh") for e in manifest if e["query_id"] in set(to_collect)}
    dropped = {r["query_id"]: r["status"] if r["status"] != "ok" else f"{r['execution_ms'] / 1000:.1f} s"
               for r in sorted(rows, key=lambda r: r["query_id"]) if not r["keep"]}
    return {
        "screened_queries": len(rows),
        "kept_queries": len(kept_parts),
        "template_groups": len(screened_groups),
        "kept_template_groups": len(kept_groups),
        "kept_families": len(kept_families),
        "min_template_groups": min_templates,
        "enough_template_groups": len(kept_groups) >= min_templates,
        "enough_families": len(kept_families) >= min_templates,
        "kept_groups": kept_groups,
        "dropped_groups": [t for t in screened_groups if t not in kept_groups],
        "dropped_queries": dropped,
        "queries_to_collect": len(to_collect),
        "draws_to_collect": len(draws),
        "collect_query_ids": to_collect,
    }


def loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        return ""


def power_state() -> str:
    """Windows battery status as run_collection.sh logs it (1 = on battery, 2 = on AC)."""
    cmd = ["powershell.exe", "-NoProfile", "-Command",
           '$b = Get-CimInstance Win32_Battery; "$($b.BatteryStatus) charge=$($b.EstimatedChargeRemaining)%"']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return p.stdout.strip() or "unknown"


def macros(rule: dict, summary: dict) -> dict[str, str]:
    return {
        "DSScreenTimeout": f"{rule['timeout_s']:g}",
        "DSScreenKeep": f"{rule['keep_s']:g}",
        "DSScreenMinGroups": str(summary["min_template_groups"]),
        "DSScreenGroups": str(summary["template_groups"]),
        "DSScreenKeptGroups": str(summary["kept_template_groups"]),
        "DSScreenKeptFamilies": str(summary["kept_families"]),
        "DSScreenKeptQueries": str(summary["kept_queries"]),
        "DSScreenCollectFiles": str(summary["queries_to_collect"]),
        "DSScreenCollectDraws": str(summary["draws_to_collect"]),
    }


def write_kept(qdir: Path, kept_dir: Path, ids: list[str], screen_json: Path, overwrite: bool) -> int:
    """The kept workload as its own query directory, for the collection pass and C4."""
    if kept_dir.exists() and any(kept_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{kept_dir} already holds a workload; pass --overwrite to replace it")
        shutil.rmtree(kept_dir)
    kept_dir.mkdir(parents=True)
    keep = set(ids)
    for q in ids:
        shutil.copyfile(qdir / f"{q}.sql", kept_dir / f"{q}.sql")
    manifest = [e for e in json.loads((qdir / "manifest.json").read_text(encoding="utf-8")) if e["query_id"] in keep]
    (kept_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    if (qdir / "families.json").exists():
        shutil.copyfile(qdir / "families.json", kept_dir / "families.json")
    (kept_dir / "kept_from.json").write_text(json.dumps({
        "screen": str(screen_json), "source": str(qdir), "queries": len(ids),
        "note": "the queries of the template parts kept by scripts/tpcds_screen.py, copied unchanged"},
        indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", default=str(HERE / "queries" / "tpcds_sf1"))
    ap.add_argument("--out", default=str(HERE / "results" / "tpcds_sf1" / "screen.csv"))
    ap.add_argument("--out-tex", default=None, help="write the rule and the counts as LaTeX macros here")
    ap.add_argument("--database-url", default=DEFAULT_URL)
    ap.add_argument("--templates", default=None, help="comma list, ranges allowed (default: all)")
    ap.add_argument("--timeout-s", type=int, default=60, help="statement timeout of the screening run")
    ap.add_argument("--keep-s", type=float, default=30.0, help="keep a query whose execution time is at most this")
    ap.add_argument("--min-templates", type=int, default=None,
                    help="kept units the test needs (default: \\PowerTemplates of the pilot)")
    ap.add_argument("--kept-dir", default=str(HERE / "queries" / "tpcds_sf1_kept"),
                    help="where a screen of all templates writes the kept workload")
    ap.add_argument("--allow-battery", action="store_true", help="smoke tests only: see run_passes.py")
    ap.add_argument("--allow-clock", action="store_true", help="smoke tests only: see run_passes.py")
    ap.add_argument("--allow-busy", action="store_true", help=f"start even if load1 >= {MAX_START_LOAD}")
    ap.add_argument("--seed", type=int, default=0, help="seed for the execution order")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    args = ap.parse_args()

    qdir = Path(args.query_dir)
    manifest = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
    templates = set(parse_templates(args.templates)) if args.templates else None
    chosen = first_queries(manifest, templates)
    if not chosen:
        raise SystemExit(f"no queries to screen in {qdir}")
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(chosen)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.min_templates is None:
        args.min_templates = power_target() or 76
    load = float(loadavg() or 0)
    if load >= MAX_START_LOAD and not args.allow_busy:
        raise SystemExit(f"the 1-minute load is {load}; wait until the machine is idle (or pass --allow-busy)")

    rule = {"timeout_s": args.timeout_s, "keep_s": args.keep_s,
            "keep": "execution time of the template part's first query, run once, at most keep_s",
            "group": "a template group is kept when at least one of its parts is kept"}
    meta = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "finished_at": None,
            "args": vars(args), "rule": rule,
            "server": server_info(args.database_url, args.timeout_s * 1000), "host": host_info(),
            "power_start": power_state(), "loadavg_start": loadavg()}
    guard = LaptopGuard(out.with_suffix(".power.log"), args.allow_battery, len(chosen) * 2 * args.timeout_s + 3600,
                        args.allow_clock) if on_laptop() else NoGuard()
    guard.before("screen")
    meta["warm_cache"] = warm_cache(args.database_url, args.timeout_s)
    print(f"read {meta['warm_cache']['bytes'] / 2**30:.2f} GiB of the database in "
          f"{meta['warm_cache']['seconds']} s", file=sys.stderr)

    rows: list[dict] = []
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, e in enumerate(chosen, 1):
            sql = (qdir / f"{e['query_id']}.sql").read_text(encoding="utf-8").strip()
            row = {"query_id": e["query_id"], "template": e["template"], "part": e.get("part") or "",
                   "order": i, "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                   "loadavg_1m": loadavg()}
            row.update(screen_query(args.database_url, sql, args.timeout_s))
            row["stalls"] = 0
            if stalled(row):
                row["stalls"] = 1
                print(f"[{i}/{len(chosen)}] {e['query_id']}: stalled ({row['wall_s']:.0f} s wall time), "
                      "running it once more", file=sys.stderr)
                row.update(screen_query(args.database_url, sql, args.timeout_s))
            row["keep"] = keeps(row, args.keep_s)
            w.writerow(row)
            f.flush()
            rows.append(row)
            took = f"{row['execution_ms'] / 1000:.2f} s" if row["status"] == "ok" else row["status"]
            print(f"[{i}/{len(chosen)}] {e['query_id']}: {took}, {'keep' if row['keep'] else 'drop'}",
                  file=sys.stderr)

    power = guard.after("screen")
    guard.close()
    why = disruption(power)
    summary = summarize(rows, manifest, args.min_templates)
    meta.update({"finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "power_end": power_state(), "loadavg_end": loadavg(), "power": power,
                 "valid": why is None, "invalid_because": why, "summary": summary})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    if why:
        raise SystemExit(f"the screen is not valid ({why}); run it again with --overwrite")
    if templates is None:
        n = write_kept(qdir, Path(args.kept_dir), summary["collect_query_ids"], out.with_suffix(".json"),
                       args.overwrite)
        print(f"wrote the kept workload ({n} queries) to {args.kept_dir}", file=sys.stderr)
    if args.out_tex:
        Path(args.out_tex).write_text(
            "% Generated by scripts/tpcds_screen.py. Do not edit by hand.\n" +
            "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros(rule, summary).items()),
            encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "collect_query_ids"}, indent=1))


if __name__ == "__main__":
    main()
