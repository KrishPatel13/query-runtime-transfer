"""Collect pre-execution plan features and measured runtimes for a query set.

For each query:
  1. EXPLAIN (FORMAT JSON) gives the plan PostgreSQL will use. The features are
     taken from this plan only, and the plan is saved for later re-extraction.
  2. The query is executed --warmups times, untimed, to warm the caches.
  3. It is executed --repeats more times with
     EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, FORMAT JSON). Result rows are not
     sent to the client, and per-node timing is off to keep overhead low.
The label is the median server-side "Execution Time" of the timed runs.

Queries run in a seeded random order so that slow drift in machine state
(thermal limits, power source) is not lined up with any template.

If the wall-clock time of a query's measurement is far longer than the
execution times it reports, the machine was most likely suspended in the
middle of it (a laptop going to sleep). Such a query is measured again, and
the stall is recorded in the output.

Each row also records load1, the 1-minute load average from /proc/loadavg
when the query starts (empty where there is no /proc): the number of
processes running or waiting to run, averaged over the last minute. On an
otherwise idle machine it reflects the measurement itself, the backend and
parallel workers of the previous queries; on a shared server it also shows
other users' work. The meta file records the machine name (--machine) and
the host: CPU model, CPU count, kernel and host name.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from features import FEATURE_NAMES, extract_features

BASE_FIELDS = [
    "query_id", "template", "status", "runtime_ms", "runtime_mean_ms", "runtime_std_ms",
    "runtime_min_ms", "runtime_max_ms", "repeats", "runs_ms", "warmup_ms",
    "planning_ms", "shared_read_blocks", "temp_written_blocks", "started_at", "order", "wall_s", "stalls",
    "load1",
]
# A measurement whose wall time exceeds its reported execution time by more than
# this many seconds (psql start-up costs about 0.05 s per call) is treated as stalled.
STALL_SLACK_S = 10.0
# psql messages that mean the server or the connection is gone, not the query.
FATAL_MARKERS = ("could not connect", "connection to server", "server closed the connection",
                 "terminating connection", "the database system is")


class QueryError(RuntimeError):
    pass


def run_psql(database_url: str, sql: str, timeout_ms: int):
    env = os.environ.copy()
    env["PGOPTIONS"] = f"-c statement_timeout={timeout_ms}"
    cmd = ["psql", database_url, "-X", "-A", "-t", "-q", "-v", "ON_ERROR_STOP=1"]
    p = subprocess.run(cmd, input=sql, text=True, capture_output=True, env=env)
    if p.returncode != 0:
        msg = p.stderr.strip() or f"psql failed with code {p.returncode}"
        if p.returncode == 2 or any(k in msg for k in FATAL_MARKERS):
            # Writing this as a per-query error would silently drop the rest of the pass.
            raise SystemExit(f"database connection failed, stopping the run: {msg}")
        raise QueryError(msg)
    return json.loads(p.stdout)


def explain(database_url: str, query: str, timeout_ms: int):
    return run_psql(database_url, f"EXPLAIN (FORMAT JSON) {query}\n", timeout_ms)


def execute_timed(database_url: str, query: str, timeout_ms: int) -> dict:
    obj = run_psql(
        database_url,
        f"EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON) {query}\n",
        timeout_ms,
    )[0]
    return {
        "execution_ms": float(obj["Execution Time"]),
        "planning_ms": float(obj.get("Planning Time", float("nan"))),
        "shared_read_blocks": int(obj["Plan"].get("Shared Read Blocks", 0)),
        # Blocks spilled to temporary files (a sort or hash that outgrew
        # work_mem); the root node's count includes every node below it.
        "temp_written_blocks": int(obj["Plan"].get("Temp Written Blocks", 0)),
    }


def measure(database_url: str, query: str, timeout_ms: int, warmups: int, repeats: int) -> dict:
    wall0 = time.time()
    warm = [execute_timed(database_url, query, timeout_ms)["execution_ms"] for _ in range(warmups)]
    runs = [execute_timed(database_url, query, timeout_ms) for _ in range(repeats)]
    wall = time.time() - wall0
    busy = (sum(warm) + sum(r["execution_ms"] for r in runs)) / 1000.0
    return {"warm": warm, "runs": runs, "wall_s": wall, "stalled": wall - busy > STALL_SLACK_S}


def server_info(database_url: str, timeout_ms: int) -> dict:
    return run_psql(
        database_url,
        "select json_build_object("
        "'version', version(),"
        "'jit_available', pg_jit_available(),"
        "'settings', (select json_object_agg(name, setting) from pg_settings"
        "             where source not in ('default', 'override')"
        "                or name in ('max_parallel_workers_per_gather', 'jit', 'random_page_cost',"
        "                            'seq_page_cost', 'cpu_tuple_cost', 'default_statistics_target')),"
        "'database_size', pg_size_pretty(pg_database_size(current_database())))",
        timeout_ms,
    )


def cpu_flags_and_model() -> tuple[set[str], str]:
    """CPU flags and model name from /proc/cpuinfo (Linux only)."""
    flags, model = set(), ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "model name" and not model:
                    model = value.strip()
                elif key.strip() == "flags" and not flags:
                    flags = set(value.split())
                if model and flags:
                    break
    except OSError:
        pass
    return flags, model or platform.processor()


def host_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        info["mem_total"] = next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split(":")[1].strip()
    except OSError:
        pass
    flags, model = cpu_flags_and_model()
    info["cpu_model"] = model
    info["kernel"] = platform.release()
    info["hostname"] = platform.node()
    if hasattr(os, "sched_getaffinity"):
        info["cpus_usable"] = len(os.sched_getaffinity(0))
    # The hypervisor flag is set inside a virtual machine (WSL2 is one).
    info["virtual_machine"] = "hypervisor" in flags
    return info


def load1(path: str = "/proc/loadavg") -> float | None:
    """1-minute load average, the first field of /proc/loadavg (None elsewhere)."""
    try:
        with open(path, encoding="ascii") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_templates(query_dir: Path) -> dict[str, int]:
    """Template of every query listed in the query directory's manifest.json."""
    manifest = query_dir / "manifest.json"
    if not manifest.exists():
        return {}
    return {e["query_id"]: int(e["template"]) for e in json.loads(manifest.read_text(encoding="utf-8"))}


def template_of(qid: str, templates: dict[str, int]) -> int:
    """From the manifest, or else from the query ID (q07_006 is template 7)."""
    if qid in templates:
        return templates[qid]
    m = re.match(r"[qQ](\d+)", qid)
    if not m:
        raise SystemExit(f"cannot tell the template of {qid}: it is not in manifest.json and does not start with q<number>")
    return int(m.group(1))


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", default="queries")
    ap.add_argument("--out", default="results/runtime_dataset.csv")
    ap.add_argument("--plan-dir", default=None, help="where to save EXPLAIN JSON (default: <out dir>/plans)")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql:///tpch"))
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=300, help="statement timeout per execution")
    ap.add_argument("--seed", type=int, default=0, help="seed for the execution order")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows with status ok in --out and measure everything else")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries (smoke tests)")
    ap.add_argument("--machine", default=None, help="machine name (M1, M2, M3), recorded in the meta file")
    args = ap.parse_args()

    qfiles = sorted(Path(args.query_dir).glob("*.sql"))
    if not qfiles:
        raise SystemExit(f"No .sql files found in {args.query_dir}")
    templates = read_templates(Path(args.query_dir))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(qfiles)
    if args.limit:
        qfiles = qfiles[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_dir = Path(args.plan_dir) if args.plan_dir else out.parent / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    timeout_ms = args.timeout_s * 1000
    fields = BASE_FIELDS + FEATURE_NAMES

    kept: list[dict] = []
    if out.exists():
        if args.resume:
            header, rows = read_rows(out)
            if header != fields:
                raise SystemExit(f"{out} has different columns than this collector writes; "
                                 "resume with the version that created it, or use --overwrite")
            kept = [r for r in rows if r["status"] == "ok"]
        elif not args.overwrite:
            raise SystemExit(f"{out} already exists; pass --resume to continue it or --overwrite to replace it")
    done = {r["query_id"] for r in kept}

    meta_path = out.with_suffix(".meta.json")
    history = []
    if args.resume and meta_path.exists():
        history = json.loads(meta_path.read_text())
        history = history if isinstance(history, list) else [history]
    meta = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "machine": args.machine,
        "args": vars(args),
        "statement_timeout_s": args.timeout_s,
        "server": server_info(args.database_url, timeout_ms),
        "host": host_info(),
        "n_queries": len(qfiles),
    }

    def write_meta():
        meta_path.write_text(json.dumps(history + [meta], indent=1) + "\n", encoding="utf-8")

    write_meta()  # so an interrupted run still records its settings
    stalls_total = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
        f.flush()
        for i, path in enumerate(qfiles, 1):
            qid = path.stem
            if qid in done:
                continue
            query = path.read_text(encoding="utf-8").strip()
            row = {"query_id": qid, "template": template_of(qid, templates), "order": i,
                   "started_at": dt.datetime.now().isoformat(timespec="seconds"), "stalls": 0,
                   "load1": load1()}
            try:
                plan = explain(args.database_url, query, timeout_ms)
                (plan_dir / f"{qid}.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
                row.update(extract_features(plan))
                m = measure(args.database_url, query, timeout_ms, args.warmups, args.repeats)
                while m["stalled"] and row["stalls"] < 3:
                    row["stalls"] += 1
                    print(f"[{i}/{len(qfiles)}] {qid}: stall of {m['wall_s']:.0f} s wall time, measuring again",
                          file=sys.stderr)
                    m = measure(args.database_url, query, timeout_ms, args.warmups, args.repeats)
            except QueryError as e:
                msg = str(e).splitlines()[0]
                row["status"] = "timeout" if "statement timeout" in msg else f"error: {msg}"
                w.writerow(row)
                f.flush()
                print(f"[{i}/{len(qfiles)}] {qid}: {row['status']}", file=sys.stderr)
                continue
            stalls_total += row["stalls"]
            vals = [r["execution_ms"] for r in m["runs"]]
            row.update({
                "status": "ok" if not m["stalled"] else "stalled",
                "runtime_ms": statistics.median(vals),
                "runtime_mean_ms": statistics.mean(vals),
                "runtime_std_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "runtime_min_ms": min(vals),
                "runtime_max_ms": max(vals),
                "repeats": len(vals),
                "runs_ms": json.dumps([round(v, 3) for v in vals]),
                "warmup_ms": json.dumps([round(v, 3) for v in m["warm"]]),
                "planning_ms": statistics.median(r["planning_ms"] for r in m["runs"]),
                "shared_read_blocks": max(r["shared_read_blocks"] for r in m["runs"]),
                "temp_written_blocks": max(r["temp_written_blocks"] for r in m["runs"]),
                "wall_s": round(m["wall_s"], 3),
            })
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(qfiles)}] {qid}: median={row['runtime_ms']:.1f} ms", file=sys.stderr)

    meta["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["stalls_detected"] = stalls_total
    write_meta()


if __name__ == "__main__":
    main()
