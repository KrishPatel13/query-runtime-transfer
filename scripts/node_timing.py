"""Record per-operator times of every query once, for explaining machine shifts.

The label passes (src/collect.py) time whole queries with TIMING OFF, which
keeps the per-row clock reads out of the labels. This script runs every query
of a directory once more with TIMING ON, after one warm-up run with TIMING
OFF as in the passes, and saves the EXPLAIN ANALYZE JSON with per-node actual
times, rows, loops and buffers. Comparing the same node of the same plan on
two machines shows which operators slow down by how much. These timings are
never used as labels: TIMING ON adds overhead that differs between machines.

Usage (standard library only; source the cluster's pg.env first):
  python3 scripts/node_timing.py --query-dir queries/tpch_sf1_v2 --out /path/node_times_M1.jsonl
  python3 scripts/node_timing.py --query-dir queries/tpcds_sf1_kept --database-url postgresql:///tpcds --out ...
The output has one JSON object per line: query_id, the plan, and the time of
the warm-up run. A run that stopped can be continued with the same command:
queries already in the file are skipped.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collect import QueryError, run_psql  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--database-url", default="postgresql:///tpch")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7, help="the queries run in a seeded random order")
    args = ap.parse_args()

    files = sorted(Path(args.query_dir).glob("*.sql"))
    random.Random(args.seed).shuffle(files)
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["query_id"])
    timeout_ms = args.timeout_s * 1000
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, path in enumerate(files, 1):
            qid = path.stem
            if qid in done:
                continue
            query = path.read_text(encoding="utf-8").strip()
            rec = {"query_id": qid}
            try:
                warm = run_psql(args.database_url,
                                f"EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, FORMAT JSON) {query}\n", timeout_ms)
                rec["warm_ms"] = float(warm[0]["Execution Time"])
                rec["plan"] = run_psql(args.database_url,
                                       f"EXPLAIN (ANALYZE, TIMING ON, SUMMARY ON, BUFFERS ON, FORMAT JSON) {query}\n",
                                       timeout_ms)
                rec["status"] = "ok"
            except QueryError as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:500]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            ms = rec["plan"][0]["Execution Time"] if rec["status"] == "ok" else float("nan")
            print(f"[{i}/{len(files)}] {qid}: {rec['status']} {ms:.1f} ms ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
