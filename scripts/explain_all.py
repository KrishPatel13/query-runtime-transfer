"""Save PostgreSQL's plan for every query of a directory, without running them.

Before the label passes on a new machine, this checks that it chooses the
same plans as M1: run it on both machines and compare the outputs with
--compare. Plans are compared with passcheck.shape (node types, relations,
join types, estimated costs, rows, widths and planned workers).

Usage (standard library only; source pg.env first):
  python3 scripts/explain_all.py --query-dir queries/tpch_sf1_v2 --out /tmp/plans_M3.json
  python3 scripts/explain_all.py --compare plans_M1.json plans_M3.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from passcheck import shape  # noqa: E402


def explain(query: str) -> list:
    p = subprocess.run(["psql", "-X", "-A", "-t", "-q", "-v", "ON_ERROR_STOP=1"],
                       input=f"EXPLAIN (FORMAT JSON) {query}\n", text=True, capture_output=True, check=True)
    return json.loads(p.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        common = sorted(set(a) & set(b))
        differ = [q for q in common if shape(a[q][0]["Plan"]) != shape(b[q][0]["Plan"])]
        print(json.dumps({"in_a": len(a), "in_b": len(b), "compared": len(common), "differ": len(differ),
                          "differing": differ[:20]}))
        sys.exit(1 if differ or len(common) != len(a) or len(common) != len(b) else 0)
    plans = {f.stem: explain(f.read_text(encoding="utf-8").strip())
             for f in sorted(Path(args.query_dir).glob("*.sql"))}
    Path(args.out).write_text(json.dumps(plans), encoding="utf-8")
    print(f"wrote {len(plans)} plans to {args.out}")


if __name__ == "__main__":
    main()
