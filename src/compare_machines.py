"""Check that two machines chose the same plan for every query.

Every machine runs on a copy of M1's database files (the same data and the
same planner statistics) with the same settings, so PostgreSQL should choose
the identical plan, with identical estimates, everywhere. That is what keeps
the plan features x the same on every machine, so that only the runtime
given x changes.

For machines A and B this compares the saved EXPLAIN plans of one pass
(default pass 1) in results/runs/<machine>/pass<p>/plans for every query both
have, by the plan shape of passcheck.shape: node types, relations, indexes,
join types, estimated startup and total costs, rows, widths and planned
workers. A mismatch is reported either as a different tree or as the same
tree with different estimates, with the first node and field where the two
differ. Queries that only one machine has are listed but do not count as
mismatches.

It also lists the server settings that differ between the two passes' meta
files, leaving out the connection settings that are meant to differ (port,
listen_addresses, the socket directory and its permissions, and
application_name). A memory or planner setting there is the likely cause of
a mismatch.

Writes results/runs/compare_<A>_<B>.json and compare_<A>_<B>.tex (LaTeX
macros \\<A><B>Shared and \\<A><B>SamePlan, e.g. \\MOneMThreeSamePlan).
Exit status 1 if any shared query has a different plan or no query is shared,
0 otherwise. Needs only the standard library.

Usage: python src/compare_machines.py M1 M3 [--runs-dir results/runs] [--pass 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passcheck import macro_prefix, read_meta, shape

HERE = Path(__file__).resolve().parents[1]
ESTIMATES = ("Startup Cost", "Total Cost", "Plan Rows", "Plan Width", "Workers Planned")
CONNECTION_SETTINGS = {"port", "listen_addresses", "unix_socket_directories", "unix_socket_permissions",
                       "application_name"}


def load_plans(plan_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))[0]["Plan"] for p in sorted(plan_dir.glob("*.json"))}


def structure(s):
    """A plan shape without its estimates: only the tree."""
    fields, children = s
    return ({k: v for k, v in fields.items() if k not in ESTIMATES}, [structure(c) for c in children])


def first_difference(a, b, path: str | None = None) -> dict | None:
    """First node, in pre-order, where two shapes (or structures) differ.

    The node is named by the node types from the root down (in A's plan),
    e.g. "Aggregate > Hash Join > Seq Scan #2" for the second child.
    """
    (fa, ca), (fb, cb) = a, b
    path = path or str(fa.get("Node Type"))
    for k in fa:
        if fa[k] != fb.get(k):
            return {"node": path, "field": k, "a": fa[k], "b": fb.get(k)}
    if len(ca) != len(cb):
        return {"node": path, "field": "number of child nodes", "a": len(ca), "b": len(cb)}
    for n, (x, y) in enumerate(zip(ca, cb)):
        child = f"{path} > {x[0].get('Node Type')}" + (f" #{n + 1}" if len(ca) > 1 else "")
        d = first_difference(x, y, child)
        if d:
            return d
    return None


def compare_plans(plans_a: dict[str, dict], plans_b: dict[str, dict]) -> dict:
    shared = sorted(set(plans_a) & set(plans_b))
    mismatches = []
    for q in shared:
        sa, sb = shape(plans_a[q]), shape(plans_b[q])
        if sa == sb:
            continue
        same_tree = structure(sa) == structure(sb)
        mismatches.append({"query_id": q, "kind": "estimates" if same_tree else "tree",
                           "total_cost": [plans_a[q].get("Total Cost"), plans_b[q].get("Total Cost")],
                           "first_difference": first_difference(sa, sb) if same_tree else
                           first_difference(structure(sa), structure(sb))})
    return {"shared": len(shared), "same_plan": len(shared) - len(mismatches),
            "only_in_a": sorted(set(plans_a) - set(plans_b)), "only_in_b": sorted(set(plans_b) - set(plans_a)),
            "mismatches": mismatches}


def server_differences(meta_a: list[dict], meta_b: list[dict]) -> dict:
    """Server version and settings that differ between two passes' last runs."""
    sa = (meta_a[-1].get("server") or {}) if meta_a else {}
    sb = (meta_b[-1].get("server") or {}) if meta_b else {}
    set_a, set_b = sa.get("settings") or {}, sb.get("settings") or {}
    keys = sorted((set(set_a) | set(set_b)) - CONNECTION_SETTINGS)
    version = [str(v).split(" on ")[0] if v else None for v in (sa.get("version"), sb.get("version"))]
    return {"version": version if version[0] != version[1] else None,
            "settings": {k: [set_a.get(k), set_b.get(k)] for k in keys if set_a.get(k) != set_b.get(k)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine_a")
    ap.add_argument("machine_b")
    ap.add_argument("--runs-dir", default=str(HERE / "results" / "runs"))
    ap.add_argument("--pass", dest="pass_no", type=int, default=1, help="which pass's plans to compare")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    a, b = args.machine_a, args.machine_b
    dirs = {m: runs / m / f"pass{args.pass_no}" for m in (a, b)}
    for m, d in dirs.items():
        if not (d / "plans").is_dir():
            raise SystemExit(f"no plans for {m} in {d / 'plans'}")
    result = compare_plans(load_plans(dirs[a] / "plans"), load_plans(dirs[b] / "plans"))
    server = server_differences(read_meta(dirs[a]), read_meta(dirs[b]))
    out = {"machines": [a, b], "pass": args.pass_no} | result | {"server": server}
    stem = runs / f"compare_{a}_{b}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    p = macro_prefix(a) + macro_prefix(b)
    stem.with_suffix(".tex").write_text(
        "% Generated by src/compare_machines.py. Do not edit by hand.\n"
        f"\\newcommand{{\\{p}Shared}}{{{result['shared']}}}\n"
        f"\\newcommand{{\\{p}SamePlan}}{{{result['same_plan']}}}\n", encoding="utf-8", newline="\n")

    print(f"{a} vs {b}, pass {args.pass_no}: {result['shared']} shared queries, "
          f"{result['same_plan']} with the same plan and estimates")
    if result["only_in_a"] or result["only_in_b"]:
        print(f"only in {a}: {len(result['only_in_a'])}, only in {b}: {len(result['only_in_b'])}")
    for mm in result["mismatches"]:
        d = mm["first_difference"] or {}
        print(f"  {mm['query_id']}: different {mm['kind']} at {d.get('node')}, {d.get('field')}: "
              f"{d.get('a')!r} on {a}, {d.get('b')!r} on {b}")
    if server["version"]:
        print(f"server versions differ: {server['version'][0]} on {a}, {server['version'][1]} on {b}")
    for k, (va, vb) in server["settings"].items():
        print(f"setting {k}: {va!r} on {a}, {vb!r} on {b}")
    print(f"wrote {stem.with_suffix('.json')}")
    if result["mismatches"] or result["shared"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
