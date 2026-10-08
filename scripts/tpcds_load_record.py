"""Write the record of a TPC-DS load: row counts, sizes and the time of each step.

scripts/tpcds_load.sh calls this at the end with the times it measured. Row
counts are exact (count(*) on every table, not the planner's estimate), sizes
come from PostgreSQL's size functions after VACUUM, the flat files are
measured on disk, and the server settings that are not at their defaults are
stored the way src/collect.py stores them for every collection run.

Writes the record as JSON and a few of its numbers as LaTeX macros (prefix
DS, for TPC-DS), so the documents never quote them by hand.

Usage (normally through scripts/tpcds_load.sh):
  python3 scripts/tpcds_load_record.py --out-json load_record.json --out-tex load_numbers.tex \
      --scale 1 --data-dir ~/cs680-pilot/tpcds-data-sf1 --seconds dsdgen=49.2 load=300.5 \
      --copy-seconds store_sales=20.3 --reencoded
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from collect import run_psql, server_info  # noqa: E402

TIMEOUT_MS = 600_000
# dsdgen writes this table to describe the run; the specification (clause
# 2.4.18) says no query uses it, so it is not counted as a benchmark table.
VERSION_TABLE = "dbgen_version"


def pairs(items: list[str], cast=float) -> dict:
    """Parse name=value arguments."""
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"expected name=value, got {item!r}")
        out[name] = cast(value)
    return out


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_stats(url: str) -> dict[str, dict]:
    """Rows (exact), heap and index bytes and the index count of every table."""
    stats = run_psql(url, """
        select json_object_agg(c.relname, json_build_object(
                 'table_bytes', pg_table_size(c.oid),
                 'index_bytes', pg_indexes_size(c.oid),
                 'indexes', (select count(*) from pg_index i where i.indrelid = c.oid)) order by c.relname)
        from pg_class c where c.relkind = 'r' and c.relnamespace = 'public'::regnamespace""", TIMEOUT_MS)
    counts = run_psql(url, "select json_object_agg(t, n) from (" + " union all ".join(
        f"select '{name}' as t, count(*) as n from {ident(name)}" for name in sorted(stats)) + ") c", TIMEOUT_MS)
    return {name: {"rows": int(counts[name]), **s} for name, s in sorted(stats.items())}


def database_stats(url: str) -> dict:
    return run_psql(url, """
        select json_build_object(
          'database', current_database(),
          'database_bytes', pg_database_size(current_database()),
          'primary_keys', (select count(*) from pg_constraint
                           where contype = 'p' and connamespace = 'public'::regnamespace),
          'foreign_keys', (select count(*) from pg_constraint
                           where contype = 'f' and connamespace = 'public'::regnamespace),
          'indexes', (select count(*) from pg_indexes where schemaname = 'public'))""", TIMEOUT_MS)


def macros(rec: dict) -> dict[str, str]:
    bench = {k: v for k, v in rec["tables"].items() if k != VERSION_TABLE}
    gb = rec["database_bytes"] / 1e9
    return {
        "DSScale": f"{rec['scale_factor']:g}",
        "DSTables": str(len(bench)),
        "DSRowsMillions": f"{sum(t['rows'] for t in bench.values()) / 1e6:.1f}",
        "DSFlatFileGB": f"{rec['flat_file_bytes'] / 1e9:.1f}",
        "DSDatabaseGB": f"{gb:.1f}",
        "DSForeignKeys": str(rec["foreign_keys"]),
        "DSIndexes": str(rec["indexes"]),
        "DSDsdgenSeconds": f"{rec['seconds']['dsdgen']:.0f}",
        "DSLoadMinutes": f"{rec['seconds']['load'] / 60:.1f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql:///tpcds"))
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--data-dir", required=True, help="where dsdgen wrote the .dat files")
    ap.add_argument("--kit-url", default="")
    ap.add_argument("--kit-commit", default="")
    ap.add_argument("--kit-version", default="")
    ap.add_argument("--seconds", nargs="*", default=[], help="step=seconds, including dsdgen and load")
    ap.add_argument("--copy-seconds", nargs="*", default=[], help="table=seconds for each COPY")
    ap.add_argument("--reencoded", nargs="*", default=[], help="table=lines changed by tpcds_fix_utf8.py")
    args = ap.parse_args()

    seconds = pairs(args.seconds)
    for step in ("dsdgen", "load"):
        if step not in seconds:
            raise SystemExit(f"--seconds needs {step}=...")
    seconds["total"] = seconds["dsdgen"] + seconds["load"]
    files = {p.stem: p.stat().st_size for p in sorted(Path(args.data_dir).glob("*.dat"))}
    tables = table_stats(args.database_url)
    for name, t in tables.items():
        t["flat_file_bytes"] = files.get(name)
    rec = {
        "scale_factor": args.scale,
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "kit": {"url": args.kit_url, "commit": args.kit_commit, "version": args.kit_version},
        "seconds": seconds,
        "copy_seconds_by_table": pairs(args.copy_seconds),
        "lines_reencoded_to_utf8": pairs(args.reencoded, int),
        **database_stats(args.database_url),
        "rows": sum(t["rows"] for t in tables.values()),
        "flat_file_bytes": sum(files.values()),
        "tables": tables,
        "server": server_info(args.database_url, TIMEOUT_MS),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    if args.out_tex:
        Path(args.out_tex).write_text(
            "% Generated by scripts/tpcds_load_record.py. Do not edit by hand.\n" +
            "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros(rec).items()),
            encoding="utf-8", newline="\n")
    print(json.dumps({k: rec[k] for k in ("database", "rows", "database_bytes", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
