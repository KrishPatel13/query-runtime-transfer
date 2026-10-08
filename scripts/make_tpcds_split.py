"""Write the pool/test split of the kept TPC-DS queries.

Machine adaptation on TPC-DS (the paper's second workload) needs the same
kind of split as TPC-H: inside every template, the query IDs are shuffled
with their own seeded stream and the first max(1, round(0.3 n)) go to the
test set, by generate_widened_queries.make_split with its seed. The two
parts of templates 14, 23, 24 and 39 belong to their template. The split
uses only query IDs and templates, so it is fixed before any target
machine's labels exist.

Usage: python scripts/make_tpcds_split.py [--query-dir queries/tpcds_sf1_kept]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location("gwq", HERE / "scripts" / "generate_widened_queries.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", default=str(HERE / "queries" / "tpcds_sf1_kept"))
    args = ap.parse_args()
    qdir = Path(args.query_dir)
    gwq = load_generator()
    manifest = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.stem for p in qdir.glob("*.sql")}
    entries = [{"query_id": e["query_id"], "template": int(e["template"]), "variant": "std"}
               for e in manifest if e["query_id"] in on_disk]
    if len(entries) != len(on_disk):
        raise SystemExit(f"{len(on_disk)} .sql files but {len(entries)} of them in manifest.json")
    rows = gwq.make_split(entries, gwq.SEED)
    out = qdir / "split.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "template", "variant", "set"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_test = sum(r["set"] == "test" for r in rows)
    print(f"wrote {out}: {len(rows)} queries, {len(rows) - n_test} pool, {n_test} test, "
          f"{len({r['template'] for r in rows})} templates")


if __name__ == "__main__":
    main()
