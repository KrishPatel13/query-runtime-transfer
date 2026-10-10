"""Write the TPC-H scale factor 10 workload of the extension (paper/PLAN.md, E1).

From queries/tpch_sf1_v2 it draws, with one generator seeded 0 and the
templates in order, 7 pool and 3 test queries of every template, keeping
their sets, and copies them to queries/tpch_sf10_sub. Q11 compares a part's
stock value with a fraction of the nation's total, and the TPC-H
specification scales that fraction with the data (0.0001 / SF); every Q11
fraction, the standard one and the widened ones, is divided by 10 and
written with qgen's ten decimals. PostgreSQL's estimates do not depend on the
fraction, so the plans do not change with it. Writes the SQL files,
manifest.json, split.csv and generation.json.

Usage, from project_proposal:
  python scripts/make_sf10_queries.py [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "queries" / "tpch_sf1_v2"
OUT = HERE / "queries" / "tpch_sf10_sub"
SF = 10
N_POOL, N_TEST, SEED = 7, 3, 0
FRACTION = re.compile(r"(sum\(ps_supplycost \* ps_availqty\) \* )(\d\.\d{10})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = ap.parse_args()
    if OUT.exists():
        if not args.force:
            raise SystemExit(f"{OUT} exists; pass --force to replace it")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    split = pd.read_csv(SRC / "split.csv")
    manifest = {e["query_id"]: e for e in json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))}
    rng = np.random.default_rng(SEED)
    chosen = []
    for template in sorted(split["template"].unique()):
        rows = split[split["template"] == template]
        for s, n in (("pool", N_POOL), ("test", N_TEST)):
            ids = sorted(rows.loc[rows["set"] == s, "query_id"])
            chosen += list(rng.choice(ids, size=n, replace=False))

    entries, fractions = [], {}
    for qid in sorted(chosen):
        sql = (SRC / f"{qid}.sql").read_text(encoding="utf-8")
        entry = json.loads(json.dumps(manifest[qid]))
        if entry["template"] == 11:
            found = FRACTION.findall(sql)
            if len(found) != 1:
                raise SystemExit(f"{qid}: expected one Q11 fraction, found {len(found)}")
            old = float(found[0][1])
            new = f"{old / SF:.10f}"
            sql = FRACTION.sub(lambda m: m.group(1) + new, sql)
            fractions[qid] = {"sf1": found[0][1], "sf10": new}
            entry.setdefault("parameters", {})["fraction"] = float(new)
        (OUT / f"{qid}.sql").write_text(sql, encoding="utf-8")
        entries.append(entry)

    sub = split[split["query_id"].isin(chosen)].sort_values("query_id")
    sub.to_csv(OUT / "split.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    (OUT / "generation.json").write_text(json.dumps({
        "source": "queries/tpch_sf1_v2", "scale_factor": SF, "seed": SEED,
        "per_template": {"pool": N_POOL, "test": N_TEST},
        "rule": "one numpy default_rng(0), templates in increasing order, pool then test, "
                "choice without replacement from the sorted query ids of the set",
        "q11_fraction": "divided by the scale factor and written with ten decimals",
        "q11_fractions": fractions,
        "queries": len(entries), "pool": int((sub["set"] == "pool").sum()), "test": int((sub["set"] == "test").sum()),
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} queries to {OUT}")


if __name__ == "__main__":
    main()
