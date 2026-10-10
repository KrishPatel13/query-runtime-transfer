"""The Join Order Benchmark (JOB) workload of the extension (paper/PLAN.md, E2).

fetch: downloads the 113 queries, the schema and the foreign-key indexes of
github.com/gregrahn/join-order-benchmark at a fixed commit. A query 17b
becomes q17_b, with template 17 and variant b. Writes queries/job_all (SQL
files and manifest.json), sql/job_schema.sql and sql/job_fkindexes.sql, and
the SHA-256 of every downloaded file in queries/job_all/source.json.

keep: reads the screening run (src/collect.py with one run and no warm-up
on M3_job, into a runtime_dataset.csv) and keeps the queries that finished
within --keep-s seconds (60). A template needs at least two kept queries to
be split, so a template with one is dropped. Inside every kept template,
with one numpy generator seeded 0 and the templates in order, round(0.3 n)
queries (at least one) go to the test set and the rest to the pool. Writes
queries/job_kept (SQL files, manifest.json, split.csv, screen.json).

Usage, from project_proposal:
  python scripts/job_queries.py fetch
  python scripts/job_queries.py keep --screen <runtime_dataset.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
COMMIT = "a39603662e023e449cb2121997a5034df9e02ebf"
RAW = f"https://raw.githubusercontent.com/gregrahn/join-order-benchmark/{COMMIT}/"
ALL, KEPT = HERE / "queries" / "job_all", HERE / "queries" / "job_kept"
NAME = re.compile(r"^(\d+)([a-z])\.sql$")
SEED, TEST_SHARE = 0, 0.3


def get(name: str) -> bytes:
    with urllib.request.urlopen(RAW + name, timeout=60) as r:
        return r.read()


def query_names() -> list[str]:
    url = f"https://api.github.com/repos/gregrahn/join-order-benchmark/contents?ref={COMMIT}"
    with urllib.request.urlopen(url, timeout=60) as r:
        names = [e["name"] for e in json.load(r)]
    return sorted((n for n in names if NAME.match(n)), key=lambda n: (int(NAME.match(n)[1]), NAME.match(n)[2]))


def fetch(_args) -> None:
    if ALL.exists():
        raise SystemExit(f"{ALL} exists; remove it to fetch again")
    ALL.mkdir(parents=True)
    names = query_names()
    if len(names) != 113:
        raise SystemExit(f"expected 113 JOB queries at {COMMIT}, found {len(names)}")
    sha, manifest = {}, []
    for n in names:
        data = get(n)
        t, v = int(NAME.match(n)[1]), NAME.match(n)[2]
        qid = f"q{t:02d}_{v}"
        (ALL / f"{qid}.sql").write_bytes(data)
        sha[n] = hashlib.sha256(data).hexdigest()
        manifest.append({"query_id": qid, "template": t, "variant": v, "parameters": {"job_name": n[:-4]}})
    for src, dest in (("schema.sql", "job_schema.sql"), ("fkindexes.sql", "job_fkindexes.sql")):
        data = get(src)
        (HERE / "sql" / dest).write_bytes(data)
        sha[src] = hashlib.sha256(data).hexdigest()
    (ALL / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (ALL / "source.json").write_text(json.dumps({"repository": "https://github.com/gregrahn/join-order-benchmark",
                                                 "commit": COMMIT, "sha256": sha}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} queries to {ALL}")


def keep(args) -> None:
    if KEPT.exists():
        if not args.force:
            raise SystemExit(f"{KEPT} exists; pass --force to replace it")
        shutil.rmtree(KEPT)
    manifest = {e["query_id"]: e for e in json.loads((ALL / "manifest.json").read_text(encoding="utf-8"))}
    screen = pd.read_csv(args.screen)
    if set(screen["query_id"]) != set(manifest):
        raise SystemExit("the screening run does not cover exactly the queries of queries/job_all")
    screen["keep"] = (screen["status"] == "ok") & (screen["runtime_ms"] <= args.keep_s * 1000)
    kept = screen.loc[screen["keep"], "query_id"]
    per_template = kept.map(lambda q: manifest[q]["template"]).value_counts()
    single = sorted(int(t) for t, n in per_template.items() if n < 2)
    kept = sorted(q for q in kept if manifest[q]["template"] not in single)

    rng = np.random.default_rng(SEED)
    rows = []
    for t in sorted({manifest[q]["template"] for q in kept}):
        ids = sorted(q for q in kept if manifest[q]["template"] == t)
        n_test = max(1, round(TEST_SHARE * len(ids)))
        test = set(rng.choice(ids, size=n_test, replace=False))
        rows += [{"query_id": q, "template": t, "variant": manifest[q]["variant"],
                  "set": "test" if q in test else "pool"} for q in ids]
    KEPT.mkdir(parents=True)
    for q in kept:
        shutil.copy2(ALL / f"{q}.sql", KEPT / f"{q}.sql")
    (KEPT / "manifest.json").write_text(json.dumps([manifest[q] for q in kept], indent=1) + "\n", encoding="utf-8")
    split = pd.DataFrame(rows)
    split.to_csv(KEPT / "split.csv", index=False)
    dropped = screen.loc[~screen["query_id"].isin(kept)]
    (KEPT / "screen.json").write_text(json.dumps({
        "keep_s": args.keep_s, "screen": str(Path(args.screen).name),
        "queries": len(screen), "kept": len(kept), "templates": int(split["template"].nunique()),
        "pool": int((split["set"] == "pool").sum()), "test": int((split["set"] == "test").sum()),
        "dropped_templates_single_query": single,
        "dropped": [{"query_id": r.query_id, "status": r.status, "runtime_ms": r.runtime_ms}
                    for r in dropped.itertuples()],
        "split_rule": "per template, numpy default_rng(0) over templates in order: round(0.3 n) test, "
                      "at least one, the rest pool",
    }, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} of {len(screen)} queries in {split['template'].nunique()} templates; wrote {KEPT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(func=fetch)
    k = sub.add_parser("keep")
    k.add_argument("--screen", required=True, help="runtime_dataset.csv of the screening run")
    k.add_argument("--keep-s", type=float, default=60.0)
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=keep)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
