"""Generate distinct parameterized TPC-H queries with qgen, rewritten for PostgreSQL.

qgen draws substitution parameters following the TPC-H specification. Its
PostgreSQL output still needs three fixes before PostgreSQL accepts it:
  * the row limit comes out as a separate "limit N;" line after the statement's
    own semicolon ("limit -1" means no limit);
  * Q1 writes "interval '90' day (3)", and PostgreSQL rejects the precision;
  * Q15 is three statements (create view, select, drop view). It is rewritten
    as one statement with a WITH clause, which is the usual single-statement
    form of Q15.

Each template is sampled with its own seeded stream of qgen seeds, and
duplicates are rejected. Two queries count as duplicates when they have the
same multiset of literal values, which also catches the same parameters in a
different order (Q12's two ship modes, Q7's pair of nations). This test can
reject a few distinct queries whose literals happen to be a permutation of
each other, but it never keeps a real duplicate. Some templates have few
distinct parameterizations (Q18 has 4, Q13 has 16), so a template can end up
with fewer than --per-template queries; the script reports this.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path

LIMIT_RE = re.compile(r";\s*limit\s+(-?\d+)\s*;\s*$", re.IGNORECASE)
VIEW_RE = re.compile(
    r"create view (\w+)\s*\(([^)]*)\)\s*as\s*(.*?);\s*(select.*?);\s*drop view \1\s*$",
    re.IGNORECASE | re.DOTALL,
)


def to_postgres(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("--")]
    sql = "\n".join(lines).strip()

    limit = None
    m = LIMIT_RE.search(sql)
    if m:
        limit = int(m.group(1))
        sql = sql[: m.start()]
    sql = sql.rstrip().rstrip(";").rstrip()

    sql = re.sub(r"(interval\s+'\d+'\s+day)\s*\(\s*\d+\s*\)", r"\1", sql, flags=re.IGNORECASE)

    v = VIEW_RE.search(sql)
    if v:
        name, cols, body, main = v.groups()
        sql = f"with {name} ({cols.strip()}) as (\n{body.strip()}\n)\n{main.strip()}"

    if limit is not None and limit > 0:
        sql += f"\nlimit {limit}"
    if ";" in sql:
        raise ValueError("more than one statement after rewriting:\n" + sql)
    return sql + ";\n"


LITERAL_RE = re.compile(r"'[^']*'|\b\d+(?:\.\d+)?\b|\.\d+\b")


def dedupe_key(sql: str) -> tuple[str, ...]:
    """Sorted literal values of a query; equal for reordered parameters."""
    return tuple(sorted(LITERAL_RE.findall(sql.lower())))


def run_qgen(qgen: str, template: int, sf: str, seed: int, env: dict, cwd: str) -> str:
    cmd = [qgen, "-r", str(seed), "-s", sf, str(template)]
    p = subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"qgen failed for template {template}: {p.stderr.strip()}")
    return p.stdout


def parse_templates(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qgen", required=True, help="path to the qgen executable")
    ap.add_argument("--query-template-dir", required=True, help="directory holding 1.sql ... 22.sql")
    ap.add_argument("--out", default="queries")
    ap.add_argument("--templates", default="1,3,5,6,14", help="comma list, ranges allowed (e.g. 1-22)")
    ap.add_argument("--per-template", type=int, default=20)
    ap.add_argument("--scale", default="1")
    ap.add_argument("--seed", type=int, default=680, help="base seed for drawing qgen seeds")
    ap.add_argument("--max-tries", type=int, default=3000, help="qgen calls per template before giving up")
    ap.add_argument("--force", action="store_true", help="delete existing .sql files in --out first")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old = list(out.glob("*.sql"))
    if old and not args.force:
        raise SystemExit(f"{out} already holds {len(old)} .sql files; pass --force to replace them")
    for f in old:
        f.unlink()
    qgen = str(Path(args.qgen).resolve())
    kit = str(Path(qgen).parent)  # qgen reads dists.dss from its own directory
    env = os.environ.copy()
    env["DSS_QUERY"] = str(Path(args.query_template_dir).resolve())
    env["DSS_CONFIG"] = kit

    manifest = []
    for t in parse_templates(args.templates):
        rng = random.Random(args.seed * 1000 + t)
        seen: set[tuple[str, ...]] = set()
        made = tries = 0
        while made < args.per_template and tries < args.max_tries:
            tries += 1
            seed = rng.randrange(1, 2**31 - 1)
            sql = to_postgres(run_qgen(qgen, t, args.scale, seed, env, kit))
            key = dedupe_key(sql)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            qid = f"q{t:02d}_{made:03d}"
            (out / f"{qid}.sql").write_text(sql, encoding="utf-8")
            manifest.append({"query_id": qid, "template": t, "qgen_seed": seed})
        note = "" if made == args.per_template else f"  (only {made} distinct after {tries} tries)"
        print(f"template {t:2d}: {made} queries{note}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
