"""Generate parameterized TPC-DS queries with dsqgen, rewritten for PostgreSQL.

dsqgen, from the TPC-DS tools, fills in each template's substitution
parameters by the rules of the specification. For every template this script
runs dsqgen with --per-template different seeds, keeps the queries whose
parameters differ, and rewrites each into single statements that PostgreSQL
18 accepts. Every rewrite is one of the minor query modifications that clause
4.2.3.4 of the specification (v2.10.0) allows without approval, or the
row-limit syntax of clause 4.2.4.1:
  * Row limit. The kit's ansi.tpl dialect writes SQL Server's "select top N".
    The PostgreSQL dialect, postgres.tpl, is ansi.tpl with "limit N" at the
    end of the query instead, as in the kit's netezza.tpl. The script derives
    it from ansi.tpl and saves a copy next to the queries.
  * Date arithmetic. Fifteen templates add days to a date as "+ 30 days",
    which PostgreSQL rejects. It becomes "+ interval '30 days'" (4.2.3.4 f.1,
    vendor-specific date expressions). Quoted column names such as
    "31-60 days" in templates 50, 62 and 99 are left alone.
  * An alias inside an ORDER BY expression. Templates 36, 70 and 86 order by
    "case when lochierarchy = 0 then ... end", where lochierarchy is a
    select-list alias for a sum of grouping() calls. PostgreSQL accepts an
    output alias in ORDER BY only as a bare name, so inside an expression the
    alias is replaced by the expression it stands for, in parentheses
    (4.2.3.4 e.2, ORDER BY <expression>).
  * Two statements. Templates 14, 23, 24 and 39 each hold two queries. Each
    becomes its own file, part a and part b. Both parts come from one dsqgen
    run, so they share the seed and the parameter values.
Nothing else needs to change on PostgreSQL 18.6: since version 16 a subquery
in FROM may omit its alias (templates 2 and 14 do), and query 30 in this kit
already uses the schema's column name c_last_review_date_sk.

Query IDs are q<template>_<n>, e.g. q07_003, with a part letter for the
two-statement templates, e.g. q14_003a and q14_003b. The part before the
first underscore is the template number that src/collect.py reads, so both
parts of a template land in one template group, which is the unit held out
together in the unseen-template test.

Queries are drawn as for TPC-H (scripts/generate_qgen_queries.py): each
template has its own seeded stream of dsqgen seeds, and a query that repeats
an earlier one is skipped. For the two-statement templates each part is
tested on its own: a draw whose parts all repeat is skipped, and a draw with
a new part writes only its new parts (part a of template 14 depends only on
the year, which has three values at SF1, so 14 has fewer a parts than b
parts). Two queries count as the same when their texts
agree apart from literal values and their literal values are the same
multiset (query_key), so a list of states drawn in another order is a
repeat, while a different choice of column or aggregate function is not.
At scale factor 1 some parameters have few possible values (template 2 only
varies the year, 1998 to 2001, and the store table has stores in one state
only), so a template can have fewer than --per-template distinct queries;
the script reports this.

dsqgen has two quirks. It reads its distributions (tpcds.idx) from its
working directory, so it runs in the kit's tools directory. It also aborts
with a heap error when the template directory's path is longer than about 80
characters, so the templates are copied to a short temporary directory.

Writes to --out: the .sql files; manifest.json with, for every query,
query_id, template (the template group), part (a, b, or null), template_file,
dsqgen_seed and parameters (the values from dsqgen's parameter log, without
its internal variables, whose names start with an underscore); postgres.tpl;
generation.json (the kit, the counts per template, the rewrites applied
and the EXPLAIN check); families.json with the template families of
scripts/tpcds_families.py, whose family field is added to every manifest
entry; and query_numbers.tex with the counts as LaTeX macros (prefix DS). Every query is then checked with plain EXPLAIN on the
TPC-DS cluster (--database-url; --no-check skips it), and the script fails if
PostgreSQL rejects any of them.

Usage (inside WSL, after scripts/tpcds_setup.sh and scripts/tpcds_load.sh):
  python3 scripts/tpcds_queries.py                     # 5 queries per template, checked
  python3 scripts/tpcds_queries.py --force             # replace an existing workload
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from generate_qgen_queries import LITERAL_RE, dedupe_key, parse_templates  # noqa: E402

# The TPC-DS cluster made by scripts/tpcds_setup.sh.
DEFAULT_URL = "postgresql:///tpcds?host=/tmp&port=5433"
# dsqgen overflows a path buffer somewhere above 80 characters.
MAX_TEMPLATE_DIR_CHARS = 60
LIMIT_DEFINES = {"__LIMITA": "", "__LIMITB": "", "__LIMITC": "limit %d"}
DEFINE_RE = re.compile(r'^(\s*define\s+(__LIMIT[ABC])\s*=\s*)"[^"]*"(\s*;.*)$', re.IGNORECASE)
DIALECT_NOTE = ("-- PostgreSQL dialect for dsqgen, written by scripts/tpcds_queries.py: the kit's\n"
                "-- ansi.tpl with the row limit as LIMIT at the end of the query, not TOP after SELECT.\n")

COMMENT_RE = re.compile(r"^[ \t]*--.*$", re.MULTILINE)
QUOTED_RE = re.compile(r"('(?:[^']|'')*'|\"[^\"]*\")")
DAYS_RE = re.compile(r"([+-])\s*(\d+)\s+days\b", re.IGNORECASE)
GROUPING_ALIAS_RE = re.compile(
    r"(grouping\s*\(\s*\w+\s*\)(?:\s*\+\s*grouping\s*\(\s*\w+\s*\))*)\s+as\s+(\w+)", re.IGNORECASE)
ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
LIMIT_TAIL_RE = re.compile(r"\s+limit\s+\d+\s*$", re.IGNORECASE)
BARE_ITEM_RE = re.compile(r"^\s*\w+(?:\s+(?:asc|desc))?(?:\s+nulls\s+(?:first|last))?\s*$", re.IGNORECASE)
# One line of dsqgen's parameter log: "\tYEAR.01 = 2001". Internal variables
# (_BEGIN, __LIMITC, ...) start with an underscore and are not matched.
PARAM_RE = re.compile(r"^\s*([A-Za-z]\w*)\.(\d+) = ?(.*?)\s*$")
PARTS = "ab"


def postgres_dialect(ansi: str) -> str:
    """The kit's ansi.tpl with the row limit written as a trailing LIMIT."""
    lines, found = [], set()
    for line in ansi.splitlines():
        m = DEFINE_RE.match(line)
        if m:
            name = m.group(2).upper()
            found.add(name)
            line = f'{m.group(1)}"{LIMIT_DEFINES[name]}"{m.group(3)}'
        lines.append(line)
    missing = sorted(set(LIMIT_DEFINES) - found)
    if missing:
        raise ValueError(f"ansi.tpl does not define {', '.join(missing)}")
    return DIALECT_NOTE + "\n".join(lines) + "\n"


def _outside_quotes(sql: str, fn) -> str:
    """Apply fn to the parts of sql that are not string literals or quoted names."""
    parts = QUOTED_RE.split(sql)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def interval_days(sql: str) -> str:
    """Write "+ 30 days" as "+ interval '30 days'"."""
    return _outside_quotes(sql, lambda s: DAYS_RE.sub(r"\1 interval '\2 days'", s))


def split_top_level(text: str) -> list[str]:
    """Split text at the commas that are not inside parentheses."""
    items, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return items


def inline_order_by_aliases(sql: str) -> str:
    """Replace a grouping() alias inside an ORDER BY expression by its expression.

    Only the statement's own ORDER BY (the last one, outside parentheses) is
    changed, and a bare alias such as "lochierarchy desc" is kept.
    """
    aliases = {m.group(2): m.group(1) for m in GROUPING_ALIAS_RE.finditer(sql)}
    orders = list(ORDER_BY_RE.finditer(sql))
    if not aliases or not orders:
        return sql
    head, clause = sql[:orders[-1].end()], sql[orders[-1].end():]
    if head.count("(") != head.count(")"):
        return sql
    tail = ""
    m = LIMIT_TAIL_RE.search(clause)
    if m:
        clause, tail = clause[:m.start()], clause[m.start():]
    items = []
    for item in split_top_level(clause):
        if not BARE_ITEM_RE.match(item):
            for alias, expr in aliases.items():
                item = re.sub(rf"\b{re.escape(alias)}\b", lambda _m, e=expr: f"({e})", item, flags=re.IGNORECASE)
        items.append(item)
    return head + ",".join(items) + tail


def to_postgres(text: str) -> tuple[list[str], list[str]]:
    """dsqgen output as PostgreSQL statements, and the names of the rewrites used.

    Each statement is returned on its own, ending in a single semicolon.
    """
    sql = COMMENT_RE.sub("", text)
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if not statements:
        raise ValueError("dsqgen wrote no statement")
    fixes: list[str] = []
    out = []
    for s in statements:
        s1 = interval_days(s)
        s2 = inline_order_by_aliases(s1)
        if s1 != s and "interval" not in fixes:
            fixes.append("interval")
        if s2 != s1 and "order_by_alias" not in fixes:
            fixes.append("order_by_alias")
        out.append(s2 + ";\n")
    if len(out) > 1:
        fixes.append("split")
    return out, fixes


def parse_param_log(text: str) -> dict:
    """Substitution values from dsqgen's -LOG file.

    A parameter with one value maps to that value; one with several (such as
    a list of states) maps to the list in dsqgen's order.
    """
    vals: dict[str, dict[int, str]] = {}
    for line in text.splitlines():
        m = PARAM_RE.match(line)
        if m:
            vals.setdefault(m.group(1), {})[int(m.group(2))] = m.group(3)
    return {k: v[1] if list(v) == [1] else [v[i] for i in sorted(v)] for k, v in sorted(vals.items())}


def query_key(sql: str) -> tuple[str, ...]:
    """Duplicate test: the text with its literals blanked out, plus the sorted literals.

    generate_qgen_queries.dedupe_key compares only the literals, which is
    enough for TPC-H, where every parameter is a value. Several TPC-DS
    templates also choose a column or an aggregate function (SELECTONE in 4,
    11, 47 and 57, AGG in 29, AGGONE in 35 and 74), which changes the text
    but not the literals.
    """
    skeleton = " ".join(LITERAL_RE.sub("?", sql.lower()).split())
    return (skeleton, *dedupe_key(sql))


def query_ids(template: int, n: int, parts: int) -> list[str]:
    """File names (without .sql) for the n-th query of a template."""
    base = f"q{template:02d}_{n:03d}"
    if parts == 1:
        return [base]
    if parts > len(PARTS):
        raise ValueError(f"template {template} has {parts} statements")
    return [base + PARTS[i] for i in range(parts)]


def run_dsqgen(tools: Path, template_dir: Path, template: int, scale: str, seed: int) -> tuple[str, dict]:
    log = template_dir / "params.log"
    log.unlink(missing_ok=True)
    cmd = [str(tools / "dsqgen"), "-DIRECTORY", str(template_dir), "-TEMPLATE", f"query{template}.tpl",
           "-DIALECT", "postgres", "-SCALE", scale, "-RNGSEED", str(seed), "-FILTER", "Y", "-QUIET", "Y",
           "-LOG", str(log)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=tools)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"dsqgen failed for template {template} with seed {seed} "
                           f"(exit {p.returncode}): {p.stderr.strip()}")
    return p.stdout, parse_param_log(log.read_text(encoding="utf-8"))


def version_line(text: str) -> str:
    """The line of a TPC tool's help text that names its version."""
    return next((ln.strip() for ln in text.splitlines() if "(Version" in ln), "")


def kit_info(kit: Path, tools: Path) -> dict:
    """Where the kit came from, its commit, and dsqgen's version line."""
    def run(cmd: list[str]) -> tuple[str, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=tools)
        except OSError:
            return "", ""
        return p.stdout, p.stderr
    # dsqgen prints its version line on stderr and the usage on stdout.
    help_out, help_err = run([str(tools / "dsqgen"), "-help"])
    return {"url": run(["git", "-C", str(kit), "remote", "get-url", "origin"])[0].strip(),
            "commit": run(["git", "-C", str(kit), "rev-parse", "HEAD"])[0].strip(),
            "dsqgen": version_line(help_err + help_out)}


def explain_check(database_url: str, out: Path, manifest: list[dict], timeout_s: int) -> dict:
    """Run plain EXPLAIN on every query; return what PostgreSQL rejected."""
    from collect import QueryError, explain, server_info
    errors = {}
    for e in manifest:
        sql = (out / f"{e['query_id']}.sql").read_text(encoding="utf-8").strip()
        if sql.count(";") != 1 or not sql.endswith(";"):
            errors[e["query_id"]] = "not exactly one statement"
            continue
        try:
            explain(database_url, sql, timeout_s * 1000)
        except QueryError as err:
            errors[e["query_id"]] = str(err).splitlines()[0]
    server = server_info(database_url, timeout_s * 1000)
    return {"checked_at": dt.datetime.now().isoformat(timespec="seconds"),
            "server_version": server["version"], "database_size": server["database_size"],
            "queries": len(manifest), "accepted": len(manifest) - len(errors), "errors": errors}


def draw_template(t: int, run_one, per_template: int, max_tries: int, rng: random.Random):
    """Draw up to per_template distinct queries of template t.

    run_one(seed) runs dsqgen and returns (text, parameters). Every part of
    a two-statement template is tested for repeats on its own: a draw whose
    parts all repeat is skipped, and a draw with a new part counts and writes
    only its new parts. Returns the files {query_id: sql}, their manifest
    entries and the counts for generation.json.
    """
    files: dict[str, str] = {}
    manifest: list[dict] = []
    seen_keys: list[set[tuple[str, ...]]] = []
    seen_seeds: set[int] = set()
    made = tries = 0
    parts, fixes = 0, []
    written: list[int] = []
    while made < per_template and tries < max_tries:
        tries += 1
        seed = rng.randrange(1, 2**31 - 1)
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        text, params = run_one(seed)
        statements, fixes = to_postgres(text)
        if parts and len(statements) != parts:
            raise SystemExit(f"template {t} gave {len(statements)} statements, earlier {parts}")
        parts = len(statements)
        while len(seen_keys) < parts:
            seen_keys.append(set())
            written.append(0)
        keys = [query_key(sql) for sql in statements]
        new = [k not in seen_keys[i] for i, k in enumerate(keys)]
        if not any(new):
            continue
        made += 1
        for i, (qid, sql) in enumerate(zip(query_ids(t, made, parts), statements)):
            if not new[i]:
                continue
            seen_keys[i].add(keys[i])
            written[i] += 1
            files[qid] = sql
            manifest.append({"query_id": qid, "template": t, "part": qid[-1] if parts > 1 else None,
                             "template_file": f"query{t}.tpl", "dsqgen_seed": seed, "parameters": params})
    info = {"queries": made, "statements": parts, "files": sum(written), "files_per_part": written,
            "tries": tries, "rewrites": fixes}
    return files, manifest, info


def annotate_families(out: Path) -> dict:
    """Template families (scripts/tpcds_families.py) written into the manifest."""
    from tpcds_families import annotate
    return annotate(out)


def write_macros(path: Path, record: dict) -> None:
    """The workload's counts as LaTeX macros, so documents never type them by hand."""
    by = record["by_template"]
    check = record.get("explain_check") or {}
    values = {
        "DSTemplates": record["templates"],
        "DSFamilies": record["families"]["families"],
        "DSQueryFiles": record["query_files"],
        "DSDraws": sum(v["queries"] for v in by.values()),
        "DSPerTemplate": record["per_template_requested"],
        "DSTwoPartTemplates": len(record["two_statement_templates"]),
        "DSShortTemplates": len(record["fewer_than_per_template"]),
        "DSExplainAccepted": check.get("accepted", "--"),
    }
    path.write_text("% Generated by scripts/tpcds_queries.py. Do not edit by hand.\n" +
                    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()),
                    encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default=str(Path.home() / "cs680-pilot" / "tpcds-kit"),
                    help="the tpcds-kit checkout, with dsqgen built in tools/")
    ap.add_argument("--out", default=str(HERE / "queries" / "tpcds_sf1"))
    ap.add_argument("--templates", default="1-99", help="comma list, ranges allowed")
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--scale", default="1")
    ap.add_argument("--seed", type=int, default=680, help="base seed for drawing dsqgen seeds")
    ap.add_argument("--max-tries", type=int, default=200, help="dsqgen calls per template before giving up")
    ap.add_argument("--database-url", default=DEFAULT_URL,
                    help="database for the EXPLAIN check (default: the TPC-DS cluster)")
    ap.add_argument("--no-check", action="store_true", help="skip the EXPLAIN check")
    ap.add_argument("--timeout-s", type=int, default=60, help="statement timeout for the EXPLAIN check")
    ap.add_argument("--force", action="store_true", help="delete existing .sql files in --out first")
    args = ap.parse_args()

    kit = Path(args.kit).expanduser().resolve()
    tools = kit / "tools"
    if not (tools / "dsqgen").exists():
        raise SystemExit(f"{tools / 'dsqgen'} not found; run scripts/tpcds_setup.sh first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old = sorted(out.glob("*.sql"))
    if old and not args.force:
        raise SystemExit(f"{out} already holds {len(old)} .sql files; pass --force to replace them")

    dialect = postgres_dialect((kit / "query_templates" / "ansi.tpl").read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    manifest: list[dict] = []
    per_template: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="dsq") as tmp:
        template_dir = Path(tmp)
        if len(str(template_dir)) > MAX_TEMPLATE_DIR_CHARS:
            raise SystemExit(f"temporary directory {template_dir} is too long for dsqgen; set TMPDIR=/tmp")
        for tpl in (kit / "query_templates").glob("*.tpl"):
            shutil.copy(tpl, template_dir / tpl.name)
        (template_dir / "postgres.tpl").write_text(dialect, encoding="utf-8")

        for t in parse_templates(args.templates):
            def run_one(seed: int, t: int = t):
                return run_dsqgen(tools, template_dir, t, args.scale, seed)
            t_files, t_manifest, info = draw_template(t, run_one, args.per_template, args.max_tries,
                                                      random.Random(args.seed * 1000 + t))
            files.update(t_files)
            manifest.extend(t_manifest)
            per_template[str(t)] = info
            made, parts, fixes = info["queries"], info["statements"], info["rewrites"]
            note = "" if made == args.per_template else f"  (only {made} distinct after {info['tries']} tries)"
            shown = f", rewrites: {', '.join(fixes)}" if fixes else ""
            print(f"template {t:2d}: {made} queries x {parts} statement(s), {info['files']} files{shown}{note}")

    for f in old:
        f.unlink()
    for qid, sql in files.items():
        (out / f"{qid}.sql").write_text(sql, encoding="utf-8", newline="\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "postgres.tpl").write_text(dialect, encoding="utf-8", newline="\n")

    short = {t: v["queries"] for t, v in per_template.items() if v["queries"] < args.per_template}
    record = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "kit": kit_info(kit, tools),
        "dialect": "postgres.tpl",
        "scale": args.scale,
        "per_template_requested": args.per_template,
        "base_seed": args.seed,
        "templates": len(per_template),
        "query_files": len(files),
        "two_statement_templates": [int(t) for t, v in per_template.items() if v["statements"] > 1],
        "fewer_than_per_template": short,
        "rewrites": {name: [int(t) for t, v in per_template.items() if name in v["rewrites"]]
                     for name in ("interval", "order_by_alias", "split")},
        "by_template": per_template,
        "explain_check": None,
    }

    def write_record():
        (out / "generation.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")

    write_record()
    fam = annotate_families(out)
    record["families"] = {"families": fam["families"], "multi_template_families": fam["multi_template_families"]}
    write_record()
    print(f"{len(files)} query files for {len(per_template)} templates ({fam['families']} families) in {out}")
    if not args.no_check:
        record["explain_check"] = explain_check(args.database_url, out, manifest, args.timeout_s)
        write_record()
    write_macros(out / "query_numbers.tex", record)
    check = record["explain_check"]
    if check:
        print(f"EXPLAIN accepted {check['accepted']} of {check['queries']} queries")
        for qid, msg in check["errors"].items():
            print(f"  {qid}: {msg}")
        if check["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
