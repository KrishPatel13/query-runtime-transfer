"""Group TPC-DS templates that are near copies of each other into families.

Many TPC-DS templates differ from another one only in the sales channel or
in one filter column: 37 and 82 differ only in catalog_sales against
store_sales, and 56 and 60 share almost every token. The unseen-template
test (C4) holds out one unit at a time. When the unit is the template, a
held-out template's near copy stays in training, the residual forest has in
effect seen it, and the two-parameter cost line gains nothing, so the test
would lean toward the correction model. C4 therefore holds out families.

The rule is fixed before any TPC-DS label exists: two templates are linked
when the token similarity of their first queries, with literal values
masked, is at least THRESHOLD; a family is a connected component of linked
templates, named by its smallest template number. Similarity is difflib's
ratio of matching tokens (2M / total) on the lower-cased token sequences,
with every string or number literal replaced by one placeholder token. For
the two-statement templates (14, 23, 24, 39) the first query is both parts
joined.

Writes "family" into every entry of the query directory's manifest.json and
the pair scores at or above REPORT_FROM, the families and the rule into
families.json there.

Usage (standard library only; no database needed):
  python3 scripts/tpcds_families.py                           # queries/tpcds_sf1
  python3 scripts/tpcds_families.py --query-dir queries/tpcds_sf1_kept
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
THRESHOLD = 0.85
REPORT_FROM = 0.70
TOKEN_RE = re.compile(r"'[^']*'|\d+(?:\.\d+)?|\.\d+|[a-z_][a-z0-9_]*|\"[^\"]*\"|<=|>=|<>|!=|\|\||\S")


def tokens(sql: str) -> list[str]:
    """Lower-cased tokens with every string or number literal as '?'."""
    out = []
    for t in TOKEN_RE.findall(sql.lower()):
        out.append("?" if t[0] == "'" or t[0].isdigit() or (t[0] == "." and t[1:2].isdigit()) else t)
    return out


def similarity(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def first_query_text(manifest: list[dict], qdir: Path) -> dict[int, str]:
    """For every template, the text of its lowest-numbered draw (all parts joined)."""
    by_draw: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for e in manifest:
        draw = e["query_id"].rstrip("abcdefgh")
        by_draw[e["template"]][draw].append(e["query_id"])
    texts = {}
    for t, draws in by_draw.items():
        first = min(draws)
        texts[t] = "\n".join((qdir / f"{q}.sql").read_text(encoding="utf-8") for q in sorted(draws[first]))
    return texts


def families(texts: dict[int, str], threshold: float = THRESHOLD) -> tuple[dict[int, int], list[dict]]:
    """Family of every template (the smallest template number of its component), and pair scores."""
    toks = {t: tokens(s) for t, s in texts.items()}
    ts = sorted(toks)
    parent = {t: t for t in ts}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            # The ratio is at most 2 min(n, m) / (n + m); skip pairs that cannot reach the report level.
            na, nb = len(toks[a]), len(toks[b])
            if 2 * min(na, nb) / (na + nb) < REPORT_FROM:
                continue
            r = similarity(toks[a], toks[b])
            if r >= REPORT_FROM:
                pairs.append({"a": a, "b": b, "similarity": round(r, 4)})
            if r >= threshold:
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    return {t: find(t) for t in ts}, sorted(pairs, key=lambda p: -p["similarity"])


def annotate(qdir: Path, threshold: float = THRESHOLD) -> dict:
    """Add 'family' to manifest.json in qdir and write families.json; return the record."""
    path = qdir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fam, pairs = families(first_query_text(manifest, qdir), threshold)
    for e in manifest:
        e["family"] = fam[e["template"]]
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    groups = defaultdict(list)
    for t, f in fam.items():
        groups[f].append(t)
    record = {
        "rule": (f"templates are linked when the token similarity (difflib ratio, literals masked) of their "
                 f"first queries is at least {threshold}; a family is a connected component of linked "
                 f"templates, named by its smallest template"),
        "threshold": threshold,
        "templates": len(fam),
        "families": len(groups),
        "multi_template_families": {str(f): ts for f, ts in sorted(groups.items()) if len(ts) > 1},
        "pairs_from": REPORT_FROM,
        "pairs": pairs,
    }
    (qdir / "families.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", default=str(HERE / "queries" / "tpcds_sf1"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    rec = annotate(Path(args.query_dir), args.threshold)
    print(f"{rec['templates']} templates in {rec['families']} families; families of more than one template:")
    for f, ts in rec["multi_template_families"].items():
        print(f"  {f}: {ts}")


if __name__ == "__main__":
    main()
