"""Turn the passes measured on one machine into one label per query.

Reads results/runs/<machine>/pass<i>/ (written by src/run_passes.py; attempts
under rejected/ are not used) and writes next to them:

  labels.csv         one row per query: query_id, template, variant, status,
                     label_ms, pass<i>_ms for every pass, and the 19 plan
                     features (FEATURE_NAMES in src/features.py) of pass 1's
                     plan (of the first pass with a plan when pass 1 has
                     none; empty when no pass has one)
  noise.json         how closely the passes agree
  noise_numbers.tex  LaTeX macros with the numbers the documents quote

label_ms is the median over the passes of each pass's label, which is itself
the median of three timed runs, and pass<i>_ms is pass i's label. status is
"ok" when the query finished with status ok in every pass and PostgreSQL
chose the same plan shape in every pass (passcheck.shape: the tree and its
estimates). Otherwise status lists each problem, separated by "; ", for
example "pass2 timeout" or "pass3 plan differs"; the row stays, label_ms is
left empty, and so is pass<i>_ms for each pass that did not finish with
status ok. No query is dropped silently: a query of the query directory that
a pass never reached is "missing" in that pass.

variant is 'std' or 'wide' in TPC-H and the query's letter in JOB, from the
query directory's manifest.json when its entries have a variant, and 'std'
otherwise. The query directory is --query-dir, or else the one recorded in
pass 1's meta file, looked up by its name under queries/ when that path does
not exist on this computer.

Noise floor. For one query and two passes, the Q-error between the two pass
labels is max(a / b, b / a). noise.json reports its median and 90th
percentile over every pair of passes and every query with status ok
(pass_to_pass), and for each pair the comparison of passcheck.compare
(pairs: agreement, identical plans, and the level-shift test in the later
pass's order, on the queries with status ok in both passes).

Usage: python src/make_labels.py M1 [--runs-dir results/runs] [--query-dir queries/tpch_sf1_v2]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, extract_features
from passcheck import compare, macro_prefix, read_meta, read_plan, shape

HERE = Path(__file__).resolve().parents[1]
PASS_RE = re.compile(r"pass(\d+)")
VARIANT_RE = re.compile(r"std|wide|[a-z]")   # TPC-H's two kinds, or a JOB query's letter


def find_passes(machine_dir: Path) -> list[tuple[int, Path]]:
    """The finished pass directories pass1 ... passK of one machine."""
    if not machine_dir.is_dir():
        raise SystemExit(f"{machine_dir} does not exist")
    found = sorted((int(m.group(1)), p) for p in machine_dir.iterdir()
                   if p.is_dir() and (m := PASS_RE.fullmatch(p.name)))
    if not found:
        raise SystemExit(f"no pass directories in {machine_dir}")
    numbers = [i for i, _ in found]
    if numbers != list(range(1, len(found) + 1)):
        raise SystemExit(f"{machine_dir} has passes {numbers}; expected 1 to {len(found)} without gaps")
    for _, d in found:
        meta = read_meta(d)
        if not (d / "runtime_dataset.csv").exists() or not meta or meta[-1].get("finished_at") is None:
            raise SystemExit(f"{d} did not finish; complete it with src/run_passes.py first")
    return found


def find_query_dir(first_pass: Path, given: str | None) -> Path | None:
    if given:
        return Path(given)
    meta = read_meta(first_pass)
    recorded = (meta[0].get("args") or {}).get("query_dir") if meta else None
    if not recorded:
        return None
    if Path(recorded).is_dir():
        return Path(recorded)
    by_name = HERE / "queries" / PurePosixPath(recorded).name
    return by_name if by_name.is_dir() else None


def read_manifest(query_dir: Path | None) -> dict[str, dict]:
    if query_dir is None or not (query_dir / "manifest.json").exists():
        return {}
    return {e["query_id"]: e for e in json.loads((query_dir / "manifest.json").read_text(encoding="utf-8"))}


def template_from_id(qid: str) -> int | None:
    m = re.match(r"[qQ](\d+)", qid)
    return int(m.group(1)) if m else None


def build_labels(passes: list[tuple[int, Path]], frames: dict[int, pd.DataFrame],
                 manifest: dict[str, dict], extra_ids: set[str]) -> pd.DataFrame:
    """One row per query of the passes and extra_ids (the query directory)."""
    ids = set(extra_ids).union(*(set(f.index) for f in frames.values()))
    rows = []
    for qid in sorted(ids):
        issues: list[tuple[int, str]] = []
        values: dict[int, float] = {}
        plans = {i: read_plan(d / "plans", qid) for i, d in passes}
        template = None
        for i, _ in passes:
            f = frames[i]
            values[i] = math.nan
            if qid not in f.index:
                issues.append((i, "missing"))
                continue
            r = f.loc[qid]
            template = int(r["template"]) if template is None else template
            if r["status"] != "ok":
                issues.append((i, str(r["status"])))
            elif plans[i] is None:
                issues.append((i, "plan missing"))
            elif not float(r["runtime_ms"]) > 0:
                issues.append((i, "runtime not positive"))
            else:
                values[i] = float(r["runtime_ms"])
        # The reference plan is pass 1's, or the first saved one when pass 1
        # failed before its EXPLAIN.
        ref = next((i for i, _ in passes if plans[i] is not None), None)
        if ref is not None:
            for i, _ in passes:
                if i != ref and plans[i] is not None and shape(plans[i]) != shape(plans[ref]):
                    issues.append((i, "plan differs"))

        entry = manifest.get(qid, {})
        if "template" in entry:
            if template is not None and int(entry["template"]) != template:
                raise SystemExit(f"{qid}: template {entry['template']} in manifest.json, {template} in the passes")
            template = int(entry["template"])
        if template is None:
            template = template_from_id(qid)
        variant = entry.get("variant", "std")
        if not VARIANT_RE.fullmatch(str(variant)):
            raise SystemExit(f"{qid}: variant {variant!r} in manifest.json is not 'std', 'wide' or one letter")

        issues.sort(key=lambda t: t[0])
        status = "; ".join(f"pass{i} {text}" for i, text in issues) or "ok"
        row = {"query_id": qid, "template": template, "variant": variant, "status": status,
               "label_ms": float(np.median(list(values.values()))) if status == "ok" else math.nan}
        if "family" in entry:
            row["family"] = int(entry["family"])
        row |= {f"pass{i}_ms": values[i] for i, _ in passes}
        if ref is not None:
            row |= extract_features({"Plan": plans[ref]})
        rows.append(row)
    # TPC-DS manifests give each query its template family (scripts/tpcds_families.py), the unit C4
    # holds out; the column is left out when the manifest has none (TPC-H).
    fam = ["family"] if any("family" in r for r in rows) else []
    cols = (["query_id", "template"] + fam + ["variant", "status", "label_ms"] + [f"pass{i}_ms" for i, _ in passes]
            + FEATURE_NAMES)
    out = pd.DataFrame(rows, columns=cols)
    out["template"] = out["template"].astype("Int64")
    if fam:
        out["family"] = out["family"].astype("Int64")
    return out


def noise(labels: pd.DataFrame, passes: list[tuple[int, Path]], raw: dict[int, pd.DataFrame]) -> dict:
    """Pass-to-pass agreement, pooled over the pairs of passes and per pair."""
    ok = labels[labels["status"] == "ok"]
    pooled, pairs = [], {}
    for (i, di), (j, dj) in combinations(passes, 2):
        a, b = ok[f"pass{i}_ms"].to_numpy(float), ok[f"pass{j}_ms"].to_numpy(float)
        pooled.append(np.maximum(a / b, b / a))
        both = raw[i][raw[i]["status"] == "ok"].merge(raw[j][raw[j]["status"] == "ok"], on="query_id")
        # compare() needs a few queries in common to compute correlations.
        pairs[f"pass{i}_vs_pass{j}"] = compare(raw[i], raw[j], di / "plans", dj / "plans") if len(both) >= 3 else None
    q = np.concatenate(pooled) if pooled else np.array([])
    return {
        "passes": [i for i, _ in passes],
        "n_queries": int(len(labels)),
        "n_ok": int(len(ok)),
        "status_counts": {str(k): int(v) for k, v in labels["status"].value_counts().items()},
        "pass_to_pass": {"pairs": len(pooled), "n": int(len(q)), "median_q": float(np.median(q)),
                         "p90_q": float(np.quantile(q, 0.9)), "max_q": float(q.max())} if len(q) else None,
        "pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine", help="machine name, the directory under --runs-dir")
    ap.add_argument("--runs-dir", default=str(HERE / "results" / "runs"))
    ap.add_argument("--query-dir", default=None, help="default: the one recorded in pass 1's meta file")
    args = ap.parse_args()

    machine_dir = Path(args.runs_dir) / args.machine
    passes = find_passes(machine_dir)
    raw = {i: pd.read_csv(d / "runtime_dataset.csv", dtype={"query_id": str, "status": str}) for i, d in passes}
    frames = {i: df.set_index("query_id", drop=False) for i, df in raw.items()}
    query_dir = find_query_dir(passes[0][1], args.query_dir)
    if query_dir is None:
        print("query directory not found: every variant is 'std' and only measured queries are listed")
    manifest = read_manifest(query_dir)
    extra = {p.stem for p in query_dir.glob("*.sql")} if query_dir is not None else set()

    labels = build_labels(passes, frames, manifest, extra)
    labels.to_csv(machine_dir / "labels.csv", index=False)
    rejected = sorted(p.name for p in (machine_dir / "rejected").glob("pass*_attempt*")) \
        if (machine_dir / "rejected").is_dir() else []
    out = {"machine": args.machine, "query_dir": str(query_dir) if query_dir else None,
           "rejected_attempts": rejected} | noise(labels, passes, raw)
    (machine_dir / "noise.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    p = macro_prefix(args.machine)
    macros = {f"{p}Queries": out["n_queries"], f"{p}Labels": out["n_ok"], f"{p}Passes": len(passes),
              f"{p}Repeated": len(rejected)}
    if out["pass_to_pass"]:
        macros |= {f"{p}NoiseMedQ": f"{out['pass_to_pass']['median_q']:.2f}",
                   f"{p}NoisePninetyQ": f"{out['pass_to_pass']['p90_q']:.2f}"}
    (machine_dir / "noise_numbers.tex").write_text(
        "% Generated by src/make_labels.py. Do not edit by hand.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8", newline="\n")

    print(f"{machine_dir / 'labels.csv'}: {out['n_ok']} of {out['n_queries']} queries labeled from "
          f"{len(passes)} passes; status: {out['status_counts']}")
    if out["pass_to_pass"]:
        print(f"pass to pass: median Q {out['pass_to_pass']['median_q']:.3f}, "
              f"P90 Q {out['pass_to_pass']['p90_q']:.3f}")


if __name__ == "__main__":
    main()
