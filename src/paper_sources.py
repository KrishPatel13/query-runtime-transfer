"""Macros and a table for the paper's other source models, from a paper_eval run over them.

Reads <run>/summary.json(.gz) of `paper_eval.py run --sources forest boosting ridge cost`
and writes <run>/sources.tex (macros) and <run>/table_sources.tex: for every
source model and workload, the geometric mean over all pairs of the speed
factor, the ridge correction and the primary method (round robin) at the
budget where every template can have a query (k = 25 for TPC-H, 100 for
TPC-DS), the oracle, and in how many pairs the primary method beats the
speed factor with an interval below 1.

Usage: python src/paper_sources.py --run results/paper/sources
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from paper_report import GENERATED, WORKLOAD_NAME, fmt, geo, tex_name

K = {"tpch": 25, "tpcds": 100}
LABEL = {"forest": "Random forest", "boosting": "Gradient boosting", "ridge": "Ridge", "cost": "Cost line",
         "hybrid": "Cost line + forest"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    p = run / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8") if p.exists()
                   else gzip.decompress((run / "summary.json.gz").read_bytes()).decode("utf-8"))
    rows, macros = [], {}
    for w in ("tpch", "tpcds"):
        k = K[w]
        for src in ("forest", "boosting", "ridge", "cost"):
            js = [j for j in s["jobs"] if j["workload"] == w and j["source"] == src]
            if not js:
                continue

            def q(key):
                return geo([j["c2"].get(key, {}).get("q") for j in js])
            prim = [j["c2"].get(f"round-robin|lowdim+template|{k}", {}) for j in js]
            wins = sum(1 for c in prim if c.get("vs_speed", {}).get("gm_hi", 9) < 1)
            r = {"workload": w, "source": src, "pairs": len(js), "k": k,
                 "none": q(f"random|none|{k}"), "speed": q(f"random|speed|{k}"), "ridge": q(f"random|ridge|{k}"),
                 "primary": q(f"round-robin|lowdim+template|{k}"), "oracle": q(f"random|oracle|{k}"), "wins": wins}
            rows.append(r)
            for f in ("none", "speed", "ridge", "primary", "oracle"):
                macros[tex_name("Src", w, src, f)] = fmt(r[f])
            macros[tex_name("Src", w, src, "wins")] = str(wins)
            macros[tex_name("Src", w, src, "pairs")] = str(len(js))
    allwins = sum(r["wins"] for r in rows)
    allpairs = sum(r["pairs"] for r in rows)
    macros["SrcWinsTotal"] = str(allwins)
    macros["SrcPairsTotal"] = str(allpairs)
    lines = [GENERATED] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (run / "sources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    t = [GENERATED, "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Workload & Source model & None & Speed & Ridge & Primary & Oracle \\\\", "\\midrule"]
    for r in rows:
        t.append(" & ".join([f"{WORKLOAD_NAME[r['workload']]} ($k={r['k']}$)", LABEL[r["source"]], fmt(r["none"]),
                             fmt(r["speed"]), fmt(r["ridge"]), fmt(r["primary"]), fmt(r["oracle"])]) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (run / "table_sources.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    for r in rows:
        print(r)
    print(f"primary beats speed (interval below 1) in {allwins} of {allpairs} source-pair cases")


if __name__ == "__main__":
    main()
