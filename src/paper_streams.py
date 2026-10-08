"""Macros for the paper's stream result, from a paper_eval run of the hybrid source model.

For every pair, paper_eval's stream switches from the source machine's test
queries to the target's without warning, and each update rule's recovery is
the number of target queries after which every batch stays within 10% of
its final level, counted per stream order and summarized by the median
(None when most orders never settle). This script writes <run>/streams.tex:

  StreamRecover          median over pairs of that recovery, for the primary
                         method refitted on the last 50 queries
  StreamRecoverRidge     the same for the ridge correction on the last 50
  StreamFinal<Rule>      geometric mean over pairs of each rule's final level
                         (the batch error at the end of the stream)
  StreamPairs            the number of pairs

Usage: python src/paper_streams.py --run results/paper/streams
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from paper_report import GENERATED, fmt, geo, tex_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    p = run / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8") if p.exists()
                   else gzip.decompress((run / "summary.json.gz").read_bytes()).decode("utf-8"))
    jobs = [j for j in s["jobs"] if j["source"] == "hybrid" and "stream" in j]
    rules = list(jobs[0]["stream"]["levels"])
    m = {"StreamPairs": str(len(jobs))}
    rec = {}
    for r in rules:
        vals = [j["stream"]["levels"][r]["recover_median"] for j in jobs]
        rec[r] = vals
        finite = [v for v in vals if v is not None]
        med = float(np.median([np.inf if v is None else v for v in vals]))
        m[tex_name("StreamRecover", r)] = "never" if not np.isfinite(med) else f"{med:.0f}"
        m[tex_name("StreamFinal", r)] = fmt(geo([j["stream"]["levels"][r]["final"] for j in jobs]))
        m[tex_name("StreamNever", r)] = str(sum(v is None for v in vals))
        print(r, "median recovery", m[tex_name("StreamRecover", r)], "final", m[tex_name("StreamFinal", r)],
              "never", sum(v is None for v in vals), "of", len(vals), "finite", sorted(finite)[:5], "...")
    m["StreamRecover"] = m[tex_name("StreamRecover", "lowdim+template-last50")]
    m["StreamRecoverRidge"] = m[tex_name("StreamRecover", "ridge-last50")]
    lines = [GENERATED] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (run / "streams.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
