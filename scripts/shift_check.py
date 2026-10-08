"""Acceptance test for a target machine after its first pass.

FINAL_PROPOSAL_PLAN.md (section 3) accepts a target machine only if its
slowdown differs between templates: the share of the variance of
r = log(target runtime / source label) between templates must exceed the
share seen between two unchanged pilot passes (about 14%). A target that
slows every template alike is one speed factor, and the ridge correction
would have nothing to learn there. This runs the test on one pass of the
target, so that it can run before the target's other passes; adapt_eval.py
reports the same statistic on the final labels.

Reads results/runs/<source>/labels.csv and
results/runs/<target>/pass<p>/runtime_dataset.csv, and prints the template
share, its one-way ANOVA p-value, the median and per-template slowdowns
and the verdict. Exit status 0 if accepted, 1 if not.

Usage: python scripts/shift_check.py M1 M2 [--pass 1] [--runs-dir results/runs]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import stats  # noqa: E402
from adapt_eval import pilot_share_before  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--pass", dest="pass_no", type=int, default=1)
    ap.add_argument("--runs-dir", default=str(HERE / "results" / "runs"))
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    src = pd.read_csv(runs / args.source / "labels.csv")
    src = src[src["status"] == "ok"].set_index("query_id")
    tgt = pd.read_csv(runs / args.target / f"pass{args.pass_no}" / "runtime_dataset.csv")
    tgt = tgt[tgt["status"] == "ok"].set_index("query_id")
    ids = src.index.intersection(tgt.index)
    r = np.log(tgt.loc[ids, "runtime_ms"].to_numpy(float) / src.loc[ids, "label_ms"].to_numpy(float))
    t = src.loc[ids, "template"].to_numpy()
    share, p = stats.template_share(r, t)
    threshold = pilot_share_before()
    per_t = pd.Series(np.exp(r)).groupby(t).median()

    print(f"{args.target} pass {args.pass_no} against {args.source}'s labels: {len(ids)} queries, "
          f"{len(per_t)} templates")
    print(f"median slowdown {np.exp(np.median(r)):.2f}; per template (median) "
          f"{per_t.min():.2f} (template {per_t.idxmin()}) to {per_t.max():.2f} (template {per_t.idxmax()})")
    print(f"template share {100 * share:.1f}% (ANOVA p = {p:.1e}); threshold "
          + ("unknown" if threshold is None else f"{100 * threshold:.1f}%"))
    accepted = threshold is not None and np.isfinite(share) and share > threshold
    print("accepted" if accepted else "not accepted")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
