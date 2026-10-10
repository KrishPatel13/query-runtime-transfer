"""The first round's numbers on the second round's environment pairs, and the second round's test on new
templates summarized as in the first (exploratory; not a registered test).

Written on 10 October 2026, before any second-round result existed, so that its definitions are fixed
blind. It reads the first round's evaluation (--final, results/paper/final) and, when it exists, the
second round's (--ext, results/paper/ext), and writes <out>/compare_numbers.tex (default out:
results/paper/extra):

  same pairs, SF1   For E1 (tpch10: the ordered pairs among M3, M4 and M6) the first round's TPC-H pairs
                    among the same environments, and for E2 (job: among M1, M2, M3, M4 and M6) the first
                    round's TPC-H and TPC-DS pairs among them. Geometric means over those pairs of the
                    error of no adaptation, the speed factor, the ridge correction and affine+parallel
                    (random), the primary method (round robin) and the oracle, at k = 10 and at one query
                    per template (25 on TPC-H, 100 on TPC-DS); the range of the template share and of
                    omega squared; the range of the parallel and serial median slowdowns over the pairs
                    that move to fewer processors on the same machine (SHRINK: M1 to M2 on the laptop,
                    M3 to M4, M3 to M6 and M6 to M4 on the server).
                    Macros XcOne<E><Wl><what>, E TpchTen or Job (the experiment whose pairs these are),
                    Wl Tpch or Tpcds (the SF1 workload), what as below.
  new templates     For each second-round workload with results: the largest average gain of any
                    correction over the speed factor on templates left out (every strategy and method of
                    the hard test except no correction, the speed factor and the oracle; percent, rounded
                    up), the ratio range of the oracle's error to the speed factor's, as
                    paper_extra_report computes them for the first round. Macros XcNew<E>HardBestPct,
                    XcNew<E>HardOracleMin and XcNew<E>HardOracleMax.
  processor pairs   For each second-round workload with results: the range of the parallel and serial
                    median slowdowns over its SHRINK pairs, and of the template share's noise
                    reference. Macros XcNew<E>ShrinkParallelMin/Max, ShrinkSerialMin/Max,
                    ShareNoiseMin/Max.

Errors are geometric means over test templates of the median Q-error (paper_eval), averaged over the
pairs geometrically.

Usage: python src/paper_ext_compare.py --final results/paper/final [--ext results/paper/ext] [--out results/paper/extra]
"""
from __future__ import annotations

import argparse
import math
from itertools import permutations
from pathlib import Path

import numpy as np

import paper_report as pr

HERE = Path(__file__).resolve().parents[1]
EXPERIMENT_ENVS = {"tpch10": ["M3", "M4", "M6"], "job": ["M1", "M2", "M3", "M4", "M6"]}
SF1_WORKLOADS = {"tpch10": ["tpch"], "job": ["tpch", "tpcds"]}
ONE_PER_TEMPLATE = {"tpch": 25, "tpcds": 100}
WORD = {"tpch10": "TpchTen", "job": "Job", "tpch": "Tpch", "tpcds": "Tpcds"}
# Moves to fewer processors on the same machine: a clean processor change, in the direction that slows.
SHRINK = {"M1->M2", "M3->M4", "M3->M6", "M6->M4"}
METHODS = [("None", "random", "none"), ("Speed", "random", "speed"), ("Ridge", "random", "ridge"),
           ("AffinePar", "random", "affine+parallel"), ("Primary", "round-robin", "lowdim+template"),
           ("Oracle", "random", "oracle")]


def rng(vals: list[float]) -> tuple[float, float] | None:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (min(v), max(v)) if v else None


def same_pair_macros(jobs: dict) -> dict:
    m = {}
    for e, envs in EXPERIMENT_ENVS.items():
        pairs = [f"{a}->{b}" for a, b in permutations(envs, 2)]
        for w in SF1_WORKLOADS[e]:
            js = [jobs[(w, "hybrid", p)] for p in pairs if (w, "hybrid", p) in jobs]
            base = ("Xc", "One", WORD[e], WORD[w])
            m[pr.tex_name(*base, "pairs")] = str(len(js))
            for k in (10, ONE_PER_TEMPLATE[w]):
                for word, s, meth in METHODS:
                    m[pr.tex_name(*base, "k", k, word)] = pr.fmt(pr.geo([(pr.cell(j, s, meth, k) or {}).get("q") for j in js]))
            for f, key in (("share", "template_share"), ("omega", "epsilon2")):
                r = rng([j.get("shift", {}).get(key) for j in js])
                if r:
                    m[pr.tex_name(*base, f, "min")], m[pr.tex_name(*base, f, "max")] = pr.fmt(r[0]), pr.fmt(r[1])
            shrink = [jobs[(w, "hybrid", p)] for p in pairs if p in SHRINK and (w, "hybrid", p) in jobs]
            for cls in ("parallel", "serial"):
                r = rng([j.get("shift", {}).get("by_plan", {}).get(cls, {}).get("median") for j in shrink])
                if r:
                    m[pr.tex_name(*base, "shrink", cls, "min")] = pr.fmt(r[0])
                    m[pr.tex_name(*base, "shrink", cls, "max")] = pr.fmt(r[1])
    return m


def new_macros(jobs: dict) -> dict:
    m = {}
    for e in EXPERIMENT_ENVS:
        pairs = pr.pairs_of(jobs, e)
        if not pairs:
            continue
        js = [jobs[(e, "hybrid", p)] for p in pairs]
        base = ("Xc", "New", WORD[e])
        best, orc = 1.0, []
        for key in js[0].get("hard", {}):
            sel, meth = key.split("|")
            r = [j["hard"][key]["q"] / j["hard"][f"{sel}|speed"]["q"] for j in js if key in j.get("hard", {})]
            if not r:
                continue
            g = math.exp(sum(math.log(x) for x in r) / len(r))
            if meth == "oracle" and sel == "random":
                orc = r
            elif meth not in ("none", "speed", "oracle"):
                best = min(best, g)
        m[pr.tex_name(*base, "hard best pct")] = str(math.ceil((1 - best) * 100))
        if orc:
            m[pr.tex_name(*base, "hard oracle min")] = pr.fmt(min(orc))
            m[pr.tex_name(*base, "hard oracle max")] = pr.fmt(max(orc))
        shrink = [j for p, j in zip(pairs, js) if p in SHRINK]
        for cls in ("parallel", "serial"):
            r = rng([j.get("shift", {}).get("by_plan", {}).get(cls, {}).get("median") for j in shrink])
            if r:
                m[pr.tex_name(*base, "shrink", cls, "min")] = pr.fmt(r[0])
                m[pr.tex_name(*base, "shrink", cls, "max")] = pr.fmt(r[1])
        r = rng([j.get("shift", {}).get("share_noise") for j in js])
        if r:
            m[pr.tex_name(*base, "share noise min")], m[pr.tex_name(*base, "share noise max")] = pr.fmt(r[0]), pr.fmt(r[1])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=HERE / "results" / "paper" / "final")
    ap.add_argument("--ext", type=Path, default=None, help="paper_eval's run of the second round, if it exists")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "paper" / "extra")
    a = ap.parse_args()
    _, jobs1, _ = pr.load(a.final)
    m = same_pair_macros(jobs1)
    if a.ext is not None:
        _, jobs2, _ = pr.load(a.ext)
        m |= new_macros(jobs2)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare_numbers.tex").write_text(
        "% Generated by src/paper_ext_compare.py. Do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())), encoding="utf-8", newline="\n")
    print(f"wrote {a.out / 'compare_numbers.tex'} ({len(m)} macros)")


if __name__ == "__main__":
    main()
