"""Numbers that explain the second round's results (exploratory, written after them; paper/PLAN.md, 10 October).

Written on 10 October 2026, after the second round's evaluation, to say why its tests came out as they
did. Nothing here is a registered test. It reads the labels (--runs), the first round's evaluation
(--final), the second round's (--ext) and the neural model's run on the second round (--nn), and writes
<out>/explain_numbers.tex (default out: results/paper/extra):

  spread       for every workload, the median over templates of the standard deviation of the log
               runtime of a template's queries, as a factor (exp), on the workload's first registered
               machine (M1, or M3 for TPC-H at SF10, which has no M1), as src/paper_report.py computes
               PSpread for the first round: XeSpread<W>
  parallel     the share of queries whose plan has parallel workers (plans are the same in every
               environment of a workload): XeParallel<W>, a percentage
  room         the geometric mean over a workload's ordered pairs of the speed factor's error (random)
               over the oracle's at one query per template (k = 25 on TPC-H, 100 on TPC-DS, kH on the
               second round): XeRoom<W>
  move         at the same budget, the primary method's error (geometric mean over the moved pairs)
               over its error on the no-move controls (geometric mean over the environments), the cost
               of the move: XeMove<W>, and XeMoveMaxPct, the largest over the four workloads in percent,
               rounded up
  pool         JOB's pool queries per template, fewest and most: XeJobPoolMin, XeJobPoolMax
  sources      on the second round at kH, for every source model of paper_eval (hybrid, the course
               study's, and forest, boosting, ridge and cost, run alongside it), the geometric mean over
               pairs of no adaptation, the speed factor (random), the primary method (round robin) and
               the oracle, and the pairs where the primary method beats the speed factor with an
               interval below 1, as src/paper_sources.py counts them for the first round:
               XeSrc<W><Src><None|Speed|Primary|Oracle|Wins>, and over the four other source models
               XeSrc<W>OtherWins of XeSrc<W>OtherPairs
  selection    on the second round at kH (course study's source model): the primary method with
               random selection, the pairs where it is below the primary method with round robin, 1-NN
               with random selection, round robin and k-center, the primary method with k-center, and
               the refits with random selection (target only, pooled, forest on the errors):
               Xe<W>PrimaryRand, Xe<W>RandBelowRr, Xe<W>KnnRand, Xe<W>KnnRr, Xe<W>KnnKc, Xe<W>PrimaryKc,
               Xe<W>TargetRand, Xe<W>PooledRand, Xe<W>ForestResidRand
  one/template on SF10 at k = 25, one query per template as at SF1: the primary method (round robin)
               and the speed factor (random): XeTpchTenPrimaryKTwoFive, XeTpchTenSpeedKTwoFive
  H2 points    on the second round, the pairs where the primary method's error is below the speed
               factor's at kH (the point estimate of H2's ratio): Xe<W>HTwoPoint
  neural       on the second round, the template correction on the neural model against the same
               model fine-tuned on the same queries (both round robin): the pairs where the correction's
               error is lower and the geometric mean over pairs of the ratio of the two errors, at
               k = 10 and kH: Xe<W>NnTplBelowFull<K>, Xe<W>NnTplOverFull<K>

W is Tpch, Tpcds, TpchTen or Job; K is KOneZero or KH.

Usage: python src/paper_ext_explain.py [--runs results/runs] [--final results/paper/final]
       [--ext results/paper/ext] [--nn results/paper/nn_ext] [--out results/paper/extra]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import paper_eval as pe
import paper_report as pr

HERE = Path(__file__).resolve().parents[1]
WORD = {"tpch": "Tpch", "tpcds": "Tpcds", "tpch10": "TpchTen", "job": "Job"}
SPREAD_RUN = {"tpch": "M1", "tpcds": "M1_tpcds", "tpch10": "M3_sf10", "job": "M1_job"}
ONE_PER_TEMPLATE = {"tpch": 25, "tpcds": 100}


def spread(labels: Path) -> float:
    d = pd.read_csv(labels)
    d = d[d["status"] == "ok"]
    sd = d.assign(l=np.log(d["label_ms"])).groupby("template")["l"].std(ddof=1).dropna()
    return float(np.exp(np.median(sd)))


def parallel_share(labels: Path) -> float:
    d = pd.read_csv(labels)
    return float((d["workers_planned"] > 0).mean())


def room(jobs: dict, w: str, k: int) -> float:
    js = [jobs[(w, "hybrid", p)] for p in pr.pairs_of(jobs, w)]
    r = [(pr.cell(j, "random", "speed", k) or {}).get("q") / (pr.cell(j, "random", "oracle", k) or {}).get("q")
         for j in js]
    return float(np.exp(np.mean(np.log(r))))


def move(jobs: dict, w: str, k: int) -> float:
    def gm(pairs):
        return pr.geo([(pr.cell(jobs[(w, "hybrid", p)], *pr.PRIMARY, k) or {}).get("q") for p in pairs])
    return gm(pr.pairs_of(jobs, w)) / gm(pr.controls_of(jobs, w))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=HERE / "results" / "runs")
    ap.add_argument("--final", type=Path, default=HERE / "results" / "paper" / "final")
    ap.add_argument("--ext", type=Path, default=HERE / "results" / "paper" / "ext")
    ap.add_argument("--nn", type=Path, default=HERE / "results" / "paper" / "nn_ext")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "paper" / "extra")
    a = ap.parse_args()
    m, doc = {}, {}
    for w, run in SPREAD_RUN.items():
        lab = a.runs / run / "labels.csv"
        m[f"XeSpread{WORD[w]}"] = pr.fmt(spread(lab))
        m[f"XeParallel{WORD[w]}"] = str(round(100 * parallel_share(lab)))
        doc[w] = {"spread_run": run, "spread": spread(lab), "parallel_share": parallel_share(lab)}
    _, jobs1, _ = pr.load(a.final)
    _, jobs2, _ = pr.load(a.ext)
    for w, k in ONE_PER_TEMPLATE.items():
        doc[w]["room"] = room(jobs1, w, k)
        m[f"XeRoom{WORD[w]}"] = pr.fmt(doc[w]["room"])
        doc[w]["move"] = move(jobs1, w, k)
        m[f"XeMove{WORD[w]}"] = pr.fmt(doc[w]["move"])
    split = pd.read_csv(HERE / pe.WORKLOADS["job"]["split"])
    per = split[split["set"] == "pool"].groupby("template").size()
    m["XeJobPoolMin"], m["XeJobPoolMax"] = str(int(per.min())), str(int(per.max()))
    doc["job"]["pool_per_template"] = {"min": int(per.min()), "max": int(per.max()), "templates": int(len(per))}
    with gzip.open(a.nn / "pairs.json.gz", "rt", encoding="utf-8") as f:
        nn_pairs = json.load(f)
    for w in ("tpch10", "job"):
        kh = pe.budget(w)
        doc[w]["room"] = room(jobs2, w, kh)
        m[f"XeRoom{WORD[w]}"] = pr.fmt(doc[w]["room"])
        doc[w]["move"] = move(jobs2, w, kh)
        m[f"XeMove{WORD[w]}"] = pr.fmt(doc[w]["move"])
        js = [jobs2[(w, "hybrid", p)] for p in pr.pairs_of(jobs2, w)]
        point = sum((pr.cell(j, *pr.PRIMARY, kh) or {}).get("vs_speed", {}).get("gm", 9) < 1 for j in js)
        m[f"Xe{WORD[w]}HTwoPoint"] = str(point)
        doc[w]["h2_point_below"] = point

        def q(jl, s, meth, k):
            return pr.geo([(pr.cell(j, s, meth, k) or {}).get("q") for j in jl])
        # Every source model at kH, as paper_sources counts the first round.
        other_wins = other_pairs = 0
        doc[w]["sources"] = {}
        for src in pe.SOURCES:
            sj = [jobs2[(w, src, p)] for p in pr.pairs_of(jobs2, w, src)]
            wins = sum((pr.cell(j, *pr.PRIMARY, kh) or {}).get("vs_speed", {}).get("gm_hi", 9) < 1 for j in sj)
            r = {"none": q(sj, "random", "none", kh), "speed": q(sj, "random", "speed", kh),
                 "primary": q(sj, *pr.PRIMARY, kh), "oracle": q(sj, "random", "oracle", kh),
                 "wins": wins, "pairs": len(sj)}
            doc[w]["sources"][src] = r
            for f in ("none", "speed", "primary", "oracle"):
                m[pr.tex_name("XeSrc", WORD[w], src, f)] = pr.fmt(r[f])
            m[pr.tex_name("XeSrc", WORD[w], src, "wins")] = str(wins)
            if src != "hybrid":
                other_wins, other_pairs = other_wins + wins, other_pairs + len(sj)
        m[f"XeSrc{WORD[w]}OtherWins"], m[f"XeSrc{WORD[w]}OtherPairs"] = str(other_wins), str(other_pairs)
        # Selection and refits on the course study's source model at kH.
        rr = [(pr.cell(j, *pr.PRIMARY, kh) or {}).get("q") for j in js]
        ra = [(pr.cell(j, "random", "lowdim+template", kh) or {}).get("q") for j in js]
        sel = {"PrimaryRand": q(js, "random", "lowdim+template", kh), "KnnRand": q(js, "random", "1-nn", kh),
               "KnnRr": q(js, "round-robin", "1-nn", kh), "KnnKc": q(js, "k-center", "1-nn", kh),
               "PrimaryKc": q(js, "k-center", "lowdim+template", kh), "TargetRand": q(js, "random", "target", kh),
               "PooledRand": q(js, "random", "pooled", kh), "ForestResidRand": q(js, "random", "forest-resid", kh)}
        for name, v in sel.items():
            m[f"Xe{WORD[w]}{name}"] = pr.fmt(v)
        m[f"Xe{WORD[w]}RandBelowRr"] = str(sum(a < b for a, b in zip(ra, rr)))
        doc[w]["selection"] = {**sel, "rand_below_rr": sum(a < b for a, b in zip(ra, rr))}
    js = [jobs2[("tpch10", "hybrid", p)] for p in pr.pairs_of(jobs2, "tpch10")]
    m["XeTpchTenPrimaryKTwoFive"] = pr.fmt(pr.geo([(pr.cell(j, *pr.PRIMARY, 25) or {}).get("q") for j in js]))
    m["XeTpchTenSpeedKTwoFive"] = pr.fmt(pr.geo([(pr.cell(j, "random", "speed", 25) or {}).get("q") for j in js]))
    for w in ("tpch10", "job"):
        kh = pe.budget(w)
        rows = [p for p in nn_pairs if p["workload"] == w]
        for k, name in ((10, "KOneZero"), (kh, "KH")):
            tpl = [p["cells"][f"round-robin|nn:lowdim+template|{k}"]["q"] for p in rows]
            full = [p["cells"][f"round-robin|nn:full-ft|{k}"]["q"] for p in rows]
            below = sum(t < f for t, f in zip(tpl, full))
            ratio = math.exp(np.mean([math.log(t / f) for t, f in zip(tpl, full)]))
            m[f"Xe{WORD[w]}NnTplBelowFull{name}"] = str(below)
            m[f"Xe{WORD[w]}NnTplOverFull{name}"] = pr.fmt(ratio)
            doc[w][f"nn_tpl_vs_full_k{k}"] = {"below": below, "pairs": len(rows), "gm_ratio": ratio}
    m["XeMoveMaxPct"] = str(math.ceil(100 * (max(doc[w]["move"] for w in WORD) - 1)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "explain.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (a.out / "explain_numbers.tex").write_text(
        "% Generated by src/paper_ext_explain.py. Do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())), encoding="utf-8", newline="\n")
    print(json.dumps(doc, indent=1))
    print(f"wrote {a.out / 'explain_numbers.tex'} ({len(m)} macros)")


if __name__ == "__main__":
    main()
