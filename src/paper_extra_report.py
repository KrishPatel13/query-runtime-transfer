"""Macros of the exploratory analyses added after the internal review of 8 October 2026.

None of these were registered (paper/PLAN.md); the paper reports them as exploratory.

Reads the first round's summary (--final, results/paper/final/summary.json.gz)
and the run of src/paper_eval.py with --extra on the first round's workloads
(--extra, its summary.json), and writes <extra>/extra_numbers.tex:

  shift transfer   on the pairs whose change was also measured on the other
                   machine (paper_eval.ANALOG: 8 per workload), the error of
                   the shift learned on that analog pair (transfer-analog,
                   round robin) and of the primary method on the same pairs,
                   and the number of pairs where transfer is lower, at k = 5,
                   10 and 25: Xtra<W>Transfer<K>, Xtra<W>PrimaryAnalog<K>,
                   Xtra<W>TransferWins<K>, XtraAnalogPairs
  hard test        on templates left out (random selection), the ratio of a
                   transfer variant's error to the speed factor's,
                   geometric mean over the pairs; the smallest and largest
                   over the six variants and two workloads:
                   XtraHardTransferMin, XtraHardTransferMax
  large budgets    the refits and the primary method (both random) at k = 100
                   and 200: Xtra<W><Method>Rand<K> for ForestResid, Target,
                   Pooled and Primary
  template-aware   the forest on the errors with template indicators
  refit            (forest-resid+template) beside the forest without them
                   and the primary method, at k = 10 and kH (25 on TPC-H,
                   100 on TPC-DS), random and round robin:
                   Xtra<W>ForestTpl<Sel><K>, Xtra<W>ForestResid<Sel><K>,
                   Xtra<W>Primary<Sel><K>, and Xtra<W>ForestTplWins<Sel><K>,
                   the pairs where it is lower than the primary method

  bounds           XtraHardBestPct, the largest average gain (percent,
                   rounded up) of any correction over the speed factor on
                   templates left out, oracle excluded; XtraNnGainPct, the
                   largest gain (percent, rounded up) of the fully
                   fine-tuned network over the template correction at any k
                   on the development pairs (--nn, results/paper/nn_dev)
  H3 by workload   Xtra<W>HThreeCI of Xtra<W>HThreeN: the confirmatory
                   pairs that change the processor count where the interval
                   of affine+parallel over the speed factor at k = 10 lies
                   below 1 (the split of the paper's interval count)

All errors are geometric means over the pairs (no-move controls excluded).
The script first checks that the --extra run reproduces the first round's
forest on the errors exactly (it runs the same code with the same seeds).

Usage: python src/paper_extra_report.py --final results/paper/final --extra results/paper/extra
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
WORKLOAD = {"tpch": "Tpch", "tpcds": "Tpcds"}
KH = {"tpch": 25, "tpcds": 100}
SEL = {"random": "Rand", "round-robin": "Rr"}
TRANSFER = ["transfer-analog", "transfer-analog+ridge", "transfer-workload", "transfer-workload+ridge",
            "transfer-analog-workload", "transfer-analog-workload+ridge"]


def kname(k: int) -> str:
    return "K" + "".join(DIGITS[int(d)] for d in str(k))


def gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load(path: Path) -> list[dict]:
    """The moved pairs of the hybrid source model (no-move controls left out)."""
    f = path / "summary.json.gz" if (path / "summary.json.gz").exists() else path / "summary.json"
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt", encoding="utf-8") as fh:
        s = json.load(fh)
    return [j for j in s["jobs"] if j["source"] == "hybrid" and j["a"] != j["b"]]


def q(j: dict, sel: str, m: str, k: int) -> float:
    return j["c2"][f"{sel}|{m}|{k}"]["q"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True, type=Path)
    ap.add_argument("--extra", required=True, type=Path)
    ap.add_argument("--nn", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "paper" / "nn_dev")
    a = ap.parse_args()
    final, extra = load(a.final), load(a.extra)
    by_pair = {(j["workload"], j["pair"]): j for j in final}

    worst = 0.0
    for j in extra:
        f = by_pair[(j["workload"], j["pair"])]
        for key, v in j["c2"].items():
            if key.split("|")[1] == "forest-resid":
                worst = max(worst, abs(v["q"] - f["c2"][key]["q"]))
    if worst > 1e-9:
        raise SystemExit(f"the --extra run does not reproduce the forest on the errors (largest difference {worst})")

    m = {}
    hard = []
    for w, W in WORKLOAD.items():
        fw = [j for j in final if j["workload"] == w]
        analog = [j for j in fw if "round-robin|transfer-analog|5" in j["c2"]]
        m["XtraAnalogPairs"] = len(analog)
        for k in (5, 10, 25):
            t = [q(j, "round-robin", "transfer-analog", k) for j in analog]
            p = [q(j, "round-robin", "lowdim+template", k) for j in analog]
            m[f"Xtra{W}Transfer{kname(k)}"] = f"{gm(t):.2f}"
            m[f"Xtra{W}PrimaryAnalog{kname(k)}"] = f"{gm(p):.2f}"
            m[f"Xtra{W}TransferWins{kname(k)}"] = sum(x < y for x, y in zip(t, p))
        for t in TRANSFER:
            r = [j["hard"][f"random|{t}"]["q"] / j["hard"]["random|speed"]["q"]
                 for j in fw if f"random|{t}" in j.get("hard", {})]
            if r:
                hard.append(gm(r))
        for k in (100, 200):
            for name, meth in (("ForestResid", "forest-resid"), ("Target", "target"), ("Pooled", "pooled"),
                               ("Primary", "lowdim+template")):
                m[f"Xtra{W}{name}Rand{kname(k)}"] = f"{gm([q(j, 'random', meth, k) for j in fw]):.2f}"
        ew = [j for j in extra if j["workload"] == w]
        for sel, S in SEL.items():
            for k in (10, KH[w]):
                ft = [q(j, sel, "forest-resid+template", k) for j in ew]
                pr = [q(j, sel, "lowdim+template", k) for j in ew]
                m[f"Xtra{W}ForestTpl{S}{kname(k)}"] = f"{gm(ft):.2f}"
                m[f"Xtra{W}Primary{S}{kname(k)}"] = f"{gm(pr):.2f}"
                m[f"Xtra{W}ForestTplWins{S}{kname(k)}"] = sum(x < y for x, y in zip(ft, pr))
                if sel == "random":
                    m[f"Xtra{W}ForestResid{S}{kname(k)}"] = f"{gm([q(j, sel, 'forest-resid', k) for j in ew]):.2f}"
        m[f"Xtra{W}Pairs"] = len(ew)
    # Rounded outward, so that the range the text quotes contains every ratio.
    m["XtraHardTransferMin"] = f"{math.floor(min(hard) * 100) / 100:.2f}"
    m["XtraHardTransferMax"] = f"{math.ceil(max(hard) * 100) / 100:.2f}"

    # The largest average gain of any correction over the speed factor on templates left out: every
    # strategy and method of the hard test except no correction, the speed factor and the oracle.
    best = 1.0
    for w in WORKLOAD:
        fw = [j for j in final if j["workload"] == w]
        for key in fw[0]["hard"]:
            sel, meth = key.split("|")
            if meth in ("none", "speed", "oracle"):
                continue
            r = [j["hard"][key]["q"] / j["hard"][f"{sel}|speed"]["q"] for j in fw if key in j["hard"]]
            best = min(best, gm(r))
    m["XtraHardBestPct"] = math.ceil((1 - best) * 100)

    # Fine-tuning against the template correction (src/paper_nn.py, development pairs): the largest
    # gain of fine-tuning all weights, from the geometric mean over pairs of the correction's error
    # over the fine-tuned network's, at any k.
    nn = json.loads((a.nn / "summary.json").read_text(encoding="utf-8"))
    gain = max(h["full"]["gm"] for by_k in nn["h6"].values() for h in by_k.values())
    m["XtraNnGainPct"] = math.ceil((gain - 1) * 100)

    # H3 counted by interval, per workload: the confirmatory pairs that change the processor count
    # (hypotheses.json), with the 95% interval of affine+parallel over the speed factor below 1 at k = 10.
    hyp = json.loads((a.final / "hypotheses.json").read_text(encoding="utf-8"))
    for w, W in WORKLOAD.items():
        procs = [p for p in hyp["pairs"] if p["workload"] == w and p["change"] in ("processors", "both")]
        ok = sum(by_pair[(w, p["pair"])]["c2"]["random|affine+parallel|10"]["vs_speed"]["gm_hi"] < 1 for p in procs)
        m[f"Xtra{W}HThreeCI"], m[f"Xtra{W}HThreeN"] = ok, len(procs)

    out = a.extra / "extra_numbers.tex"
    out.write_text("% Generated by src/paper_extra_report.py. Do not edit by hand.\n"
                   + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(m)} macros); the --extra run matches the first round's forest on the errors")
    for k, v in m.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
