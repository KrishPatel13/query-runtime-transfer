"""Tests for the second round of the paper's evaluation: the workloads' k grids, budgets and change
labels in src/paper_eval.py, how C2 and the hard test treat a k above the pool, and the rules of
src/paper_ext_report.py (H3's processor pairs and the counts of H1 to H6b).

The adaptation fixture is a slice of the pilot's measurements (as in tests/test_adapt_eval.py) with
a second machine made from it; the report's inputs are small hand-made summaries. No label of the
second round is read: only its split files.
"""
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adapt_eval as ae
import paper_eval as pe
import paper_ext_report as per
import paper_report as pr
from features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/pilot/runtime_dataset.csv"
TEMPLATES = [1, 3, 5, 6, 14, 18]


# ------------------------------------------------------- grids and budgets

def test_k_grids_and_budgets():
    assert pe.ks_of("tpch") == pe.ks_of("tpcds") == pe.ks_of("tpch10") == pe.KS
    assert pe.ks_of("job") == [5, 10, 25, 33, 50]
    assert {w: pe.budget(w) for w in pe.WORKLOADS} == {"tpch": 100, "tpcds": 100, "tpch10": 100, "job": 33}
    for w in pe.WORKLOADS:
        assert pe.budget(w) in pe.ks_of(w) and 10 in pe.ks_of(w)


@pytest.mark.parametrize("workload", ["tpch10", "job"])
def test_budget_fits_the_split(workload):
    split = ae.load_split(ROOT / pe.WORKLOADS[workload]["split"])
    pool = split[split["set"] == "pool"]
    kept = [k for k in pe.ks_of(workload) if k <= len(pool)]       # what run_c2 keeps
    assert pe.budget(workload) in kept
    if workload == "job":
        # kH is the number of templates with a pool query: one query per template (paper/PLAN.md).
        assert pe.budget(workload) == pool["template"].nunique() == 33
        assert kept == pe.ks_of(workload) and len(pool) == 78
    else:
        assert kept == [5, 10, 25, 50, 100] and len(pool) == 154


def test_default_workloads_are_the_first_round(monkeypatch):
    seen = {}
    monkeypatch.setattr(pe, "run", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["paper_eval.py", "run"])
    pe.main()
    assert seen["workloads"] == ["tpch", "tpcds"]
    monkeypatch.setattr(sys, "argv", ["paper_eval.py", "run", "--workloads", "tpch10", "job"])
    pe.main()
    assert seen["workloads"] == ["tpch10", "job"]


def test_change_labels():
    # The first round keeps its labels (results/paper/final): the other pairs of M6 stay "mixed".
    for w in ("tpch", "tpcds"):
        assert pe.change_of(w, "M4", "M6") == pe.change_of(w, "M6", "M1") == pe.change_of(w, "M2", "M6") == "mixed"
        assert pe.change_of(w, "M3", "M6") == "processors (2)"
    assert pe.change_of("tpch10", "M6", "M4") == pe.change_of("job", "M4", "M6") == "processors (1 vs 2)"
    assert pe.change_of("job", "M1", "M6") == "both (2)"
    assert pe.change_of("job", "M6", "M2") == "both (1 vs 2)"
    assert pe.change_of("job", "M3", "M3") == "none"
    for w in ("tpch10", "job"):
        for a, b in permutations(pe.WORKLOADS[w]["runs"], 2):
            assert pe.change_of(w, a, b) != "mixed"


# ------------------------------------------------- k larger than the pool

@pytest.fixture(scope="module")
def pair() -> ae.Pair:
    """Two machines on 64 pilot queries: the second slows every template by its own factor."""
    d = pd.read_csv(PILOT)
    d = d[(d["status"] == "ok") & d["template"].isin(TEMPLATES)].groupby("template", sort=False).head(12)
    d = d.reset_index(drop=True)
    factor = d["template"].map(dict(zip(TEMPLATES, [1.5, 2.0, 0.8, 1.2, 3.0, 1.1]))).to_numpy(float)

    def machine(f):
        return pd.concat([d[["query_id", "template"]].assign(variant="std", status="ok"),
                          pd.DataFrame({ae.LABEL: d["runtime_ms"].to_numpy(float) * f}), d[FEATURE_NAMES]], axis=1)
    labels = {"A": machine(1.0), "B": machine(factor)}
    return ae.Pair(labels, ae.make_split(labels["A"]), "A", "B", fit=pe.fit_cost)


def test_run_c2_leaves_out_every_k_above_the_pool(pair):
    orders = [np.random.default_rng(r).permutation(pair.pool_ids) for r in range(2)]
    ks = [5, 10, 25, 33, pair.n_pool, pair.n_pool + 1, 100]
    plan = {"random": [pe.NONE, pe.SPEED, pe.ORACLE], "round-robin": [pe.LOWDIM_TPL]}
    logs, cost = pe.run_c2(pair, pe.fit_cost, orders, ks, plan, {})
    assert sorted({k for _, _, k in logs}) == [5, 10, 25, 33, pair.n_pool]
    assert sorted({k for _, k in cost}) == [5, 10, 25, 33, pair.n_pool]
    assert logs[("random", pe.SPEED, 5)].shape == (2, len(pair.templates))


def test_round_robin_takes_one_query_per_template_first(pair):
    pos = pair.positions(np.random.default_rng(0).permutation(pair.pool_ids))
    n_tpl = pair.pool["template"].nunique()
    seq = pe.strategy_sequence("round-robin", pair, pos, 0, 1000)      # more than the pool: all of it
    assert sorted(seq.tolist()) == list(range(pair.n_pool))
    assert pair.pool["template"].iloc[seq[:n_tpl]].nunique() == n_tpl


def test_hard_test_with_k_above_the_pool_uses_the_whole_rest(pair):
    orders = [np.random.default_rng(0).permutation(pair.pool_ids)]
    logs = pe.run_hard(pair, pe.fit_cost, orders, ["random"], [pe.NONE, pe.SPEED, pe.ORACLE], pe.unit_map(pair),
                       k=1000, n_orders=1)
    assert np.isfinite(logs[("random", pe.SPEED)]).all()


# ----------------------------------------------------- H3's processor pairs

def test_processor_pairs_reproduce_the_first_round():
    first = [f"{a}->{b}" for a, b in permutations(["M1", "M2", "M3", "M4"], 2)]
    for p in first:
        a, b = p.split("->")
        assert per.processor_pair(p) == (pe.CHANGE.get(frozenset({a, b})) in ("processors", "both"))
    assert sorted(p for p in first if per.processor_pair(p)) == \
        ["M1->M2", "M1->M4", "M2->M1", "M2->M3", "M3->M2", "M3->M4", "M4->M1", "M4->M3"]
    # On the first round's confirmatory pairs (paper_report.confirmatory) the rule gives its H3 set.
    for w in ("tpch", "tpcds"):
        conf = [f"{a}->{b}" for a, b in permutations([f"M{i}" for i in range(1, 7)], 2) if pr.confirmatory(w, f"{a}->{b}")]
        assert [p for p in conf if per.processor_pair(p)] == \
            [p for p in conf if pe.change_of(w, *p.split("->")) in ("processors", "both")]


def test_processor_pairs_of_the_second_round():
    pairs = {w: [f"{a}->{b}" for a, b in permutations(pe.WORKLOADS[w]["runs"], 2)] for w in ("tpch10", "job")}
    assert len(pairs["tpch10"]) == 6 and all(per.processor_pair(p) for p in pairs["tpch10"])
    assert sorted(p for p in pairs["job"] if not per.processor_pair(p)) == ["M1->M3", "M2->M4", "M3->M1", "M4->M2"]
    assert sum(per.processor_pair(p) for p in pairs["job"]) == 16


# ------------------------------------------------------------- the rules

@pytest.mark.parametrize("rule,ok,n,expect", [
    ("every", 6, 6, True), ("every", 5, 6, False), ("every", 0, 0, False),
    ("two_thirds", 4, 6, True), ("two_thirds", 3, 6, False), ("two_thirds", 14, 20, True),
    ("two_thirds", 13, 20, False), ("three_quarters", 12, 16, True), ("three_quarters", 11, 16, False),
    ("three_quarters", 5, 6, True), ("three_quarters", 4, 6, False), ("more_than_half", 4, 6, True),
    ("more_than_half", 3, 6, False), ("more_than_half", 11, 20, True), ("more_than_half", 10, 20, False)])
def test_counting_rules(rule, ok, n, expect):
    assert getattr(per, rule)(ok, n) is expect


def ratio(gm, hi):
    return {"gm": gm, "gm_lo": gm * gm / hi, "gm_hi": hi, "success": hi < 1}


def fake_run():
    """paper_eval and paper_nn results of six JOB pairs (M1, M3 and M4) with known outcomes."""
    pairs = ["M1->M3", "M1->M4", "M3->M1", "M3->M4", "M4->M1", "M4->M3"]
    # Per pair: share, H2 vs speed, H2 vs ridge, H3 ratio, H4 speed / oracle, H5 ratio, H6 at kH, H6 at 10.
    spec = {"M1->M3": (0.7, ratio(.8, .9), ratio(.9, .95), ratio(1.2, 1.3), 1.05, ratio(.9, 1.1), ratio(.9, .95), ratio(.9, 1.0)),
            "M1->M4": (0.8, ratio(.8, .9), ratio(.9, .95), ratio(.9, 1.1), 1.10, ratio(.9, 1.1), ratio(.9, .95), ratio(.9, 1.0)),
            "M3->M1": (0.6, ratio(.8, .9), ratio(.9, .95), ratio(1.2, 1.3), 1.12, ratio(.9, 1.1), ratio(.9, .95), ratio(.9, 1.0)),
            "M3->M4": (0.9, ratio(.8, .9), ratio(.9, .95), ratio(.95, .99), 1.14, ratio(1.1, 1.2), ratio(.9, .95), ratio(1.1, 1.2)),
            "M4->M1": (0.4, ratio(.8, .9), ratio(1.0, 1.1), ratio(.9, 1.1), 1.16, ratio(1.1, 1.2), ratio(1.0, 1.2), ratio(1.1, 1.2)),
            "M4->M3": (0.7, ratio(.8, .9), ratio(1.0, 1.1), ratio(1.1, 1.2), 1.00, ratio(1.1, 1.2), ratio(1.0, 1.1), ratio(1.1, 1.2))}
    jobs, nn_pairs = {}, []
    for p, (share, h2s, h2r, h3, h4, h5, h6k, h6s) in spec.items():
        c2 = {}
        for k in (10, 33):
            for s, m, q in (("random", "none", 2.0), ("random", "speed", 1.5), ("random", "affine+parallel", 1.4),
                            ("random", "ridge", 1.45), ("round-robin", "lowdim+template", 1.2),
                            ("k-center", "1-nn", 1.3), ("random", "oracle", 1.1)):
                c2[f"{s}|{m}|{k}"] = {"q": q}
        c2["round-robin|lowdim+template|33"].update(vs_speed=h2s, vs_ridge=h2r)
        c2["random|affine+parallel|10"]["vs_speed"] = h3
        c2["k-center|1-nn|33"]["vs_ridge"] = h5
        by_plan = {"serial": {"median": 1.3}, "parallel": {"median": 2.1}}
        jobs[("job", "hybrid", p)] = {
            "change": pe.change_of("job", *p.split("->")), "c2": c2,
            "shift": {"template_share": share, "share_noise": 0.1, "epsilon2": share - 0.05, "median": 1.5,
                      "by_plan": by_plan},
            "hard": {"random|speed": {"q": 1.2 * h4}, "random|oracle": {"q": 1.2}, "random|none": {"q": 2.0},
                     "random|affine+parallel": {"q": 1.3}, "random|ridge": {"q": 1.3}}}
        nn_pairs.append({"workload": "job", "pair": p, "h6": {"33": {"full": h6k, "head": h6k},
                                                               "10": {"full": h6s, "head": h6s}}})
    for m in ("M1", "M3", "M4"):
        jobs[("job", "hybrid", f"{m}->{m}")] = {"change": "none",
                                                "c2": {f"round-robin|lowdim+template|{k}": {"q": 1.05} for k in (10, 33)}}
    nn = {"gm_over_pairs": {"job": {"round-robin": {"nn:full-ft": {"10": 1.3, "33": 1.15},
                                                    "nn:head-ft": {"10": 1.35, "33": 1.2}}}}}
    return jobs, nn_pairs, nn


def test_hypotheses_on_known_counts(tmp_path):
    jobs, nn_pairs, nn = fake_run()
    h = per.hypotheses(jobs, nn_pairs, "job")
    assert h["kh"] == 33 and h["n_pairs"] == 6 and h["processor_pairs"] == 4
    assert (h["H1"]["ok"], h["H1"]["min"], h["H1"]["pass"]) == (5, 0.4, False)
    assert (h["H2"]["vs_speed_ok"], h["H2"]["vs_ridge_ok"], h["H2"]["pass"]) == (6, 4, True)
    # H3 counts the four processor pairs only: three below 1 (one of them by interval).
    assert (h["H3"]["ok"], h["H3"]["ok_ci"], h["H3"]["n"], h["H3"]["pass"]) == (3, 1, 4, True)
    assert (h["H4"]["ok"], h["H4"]["pass"]) == (5, False) and h["H4"]["max"] == pytest.approx(1.16)
    assert (h["H5"]["ok"], h["H5"]["ok_ci"], h["H5"]["pass"]) == (3, 0, False)
    # H6 per experiment is a descriptive count; its result is over all new pairs (pooled_h6).
    assert h["H6"]["ok"] == 4 and "pass" not in h["H6"]
    assert (h["H6a"]["point"], h["H6a"]["pass"]) == (3, False)
    assert (h["H6b"]["ok"], h["H6b"]["pass"]) == (5, False) and h["H6b"]["max_hi"] == pytest.approx(1.2)
    assert h["H6_head"]["ok"] == 4 and "pass" not in h["H6_head"]
    assert "H6" not in per.hypotheses(jobs, None, "job")
    c = per.input_checks({"job": h})["job"]
    assert c["present"] == ["M1", "M3", "M4"] and c["missing"] == ["M2", "M6"]
    assert c["split"] == {"pool": 78, "test": 35} and c["pairs_short"] == []

    m = per.macros(jobs, {"job": h}, nn)
    assert m["XJobHThreePass"] == "holds" and m["XJobHFivePass"] == "fails"
    assert m["XJobHTwoVsSpeedOk"] == "6" and m["XJobHThreeOkCi"] == "1" and m["XJobProcPairs"] == "4"
    assert m["XJobHFourMax"] == "1.16" and m["XJobHSixbMaxHi"] == "1.200"
    assert m["XJobKThreeThreePrimary"] == m["XJobKHPrimary"] == "1.20"
    assert m["XJobKOneZeroNnFull"] == "1.30" and m["XJobKHControl"] == "1.05"
    assert m["XJobMOnetoMFourSerial"] == "1.30" and m["XJobMOnetoMFourParallel"] == "2.10"
    assert m["XJobShareMin"] == "0.40" and m["XJobHardSpeed"] != "--"
    assert all(k.isalpha() for k in m)
    per.table_hypotheses({"job": h}, tmp_path)
    per.table_errors(jobs, {"job": h}, nn, tmp_path)
    text = (tmp_path / "table_hypotheses_ext.tex").read_text(encoding="utf-8")
    assert "E2: JOB" in text and "3/4 & holds" in text and "6/6, 4/6 & holds" in text
    assert "$k=33$" in (tmp_path / "table_ext.tex").read_text(encoding="utf-8")


def test_h6_is_one_count_over_all_new_pairs():
    """H6 is registered "in at least two thirds of the new pairs", without "of each experiment": one
    count over E1's and E2's pairs, which per-experiment verdicts cannot give."""
    def exp(n, ok):
        rows = [{"pair": f"p{i}", "h6_kh": {m: ratio(.9, .95 if i < ok else 1.05) for m in per.FT_MODES}}
                for i in range(n)]
        return {"n_pairs": n, "H6": {}, "H6_head": {}, "pairs": rows}

    p = per.pooled_h6({"tpch10": exp(6, 3), "job": exp(20, 15)})["H6"]       # E1 alone 3/6, E2 alone 15/20
    assert (p["ok"], p["n"], p["n_pairs"], p["pass"]) == (18, 26, 26, True)
    p = per.pooled_h6({"tpch10": exp(6, 4), "job": exp(20, 13)})["H6"]       # E1 alone 4/6, E2 alone 13/20
    assert (p["ok"], p["n"], p["pass"]) == (17, 26, False)
    assert per.pooled_h6({"job": exp(20, 20)})["H6"]["pass"] is False      # an experiment is missing
    m = per.pooled_macros(per.pooled_h6({"tpch10": exp(6, 3), "job": exp(20, 15)}))
    assert m["XAllHSixOk"] == "18" and m["XAllHSixN"] == "26" and m["XAllHSixPass"] == "holds"
    assert m["XAllHSixHeadPass"] == "holds" and all(k.isalpha() for k in m)


def test_share_macros_never_round_across_the_threshold():
    jobs, nn_pairs, nn = fake_run()
    h = per.hypotheses(jobs, nn_pairs, "job")
    for r in h["pairs"]:
        if r["pair"] == "M4->M1":
            r["share"] = 0.4996
    m = per.macros(jobs, {"job": h}, nn)
    assert m["XJobShareMin"] == "0.4996"


def test_fmt_level_never_contradicts_the_test():
    assert per.fmt_level(1.1496, 1.15) == "1.1496"
    assert per.fmt_level(1.151, 1.15) == "1.151"
    assert per.fmt_level(0.987, 1.0) == "0.99"
    assert per.fmt_level(0.996, 1.0) == "0.996"
    assert per.fmt_level(None, 1.0) == "--"
