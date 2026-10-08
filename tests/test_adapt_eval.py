"""Tests for the adaptation evaluation (src/adapt_eval.py) and its statistics (src/stats.py).

The fixture is a slice of the pilot's measurements: 12 queries from each of
five templates and the 4 queries of Q18, 64 in all. Forests have 10 trees, so
the file runs in well under a minute.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import adapt_eval as ae
import stats
import train_eval as te
from features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/pilot/runtime_dataset.csv"
TEMPLATES = [1, 3, 5, 6, 14, 18]


@pytest.fixture(autouse=True)
def small_forests(monkeypatch):
    monkeypatch.setattr(te, "forest", lambda seed: RandomForestRegressor(
        n_estimators=10, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1))


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    d = pd.read_csv(PILOT)
    d = d[(d["status"] == "ok") & d["template"].isin(TEMPLATES)]
    return d.groupby("template", sort=False).head(12).reset_index(drop=True)


def machine(base: pd.DataFrame, factor=1.0) -> pd.DataFrame:
    """A labels.csv frame whose labels are the pilot's times factor (a number or one per query)."""
    return pd.concat([base[["query_id", "template"]].assign(variant="std", status="ok"),
                      pd.DataFrame({ae.LABEL: base["runtime_ms"].to_numpy(float) * factor}),
                      base[FEATURE_NAMES]], axis=1)


def template_factor(base: pd.DataFrame, per_template: dict) -> np.ndarray:
    return base["template"].map(per_template).to_numpy(float)


# ------------------------------------------------------------------ the data

def test_split_is_made_inside_every_template_and_variant():
    q = pd.DataFrame({"query_id": [f"q{i:03d}" for i in range(41)],
                      "template": [1] * 20 + [2] * 16 + [3] * 4 + [4],
                      "variant": ["std", "wide"] * 10 + ["std"] * 21})
    s = ae.make_split(q, seed=0)
    assert s.equals(ae.make_split(q, seed=0))
    assert not s.equals(ae.make_split(q, seed=1))
    assert sorted(s["query_id"]) == sorted(q["query_id"])
    n_test = s[s["set"] == "test"].groupby(["template", "variant"]).size().to_dict()
    assert n_test == {(1, "std"): 3, (1, "wide"): 3, (2, "std"): 5, (3, "std"): 1}
    assert s.loc[s["template"] == 4, "set"].tolist() == ["pool"]


def test_loaders_enforce_the_contract(tmp_path, base):
    good = machine(base)
    for broken, message in ((good.drop(columns=["variant"]), "missing columns"),
                            (good.assign(label_ms=np.where(good.index == 3, np.nan, good[ae.LABEL])), "no positive"),
                            (pd.concat([good, good.iloc[:1]]), "duplicate")):
        broken.to_csv(tmp_path / "labels.csv", index=False)
        with pytest.raises(ValueError, match=message):
            ae.load_labels(tmp_path / "labels.csv")
    # A failed query may have no label.
    good.assign(status=np.where(good.index == 3, "timeout", "ok"),
                label_ms=np.where(good.index == 3, np.nan, good[ae.LABEL])).to_csv(tmp_path / "labels.csv", index=False)
    assert len(ae.load_labels(tmp_path / "labels.csv")) == len(good)

    split = ae.make_split(good)
    split.assign(set=np.where(split.index == 0, "train", split["set"])).to_csv(tmp_path / "split.csv", index=False)
    with pytest.raises(ValueError, match="pool"):
        ae.load_split(tmp_path / "split.csv")
    with pytest.raises(ValueError, match="differs from the split"):
        ae.check_keys({"M1": good.assign(template=np.where(good.index == 0, 99, good["template"]))}, split)


def test_a_pair_uses_only_queries_ok_on_both_machines(base):
    m1, m2 = machine(base), machine(base, 2.0)
    m1.loc[0, "status"] = "error: x"
    m2.loc[[5, 40], "status"] = "timeout"
    split = ae.make_split(m1)
    p = ae.Pair({"M1": m1, "M2": m2}, split, "M1", "M2")
    dropped = set(base.loc[[0, 5, 40], "query_id"])
    assert dropped.isdisjoint(p.pool_ids) and dropped.isdisjoint(p.test_ids)
    assert p.n_pool + p.n_test == len(base) - 3
    assert set(p.pool_ids) <= set(split.loc[split["set"] == "pool", "query_id"])
    assert set(p.test_ids) <= set(split.loc[split["set"] == "test", "query_id"])
    assert p.features_differ == []
    # The target rows carry the target's labels, the source pool the source's.
    assert np.allclose(p.pool[ae.LABEL].to_numpy(), 2 * p.source_pool[ae.LABEL].to_numpy())
    # The source model trains on every source-pool query that is ok on the source, also those
    # that time out on the target.
    pool = set(split.loc[split["set"] == "pool", "query_id"])
    assert set(p.source_train.index) == pool - {base.loc[0, "query_id"]}


def test_every_pair_from_one_source_starts_from_the_same_model(base):
    m3 = machine(base, 3.0)
    split = ae.make_split(machine(base))
    pool_ids = split.loc[split["set"] == "pool", "query_id"]
    m3.loc[m3["query_id"].isin(pool_ids.iloc[:4]), "status"] = "timeout"
    labels = {"M1": machine(base), "M2": machine(base, 2.0), "M3": m3}
    a, b = ae.Pair(labels, split, "M1", "M2"), ae.Pair(labels, split, "M1", "M3")
    assert a.source_train.index.equals(b.source_train.index)
    probe = machine(base).iloc[:20]
    assert np.allclose(a.f_s(probe), b.f_s(probe))
    assert b.n_pool == a.n_pool - 4


def test_a_query_whose_plan_differs_is_left_out(base):
    m2 = machine(base, 2.0)
    m2.loc[3, "total_cost"] *= 1.5
    split = ae.make_split(machine(base))
    p = ae.Pair({"M1": machine(base), "M2": m2}, split, "M1", "M2")
    q = base.loc[3, "query_id"]
    assert p.features_differ == [q]
    assert q not in p.pool_ids and q not in p.test_ids
    assert p.n_pool + p.n_test == len(base) - 1


def test_arrival_orders_are_seeded_and_shared_by_pairs(base):
    split = ae.make_split(machine(base))
    orders = ae.arrival_orders(split, 3)
    again = ae.arrival_orders(split, 3)
    pool = sorted(split.loc[split["set"] == "pool", "query_id"])
    for a, b in zip(orders, again):
        assert list(a) == list(b) and sorted(a) == pool
    assert list(orders[0]) != list(orders[1])
    m2 = machine(base, 2.0)
    m2.loc[m2["query_id"] == orders[0][0], "status"] = "timeout"
    p = ae.Pair({"M1": machine(base), "M2": m2}, split, "M1", "M2")
    kept = [p.pool_ids[i] for i in p.positions(orders[0])]
    assert kept == [q for q in orders[0] if q != orders[0][0]]


# --------------------------------------------------------------- the methods

def test_a_uniform_slowdown_is_one_speed_factor(base):
    labels = {"M1": machine(base), "M2": machine(base, 2.0)}
    split = ae.make_split(labels["M1"])
    p = ae.Pair(labels, split, "M1", "M2")
    logs, runs, _ = ae.c2_pair(p, ae.arrival_orders(split, 3), [5, 25])
    E = {key: v.mean() for key, v in logs.items()}
    assert np.exp(E[(ae.NONE, 25)]) > 1.6
    assert E[(ae.SPEED, 25)] < E[(ae.NONE, 25)] - 0.4
    assert abs(E[(ae.RIDGE, 25)] - E[(ae.SPEED, 25)]) < 0.1
    # No adaptation and the oracle do not depend on k or the order.
    assert np.allclose(logs[(ae.NONE, 5)], logs[(ae.NONE, 25)])
    assert np.allclose(logs[(ae.ORACLE, 5)][0], logs[(ae.ORACLE, 25)][2])
    assert set(runs["method"]) == set(ae.METHODS) and len(runs) == 3 * 2 * 5


def test_ridge_correction_finds_slowdowns_that_differ_by_template(base):
    slow = {t: np.exp(0.3 * i) for i, t in enumerate(TEMPLATES)}
    labels = {"M1": machine(base), "M2": machine(base, template_factor(base, slow))}
    split = ae.make_split(labels["M1"])
    p = ae.Pair(labels, split, "M1", "M2")
    logs, runs, qerr = ae.c2_pair(p, ae.arrival_orders(split, 3), [25])
    s = ae.c2_summary(p, logs, runs, qerr)
    pooled = s["pooled"][ae.RIDGE]["25"]
    assert pooled["median_lo"] <= pooled["median_q"] <= pooled["median_hi"]
    assert pooled["p90_lo"] <= pooled["p90_q"] <= pooled["p90_hi"]
    assert s["error"][ae.RIDGE]["25"]["q"] < s["error"][ae.SPEED]["25"]["q"] / 1.1
    assert s["ridge_vs_speed"]["25"]["gm"] < 1
    assert s["gap"][ae.RIDGE]["25"]["g"] > s["gap"][ae.SPEED]["25"]["g"]


def test_the_harder_test_never_trains_on_the_held_out_template(base, monkeypatch):
    labels = {"M1": machine(base), "M2": machine(base, 1.5)}
    split = ae.make_split(labels["M1"])
    p = ae.Pair(labels, split, "M1", "M2")
    seen = []
    original = ae.fit_source
    monkeypatch.setattr(ae, "fit_source", lambda df, seed: seen.append(set(df["template"])) or original(df, seed))
    n_orders = 2
    logs = ae.c2_hard(p, ae.arrival_orders(split, n_orders), k=10, n_orders=n_orders)
    per_template = 2 + n_orders  # source without T, oracle without T, one target-only fit per order
    assert len(seen) == per_template * len(p.templates)
    for j, t in enumerate(p.templates):
        for used in seen[j * per_template:(j + 1) * per_template]:
            assert t not in used
    assert all(v.shape == (n_orders, len(p.templates)) for v in logs.values())


# --------------------------------------------------------- continued learning

def test_refit_steps():
    assert ae.refit_steps(293) == [50, 100, 150, 200, 250, 293]
    assert ae.refit_steps(100) == [50, 100]
    assert ae.refit_steps(40) == [40]


def test_the_stream_is_scored_test_then_train(base, monkeypatch):
    monkeypatch.setattr(ae, "WINDOW", 10)
    labels = {"M1": machine(base), "M2": machine(base, 3.0)}
    split = ae.make_split(labels["M1"])
    p = ae.Pair(labels, split, "M1", "M2")
    rows = ae.c3_stream(p, split, n_orders=2, batch=5)
    n = p.n_test
    for _, r in rows.iterrows():
        # Each batch is predicted from the queries before it only.
        expected = {ae.NEVER: 0, ae.ALL_SEEN: r["start"], ae.LAST: min(10, r["start"])}[r["method"]]
        assert r["n_train"] == expected
    assert set(rows.loc[rows["start"] < n, "machine"]) == {"M1"}
    assert set(rows.loc[rows["start"] >= n, "machine"]) == {"M2"}
    first_after = rows[rows["after_switch"] == 0]
    assert (first_after["median_q"] > 2.0).all()  # nobody has seen M2 yet
    s = ae.c3_stream_summary(rows, "M1", "M2", 5)
    r = s["methods"]
    assert min(s["batch_sizes"]) >= 5
    assert r[ae.NEVER]["jump"] > 2.0
    assert r[ae.LAST]["final"] < r[ae.ALL_SEEN]["final"] < r[ae.NEVER]["final"]
    assert s["batch"] == 5 and s["queries_each"] == n


def test_stream_batches_leave_no_short_remainder():
    sizes = [len(b) for b in ae.stream_batches(204, 10)]
    assert sum(sizes) == 204 and sorted(set(sizes)) == [10, 11] and sizes.count(11) == 4
    assert [len(b) for b in ae.stream_batches(7, 10)] == [7]
    assert np.concatenate(ae.stream_batches(23, 5)).tolist() == list(range(23))


def test_recovery_and_stream_levels():
    assert stats.recovery_index(np.array([2.0, 1.5, 1.12, 1.05, 1.1, 1.0]), 1.05) == 2
    assert stats.recovery_index(np.array([1.0, 1.05]), 1.05) == 0
    assert stats.recovery_index(np.array([1.0, 1.5]), 1.0) is None
    curve = np.array([1.1, 1.0, 2.0, 1.6, 1.2, 1.2, 1.2, 1.2, 1.2])
    post = np.array([False, False] + [True] * 7)
    after = np.array([-20, -10, 0, 10, 20, 30, 40, 50, 60])
    r = ae.stream_levels(curve, post, after)
    # The level before the switch is the geometric mean of the batches before it (at most 5).
    assert r["before"] == pytest.approx(np.sqrt(1.1)) and r["first_after"] == pytest.approx(2.0)
    assert r["jump"] == pytest.approx(2.0 / np.sqrt(1.1)) and r["final"] == pytest.approx(1.2)
    assert r["recover_queries"] == 20


# ---------------------------------------------------------------- statistics

def test_paired_ratio_is_train_evals_computation():
    rng = np.random.default_rng(7)
    rows = []
    for t in range(9):
        y = np.exp(rng.normal(5, 1, 6))
        for model, spread in (("Cost-only", 0.6), ("Ridge", 0.4)):
            rows.append(pd.DataFrame({"model": model, "template": t, "y": y,
                                      "yhat": y * np.exp(rng.normal(0.2, spread, 6))}))
    e2 = pd.concat(rows, ignore_index=True)
    ref = te.paired_vs_cost(e2, np.random.default_rng(4000))["Ridge"]
    med = e2.assign(q=te.qerror(e2["y"], e2["yhat"])).groupby(["model", "template"])["q"].median()
    got = stats.paired_ratio(np.log(med["Ridge"].to_numpy()), np.log(med["Cost-only"].to_numpy()),
                             stats.resamples(9, np.random.default_rng(4000)))
    for k in ("gm", "gm_lo", "gm_hi", "sign_p", "wilcoxon_p"):
        assert got[k] == pytest.approx(ref[k], rel=1e-12)
    assert (got["wins"], got["n"]) == (ref["wins"], ref["n"])
    assert got["success"] == (got["gm_hi"] < 1)


def test_template_share_is_the_pilots_acceptance_statistic():
    spec = importlib.util.spec_from_file_location("pilot_adaptation", ROOT / "scripts" / "pilot_adaptation.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    rng = np.random.default_rng(3)
    groups = np.repeat([1, 3, 5, 18], [7, 5, 6, 1])
    values = rng.normal(0, 0.1, len(groups)) + 0.2 * (groups == 3)
    assert stats.template_share(values, groups) == pytest.approx(pilot.template_share(values, groups), rel=1e-12)
    share, p = stats.template_share(np.log(1.5) + 0 * values, groups)
    assert np.isnan(share) and np.isnan(p)


def test_the_shift_summary_tells_template_slowdowns_from_one_factor(base):
    slow = {t: np.exp(0.3 * i) for i, t in enumerate(TEMPLATES)}
    noise = np.exp(np.random.default_rng(0).normal(0, 0.05, len(base)))
    labels = {"M1": machine(base), "M2": machine(base, template_factor(base, slow) * noise),
              "M3": machine(base, 1.5 * noise)}
    split = ae.make_split(labels["M1"])
    by_template = ae.shift_summary(ae.Pair(labels, split, "M1", "M2"), threshold=0.14)
    uniform = ae.shift_summary(ae.Pair(labels, split, "M1", "M3"), threshold=0.14)
    assert by_template["queries"] == len(base)
    assert by_template["template_share"] > 0.9 and by_template["accept"] is True
    assert by_template["template_min"] == pytest.approx(1.0, abs=0.05)
    assert by_template["template_max"] == pytest.approx(np.exp(0.3 * 5), rel=0.05)  # Q18, with its 4 queries
    assert uniform["template_share"] < 0.5 and uniform["accept"] is False
    assert uniform["median"] == pytest.approx(1.5, rel=0.05)
    assert ae.shift_summary(ae.Pair(labels, split, "M1", "M3"), threshold=None)["accept"] is None


def test_gap_closed_and_first_k():
    picks = stats.resamples(3, np.random.default_rng(0), draws=50)
    g = stats.gap_closed(np.ones(3), np.full(3, 0.5), np.zeros(3), picks)
    assert g["g"] == pytest.approx(0.5) and g["g_lo"] == pytest.approx(0.5) and g["g_hi"] == pytest.approx(0.5)
    assert np.isnan(stats.gap_closed(np.ones(3), np.zeros(3), np.full(3, 2.0), picks)["g"])
    assert stats.first_reaching([5, 10, 25], [0.5, 0.95, 0.99], 0.9) == 10
    assert stats.first_reaching([5, 10], [0.5, np.nan], 0.9) is None


# ------------------------------------------------------------------------ C4

def test_template_groups_join_the_two_part_tpcds_templates():
    assert ae.template_group("14a") == ae.template_group("14b") == ae.template_group(14) == "14"
    assert ae.template_group("query23_2") == ae.template_group("q23") == "23"
    assert ae.template_group("39b") == "39"
    assert ae.template_group("7a") == "7a" and ae.template_group(7) == "7"


def test_c4_holds_out_both_parts_together(base):
    labels = machine(base)
    parts = np.where(labels["template"] == 14, np.where(labels.index % 2 == 0, "14a", "14b"),
                     labels["template"].astype(str))
    res, pred = ae.c4_unseen(labels.assign(template=parts))
    assert res["groups"] == len(TEMPLATES) and res["multi_part_groups"] == ["14"]
    assert "14" in res["per_group"] and "14a" not in res["per_group"]
    assert len(pred) == 2 * len(labels)
    # Group labels that are all numbers are stored as numbers, ordered as train_eval orders templates.
    assert sorted(pred.loc[pred["template"] == 14, "source_template"].unique()) == ["14a", "14b"]


def test_c4_finds_two_part_groups_from_the_query_ids(base):
    # The TPC-DS manifest keeps the integer template and puts the part in the query ID.
    labels = machine(base)
    is14 = labels["template"] == 14
    labels.loc[is14, "query_id"] = [f"q14_{i:03d}{'ab'[i % 2]}" for i in range(int(is14.sum()))]
    res, _ = ae.c4_unseen(labels)
    assert res["multi_part_groups"] == ["14"]


def test_c4_holds_out_families_and_reports_templates_beside_them(base):
    labels = machine(base)
    family = {t: t for t in TEMPLATES} | {TEMPLATES[1]: TEMPLATES[0]}
    res, pred, by_template = ae.c4_families(labels.assign(family=labels["template"].map(family)))
    assert res["unit"] == "family" and res["groups"] == len(TEMPLATES) - 1
    assert by_template["unit"] == "template" and by_template["groups"] == len(TEMPLATES)
    # Both templates of the family are predicted by a model that saw neither.
    pair = pred[pred["template"].astype(str) == str(TEMPLATES[0])]
    assert set(pair["source_template"].astype(str)) == {str(TEMPLATES[0]), str(TEMPLATES[1])}
    res2, _, none = ae.c4_families(labels)
    assert res2["unit"] == "template" and none is None


def test_numbers_near_a_decision_level_get_more_decimals():
    assert ae.fmt_level(0.8978, 0.9) == "0.898" and ae.fmt_level(0.95, 0.9) == "0.95"
    assert ae.fmt_level(0.996, 1.0) == "0.996" and ae.fmt_level(1.004, 1.0) == "1.004"
    assert ae.fmt_level(1.2, 1.0) == "1.20" and ae.fmt_level(float("nan"), 1.0) == "--"
    assert ae.fmt_p(0.0) == ae.fmt_p(1e-320) == "\\ensuremath{<10^{-300}}"


def test_c4_is_the_pilots_leave_one_template_out_comparison(base):
    labels = machine(base)
    res, _ = ae.c4_unseen(labels)
    d = ae.ok_rows(labels)
    recs = []
    for t in sorted(d["template"].unique()):
        tr, ts = d[d["template"] != t], d[d["template"] == t]
        recs += te.records("E2", f"Q{t}", ts, te.predict_all(tr, ts, int(t), reference=False, unseen=True))
    ref = te.paired_vs_cost(pd.concat(recs, ignore_index=True), np.random.default_rng(4000))[te.HYBRID]
    for k in ("gm", "gm_lo", "gm_hi", "sign_p", "wilcoxon_p"):
        assert res["paired"][k] == pytest.approx(ref[k], rel=1e-9)


# ------------------------------------------------------------------- reports

def test_macro_names_and_number_formats():
    assert ae.number_word(25) == "TwentyFive" and ae.number_word(100) == "OneHundred"
    assert ae.number_word(293) == "TwoHundredNinetyThree" and ae.number_word(1000) == "OneThousand"
    assert ae.tex_name("CTwo", "M1->M3", "Ridge", "K", 25) == "CTwoMOneMThreeRidgeKTwentyFive"
    assert ae.fmt_p(0.01234) == "0.012"
    assert ae.fmt_p(1.2e-15) == "\\ensuremath{1\\times 10^{-15}}"
    assert ae.tex("M1->M2 (descr.)") == "M1$\\to$M2 (descr.)" and ae.tex(">100") == "$>$100"
    assert ae.tex("median Q [95% CI]") == "median $Q$ [95\\% CI]"
    assert ae.tex("cost R^2 of M1_x & y") == "cost $R^2$ of M1\\_x \\& y"
    assert ae.tex(ae.fmt_p(1.2e-15)) == ae.fmt_p(1.2e-15)
    assert ae.tex("-0.52") == "$-$0.52" and ae.tex("--") == "--" and ae.tex("1.11 to 2.85") == "1.11 to 2.85"
    assert ae.tex("G >= 0.9 from k") == "$G \\ge 0.9$ from $k$"
    assert ae.fmt_p(float("nan")) == "--" and ae.pct(0.534) == "53%" and ae.pct(float("nan")) == "--"


def test_shown_numbers_never_contradict_their_decision():
    # A ratio's interval end against 1 (the preliminary M3->M1 run had 0.99785, shown before as 1.00).
    assert ae.fmt_ratio({"gm": 0.96, "gm_lo": 0.91, "gm_hi": 0.99785}) == "0.96 [0.91, 0.998]"
    assert ae.fmt_ratio({"gm": 0.93, "gm_lo": 0.89, "gm_hi": 0.96}) == "0.93 [0.89, 0.96]"
    assert ae.fmt_level(0.896, ae.GAP_LEVEL) == "0.896"
    # A template share against the acceptance threshold (the pilot's 0.13556, shown as 13.6%).
    assert ae.pct_threshold(0.13556) == "13.6%"
    assert ae.pct_level(0.80, 0.13556) == "80%" and ae.pct_level(0.1358, 0.13556) == "14%"
    assert ae.pct_level(0.1355, 0.13556) == "13.55%"
    assert ae.pct_level(0.5, None) == "50%" and ae.pct_level(float("nan"), 0.13556) == "--"


def test_end_to_end_run_writes_every_output(tmp_path, base, monkeypatch):
    monkeypatch.setattr(te, "SEEDS", range(2))
    monkeypatch.setattr(ae, "REFIT_EVERY", 20)
    monkeypatch.setattr(ae, "WINDOW", 20)
    slow = {t: np.exp(0.2 * i) for i, t in enumerate(TEMPLATES)}
    labels = {"M1": machine(base),
              "M2": machine(base, template_factor(base, slow) * np.exp(0.2 * base["workers_planned"].to_numpy())),
              "M3": machine(base, 1.5)}
    labels["M3"].loc[3, ["status", ae.LABEL]] = ["timeout", np.nan]
    split = ae.make_split(labels["M1"])
    args = argparse.Namespace(pairs="M1:M2,M1:M3,M3:M1", descriptive="M3:M1", primary="M1:M3", primary_k=10,
                              stream="M1:M2", ks="5,10,25", orders=3, hard_orders=2, stream_orders=2, skip="")
    cfg = ae.config(args)
    s, frames = ae.evaluate(labels, split, cfg, c4_labels=labels["M1"], jobs=1, log=lambda *_: None)
    out = tmp_path / "out"
    ae.save_run(s, frames, out)

    for name in ("summary.json", "numbers.tex", "results.md", "c2_runs.csv", "c2_per_template.csv",
                 "c2_hard_per_template.csv", "c3_continued_per_template.csv", "c3_stream.csv", "c4_predictions.csv",
                 *[f"table_{t}.tex" for t in ("c1", "shift", "c2", "c2_gap", "c2_ratio", "c2_pooled", "c2_hard",
                                               "c3_continued", "c3_stream", "c4")],
                 *[f"fig_{f}.{ext}" for f in ("c2_error_vs_k", "c3_continued", "c3_stream") for ext in ("pdf", "png")]):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["c2_primary"]["pair"] == "M1->M3" and summary["c2_primary"]["k"] == 10
    assert summary["c2"]["M1->M3"]["test"] == summary["c2"]["M1->M2"]["test"] - (split.set_index("query_id").loc[
        base.loc[3, "query_id"], "set"] == "test")
    assert summary["c3_continued"]["M1->M2"]["steps"] == ae.refit_steps(summary["c2"]["M1->M2"]["pool"], 20)
    assert set(summary["c1"]) == {"M1", "M2", "M3"}

    lines = (out / "numbers.tex").read_text(encoding="utf-8").splitlines()
    names = [re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{.*\}", line).group(1) for line in lines[1:]]
    assert len(names) == len(set(names)) and "PrimaryGeo" in names and "CFourGeo" in names
    assert "CTwoMOneMTwoShiftShare" in names and "ShiftShareThreshold" in names
    assert summary["c2"]["M1->M2"]["shift"]["accept"] is True

    # The report subcommand rebuilds the same tables and macros from summary.json alone.
    again = tmp_path / "again"
    ae.write_reports(summary, again)
    for name in ("numbers.tex", "table_c2.tex", "table_c3_stream.tex"):
        assert (again / name).read_text(encoding="utf-8") == (out / name).read_text(encoding="utf-8")
