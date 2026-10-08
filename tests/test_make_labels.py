"""Tests for src/make_labels.py on synthetic passes of one machine."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import make_labels
from features import FEATURE_NAMES, extract_features
from synthetic_passes import make_plan, write_pass

PASS_LABELS = {
    1: {"q01_001": 100.0, "q01_w01": 50.0, "q02_001": 10.0, "q02_002": 20.0, "q03_001": 30.0, "q04_001": 40.0,
        "q06_001": 60.0},
    2: {"q01_001": 110.0, "q01_w01": 55.0, "q02_001": 9.0, "q02_002": 22.0, "q04_001": 44.0, "q06_001": 66.0},
    3: {"q01_001": 121.0, "q01_w01": 45.0, "q02_001": 11.0, "q02_002": 21.0, "q03_001": 33.0, "q04_001": 41.0,
        "q06_001": 63.0},
}
OK = ["q01_001", "q01_w01", "q02_001", "q02_002"]


def build(tmp_path, manifest=True):
    """Machine M1 with three passes: q03_001 times out in pass 2, q04_001 gets
    another plan in pass 3, q06_001's EXPLAIN fails in pass 1, and q05_001 is
    in the query directory but was never measured."""
    qd = tmp_path / "queries" / "tpch_sf1_v2"
    qd.mkdir(parents=True)
    ids = sorted(set(PASS_LABELS[1]) | {"q05_001"})
    for q in ids:
        (qd / f"{q}.sql").write_text("select 1;\n", encoding="utf-8")
    if manifest:
        entries = [{"query_id": q, "template": int(q[1:3]), "variant": "wide" if "_w" in q else "std"} for q in ids]
        (qd / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    m = tmp_path / "runs" / "M1"
    write_pass(m / "pass1", {q: v for q, v in PASS_LABELS[1].items() if q != "q06_001"}, query_dir=str(qd),
               status={"q06_001": "error: syntax error"}, plans={"q06_001": None})
    write_pass(m / "pass2", PASS_LABELS[2], status={"q03_001": "timeout"}, seed=1, query_dir=str(qd))
    write_pass(m / "pass3", PASS_LABELS[3], seed=2, query_dir=str(qd),
               plans={"q04_001": make_plan("q04_001", node="Merge Join")})
    return m, qd


def main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["make_labels.py", *argv])
    make_labels.main()


def test_labels_csv_follows_the_contract(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    lab = pd.read_csv(m / "labels.csv").set_index("query_id", drop=False)
    assert list(lab.columns) == (["query_id", "template", "variant", "status", "label_ms", "pass1_ms", "pass2_ms",
                                  "pass3_ms"] + FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 19
    assert sorted(lab.index) == sorted(set(PASS_LABELS[1]) | {"q05_001"})
    for q in OK:
        r = lab.loc[q]
        assert r.status == "ok"
        assert r.label_ms == np.median([PASS_LABELS[i][q] for i in (1, 2, 3)])
        assert [r.pass1_ms, r.pass2_ms, r.pass3_ms] == [PASS_LABELS[i][q] for i in (1, 2, 3)]
    assert lab.loc["q01_001", "label_ms"] == 110.0
    assert lab.loc["q01_w01", "variant"] == "wide" and set(lab.drop("q01_w01").variant) == {"std"}
    assert lab.loc["q01_w01", "template"] == 1 and lab.loc["q06_001", "template"] == 6


def test_failed_and_missing_queries_are_kept_with_their_status(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    lab = pd.read_csv(m / "labels.csv").set_index("query_id")
    t = lab.loc["q03_001"]
    assert t.status == "pass2 timeout" and math.isnan(t.label_ms)
    assert t.pass1_ms == 30.0 and math.isnan(t.pass2_ms) and t.pass3_ms == 33.0
    d = lab.loc["q04_001"]
    assert d.status == "pass3 plan differs" and math.isnan(d.label_ms)
    assert lab.loc["q05_001", "status"] == "pass1 missing; pass2 missing; pass3 missing"
    e = lab.loc["q06_001"]
    assert e.status == "pass1 error: syntax error" and math.isnan(e.pass1_ms) and e.pass2_ms == 66.0
    noise = json.loads((m / "noise.json").read_text(encoding="utf-8"))
    assert noise["n_queries"] == 8 and noise["n_ok"] == 4
    assert sum(noise["status_counts"].values()) == 8


def test_features_come_from_pass_1(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    lab = pd.read_csv(m / "labels.csv").set_index("query_id")
    # q04_001 changed plan in pass 3; its features are still pass 1's.
    for q in ("q01_001", "q04_001", "q03_001"):
        want = extract_features([{"Plan": make_plan(q)}])
        assert lab.loc[q, FEATURE_NAMES].to_dict() == pytest.approx(want)
    # Pass 1 has no plan for q06_001, so its features are pass 2's.
    assert lab.loc["q06_001", FEATURE_NAMES].to_dict() == pytest.approx(extract_features([{"Plan": make_plan("q06_001")}]))
    assert lab.loc["q05_001", FEATURE_NAMES].isna().all()


def test_noise_is_the_pass_to_pass_q_error(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    noise = json.loads((m / "noise.json").read_text(encoding="utf-8"))
    q = [max(PASS_LABELS[i][k] / PASS_LABELS[j][k], PASS_LABELS[j][k] / PASS_LABELS[i][k])
         for i, j in ((1, 2), (1, 3), (2, 3)) for k in OK]
    p = noise["pass_to_pass"]
    assert p["pairs"] == 3 and p["n"] == 12
    assert p["median_q"] == pytest.approx(np.median(q)) and p["p90_q"] == pytest.approx(np.quantile(q, 0.9))
    assert set(noise["pairs"]) == {"pass1_vs_pass2", "pass1_vs_pass3", "pass2_vs_pass3"}
    assert noise["pairs"]["pass1_vs_pass2"]["n_compared"] == 5
    tex = (m / "noise_numbers.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\MOneLabels}{4}" in tex and "\\newcommand{\\MOneQueries}{8}" in tex
    assert f"\\newcommand{{\\MOneNoiseMedQ}}{{{np.median(q):.2f}}}" in tex
    assert "\\newcommand{\\MOneRepeated}{0}" in tex


def test_without_a_manifest_every_variant_is_std(tmp_path, monkeypatch):
    m, qd = build(tmp_path, manifest=False)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    lab = pd.read_csv(m / "labels.csv")
    assert set(lab.variant) == {"std"} and len(lab) == 8


def test_the_query_dir_recorded_in_pass_1_is_found(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"))
    lab = pd.read_csv(m / "labels.csv").set_index("query_id")
    assert lab.loc["q01_w01", "variant"] == "wide" and "q05_001" in lab.index


def test_unfinished_or_missing_passes_are_refused(tmp_path):
    m, _ = build(tmp_path)
    write_pass(m / "pass4", PASS_LABELS[1], finished=False)
    with pytest.raises(SystemExit, match="did not finish"):
        make_labels.find_passes(m)
    (m / "pass4").rename(m / "pass5")
    with pytest.raises(SystemExit, match="without gaps"):
        make_labels.find_passes(m)
    with pytest.raises(SystemExit, match="does not exist"):
        make_labels.find_passes(tmp_path / "runs" / "M9")


def test_a_single_pass_machine(tmp_path, monkeypatch):
    # TPC-DS on M1 gets one pass; IDs like q14_003a belong to template 14.
    m = tmp_path / "runs" / "M1_tpcds"
    write_pass(m / "pass1", {"q14_003a": 12.0, "q14_003b": 30.0, "q07_001": 0.0})
    main(monkeypatch, "M1_tpcds", "--runs-dir", str(tmp_path / "runs"))
    lab = pd.read_csv(m / "labels.csv").set_index("query_id")
    assert list(lab.columns[:5]) == ["template", "variant", "status", "label_ms", "pass1_ms"]
    assert lab.loc["q14_003a", "label_ms"] == 12.0 and lab.loc["q14_003b", "template"] == 14
    assert lab.loc["q07_001", "status"] == "pass1 runtime not positive"
    noise = json.loads((m / "noise.json").read_text(encoding="utf-8"))
    assert noise["pass_to_pass"] is None and noise["pairs"] == {}
    tex = (m / "noise_numbers.tex").read_text(encoding="utf-8")
    assert "\\MOnetpcdsLabels}{2}" in tex and "NoiseMedQ" not in tex


def test_rejected_attempts_are_listed_not_used(tmp_path, monkeypatch):
    m, qd = build(tmp_path)
    write_pass(m / "rejected" / "pass2_attempt1", {q: 5 * v for q, v in PASS_LABELS[2].items()}, seed=1)
    main(monkeypatch, "M1", "--runs-dir", str(tmp_path / "runs"), "--query-dir", str(qd))
    noise = json.loads((m / "noise.json").read_text(encoding="utf-8"))
    assert noise["rejected_attempts"] == ["pass2_attempt1"]
    assert pd.read_csv(m / "labels.csv").set_index("query_id").loc["q01_001", "pass2_ms"] == 110.0
    assert "\\newcommand{\\MOneRepeated}{1}" in (m / "noise_numbers.tex").read_text(encoding="utf-8")
