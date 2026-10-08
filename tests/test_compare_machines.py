"""Tests for src/compare_machines.py: identical plans pass, any difference in
the tree or the estimates fails with exit status 1."""
import json
import sys

import pytest

import compare_machines
from synthetic_passes import SETTINGS, make_plan, write_pass

LABELS = {f"q{t:02d}_{i:03d}": 10.0 * t + i for t in range(1, 6) for i in range(1, 5)}
M3_SETTINGS = SETTINGS | {"port": "5433", "listen_addresses": "", "unix_socket_permissions": "0700",
                          "unix_socket_directories": "/var/tmp/pgstudy/socket"}


def setup(tmp_path, plans_b=None, labels_b=None, settings_b=None):
    runs = tmp_path / "runs"
    write_pass(runs / "M1" / "pass1", LABELS)
    write_pass(runs / "M3" / "pass1", labels_b or {q: 2 * v for q, v in LABELS.items()}, plans=plans_b,
               settings=settings_b or M3_SETTINGS, machine="M3")
    return runs


def main(monkeypatch, runs, a="M1", b="M3"):
    monkeypatch.setattr(sys, "argv", ["compare_machines.py", a, b, "--runs-dir", str(runs)])
    try:
        compare_machines.main()
    except SystemExit as e:
        return e.code
    return 0


def test_identical_plans_pass(tmp_path, monkeypatch, capsys):
    runs = setup(tmp_path)
    assert main(monkeypatch, runs) == 0
    out = json.loads((runs / "compare_M1_M3.json").read_text(encoding="utf-8"))
    assert out["shared"] == out["same_plan"] == len(LABELS) and out["mismatches"] == []
    # Only the connection settings differ, and those are meant to.
    assert out["server"] == {"version": None, "settings": {}}
    tex = (runs / "compare_M1_M3.tex").read_text(encoding="utf-8")
    assert f"\\newcommand{{\\MOneMThreeSamePlan}}{{{len(LABELS)}}}" in tex
    assert f"\\newcommand{{\\MOneMThreeShared}}{{{len(LABELS)}}}" in tex
    assert "20 with the same plan" in capsys.readouterr().out


def test_a_different_estimate_fails(tmp_path, monkeypatch, capsys):
    p = make_plan("q02_003")
    p["Plans"][0]["Plans"][1]["Plan Rows"] = 1499
    runs = setup(tmp_path, plans_b={"q02_003": p})
    assert main(monkeypatch, runs) == 1
    out = json.loads((runs / "compare_M1_M3.json").read_text(encoding="utf-8"))
    assert out["same_plan"] == len(LABELS) - 1
    [mm] = out["mismatches"]
    assert mm["query_id"] == "q02_003" and mm["kind"] == "estimates"
    assert mm["first_difference"] == {"node": "Aggregate > Hash Join > Seq Scan #2", "field": "Plan Rows",
                                      "a": 1500, "b": 1499}
    assert "q02_003: different estimates" in capsys.readouterr().out


def test_a_different_tree_fails(tmp_path, monkeypatch):
    runs = setup(tmp_path, plans_b={"q04_002": make_plan("q04_002", node="Merge Join")})
    assert main(monkeypatch, runs) == 1
    [mm] = json.loads((runs / "compare_M1_M3.json").read_text(encoding="utf-8"))["mismatches"]
    assert mm["kind"] == "tree"
    assert mm["first_difference"]["field"] == "Node Type"
    assert (mm["first_difference"]["a"], mm["first_difference"]["b"]) == ("Hash Join", "Merge Join")


def test_queries_of_one_machine_only_are_listed_but_pass(tmp_path, monkeypatch):
    runs = setup(tmp_path, labels_b={q: v for q, v in LABELS.items() if q != "q05_004"})
    assert main(monkeypatch, runs) == 0
    out = json.loads((runs / "compare_M1_M3.json").read_text(encoding="utf-8"))
    assert out["only_in_a"] == ["q05_004"] and out["only_in_b"] == []
    assert out["shared"] == len(LABELS) - 1


def test_no_shared_query_fails(tmp_path, monkeypatch):
    runs = setup(tmp_path, labels_b={"q09_001": 1.0})
    assert main(monkeypatch, runs) == 1


def test_a_memory_setting_difference_is_reported(tmp_path, monkeypatch, capsys):
    runs = setup(tmp_path, settings_b=M3_SETTINGS | {"shared_buffers": "16384"})
    assert main(monkeypatch, runs) == 0
    out = json.loads((runs / "compare_M1_M3.json").read_text(encoding="utf-8"))
    assert out["server"]["settings"] == {"shared_buffers": ["262144", "16384"]}
    assert "setting shared_buffers" in capsys.readouterr().out


def test_missing_plans_are_refused(tmp_path, monkeypatch):
    runs = setup(tmp_path)
    with pytest.raises(SystemExit, match="no plans for M2"):
        main_raises(monkeypatch, runs, "M1", "M2")


def main_raises(monkeypatch, runs, a, b):
    monkeypatch.setattr(sys, "argv", ["compare_machines.py", a, b, "--runs-dir", str(runs)])
    compare_machines.main()
