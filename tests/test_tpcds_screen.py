import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tpcds_screen", Path(__file__).resolve().parents[1] / "scripts" / "tpcds_screen.py")
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)

MANIFEST = [
    {"query_id": "q03_002", "template": 3, "part": None},
    {"query_id": "q03_001", "template": 3, "part": None},
    {"query_id": "q14_001a", "template": 14, "part": "a"},
    {"query_id": "q14_001b", "template": 14, "part": "b"},
    {"query_id": "q14_002a", "template": 14, "part": "a"},
    {"query_id": "q14_002b", "template": 14, "part": "b"},
]

# EXPLAIN (ANALYZE, FORMAT JSON) output, shortened.
PLAN = [{"Plan": {"Node Type": "Limit", "Total Cost": 1234.5, "Plan Rows": 100, "Actual Rows": 42},
         "Planning Time": 1.5, "Execution Time": 2500.0}]


def test_first_query_of_every_part():
    assert [e["query_id"] for e in screen.first_queries(MANIFEST)] == ["q03_001", "q14_001a", "q14_001b"]
    assert [e["query_id"] for e in screen.first_queries(MANIFEST, {14})] == ["q14_001a", "q14_001b"]


def test_keep_rule():
    assert screen.keeps({"status": "ok", "execution_ms": 30000.0}, 30)
    assert not screen.keeps({"status": "ok", "execution_ms": 30000.1}, 30)
    assert not screen.keeps({"status": "timeout"}, 30)
    assert not screen.keeps({"status": "error: ERROR:  division by zero"}, 30)


def test_screen_query_times_like_the_collector(monkeypatch):
    seen = {}

    def fake(url, sql, timeout_ms):
        seen.update(sql=sql, timeout_ms=timeout_ms)
        return PLAN

    monkeypatch.setattr(screen, "run_psql", fake)
    r = screen.screen_query("postgresql:///tpcds", "select 1;", 60)
    assert r["status"] == "ok"
    assert (r["execution_ms"], r["planning_ms"], r["total_cost"], r["actual_rows"]) == (2500.0, 1.5, 1234.5, 42)
    assert seen["timeout_ms"] == 60000
    assert seen["sql"] == "EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON) select 1;\n"


def test_timeouts_and_errors_are_recorded(monkeypatch):
    def timeout(*_):
        raise screen.QueryError("ERROR:  canceling statement due to statement timeout")

    def failure(*_):
        raise screen.QueryError("ERROR:  division by zero\nCONTEXT:  parallel worker")

    monkeypatch.setattr(screen, "run_psql", timeout)
    assert screen.screen_query("u", "select 1;", 60)["status"] == "timeout"
    monkeypatch.setattr(screen, "run_psql", failure)
    assert screen.screen_query("u", "select 1;", 60)["status"] == "error: ERROR:  division by zero"


def test_a_group_is_kept_when_one_part_is():
    rows = [
        {"query_id": "q03_001", "template": 3, "part": "", "status": "ok", "execution_ms": 31000.0, "keep": False},
        {"query_id": "q14_001a", "template": 14, "part": "a", "status": "ok", "execution_ms": 1000.0, "keep": True},
        {"query_id": "q14_001b", "template": 14, "part": "b", "status": "timeout", "keep": False},
    ]
    s = screen.summarize(rows, MANIFEST, min_templates=2)
    assert s["kept_groups"] == [14] and s["dropped_groups"] == [3]
    assert s["kept_queries"] == 1
    assert s["queries_to_collect"] == 2
    assert s["collect_query_ids"] == ["q14_001a", "q14_002a"]
    assert s["dropped_queries"] == {"q03_001": "31.0 s", "q14_001b": "timeout"}
    assert not s["enough_template_groups"]
    assert all(k.isalpha() for k in screen.macros({"timeout_s": 60, "keep_s": 30.0}, s))


def test_summary_counts_families_and_draws():
    manifest = [dict(e, family=3 if e["template"] == 3 else 14) for e in MANIFEST] + [
        {"query_id": "q42_001", "template": 42, "part": None, "family": 3}]
    rows = [{"query_id": "q03_001", "template": 3, "part": "", "status": "ok", "execution_ms": 10.0, "keep": True},
            {"query_id": "q42_001", "template": 42, "part": "", "status": "ok", "execution_ms": 20.0, "keep": True},
            {"query_id": "q14_001a", "template": 14, "part": "a", "status": "timeout", "keep": False},
            {"query_id": "q14_001b", "template": 14, "part": "b", "status": "ok", "execution_ms": 9.0, "keep": True}]
    s = screen.summarize(rows, manifest, min_templates=2)
    assert s["kept_template_groups"] == 3 and s["kept_families"] == 2 and s["enough_families"]
    assert s["collect_query_ids"] == ["q03_002", "q03_001", "q14_001b", "q14_002b", "q42_001"]
    assert s["queries_to_collect"] == 5 and s["draws_to_collect"] == 5


def test_a_stalled_run_is_recognized():
    assert screen.stalled({"status": "ok", "execution_ms": 2000.0, "wall_s": 2.0 + screen.STALL_SLACK_S + 1})
    assert not screen.stalled({"status": "ok", "execution_ms": 2000.0, "wall_s": 2.3})
    assert not screen.stalled({"status": "timeout", "wall_s": 999.0})


def test_the_power_target_comes_from_the_pilot_macros(tmp_path):
    f = tmp_path / "numbers.tex"
    f.write_text(r"\newcommand{\NQueries}{420}" "\n" r"\newcommand{\PowerTemplates}{76}" "\n", encoding="utf-8")
    assert screen.power_target(f) == 76
    assert screen.power_target(tmp_path / "missing.tex") is None


def test_the_kept_workload_is_its_own_directory(tmp_path):
    import json
    src = tmp_path / "all"
    src.mkdir()
    for e in MANIFEST:
        (src / f"{e['query_id']}.sql").write_text(f"select 1; -- {e['query_id']}\n", encoding="utf-8")
    (src / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    kept = tmp_path / "kept"
    assert screen.write_kept(src, kept, ["q03_001", "q14_001b"], tmp_path / "screen.json", overwrite=False) == 2
    assert sorted(p.name for p in kept.glob("*.sql")) == ["q03_001.sql", "q14_001b.sql"]
    assert [e["query_id"] for e in json.loads((kept / "manifest.json").read_text(encoding="utf-8"))] == \
        ["q03_001", "q14_001b"]
    import pytest
    with pytest.raises(SystemExit, match="overwrite"):
        screen.write_kept(src, kept, ["q03_001"], tmp_path / "screen.json", overwrite=False)
    assert screen.write_kept(src, kept, ["q03_001"], tmp_path / "screen.json", overwrite=True) == 1
    assert sorted(p.name for p in kept.glob("*.sql")) == ["q03_001.sql"]
