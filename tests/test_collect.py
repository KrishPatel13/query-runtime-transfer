"""Tests for src/collect.py without a database: psql is replaced by a fake
that returns plans and timings, so the measurement protocol, the load1
column and the meta file can be checked. Also checks that old (pilot) and new
runtime_dataset.csv files load the way src/train_eval.py loads them."""
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import collect
from collect import BASE_FIELDS, load1, read_templates, template_of
from features import FEATURE_NAMES, extract_features
from synthetic_passes import make_plan

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilot"


class FakePsql:
    """Answers collect.py's three kinds of psql calls."""

    def __init__(self, timeout_for=()):
        self.sql: list[str] = []
        self.timeout_for = timeout_for

    def __call__(self, database_url, sql, timeout_ms):
        self.sql.append(sql)
        if sql.startswith("select json_build_object"):
            return {"version": "PostgreSQL 18.6", "jit_available": False, "database_size": "1 GB",
                    "settings": {"shared_buffers": "262144", "max_parallel_workers_per_gather": "2"}}
        query = sql.split(") ", 1)[1].strip()
        if sql.startswith("EXPLAIN (FORMAT JSON) "):
            return [{"Plan": make_plan(query)}]
        assert sql.startswith("EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON, BUFFERS ON, FORMAT JSON) ")
        if any(t in query for t in self.timeout_for):
            raise collect.QueryError("ERROR:  canceling statement due to statement timeout")
        # 10 ms for the first execution of a query, 11, 12, 13 ms after it.
        earlier = sum(s.endswith(query + "\n") for s in self.sql[:-1] if s.startswith("EXPLAIN (ANALYZE"))
        return [{"Plan": dict(make_plan(query), **{"Shared Read Blocks": 0}),
                 "Execution Time": 10.0 + earlier, "Planning Time": 0.25}]


@pytest.fixture
def query_dir(tmp_path):
    qd = tmp_path / "queries"
    qd.mkdir()
    for q in ("q01_001", "q01_w01", "q14_002"):
        (qd / f"{q}.sql").write_text(f"select '{q}';\n", encoding="utf-8")
    (qd / "manifest.json").write_text(json.dumps(
        [{"query_id": "q01_001", "template": 1, "variant": "std"},
         {"query_id": "q01_w01", "template": 1, "variant": "wide"},
         {"query_id": "q14_002", "template": 14, "variant": "std"}]), encoding="utf-8")
    return qd


def run_collect(monkeypatch, tmp_path, query_dir, fake, *extra):
    out = tmp_path / "out" / "runtime_dataset.csv"
    monkeypatch.setattr(collect, "run_psql", fake)
    monkeypatch.setattr(collect, "load1", lambda: 1.25)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--query-dir", str(query_dir), "--out", str(out), *extra])
    collect.main()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return out, rows, meta


def test_the_protocol_and_the_new_columns(monkeypatch, tmp_path, query_dir):
    fake = FakePsql()
    out, rows, meta = run_collect(monkeypatch, tmp_path, query_dir, fake, "--machine", "M3", "--seed", "1")
    assert list(rows[0]) == BASE_FIELDS + FEATURE_NAMES and BASE_FIELDS[-1] == "load1"
    assert sorted(r["query_id"] for r in rows) == ["q01_001", "q01_w01", "q14_002"]
    for r in rows:
        q = r["query_id"]
        # EXPLAIN once, then one warm-up and three timed runs; the label is the
        # median of the three timed runs (11, 12, 13 ms after the warm-up's 10).
        mine = [s for s in fake.sql if f"'{q}'" in s]
        assert [s.startswith("EXPLAIN (ANALYZE") for s in mine] == [False, True, True, True, True]
        assert mine[0] == f"EXPLAIN (FORMAT JSON) select '{q}';\n"
        assert r["status"] == "ok" and float(r["runtime_ms"]) == 12.0
        assert json.loads(r["runs_ms"]) == [11.0, 12.0, 13.0] and json.loads(r["warmup_ms"]) == [10.0]
        assert r["load1"] == "1.25"
        assert int(r["template"]) == (14 if q.startswith("q14") else 1)
        assert float(r["total_cost"]) == extract_features([{"Plan": make_plan(f"select '{q}';")}])["total_cost"]
        assert (out.parent / "plans" / f"{q}.json").exists()
    [m] = meta
    assert m["machine"] == "M3" and m["args"]["machine"] == "M3" and m["finished_at"]
    for k in ("cpu_model", "kernel", "hostname", "cpu_count", "platform"):
        assert k in m["host"]


def test_a_timeout_is_recorded_with_its_load(monkeypatch, tmp_path, query_dir):
    _, rows, _ = run_collect(monkeypatch, tmp_path, query_dir, FakePsql(timeout_for=("q01_w01",)))
    r = {x["query_id"]: x for x in rows}["q01_w01"]
    assert r["status"] == "timeout" and r["runtime_ms"] == "" and r["load1"] == "1.25"
    assert r["total_cost"] != ""


def test_load1_reads_the_first_field(tmp_path):
    p = tmp_path / "loadavg"
    p.write_text("2.40 9.93 12.04 2/3922 102684\n")
    assert load1(str(p)) == 2.40
    assert load1(str(tmp_path / "missing")) is None
    p.write_text("")
    assert load1(str(p)) is None


def test_templates_from_the_manifest_or_the_query_id(query_dir, tmp_path):
    t = read_templates(query_dir)
    assert t == {"q01_001": 1, "q01_w01": 1, "q14_002": 14}
    assert read_templates(tmp_path) == {}
    assert template_of("q01_w01", t) == 1
    assert template_of("q07_006", {}) == 7 and template_of("Q12_001", {}) == 12
    with pytest.raises(SystemExit, match="template"):
        template_of("ds_014", {})


def test_host_info_has_cpu_model_and_kernel():
    h = collect.host_info()
    assert h["cpu_model"] is not None and h["kernel"] and h["hostname"] and h["cpu_count"] >= 1


# ---------------------------------------------- loading in train_eval.py

def load_like_train_eval(path: Path, plan_dir: Path):
    """The loading steps of train_eval.main(), up to data_summary()."""
    import train_eval
    raw = pd.read_csv(path)
    df = raw[raw["status"] == "ok"].reset_index(drop=True)
    df["template"] = df["template"].astype(int)
    feats = [extract_features(json.loads((plan_dir / f"{q}.json").read_text(encoding="utf-8")))
             for q in df["query_id"]]
    df = df.drop(columns=[c for c in df.columns if c in FEATURE_NAMES or c in ("hash_count", "join_count")])
    df = pd.concat([df, pd.DataFrame(feats, index=df.index)[FEATURE_NAMES]], axis=1)
    return df, train_eval.data_summary(raw, df)


@pytest.mark.skipif(not (PILOT / "runtime_dataset.csv").exists(), reason="pilot data not present")
def test_old_and_new_csvs_load_in_train_eval(tmp_path):
    header = next(csv.reader(open(PILOT / "runtime_dataset.csv", newline="", encoding="utf-8")))
    assert "load1" not in header and set(header) <= set(BASE_FIELDS + FEATURE_NAMES)
    old, s_old = load_like_train_eval(PILOT / "runtime_dataset.csv", PILOT / "plans")
    assert s_old["n_ok"] == len(old) == 420
    # The same rows written with this collector's columns, load1 and the
    # spilled blocks included.
    new_path = tmp_path / "runtime_dataset.csv"
    pd.read_csv(PILOT / "runtime_dataset.csv").assign(load1=0.7, temp_written_blocks=0).reindex(
        columns=BASE_FIELDS + FEATURE_NAMES).to_csv(new_path, index=False)
    new, s_new = load_like_train_eval(new_path, PILOT / "plans")
    assert s_new == s_old
    pd.testing.assert_frame_equal(new.drop(columns=["load1", "temp_written_blocks"]), old)
