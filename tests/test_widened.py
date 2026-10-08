import csv
import datetime as dt
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "generate_widened_queries", ROOT / "scripts" / "generate_widened_queries.py")
gw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gw)

V2 = ROOT / "queries" / "tpch_sf1_v2"
STD = ROOT / "queries" / "tpch_sf1"
CHECK = ROOT / "results" / "workload_check"
LISTED = [1, 3, 4, 5, 6, 10, 11, 12, 14, 15, 18, 19, 20]


@pytest.fixture(scope="module")
def workload():
    return gw.make_workload(gw.SEED)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split_rows():
    with open(V2 / "split.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ determinism

def test_generation_is_deterministic(workload):
    again = gw.make_workload(gw.SEED)
    assert again == workload
    assert gw.make_workload(gw.SEED + 1)[0] != workload[0]


def test_committed_files_match_a_fresh_generation(workload, manifest):
    entries, sql = workload
    assert manifest == json.loads(json.dumps(entries))
    for e in entries:
        path = V2 / f"{e['query_id']}.sql"
        if e["variant"] == "std":
            assert path.read_bytes() == (STD / path.name).read_bytes()
        else:
            assert path.read_bytes() == sql[e["query_id"]].encode("utf-8")


def test_templates_reproduce_every_pilot_query():
    std = json.loads((STD / "manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for e in std:
        if e["template"] in gw.TEMPLATES:
            text = (STD / f"{e['query_id']}.sql").read_text(encoding="utf-8")
            assert gw.render(e["template"], gw.parse(e["template"], text)) == text
            checked += 1
    assert checked == 12 * 20 + 4    # Q18 has only 4 standard queries


def test_parse_reads_back_widened_values(workload):
    entries, sql = workload
    for e in entries:
        if e["variant"] == "wide":
            assert gw.parse(e["template"], sql[e["query_id"]]) == e["parameters"]
    with pytest.raises(ValueError):
        gw.parse(4, sql["q05_w01"])


# ------------------------------------------------------------------ counts

def test_counts(manifest):
    assert len(manifest) == 680
    assert len(list(V2.glob("*.sql"))) == 680
    variants = Counter(e["variant"] for e in manifest)
    assert variants == {"std": 420, "wide": 260}
    wide = Counter(e["template"] for e in manifest if e["variant"] == "wide")
    assert sorted(wide) == LISTED == gw.WIDENED_TEMPLATES
    assert set(wide.values()) == {20}
    std_ids = {e["query_id"] for e in json.loads((STD / "manifest.json").read_text(encoding="utf-8"))}
    assert {e["query_id"] for e in manifest if e["variant"] == "std"} == std_ids
    for e in manifest:
        if e["variant"] == "wide":
            assert re.fullmatch(rf"q{e['template']:02d}_w(0[1-9]|1\d|20)", e["query_id"])
        # src/collect.py reads the template from the part of the ID before the first "_".
        assert int(e["query_id"].split("_")[0].lstrip("qQ")) == e["template"]


def test_no_duplicated_parameters(workload):
    entries, sql = workload
    by_template = defaultdict(list)
    for e in entries:
        by_template[e["template"]].append(e)
    for t, group in by_template.items():
        texts = [sql[e["query_id"]] for e in group]
        assert len({gw.dedupe_key(s) for s in texts}) == len(group), t
        if t in gw.TEMPLATES:
            keys = [k for e in group for k in gw.same_query_keys(e["parameters"], sql[e["query_id"]])]
            assert len(keys) == len(set(keys)), t


def test_same_window_in_other_units_counts_as_duplicate():
    months = {"start": "1994-08-01", "window": 3, "unit": "month"}
    days = {"start": "1994-08-01", "window": 92, "unit": "day"}
    assert gw.same_query_keys(months, gw.render(4, months)) & gw.same_query_keys(days, gw.render(4, days))
    other = {"start": "1994-08-01", "window": 91, "unit": "day"}
    assert not gw.same_query_keys(months, gw.render(4, months)) & gw.same_query_keys(other, gw.render(4, other))


def test_swapped_values_and_other_units_together_count_as_duplicate():
    # q12_001's values, with the two ship modes swapped and its year written as
    # 366 days (1996 is a leap year): the same rows, so the same query.
    std = {"shipmode1": "MAIL", "shipmode2": "AIR", "start": "1996-01-01", "window": 1, "unit": "year"}
    swapped = {"shipmode1": "AIR", "shipmode2": "MAIL", "start": "1996-01-01", "window": 366, "unit": "day"}
    assert gw.same_query_keys(std, gw.render(12, std)) & gw.same_query_keys(swapped, gw.render(12, swapped))
    shorter = dict(swapped, window=365)
    assert not gw.same_query_keys(std, gw.render(12, std)) & gw.same_query_keys(shorter, gw.render(12, shorter))


def test_window_end_follows_postgres_month_arithmetic():
    assert gw.window_end({"start": "1993-01-01", "window": 3, "unit": "month"}) == dt.date(1993, 4, 1)
    assert gw.window_end({"start": "1995-01-31", "window": 1, "unit": "month"}) == dt.date(1995, 2, 28)
    assert gw.window_end({"start": "1996-02-29", "window": 1, "unit": "year"}) == dt.date(1997, 2, 28)
    assert gw.window_end({"start": "1997-12-15", "window": 20, "unit": "day"}) == dt.date(1998, 1, 4)


# ------------------------------------------------------ one valid statement

def test_one_statement_per_file(manifest):
    for e in manifest:
        text = (V2 / f"{e['query_id']}.sql").read_text(encoding="utf-8")
        assert text.endswith(";\n") and text.count(";") == 1, e["query_id"]
        assert text.split(None, 1)[0] in ("select", "with"), e["query_id"]
        assert "--" not in text and "\r" not in text, e["query_id"]
        assert not re.search(r"\{\w+\}|:\d|:[xnos]\b", text), e["query_id"]
        assert gw.to_postgres(text) == text, e["query_id"]


def test_widened_sql_keeps_the_template_text():
    params = {"start": "1995-02-10", "window": 45, "unit": "day"}
    sql = gw.render(14, params)
    assert "l_shipdate >= date '1995-02-10'" in sql
    assert "l_shipdate < date '1995-02-10' + interval '45' day;" in sql
    q1 = gw.render(1, {"delta": 1500})
    assert "interval '1500' day\n" in q1 and "(3)" not in q1


# ------------------------------------------------------------------ ranges

def test_widened_values_lie_in_the_documented_ranges(manifest):
    column = {4: gw.ORDER_DATES, 5: gw.ORDER_DATES, 10: gw.ORDER_DATES, 6: gw.SHIP_DATES, 14: gw.SHIP_DATES,
              15: gw.SHIP_DATES, 20: gw.SHIP_DATES, 12: gw.RECEIPT_DATES}
    for e in manifest:
        t, p = e["template"], e["parameters"]
        if e["variant"] != "wide":
            continue
        if "window" in p:
            assert p["unit"] == "day" and gw.SHORTEST <= p["window"] <= gw.LONGEST
            first, last = column[t]
            start = dt.date.fromisoformat(p["start"])
            assert first <= start and gw.window_end(p) - dt.timedelta(days=1) <= last
        if t == 1:
            assert 0 <= p["delta"] <= 2400
        if t == 3:
            assert gw.ORDER_DATES[0] <= dt.date.fromisoformat(p["date"]) <= gw.ORDER_DATES[1]
            assert p["segment"] in gw.SEGMENTS
        if t == 6:
            assert 0 <= p["discount"] <= 0.10 and 0 <= p["discount_band"] <= 0.05 and 2 <= p["quantity"] <= 51
        if t == 11:
            assert 1e-7 <= p["fraction"] <= 1e-3 and p["nation"] in gw.NATIONS
        if t == 12:
            assert p["shipmode1"] != p["shipmode2"] and {p["shipmode1"], p["shipmode2"]} <= set(gw.SHIP_MODES)
        if t == 18:
            assert 30 <= p["quantity"] <= 311
        if t == 19:
            for i in (1, 2, 3):
                assert 1 <= p[f"quantity{i}"] <= 50 and 1 <= p[f"size{i}"] <= 50
                assert re.fullmatch(r"Brand#[1-5][1-5]", p[f"brand{i}"])
            assert 0 <= p["quantity_band"] <= 49
        if t == 20:
            words = p["prefix"].split(" ")
            assert 1 <= len(words) <= 2 and p["prefix"] == p["prefix"].strip()
            assert any(c.startswith(words[0]) for c in gw.COLORS) and p["nation"] in gw.NATIONS


def test_widened_ranges_reach_beyond_the_standard_ones(manifest):
    """For every numeric or date parameter the widened values leave the range
    the standard queries use, and date windows get both shorter and longer."""
    values = defaultdict(lambda: defaultdict(list))
    for e in manifest:
        for k, v in e["parameters"].items():
            values[(e["template"], k)][e["variant"]].append(v)
    for (t, k), v in values.items():
        if k in ("qgen_seed", "unit") or gw.kind(k) == "text" or not v["wide"]:
            continue
        std = v["std"]
        if k == "window":
            std_days = [(gw.window_end(e["parameters"]) - dt.date.fromisoformat(e["parameters"]["start"])).days
                        for e in manifest if e["template"] == t and e["variant"] == "std"]
            assert min(v["wide"]) < min(std_days) and max(v["wide"]) > max(std_days), t
        else:
            assert min(v["wide"]) < min(std) or max(v["wide"]) > max(std), (t, k)


# ------------------------------------------------------------------- split

def test_split_is_stratified(manifest, split_rows):
    assert [r["query_id"] for r in split_rows] == sorted(e["query_id"] for e in manifest)
    assert {r["set"] for r in split_rows} == {"pool", "test"}
    groups = defaultdict(Counter)
    for r in split_rows:
        groups[(int(r["template"]), r["variant"])][r["set"]] += 1
    for (t, variant), c in groups.items():
        n = c["pool"] + c["test"]
        assert c["test"] == (0 if n < 2 else max(1, round(0.3 * n))), (t, variant)
    assert groups[(18, "std")] == {"pool": 3, "test": 1}
    assert groups[(13, "std")] == {"pool": 11, "test": 5}
    assert all(groups[(t, "wide")] == {"pool": 14, "test": 6} for t in LISTED)
    total = Counter(r["set"] for r in split_rows)
    assert total == {"pool": 476, "test": 204}


def test_split_is_deterministic_and_matches_the_file(manifest, split_rows):
    rows = gw.make_split(manifest, gw.SEED)
    assert rows == gw.make_split(manifest, gw.SEED)
    assert [{k: str(v) for k, v in r.items()} for r in rows] == split_rows
    assert gw.make_split(manifest, gw.SEED + 1) != rows


def test_single_query_group_goes_to_the_pool():
    entries = [{"query_id": "q18_001", "template": 18, "variant": "std"},
               {"query_id": "q18_w01", "template": 18, "variant": "wide"},
               {"query_id": "q18_w02", "template": 18, "variant": "wide"}]
    sets = {r["query_id"]: r["set"] for r in gw.make_split(entries)}
    assert sets["q18_001"] == "pool"
    assert sorted(sets[q] for q in ("q18_w01", "q18_w02")) == ["pool", "test"]


# ------------------------------------------------------------ check on M1

def read_check(name):
    with open(CHECK / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_workload_check_ran_on_these_queries():
    """results/workload_check belongs to the committed queries: every query's
    text hashes to what the check ran, the timing pass covers exactly the
    widened queries, and the Q11 step the widened Q11 queries and both ends of
    the fraction for every nation. Regenerating the workload without running
    scripts/check_workload.py again fails here."""
    plans = read_check("explain.csv")
    assert sorted(r["query_id"] for r in plans) == sorted(p.stem for p in V2.glob("*.sql"))
    for r in plans:
        text = (V2 / f"{r['query_id']}.sql").read_text(encoding="utf-8").strip()
        assert r["sql_sha1"] == hashlib.sha1(text.encode("utf-8")).hexdigest(), r["query_id"]
    wide = {r["query_id"] for r in plans if r["variant"] == "wide"}
    assert {r["query_id"] for r in read_check("widened_runtimes.csv")} == wide
    q11 = read_check("q11_bounds.csv")
    assert {r["query_id"] for r in q11 if r["case"] == "widened"} == {q for q in wide if q.startswith("q11_")}
    assert sorted(r["nation"] for r in q11 if r["case"] == "fraction 0") == sorted(gw.NATIONS)
    assert sorted(r["nation"] for r in q11 if r["case"] == "fraction 1") == sorted(gw.NATIONS)
