"""Tests for src/passcheck.py: the plan shape, the level-shift test and the
pass comparisons, on synthetic passes with and without shifts."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import passcheck
from passcheck import (MIN_SEGMENT, MIN_SHIFT, change_point, check_pass, compare, macro_prefix, quantile,
                       same_shape, shape)
from synthetic_passes import make_plan, write_pass

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilot"


def change_point_numpy(log_ratio):
    """The numpy version that scripts/compare_passes.py used before the move."""
    n = len(log_ratio)
    best, cut = np.inf, None
    for c in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        a, b = log_ratio[:c], log_ratio[c:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best:
            best, cut = sse, c
    if cut is None or abs(np.median(log_ratio[cut:]) - np.median(log_ratio[:cut])) < MIN_SHIFT:
        return None
    return cut


def stepped(n: int, at: int, factor: float, noise: float = 0.03) -> list[float]:
    """Log ratios with a small alternating noise and a level step at index at."""
    return [noise * (-1) ** i + (math.log(factor) if i >= at else 0.0) for i in range(n)]


# ----------------------------------------------------------- change_point

def test_change_point_finds_the_step():
    assert change_point(stepped(300, 180, 1.6)) == 180
    assert change_point(stepped(300, 40, 0.7)) == 40


def test_change_point_ignores_small_or_no_steps():
    assert change_point(stepped(300, 150, 1.05)) is None
    assert change_point(stepped(300, 150, 1.0)) is None
    assert change_point(stepped(300, 150, 1.12)) == 150


def test_change_point_needs_min_segment_on_each_side():
    assert change_point(stepped(2 * MIN_SEGMENT - 1, 20, 2.0)) is None
    assert change_point([]) is None
    # A step closer to the end than MIN_SEGMENT cannot be placed there.
    cut = change_point(stepped(200, 190, 3.0))
    assert cut is None or cut >= 200 - MIN_SEGMENT


def test_change_point_matches_the_numpy_version():
    rng = np.random.default_rng(680)
    for trial in range(200):
        n = int(rng.integers(40, 700))
        x = rng.normal(0, rng.uniform(0.01, 0.3), n)
        if trial % 3:
            at = int(rng.integers(1, n))
            x[at:] += rng.normal(0, 0.5)
        assert change_point(x) == change_point_numpy(x), trial


@pytest.mark.skipif(not (PILOT / "pass2" / "runtime_dataset.csv").exists(), reason="pilot data not present")
def test_change_point_on_the_pilot_passes():
    a = pd.read_csv(PILOT / "runtime_dataset.csv")
    b = pd.read_csv(PILOT / "pass2" / "runtime_dataset.csv")
    m = a.merge(b, on="query_id", suffixes=("_1", "_2"))
    m = m[(m.status_1 == "ok") & (m.status_2 == "ok")].sort_values("order_2")
    x = np.log(m.runtime_ms_2 / m.runtime_ms_1).to_numpy()
    assert change_point(x) == change_point_numpy(x) == 250


# ------------------------------------------------------------------ shape

def test_shape_ignores_execution_fields_but_not_estimates():
    p = make_plan("q01_001")
    q = json.loads(json.dumps(p))
    q["Actual Rows"] = 12
    q["Plans"][0]["Actual Loops"] = 3
    assert shape(p) == shape(q) and same_shape(p, q)
    q["Plans"][0]["Plans"][1]["Total Cost"] += 0.01
    assert not same_shape(p, q)
    r = make_plan("q01_001", node="Merge Join")
    assert not same_shape(p, r)
    assert not same_shape(p, None) and not same_shape(None, None)


# ------------------------------------------------------------- check_pass

def labels(n: int) -> dict[str, float]:
    return {f"q{t:02d}_{i:03d}": 10.0 * t + i for t in range(1, 11) for i in range(1, n // 10 + 1)}


def test_check_pass_finds_a_shift_in_the_new_pass(tmp_path):
    ref = labels(200)
    ids = sorted(ref)
    order = list(np.random.default_rng(1).permutation(ids))
    # The new pass slows down by 1.6 after its 120th query, like the pilot's
    # energy saver, with a little noise before and after.
    new = {q: ref[q] * (1.6 if k >= 120 else 1.0) * (1.01 if k % 2 else 0.99) for k, q in enumerate(order)}
    write_pass(tmp_path / "pass1", ref)
    write_pass(tmp_path / "pass2", new, order=order, seed=1)
    chk = check_pass(tmp_path / "pass1", tmp_path / "pass2")
    assert chk["n_compared"] == 200 and chk["same_plan"] == 200
    s = chk["shift"]
    assert s["after_query_number"] == 120 and s["of"] == 200
    assert s["level_change"] == pytest.approx(1.6, rel=0.03)
    rows = passcheck.read_pass(tmp_path / "pass2")
    assert s["at"] == rows[order[120]]["started_at"]
    assert chk["shift_in_reference_order"] is None


def test_check_pass_without_a_shift(tmp_path):
    ref = labels(200)
    order = list(np.random.default_rng(2).permutation(sorted(ref)))
    new = {q: ref[q] * (1.04 if k % 2 else 0.97) for k, q in enumerate(order)}
    write_pass(tmp_path / "pass1", ref)
    write_pass(tmp_path / "pass3", new, order=order, seed=2)
    chk = check_pass(tmp_path / "pass1", tmp_path / "pass3")
    assert chk["shift"] is None and chk["shift_in_reference_order"] is None
    q = [max(new[k] / ref[k], ref[k] / new[k]) for k in ref]
    assert chk["median_q"] == pytest.approx(np.median(q))
    assert chk["p90_q"] == pytest.approx(np.quantile(q, 0.9))


def test_check_pass_points_at_a_shift_inside_the_reference(tmp_path):
    true = labels(200)
    ref_order = sorted(true)
    # The reference pass itself ran 1.5 times slower from its 100th query on.
    ref = {q: true[q] * (1.5 if k >= 100 else 1.0) for k, q in enumerate(ref_order)}
    order = list(np.random.default_rng(3).permutation(ref_order))
    write_pass(tmp_path / "pass1", ref, order=ref_order)
    write_pass(tmp_path / "pass2", true, order=order, seed=1)
    chk = check_pass(tmp_path / "pass1", tmp_path / "pass2")
    assert chk["shift"] is None
    r = chk["shift_in_reference_order"]
    assert r["after_query_number"] == 100
    assert r["level_change"] == pytest.approx(1 / 1.5)


def test_check_pass_skips_failed_queries_and_counts_plan_changes(tmp_path):
    ref = labels(100)
    new = dict(ref)
    write_pass(tmp_path / "pass1", ref, status={"q01_001": "timeout"})
    write_pass(tmp_path / "pass2", new, status={"q02_001": "error: boom"},
               plans={"q03_001": make_plan("q03_001", cost=5.0), "q04_001": None})
    chk = check_pass(tmp_path / "pass1", tmp_path / "pass2")
    assert chk["n_compared"] == 98
    assert chk["same_plan"] == 96
    assert chk["median_q"] == 1.0


# ----------------------------------------------------------------- compare

def frames(tmp_path, ref, new, order, **kw):
    write_pass(tmp_path / "a", ref)
    write_pass(tmp_path / "b", new, order=order, **kw)
    return pd.read_csv(tmp_path / "a" / "runtime_dataset.csv"), pd.read_csv(tmp_path / "b" / "runtime_dataset.csv")


def test_compare_reports_the_shift_and_agreement(tmp_path):
    ref = labels(200)
    order = list(np.random.default_rng(4).permutation(sorted(ref)))
    new = {q: ref[q] * (1.7 if k >= 150 else 1.0) * (1.02 if k % 2 else 0.98) for k, q in enumerate(order)}
    a, b = frames(tmp_path, ref, new, order)
    out = compare(a, b, tmp_path / "a" / "plans", tmp_path / "b" / "plans")
    assert out["n_compared"] == 200 and out["same_plan"] == 200
    assert out["shift"]["after_query_number"] == 150
    assert out["shift"]["after"]["median_ratio"] == pytest.approx(1.7, rel=0.03)
    assert out["shift"]["before"]["median_q"] == pytest.approx(1.02, rel=0.001)
    assert out["shift"]["share_slower_after"] == 1.0
    assert out["longest_gap_s"] == 2.0


def test_compare_counts_a_missing_plan_as_different(tmp_path):
    ref = labels(100)
    a, b = frames(tmp_path, ref, dict(ref), sorted(ref), plans={"q05_002": None})
    out = compare(a, b, tmp_path / "a" / "plans", tmp_path / "b" / "plans")
    assert out["n_compared"] == 100 and out["same_plan"] == 99 and out["shift"] is None


def assert_same(out, saved, path=""):
    """Equal, except that floats may differ in the last bits: numpy's sums
    run in a different order on Linux and Windows (correlations differ by
    about 1e-16 there)."""
    if isinstance(saved, dict):
        assert isinstance(out, dict) and set(out) == set(saved), path
        for k in saved:
            assert_same(out[k], saved[k], f"{path}/{k}")
    elif isinstance(saved, list):
        assert isinstance(out, list) and len(out) == len(saved), path
        for i, (u, v) in enumerate(zip(out, saved)):
            assert_same(u, v, f"{path}[{i}]")
    elif isinstance(saved, float):
        assert out == pytest.approx(saved, rel=1e-12, abs=1e-15), path
    else:
        assert out == saved, path


@pytest.mark.skipif(not (PILOT / "pass2" / "reproducibility.json").exists(), reason="pilot data not present")
def test_compare_reproduces_the_pilot_report():
    a = pd.read_csv(PILOT / "runtime_dataset.csv")
    b = pd.read_csv(PILOT / "pass2" / "runtime_dataset.csv")
    out = compare(a, b, PILOT / "plans", PILOT / "pass2" / "plans")
    saved = json.loads((PILOT / "pass2" / "reproducibility.json").read_text(encoding="utf-8"))
    assert_same(json.loads(json.dumps(out)), saved)


# ------------------------------------------------------------------ small

def test_quantile_matches_numpy():
    rng = np.random.default_rng(5)
    for n in (1, 2, 7, 100):
        x = list(rng.lognormal(size=n))
        for p in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert quantile(x, p) == pytest.approx(np.quantile(x, p))


def test_macro_prefix_has_no_digits():
    assert macro_prefix("M1") == "MOne"
    assert macro_prefix("M3") == "MThree"
    assert macro_prefix("m-2") == "MTwo"
