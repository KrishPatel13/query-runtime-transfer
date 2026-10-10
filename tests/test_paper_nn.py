"""Tests for the data plumbing of src/paper_nn.py on the second round's workloads: the k grids, the
budget kH, the directory the plans come from, the command line and the summaries over pairs. No
network is trained, but paper_nn imports PyTorch, so without it (requirements-nn.txt) the file is
skipped."""
import sys

import pytest

torch = pytest.importorskip("torch")

import nn_model as nn  # noqa: E402
import paper_eval as pe  # noqa: E402
import paper_nn as pn  # noqa: E402


def test_k_grids_hold_the_budget():
    assert pn.ks_of("tpch") == pn.ks_of("tpcds") == pn.ks_of("tpch10") == pn.KS
    assert pn.ks_of("job") == [5, 10, 25, 33, 50]
    for w in list(pn.DEV) + pn.EXT:
        assert pe.budget(w) in pn.ks_of(w) and 10 in pn.ks_of(w)


def test_job_budget_takes_the_frozen_rule_of_k_25():
    # The nearest entry on a log scale: 33 is nearer to 25 than to 50.
    for mode in pn.FT_MODES:
        assert nn.fine_tune_settings(mode, 33) == tuple(nn.FINE_TUNE[mode][25])


def test_plans_come_from_the_first_machine_that_has_them(tmp_path):
    with pytest.raises(ValueError, match="pass1/plans"):
        pn.plan_dir("job", tmp_path)
    for d in ("M4_job", "M3_job"):
        (tmp_path / d / "pass1" / "plans").mkdir(parents=True)
    assert pn.plan_dir("job", tmp_path) == tmp_path / "M3_job" / "pass1" / "plans"
    (tmp_path / "M1_job" / "pass2" / "plans").mkdir(parents=True)        # not pass 1
    assert pn.plan_dir("job", tmp_path) == tmp_path / "M3_job" / "pass1" / "plans"
    (tmp_path / "M1_job" / "pass1" / "plans").mkdir(parents=True)
    assert pn.plan_dir("job", tmp_path) == tmp_path / "M1_job" / "pass1" / "plans"


def test_command_line(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(pn, "run", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["paper_nn.py", "run", "--out", str(tmp_path)])
    pn.main()
    assert seen["workloads"] == ["tpch", "tpcds"]
    monkeypatch.setattr(sys, "argv", ["paper_nn.py", "run", "--out", str(tmp_path), "--workloads", "tpch10", "job"])
    pn.main()
    assert seen["workloads"] == ["tpch10", "job"]
    # cv and tune never take the second round's workloads.
    for cmd in ("cv", "tune"):
        monkeypatch.setattr(sys, "argv", ["paper_nn.py", cmd, "--out", str(tmp_path), "--workloads", "job"])
        with pytest.raises(SystemExit):
            pn.main()


def test_summaries_over_pairs_skip_a_k_a_pair_lacks():
    def r(gm):
        return {"gm": gm, "gm_lo": gm * 0.9, "gm_hi": gm * 1.1, "success": gm * 1.1 < 1}
    a = {"workload": "job", "cells": {"round-robin|nn:full-ft|10": {"q": 1.2}, "round-robin|nn:full-ft|50": {"q": 1.1}},
         "h6": {"10": {"full": r(0.9), "head": r(0.95)}, "50": {"full": r(1.0), "head": r(1.0)}}}
    b = {"workload": "job", "cells": {"round-robin|nn:full-ft|10": {"q": 1.3}},
         "h6": {"10": {"full": r(0.8), "head": r(0.9)}}}
    g = pn.over_pairs([a, b])["job"]["round-robin"]["nn:full-ft"]
    assert g["50"] == pytest.approx(1.1) and g["10"] == pytest.approx((1.2 * 1.3) ** 0.5)
    h = pn.h6_counts([a, b])["job"]
    assert h["50"]["full"]["pairs"] == 1 and h["10"]["full"]["pairs"] == 2
    assert h["10"]["full"]["point_below_1"] == 2 and h["10"]["full"]["below_1"] == 2
