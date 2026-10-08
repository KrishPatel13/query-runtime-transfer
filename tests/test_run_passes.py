"""Tests for src/run_passes.py with a fake collector that writes synthetic
passes, so the pass logic (seeds, level-shift checks, repeats, resuming) and
the laptop power checks run without a database or Windows."""
import json
import random
from pathlib import Path

import pytest

import run_passes
from run_passes import LAPTOP_CLOCK, LaptopGuard, check_inputs, disruption, parse_args, power_summary, run
from synthetic_passes import HOST, write_pass

LABELS = {f"q{t:02d}_{i:03d}": 20.0 * t + 3 * i for t in range(1, 13) for i in range(1, 16)}   # 180 queries


class FakeCollector:
    """Stands in for collect.py. shifts[(pass, attempt)] = (k, factor): that
    attempt runs `factor` times slower from its k-th query on."""

    def __init__(self, shifts=None):
        self.shifts = shifts or {}
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, pass_dir: Path, seed: int, resume: bool, args) -> None:
        i = seed + 1
        attempt = sum(s == seed and not r for s, r in self.calls) + (0 if resume else 1)
        self.calls.append((seed, resume))
        order = sorted(LABELS)
        random.Random(seed).shuffle(order)
        at, factor = self.shifts.get((i, attempt), (len(order), 1.0))
        vals = {q: LABELS[q] * (1.01 if (k + seed) % 2 else 0.99) * (factor if k >= at else 1.0)
                for k, q in enumerate(order)}
        # A distinct start time per call, as a real measurement has.
        write_pass(pass_dir, vals, order=order, seed=seed, query_dir=str(args.query_dir),
                   started_at=f"2026-10-01T{10 + len(self.calls):02d}:00:00")


class FakeGuard:
    """powers: power summaries to return from after(), in call order; then a clean one."""

    def __init__(self, powers=None):
        self.events = []
        self.closed = False
        self.powers = list(powers or [])

    def before(self, label):
        self.events.append(("before", label))

    def after(self, label):
        self.events.append(("after", label))
        if self.powers:
            return self.powers.pop(0)
        return {"lines": 4, "on_battery": 0, "saver_on": 0, "charge_min": 90, "max_gap_s": 30.0, "gaps": 0}

    def close(self):
        self.closed = True


def args_for(tmp_path, *extra):
    return parse_args(["--machine", "M1", "--query-dir", str(tmp_path / "queries"),
                       "--out-root", str(tmp_path / "runs"), *extra])


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_passes_with_seeds_0_1_2(tmp_path):
    collector, guard = FakeCollector(), FakeGuard()
    summary = run(args_for(tmp_path), collector=collector, guard=guard, host=HOST)
    assert collector.calls == [(0, False), (1, False), (2, False)]
    assert guard.closed and [e for e in guard.events if e[0] == "before"] == [
        ("before", "pass1"), ("before", "pass2"), ("before", "pass3")]
    m = tmp_path / "runs" / "M1"
    assert [p["dir"] for p in summary["passes"]] == ["pass1", "pass2", "pass3"]
    assert all(p["finished"] for p in summary["passes"]) and summary["rejected"] == []
    assert summary["passes"][0]["load1"] == {"median": 0.5, "max": 0.5}
    for i in (2, 3):
        rec = read(m / f"pass{i}" / "run.json")
        assert rec["seed"] == i - 1 and rec["attempt"] == 1
        assert rec["check"]["shift"] is None and rec["check"]["n_compared"] == len(LABELS)
        assert rec["power"][0]["charge_min"] == 90
    assert read(m / "passes.json")["passes_planned"] == 3


def test_a_shifted_pass_is_measured_once_more(tmp_path):
    collector = FakeCollector({(2, 1): (120, 1.6)})
    summary = run(args_for(tmp_path), collector=collector, guard=FakeGuard(), host=HOST)
    assert collector.calls == [(0, False), (1, False), (1, False), (2, False)]
    m = tmp_path / "runs" / "M1"
    old = read(m / "rejected" / "pass2_attempt1" / "run.json")
    assert old["attempt"] == 1 and old["check"]["shift"]["after_query_number"] == 120
    assert old["rejected"]["reason"].startswith("level shift of 1.6")
    new = read(m / "pass2" / "run.json")
    assert new["attempt"] == 2 and new["check"]["shift"] is None
    assert [r["dir"] for r in summary["rejected"]] == ["rejected/pass2_attempt1"]


def test_a_pass_that_shifts_twice_is_kept_and_recorded(tmp_path, capsys):
    collector = FakeCollector({(3, 1): (90, 0.7), (3, 2): (60, 1.5)})
    run(args_for(tmp_path), collector=collector, guard=FakeGuard(), host=HOST)
    assert collector.calls == [(0, False), (1, False), (2, False), (2, False)]
    m = tmp_path / "runs" / "M1"
    kept = read(m / "pass3" / "run.json")
    assert kept["attempt"] == 2 and kept["check"]["shift"]["after_query_number"] == 60
    assert "rejected" not in kept
    assert (m / "rejected" / "pass3_attempt1").is_dir()
    assert "shifts again" in capsys.readouterr().err


def test_a_shift_inside_pass_1_measures_pass_1_again(tmp_path):
    collector = FakeCollector({(1, 1): (100, 1.5)})
    run(args_for(tmp_path), collector=collector, guard=FakeGuard(), host=HOST)
    # Pass 2 shows the shift in pass 1's order, so pass 1 is measured again
    # and pass 2 is checked against the new pass 1 without being measured.
    assert collector.calls == [(0, False), (1, False), (0, False), (2, False)]
    m = tmp_path / "runs" / "M1"
    old = read(m / "rejected" / "pass1_attempt1" / "run.json")
    assert "found against pass 2" in old["rejected"]["reason"]
    assert old["rejected"]["reason"].startswith("level shift of 1.50x after query 100")
    assert read(m / "pass1" / "run.json")["attempt"] == 2
    for i in (2, 3):
        chk = read(m / f"pass{i}" / "run.json")["check"]
        assert chk["shift"] is None and chk["shift_in_reference_order"] is None


def test_pass_1_found_shifted_late_rechecks_the_earlier_passes(tmp_path, monkeypatch):
    # The pass 1 shift is only noticed against pass 3 (the check against pass 2
    # is made to miss it), so pass 2 must be checked again afterwards.
    collector = FakeCollector({(1, 1): (100, 1.5)})
    real = run_passes.check_pass
    seen = []

    def check(ref, new):
        out = real(ref, new)
        seen.append((Path(new).name, out["shift_in_reference_order"] is not None))
        if Path(new).name == "pass2" and len(seen) == 1:
            out["shift_in_reference_order"] = None
        return out

    monkeypatch.setattr(run_passes, "check_pass", check)
    run(args_for(tmp_path), collector=collector, guard=FakeGuard(), host=HOST)
    assert collector.calls == [(0, False), (1, False), (2, False), (0, False)]
    # pass 2 (shift hidden), pass 3 (shift found, pass 1 measured again),
    # pass 3 against the new pass 1, then pass 2 against it.
    assert seen == [("pass2", True), ("pass3", True), ("pass3", False), ("pass2", False)]
    m = tmp_path / "runs" / "M1"
    for i in (2, 3):
        chk = read(m / f"pass{i}" / "run.json")["check"]
        assert chk["identity"]["reference"].startswith(read_meta_start(m / "pass1"))
        assert chk["shift"] is None
    assert read(m / "pass1" / "run.json")["attempt"] == 2


def read_meta_start(d: Path) -> str:
    return read(d / "runtime_dataset.meta.json")[0]["started_at"]


def test_running_again_measures_and_checks_nothing_new(tmp_path, monkeypatch):
    run(args_for(tmp_path), collector=FakeCollector(), guard=FakeGuard(), host=HOST)
    calls = []
    monkeypatch.setattr(run_passes, "check_pass", lambda *a: calls.append(a))
    collector, guard = FakeCollector(), FakeGuard()
    summary = run(args_for(tmp_path), collector=collector, guard=guard, host=HOST)
    assert collector.calls == [] and calls == [] and guard.events == []
    assert all(p["finished"] for p in summary["passes"])


def test_an_interrupted_pass_is_resumed(tmp_path):
    m = tmp_path / "runs" / "M1"
    write_pass(m / "pass1", dict(list(LABELS.items())[:50]), finished=False,
               query_dir=str((tmp_path / "queries").resolve()))
    collector = FakeCollector()
    run(args_for(tmp_path, "--passes", "1"), collector=collector, guard=FakeGuard(), host=HOST)
    assert collector.calls == [(0, True)]


def test_passes_of_one_machine_come_from_one_host_and_query_dir(tmp_path):
    m = tmp_path / "runs" / "M1"
    qd = (tmp_path / "queries").resolve()
    write_pass(m / "pass1", LABELS, query_dir=str(qd))
    check_inputs(m, qd, dict(HOST))
    with pytest.raises(SystemExit, match="another host"):
        check_inputs(m, qd, HOST | {"hostname": "server-x"})
    with pytest.raises(SystemExit, match="another host"):
        check_inputs(m, qd, HOST | {"cpu_count": 1})
    with pytest.raises(SystemExit, match="measured from"):
        check_inputs(m, (tmp_path / "other").resolve(), dict(HOST))


def test_arguments():
    assert parse_args(["--machine", "M3"]).passes == 3
    for bad in (["--machine", "../M1"], ["--machine", "M1", "--passes", "0"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_the_keep_awake_window_covers_all_passes_and_repeats(tmp_path, monkeypatch):
    made = []

    class Recorder(FakeGuard):
        def __init__(self, log, allow_battery, seconds, allow_clock=False):
            super().__init__()
            made.append((log, allow_battery, seconds))

    monkeypatch.setattr(run_passes, "on_laptop", lambda: True)
    monkeypatch.setattr(run_passes, "LaptopGuard", Recorder)
    run(args_for(tmp_path), collector=FakeCollector(), host=HOST)
    run(args_for(tmp_path, "--machine", "M2", "--passes", "2", "--allow-battery"), collector=FakeCollector(),
        host=HOST)
    assert made[0][2] == 20 * 3600 and made[0][0] == tmp_path / "runs" / "M1" / "power.log"
    assert made[1][1:] == (True, 12 * 3600)


# ----------------------------------------------------------- laptop guard

@pytest.fixture
def guard(tmp_path, monkeypatch):
    monkeypatch.setattr(run_passes, "windows_path", lambda p: str(p))
    monkeypatch.setattr(run_passes, "laptop_clock", lambda: {"clocksource": LAPTOP_CLOCK, "ok": True})
    g = LaptopGuard(tmp_path / "M1" / "power.log", allow_battery=False, seconds=3600)
    g.helper_starts = 0

    def start_helper():
        g.helper_starts += 1
    monkeypatch.setattr(g, "start_helper", start_helper)
    return g


@pytest.mark.parametrize("state, message", [("ac=0 charge=80% saver=0", "on battery"),
                                            ("ac=1 charge=80% saver=1", "energy saver"),
                                            ("unknown", "could not read")])
def test_the_laptop_refuses_battery_and_energy_saver(guard, monkeypatch, state, message):
    monkeypatch.setattr(guard, "status", lambda: state)
    with pytest.raises(SystemExit, match=message):
        guard.before("pass1")
    assert guard.helper_starts == 0
    assert f"pass1 start {state}" in guard.log.read_text(encoding="utf-8")
    guard.allow_battery = True
    guard.before("pass1")
    assert guard.helper_starts == 1


def test_on_ac_the_helper_runs_and_its_lines_are_summarized(guard, monkeypatch):
    monkeypatch.setattr(guard, "status", lambda: "ac=1 charge=97% saver=0")
    guard.before("pass2")
    assert guard.helper_starts == 1
    with open(guard.log, "a", encoding="utf-8") as f:
        f.write("2026-10-01T10:00:30-04:00 ac=1 charge=97% saver=0\r\n"
                "2026-10-01T10:01:00-04:00 ac=0 charge=96% saver=0\r\n"
                "2026-10-01T10:01:30-04:00 ac=0 charge=19% saver=1\r\n")
    s = guard.after("pass2")
    assert (s["lines"], s["on_battery"], s["saver_on"], s["charge_min"]) == (3, 2, 1, 19)
    assert guard.log.read_text(encoding="utf-8").splitlines()[-1].endswith("pass2 end ac=1 charge=97% saver=0")


def test_power_summary_skips_the_start_and_end_marks():
    lines = ["2026-10-01T10:00:00-04:00 pass1 start ac=0 charge=50% saver=1",
             "2026-10-01T10:00:00-04:00 ac=1 charge=50% saver=0",
             "\ufeff2026-10-01T10:00:30-04:00 ac=1 charge=49% saver=0"]
    assert power_summary(lines) == {"lines": 2, "on_battery": 0, "saver_on": 0, "charge_min": 49,
                                    "max_gap_s": 30.0, "gaps": 0, "host_cpu": None}
    lines += ["2026-10-01T10:01:00-04:00 ac=1 charge=49% saver=0 cpu=12%",
              "2026-10-01T10:01:30-04:00 ac=1 charge=49% saver=0 cpu=40%"]
    assert power_summary(lines)["host_cpu"] == {"median": 26.0, "max": 40}


def test_a_gap_in_the_power_log_means_the_laptop_slept():
    lines = ["2026-10-01T10:00:00-04:00 pass1 start ac=1 charge=50% saver=0",
             "2026-10-01T10:00:30-04:00 ac=1 charge=50% saver=0",
             "2026-10-01T10:51:00-04:00 ac=1 charge=49% saver=0",
             "2026-10-01T10:51:30-04:00 pass1 end"]
    s = power_summary(lines)
    assert (s["max_gap_s"], s["gaps"]) == (3030.0, 1)
    assert "slept" in disruption(s)
    assert disruption(power_summary(lines[:2])) is None
    assert "on battery" in disruption({"on_battery": 2, "saver_on": 0, "gaps": 0})


def test_the_laptop_refuses_a_clock_that_is_not_set_up(guard, monkeypatch):
    monkeypatch.setattr(guard, "status", lambda: "ac=1 charge=97% saver=0")
    monkeypatch.setattr(run_passes, "laptop_clock", lambda: {"clocksource": "tsc", "ok": False})
    with pytest.raises(SystemExit, match="WSL clock is not set up"):
        guard.before("pass1")
    assert guard.helper_starts == 0
    guard.allow_clock = True
    guard.before("pass1")
    assert guard.helper_starts == 1


def test_a_disrupted_pass_is_measured_once_more(tmp_path):
    slept = {"lines": 40, "on_battery": 0, "saver_on": 0, "charge_min": 90, "max_gap_s": 3000.0, "gaps": 1}
    collector, guard = FakeCollector(), FakeGuard(powers=[{"lines": 4, "on_battery": 0, "saver_on": 0,
                                                            "charge_min": 90, "max_gap_s": 30.0, "gaps": 0},
                                                           slept])
    summary = run(args_for(tmp_path), collector=collector, guard=guard, host=HOST)
    assert collector.calls == [(0, False), (1, False), (1, False), (2, False)]
    assert [r["dir"] for r in summary["rejected"]] == ["rejected/pass2_attempt1"]
    m = tmp_path / "runs" / "M1"
    assert "slept" in read(m / "rejected" / "pass2_attempt1" / "run.json")["rejected"]["reason"]
    assert read(m / "pass2" / "run.json")["attempt"] == 2


def test_a_pass_disrupted_twice_is_kept(tmp_path, capsys):
    slept = {"lines": 40, "on_battery": 3, "saver_on": 0, "charge_min": 60, "max_gap_s": 30.0, "gaps": 0}
    collector, guard = FakeCollector(), FakeGuard(powers=[slept, slept])
    summary = run(args_for(tmp_path, "--passes", "1"), collector=collector, guard=guard, host=HOST)
    assert collector.calls == [(0, False), (0, False)]
    assert summary["passes"][0]["finished"] and len(summary["rejected"]) == 1
    assert "disrupted again" in capsys.readouterr().err


class FakeHelper:
    """Stands in for powershell.exe running keep_awake.ps1."""
    mode = "writes"

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.returncode = None
        if self.mode == "writes":
            log = Path(cmd[cmd.index("-LogPath") + 1])
            with open(log, "a", encoding="utf-8") as f:
                f.write("2026-10-01T10:00:00-04:00 ac=1 charge=97% saver=0\r\n")
        elif self.mode == "exits":
            self.returncode = 1

    def poll(self):
        return self.returncode


def real_guard(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(run_passes, "windows_path", lambda p: str(p))
    monkeypatch.setattr(FakeHelper, "mode", mode)
    monkeypatch.setattr(run_passes.subprocess, "Popen", FakeHelper)
    g = LaptopGuard(tmp_path / "power.log", allow_battery=False, seconds=7200)
    g.mark("pass1 start ac=1 charge=97% saver=0")
    return g


def test_the_helper_is_started_with_the_window_and_log(tmp_path, monkeypatch):
    g = real_guard(tmp_path, monkeypatch, "writes")
    g.start_helper(wait_s=5)
    assert g.proc.cmd[g.proc.cmd.index("-Seconds") + 1] == "7200"
    assert g.proc.cmd[-1] == str(tmp_path / "power.log")


def test_a_helper_that_exits_at_once_stops_the_run(tmp_path, monkeypatch):
    g = real_guard(tmp_path, monkeypatch, "exits")
    with pytest.raises(SystemExit, match="keep_awake.ps1 exited"):
        g.start_helper(wait_s=5)


def test_a_silent_helper_only_warns(tmp_path, monkeypatch, capsys):
    g = real_guard(tmp_path, monkeypatch, "silent")
    g.start_helper(wait_s=0.3)
    assert "has not written" in capsys.readouterr().err
