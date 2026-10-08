"""Measure a query set several times on one machine and check every pass.

In the full study every query gets three passes on every machine. Pass i runs
src/collect.py over the whole query directory in its own seeded random order
(seed i - 1, so every machine runs the same orders), and src/make_labels.py
later takes the median of the pass labels. The measurement itself is
collect.py's: EXPLAIN, one warm-up, three timed EXPLAIN ANALYZE runs with
TIMING OFF, and the median of the three.

Layout, for --machine M1:
  results/runs/M1/pass<i>/            runtime_dataset.csv, runtime_dataset.meta.json
                                      and plans/ from collect.py, run.json from
                                      this script (seed, attempt, power, check)
  results/runs/M1/rejected/pass<i>_attempt<a>/   a pass that was measured again
  results/runs/M1/passes.json         summary of all passes and attempts
  results/runs/M1/power.log           laptop only, see below

Level shifts. A pass can change level partway through: in the pilot every
query after the 250th of pass 2 ran 1.7 times slower, most likely because
Windows energy saver switched on. From the second pass on, each pass is
compared with pass 1 by the level-shift test of scripts/compare_passes.py
(passcheck.check_pass): the log ratio of its labels to pass 1's labels, in
its own execution order, is split where the squared error around the two
segment means is smallest, with at least 30 queries on each side, and it is a
shift when the medians of the two segments differ by at least 10%. The same
test in pass 1's order finds a shift inside pass 1; when both orders show
one, the order whose split explains more of the squared error decides which
pass shifted. A pass with a shift, pass 1 included, is moved to
rejected/pass<i>_attempt1 and measured once more with the same seed, and
after a new pass 1 every pass is checked against it again. If the second
attempt shifts too, it is kept and the shift stays on record in its
run.json and in passes.json. To measure a pass again by hand, move its
directory into rejected/ (named pass<i>_attempt<a>, after the last attempt
there) and run this script again.

The laptop (WSL with powershell.exe). Before each pass the power state is
read with scripts/keep_awake.ps1 -Status, the check run_collection.sh uses,
and the run stops when the laptop is on battery or Windows energy saver is
on, unless --allow-battery is given. From the first pass to the end of the
run keep_awake.ps1 keeps Windows awake and appends the power source, charge
and energy saver state to power.log every 30 s; the run stops if the helper
exits before its first line. Its window bounds it in case this script is
killed: by default 4 hours per pass, for the K passes and one possible repeat
of each pass after the first, so 20 hours for three passes (the pilot's
passes of 420 queries took 9 and 11 minutes on M1). Each pass's run.json
summarizes the power lines written while it ran. A pass that ran partly on
battery or with energy saver on, or whose power log has a gap of more than
GAP_S seconds (the laptop slept: with Modern Standby, closing the lid
suspends it whatever keep_awake.ps1 asks for), is moved to rejected/ and
measured once more, as a pass with a level shift is.

Clock. EXPLAIN ANALYZE times queries with the kernel's monotonic clock. In
WSL on the laptop it ran 1.2% to 2% fast against Windows until
scripts/wsl_clock.py stopped systemd-timesyncd and selected the
hyperv_clocksource_tsc_page clock source; after that it ran about 0.5% fast.
On the laptop the run stops unless wsl_clock.py's state is ok (run it as
root after every WSL restart). Every pass records in run.json its clock
source and its average clock rate: the monotonic time it took over the wall
time it took, where the wall clock is kept on the host's (or, on the server,
NTP's) time. A rate of 1.005 means the pass's runtimes read 0.5% long.

Continuing. collect.py records in the meta file whether a pass finished, so
running the same command again measures only what is missing and resumes an
interrupted pass with collect.py --resume. All passes of one machine must
come from one host and one query directory: the script refuses to go on when
the host name, CPU model or CPU count differs from a pass already measured
(the UW server's shared host name points to two different machines).

Needs only the standard library (the laptop's WSL and the UW server have no
numpy or pandas), and psql on PATH (source pg.env first).

Usage:
  python3 src/run_passes.py --machine M1 --query-dir queries/tpch_sf1_v2
  python3 src/run_passes.py --machine M1 --limit 5 --out-root /tmp/runs --allow-battery   # smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from collect import host_info
from passcheck import check_pass, read_meta, read_pass

HERE = Path(__file__).resolve().parents[1]
CSV = "runtime_dataset.csv"
HOURS_PER_PASS = 4
HOST_KEYS = ("hostname", "cpu_model", "cpu_count")
ATTEMPT_RE = re.compile(r"pass(\d+)_attempt(\d+)")
POWER_RE = re.compile(r"ac=(\d+) charge=(\d+)% saver=(\d+)")
CPU_RE = re.compile(r"cpu=(\d+)%")
GAP_S = 90           # keep_awake.ps1 logs every 30 s; a longer gap means the laptop slept
CLOCKSOURCE = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
LAPTOP_CLOCK = "hyperv_clocksource_tsc_page"
MIN_RATE_S = 60      # a clock rate needs at least this much wall time to mean anything


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def clocksource() -> str | None:
    """The kernel's current clock source, or None where it cannot be read."""
    try:
        return CLOCKSOURCE.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def laptop_clock() -> dict:
    """scripts/wsl_clock.py's state(): clock source, tick, timesyncd and ok."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsl_clock", HERE / "scripts" / "wsl_clock.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.state()


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def finished(d: Path) -> bool:
    meta = read_meta(d)
    return (d / CSV).exists() and bool(meta) and meta[-1].get("finished_at") is not None


def identity(d: Path) -> str:
    """Changes whenever collect.py writes to the pass again."""
    meta = read_meta(d)
    return f"{meta[0].get('started_at')}/{meta[-1].get('finished_at')}"


def rejected_attempts(machine_dir: Path, i: int) -> list[int]:
    """Attempt numbers of pass i that were moved to rejected/."""
    r = machine_dir / "rejected"
    if not r.is_dir():
        return []
    return sorted(int(m.group(2)) for p in r.iterdir()
                  if (m := ATTEMPT_RE.fullmatch(p.name)) and int(m.group(1)) == i)


def run_collect(pass_dir: Path, seed: int, resume: bool, args) -> None:
    """One pass of collect.py into pass_dir."""
    cmd = [sys.executable, str(HERE / "src" / "collect.py"), "--query-dir", str(args.query_dir),
           "--out", str(pass_dir / CSV), "--seed", str(seed), "--machine", args.machine]
    if resume:
        cmd.append("--resume")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.timeout_s:
        cmd += ["--timeout-s", str(args.timeout_s)]
    if args.database_url:
        cmd += ["--database-url", args.database_url]
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------- laptop

def on_laptop() -> bool:
    """WSL with Windows interop, where the power state can be read."""
    return shutil.which("powershell.exe") is not None and shutil.which("wslpath") is not None


def windows_path(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def power_summary(lines: list[str]) -> dict:
    """Summary of keep_awake.ps1's 30 s lines (not the start and end marks).

    max_gap_s is the longest time between two consecutive lines, the marks
    included, and gaps counts the gaps longer than GAP_S: the helper writes
    every 30 s, so a longer gap means Windows was asleep or the helper had
    stopped.
    """
    states, times, cpu = [], [], []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            times.append(dt.datetime.fromisoformat(parts[0].lstrip("\ufeff")))
        except ValueError:
            pass
        if len(parts) >= 2 and parts[1].startswith("ac=") and (m := POWER_RE.search(line)):
            states.append(tuple(int(g) for g in m.groups()))
            if c := CPU_RE.search(line):
                cpu.append(int(c.group(1)))
    steps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    return {"lines": len(states), "on_battery": sum(ac == 0 for ac, _, _ in states),
            "saver_on": sum(saver == 1 for _, _, saver in states),
            "charge_min": min((c for _, c, _ in states), default=None),
            "max_gap_s": round(max(steps), 1) if steps else None,
            "gaps": sum(x > GAP_S for x in steps),
            "host_cpu": {"median": statistics.median(cpu), "max": max(cpu)} if cpu else None}


def disruption(power: dict | None) -> str | None:
    """Why a pass cannot be trusted, from its power summary, or None."""
    if not power:
        return None
    why = []
    if power.get("on_battery"):
        why.append(f"{power['on_battery']} power lines on battery")
    if power.get("saver_on"):
        why.append(f"{power['saver_on']} power lines with energy saver on")
    if power.get("gaps"):
        why.append(f"{power['gaps']} gaps in the power log (longest {power['max_gap_s']:.0f} s): the laptop slept")
    return "; ".join(why) or None


class NoGuard:
    """Off the laptop there is no power state to check."""

    def before(self, label: str) -> None:
        pass

    def after(self, label: str) -> dict | None:
        return None

    def close(self) -> None:
        pass


class LaptopGuard:
    """Power checks and the keep-awake helper for a run in WSL on the laptop."""

    def __init__(self, log: Path, allow_battery: bool, seconds: int, allow_clock: bool = False):
        self.log, self.allow_battery, self.seconds = log, allow_battery, seconds
        self.allow_clock = allow_clock
        self.started = ""
        self.proc: subprocess.Popen | None = None
        self.offset = 0

    def _script(self) -> str:
        return windows_path(HERE / "scripts" / "keep_awake.ps1")

    def status(self) -> str:
        """keep_awake.ps1 -Status: 'ac=1 charge=85% saver=0', or 'unknown'."""
        try:
            p = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                self._script(), "-Status"], capture_output=True, text=True, timeout=120,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        out = p.stdout.strip()
        return out if p.returncode == 0 and POWER_RE.fullmatch(out) else "unknown"

    def mark(self, text: str) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"{now()} {text}\n")

    def start_helper(self, wait_s: float = 60) -> None:
        """Start keep_awake.ps1 and wait for its first power line.

        The helper writes a line as soon as its keep-awake request is in
        place, so a line in the log shows that it works. If it exits first,
        nothing keeps Windows awake and the run stops, since a laptop that
        sleeps in the middle of a pass stalls it (the pilot's first attempt).
        """
        size = self.log.stat().st_size
        cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", self._script(),
               "-Seconds", str(self.seconds), "-LogPath", windows_path(self.log)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.log.stat().st_size > size:
                return
            if self.proc.poll() is not None:
                raise SystemExit(f"keep_awake.ps1 exited with status {self.proc.returncode} before writing to "
                                 f"{self.log}; run it by hand to see why")
            time.sleep(0.5)
        print(f"warning: keep_awake.ps1 is running but has not written to {self.log} after {wait_s:.0f} s",
              file=sys.stderr, flush=True)

    def before(self, label: str) -> None:
        clock = laptop_clock()
        if not clock["ok"] and not self.allow_clock:
            raise SystemExit(f"the WSL clock is not set up ({clock}); run scripts/wsl_clock.py as root "
                             "(wsl.exe -u root) or pass --allow-clock for a smoke test")
        state = self.status()
        self.mark(f"{label} start {state}")
        m = POWER_RE.fullmatch(state)
        if not self.allow_battery:
            if m is None:
                raise SystemExit("could not read the power state; pass --allow-battery to measure anyway")
            if m.group(1) != "1":
                raise SystemExit(f"on battery ({state}); plug in the laptop, or pass --allow-battery for a smoke test")
            if m.group(3) == "1":
                raise SystemExit(f"Windows energy saver is on ({state}); turn it off, or pass --allow-battery")
        # The helper's lines from here on belong to this pass.
        self.offset = self.log.stat().st_size
        self.started = f"{now()} {label} start"
        if self.proc is None or self.proc.poll() is not None:
            self.start_helper()

    def after(self, label: str) -> dict:
        with open(self.log, encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            lines = f.read().splitlines()
        state = self.status()
        self.mark(f"{label} end {state}")
        # The start and end marks bound the pass, so a sleep before the
        # helper's first line or after its last one still shows as a gap.
        return {"ended_at": now(), **power_summary([self.started] + lines + [f"{now()} {label} end"])}

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ------------------------------------------------------------------ passes

def check_inputs(machine_dir: Path, query_dir: Path, host: dict) -> None:
    """Refuse to mix query directories or hosts within one machine's passes."""
    dirs = [d for d in machine_dir.glob("pass*") if d.is_dir()]
    for d in sorted(dirs):
        meta = read_meta(d)
        if not meta:
            continue
        recorded = (meta[0].get("args") or {}).get("query_dir")
        if recorded and Path(recorded).resolve() != query_dir:
            raise SystemExit(f"{d} was measured from {recorded}, not {query_dir}")
        old = meta[0].get("host") or {}
        diff = {k: (old[k], host.get(k)) for k in HOST_KEYS if k in old and old[k] != host.get(k)}
        if diff:
            raise SystemExit(f"{d} was measured on another host: " +
                             ", ".join(f"{k} {a!r} there, {b!r} here" for k, (a, b) in diff.items()))


def describe(d: Path, machine_dir: Path) -> dict:
    """One pass directory for passes.json."""
    rec = read_json(d / "run.json", {})
    meta = read_meta(d)
    rows = read_pass(d) if (d / CSV).exists() else {}
    loads = [float(r["load1"]) for r in rows.values() if r.get("load1")]
    return {"dir": d.relative_to(machine_dir).as_posix(), "pass": rec.get("pass"), "seed": rec.get("seed"),
            "attempt": rec.get("attempt"), "finished": finished(d),
            "started_at": meta[0].get("started_at") if meta else None,
            "finished_at": meta[-1].get("finished_at") if meta else None,
            "rows": len(rows), "status_counts": dict(Counter(r["status"] for r in rows.values())),
            "load1": {"median": statistics.median(loads), "max": max(loads)} if loads else None,
            "power": rec.get("power", []), "check": rec.get("check"), "rejected": rec.get("rejected")}


def write_summary(machine_dir: Path, args, host: dict) -> dict:
    passes = sorted((d for d in machine_dir.glob("pass*") if d.is_dir() and re.fullmatch(r"pass\d+", d.name)),
                    key=lambda d: int(d.name[4:]))
    rejected = sorted(p for p in (machine_dir / "rejected").glob("pass*_attempt*") if p.is_dir()) \
        if (machine_dir / "rejected").is_dir() else []
    summary = {"machine": args.machine, "query_dir": str(args.query_dir), "passes_planned": args.passes,
               "updated_at": now(), "host": host,
               "passes": [describe(d, machine_dir) for d in passes],
               "rejected": [describe(d, machine_dir) for d in rejected]}
    if machine_dir.is_dir():
        write_json(machine_dir / "passes.json", summary)
    return summary


def report(chk: dict) -> None:
    line = (f"{chk['pass']} vs {chk['reference']}: {chk['n_compared']} queries, same plan {chk['same_plan']}, "
            f"median Q {chk['median_q']:.3f}, P90 Q {chk['p90_q']:.3f}" if chk["n_compared"] else
            f"{chk['pass']} vs {chk['reference']}: no query with status ok in both")
    s, r = chk["shift"], chk.get("shift_in_reference_order")
    if s:
        line += f"; {chk['pass']} shifts by {s['level_change']:.2f}x after query {s['after_query_number']} of {s['of']}"
    elif r:
        line += (f"; {chk['reference']} shifts by {1 / r['level_change']:.2f}x after its query "
                 f"{r['after_query_number']} of {r['of']}")
    else:
        line += "; no level shift"
    print(line, file=sys.stderr, flush=True)


def shift_reason(s: dict, factor: float) -> str:
    return f"level shift of {factor:.2f}x after query {s['after_query_number']} of {s['of']} (at {s['at']})"


def reject(machine_dir: Path, i: int, reason: str) -> Path:
    """Move pass i to rejected/pass<i>_attempt<a>, with the reason in its
    run.json. Attempts are numbered after the last rejected one, so the
    target never exists."""
    d = machine_dir / f"pass{i}"
    rec = read_json(d / "run.json", {})
    rec["rejected"] = {"at": now(), "reason": reason}
    write_json(d / "run.json", rec)
    target = machine_dir / "rejected" / f"pass{i}_attempt{max(rejected_attempts(machine_dir, i), default=0) + 1}"
    target.parent.mkdir(exist_ok=True)
    d.rename(target)
    print(f"pass {i}: {reason}; moved to {target} and measuring it again", file=sys.stderr, flush=True)
    return target


def measure_pass(i: int, machine_dir: Path, args, collector, guard) -> None:
    """Measure pass i if needed and check it against pass 1. A pass with a
    level shift is measured once more; so is pass 1 when the shift is inside
    it, after which pass i is checked against the new pass 1."""
    seed = i - 1
    d = machine_dir / f"pass{i}"
    while True:
        attempt = max(rejected_attempts(machine_dir, i), default=0) + 1
        rec = read_json(d / "run.json", {})
        rec.update({"pass": i, "seed": seed, "attempt": attempt, "machine": args.machine})
        if not finished(d):
            rec["clocksource"] = clocksource()
            mono0, wall0 = time.monotonic(), time.time()
            resume = (d / CSV).exists()
            print(f"pass {i} of {args.passes} (seed {seed}, attempt {attempt})" + (", resuming" if resume else ""),
                  file=sys.stderr, flush=True)
            guard.before(f"pass{i}")
            power = None
            try:
                collector(d, seed, resume, args)
            finally:
                wall = time.time() - wall0
                if wall >= MIN_RATE_S and d.is_dir():
                    rec.setdefault("clock_rate", []).append(
                        {"rate": round((time.monotonic() - mono0) / wall, 6), "wall_s": round(wall, 1)})
                power = guard.after(f"pass{i}")
                if power is not None and d.is_dir():
                    rec.setdefault("power", []).append(power)
                if d.is_dir():
                    write_json(d / "run.json", rec)
            why = disruption(power)
            if why:
                if attempt > 1:
                    print(f"warning: pass {i} was disrupted again ({why}); it is kept and the power log is "
                          f"summarized in {d / 'run.json'}", file=sys.stderr, flush=True)
                else:
                    reject(machine_dir, i, why)
                    continue
        if i == 1:
            write_json(d / "run.json", rec)
            return
        key = {"reference": identity(machine_dir / "pass1"), "pass": identity(d)}
        chk = rec.get("check")
        if not chk or chk.get("identity") != key:
            chk = check_pass(machine_dir / "pass1", d) | {"identity": key}
            rec["check"] = chk
        write_json(d / "run.json", rec)
        report(chk)
        s, r = chk["shift"], chk.get("shift_in_reference_order")
        if s is not None:
            if attempt > 1:
                print(f"warning: pass {i} shifts again after being measured {attempt} times; it is kept and "
                      f"the shift is recorded in {d / 'run.json'}", file=sys.stderr, flush=True)
                return
            reject(machine_dir, i, shift_reason(s, s["level_change"]))
        elif r is not None:
            # The ratio is new / pass 1, so pass 1's own change is the inverse.
            if max(rejected_attempts(machine_dir, 1), default=0) >= 1:
                print(f"warning: pass 1 shifts again after being measured twice; it is kept and the shift is "
                      f"recorded in {d / 'run.json'}", file=sys.stderr, flush=True)
                return
            reject(machine_dir, 1, shift_reason(r, 1 / r["level_change"]) + f", found against pass {i}")
            measure_pass(1, machine_dir, args, collector, guard)
        else:
            return


def run(args, collector=run_collect, guard=None, host: dict | None = None) -> dict:
    """All passes of one machine; returns the summary written to passes.json."""
    args.query_dir = Path(args.query_dir).resolve()
    machine_dir = Path(args.out_root) / args.machine
    host = host_info() if host is None else host
    check_inputs(machine_dir, args.query_dir, host)
    if guard is None:
        hours = args.keep_awake_hours or HOURS_PER_PASS * (2 * args.passes - 1)
        guard = LaptopGuard(machine_dir / "power.log", args.allow_battery, int(hours * 3600),
                            getattr(args, "allow_clock", False)) if on_laptop() else NoGuard()
    try:
        while True:
            replaced = len(rejected_attempts(machine_dir, 1))
            for i in range(1, args.passes + 1):
                measure_pass(i, machine_dir, args, collector, guard)
            # If pass 1 was measured again on the way, the passes checked
            # against the old pass 1 are checked once more (finished passes
            # are not measured again unless the new check finds a shift).
            if len(rejected_attempts(machine_dir, 1)) == replaced:
                break
    finally:
        guard.close()
        summary = write_summary(machine_dir, args, host)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, help="machine name, e.g. M1; also the output directory name")
    ap.add_argument("--query-dir", default=str(HERE / "queries" / "tpch_sf1_v2"))
    ap.add_argument("--passes", type=int, default=3, help="number of passes; pass i uses seed i - 1")
    ap.add_argument("--out-root", default=str(HERE / "results" / "runs"))
    ap.add_argument("--allow-battery", action="store_true",
                    help="on the laptop, measure even on battery or with energy saver on (smoke tests only)")
    ap.add_argument("--allow-clock", action="store_true",
                    help="on the laptop, measure even when scripts/wsl_clock.py has not set up the clock "
                         "(smoke tests only)")
    ap.add_argument("--keep-awake-hours", type=float, default=None,
                    help=f"keep-awake window (default {HOURS_PER_PASS} h per pass, counting one repeat of each "
                         "pass after the first)")
    ap.add_argument("--limit", type=int, default=None, help="collect.py --limit: first N queries only (smoke tests)")
    ap.add_argument("--timeout-s", type=int, default=None, help="collect.py --timeout-s (default 300)")
    ap.add_argument("--database-url", default=None, help="collect.py --database-url")
    args = ap.parse_args(argv)
    if args.passes < 1:
        ap.error("--passes must be at least 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.machine):
        ap.error("--machine must be letters, digits, _ and - only (it names a directory)")
    return args


def main() -> None:
    args = parse_args()
    if shutil.which("psql") is None:
        raise SystemExit("psql is not on PATH; source pg.env first (e.g. source ~/cs680-pilot/pg.env)")
    if not list(Path(args.query_dir).glob("*.sql")):
        raise SystemExit(f"no .sql files in {args.query_dir}")

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)
    try:
        summary = run(args)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"collect.py stopped with status {e.returncode}; run the same command again to continue")
    done = [p for p in summary["passes"] if p["finished"]]
    print(f"{len(done)} of {args.passes} passes finished in {Path(args.out_root) / args.machine}", file=sys.stderr)


if __name__ == "__main__":
    main()
