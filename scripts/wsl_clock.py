"""Make the WSL guest's clock run at the host's rate before measuring.

EXPLAIN ANALYZE times queries with CLOCK_MONOTONIC. In WSL on the laptop it
ran 1.2% to 2% fast against Windows (whose clock agrees with NTP to 0.01%):
systemd-timesyncd in the guest syncs with ntp.ubuntu.com while Hyper-V's
time sync steps the guest's clock to the host's every 30 s or so, and
timesyncd answers the steps by speeding the kernel clock up (the adjtimex
tick was 10202 us instead of 10000). This script, run as root, stops
timesyncd (Hyper-V's time sync still keeps the wall clock right), resets the
tick and frequency, and selects the hyperv_clocksource_tsc_page clock source,
which follows the hypervisor's reference time. Everything lasts until WSL
restarts, so run it again after every restart (M2 needs one). The kernel
still slews the clock toward the host's time by changing the tick, so the
rate is not exact: measured against Windows over 2 minutes it ran 1.2% fast
before this script and 0.5% fast after it. src/run_passes.py records each
pass's average rate (monotonic time over wall time, which Hyper-V keeps on
the host's time) so labels can be rescaled.

Usage, from Windows:
  wsl.exe -d Ubuntu-24.04 -u root -e python3 scripts/wsl_clock.py           # set up
  wsl.exe -d Ubuntu-24.04 -e python3 scripts/wsl_clock.py --check           # show only
src/run_passes.py refuses to measure on the laptop unless state() is right.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import subprocess
import sys
from pathlib import Path

CLOCKSOURCE = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
WANT_CLOCK = "hyperv_clocksource_tsc_page"
NOMINAL_TICK = 10000      # microseconds per tick at USER_HZ = 100
ADJ_FREQUENCY, ADJ_TICK = 0x0002, 0x4000


class Timex(ctypes.Structure):
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
                ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
                ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
                ("time_sec", ctypes.c_long), ("time_usec", ctypes.c_long), ("tick", ctypes.c_long),
                ("ppsfreq", ctypes.c_long), ("jitter", ctypes.c_long), ("shift", ctypes.c_int),
                ("stabil", ctypes.c_long), ("jitcnt", ctypes.c_long), ("calcnt", ctypes.c_long),
                ("errcnt", ctypes.c_long), ("stbcnt", ctypes.c_long), ("tai", ctypes.c_int),
                ("pad", ctypes.c_int * 11)]


def adjtimex(modes: int = 0, tick: int = 0, freq: int = 0) -> Timex:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    tx = Timex(modes=modes, tick=tick, freq=freq)
    if libc.adjtimex(ctypes.byref(tx)) < 0:
        raise OSError(ctypes.get_errno(), "adjtimex failed (setting it needs root)")
    return tx


def timesyncd_active() -> bool | None:
    try:
        p = subprocess.run(["systemctl", "is-active", "systemd-timesyncd"], capture_output=True, text=True)
    except OSError:
        return None
    return p.stdout.strip() == "active"


def state() -> dict:
    """The clock settings that decide whether timings run at the host's rate."""
    try:
        source = CLOCKSOURCE.read_text(encoding="utf-8").strip()
    except OSError:
        source = None
    tx = adjtimex()
    s = {"clocksource": source, "tick_us": int(tx.tick), "freq_ppm": round(tx.freq / 65536, 2),
         "timesyncd_active": timesyncd_active()}
    # The tick is not part of the test: with timesyncd stopped the kernel
    # still moves it (9877 to 10202 us were seen) while it slews toward the
    # host's time, so run_passes.py records each pass's average clock rate.
    s["ok"] = source == WANT_CLOCK and not s["timesyncd_active"]
    return s


def set_up() -> dict:
    subprocess.run(["systemctl", "stop", "systemd-timesyncd"], check=False)
    adjtimex(ADJ_TICK | ADJ_FREQUENCY, tick=NOMINAL_TICK, freq=0)
    CLOCKSOURCE.write_text(WANT_CLOCK + "\n", encoding="utf-8")
    return state()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the clock state and change nothing")
    args = ap.parse_args()
    s = state() if args.check else set_up()
    print(json.dumps(s))
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
