# Hardware of the environments

Read on 2026-10-08 at about 22:20 with `uname -r`, `systemd-detect-virt`,
`lscpu` and `/sys/devices/system/cpu` (and on the laptop's host with
`Get-CimInstance Win32_Processor`). The machines had not changed since the
passes; kernel versions may have been updated by the server's maintainers in
the meantime.

## Server (M3, M4, M6; the SF10 and JOB server environments)

- Virtual machine: KVM, full virtualization, clock source `kvm-clock`.
- Kernel 6.8.0-117-generic (Ubuntu 24.04).
- 64 virtual processors of an AMD EPYC 9754 128-Core Processor, presented as
  64 sockets of one core with one thread each; one NUMA node; the guest sees
  1 GiB of L3 in 64 instances.
- CPUs 62 and 63 (M6; M4 uses 63) have no SMT siblings in the guest's view
  (each is its own package).
- No cpufreq interface in the guest, so frequency scaling and turbo are the
  host's and not visible.

## Laptop (M1, M2, M5)

- Host: Intel Core Ultra 7 155H, 16 cores (6 performance, 8 efficiency, 2
  low-power efficiency) and 22 logical processors, Windows 11 Education
  10.0.26300.
- Guest: WSL2 (Hyper-V), kernel 6.18.33.2-microsoft-standard-WSL2, 22
  virtual processors shown as 11 cores of 2 threads; virtual CPU 19 (M2)
  pairs with 18 in that view. Windows schedules the virtual processors on
  any host core, so pinning in the guest fixes a virtual processor, not a
  core type.
- No cpufreq interface in the guest. The clock source after a WSL restart
  is `tsc`; before every laptop pass `scripts/wsl_clock.py` selected
  `hyperv_clocksource_tsc_page` (STUDY.md, "Clock"), and each pass records
  its clock rate in `run.json`.
