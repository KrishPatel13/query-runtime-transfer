# Keep Windows from going to sleep on idle while a measurement runs.
# A busy process does not count as user activity, so on battery Windows would
# otherwise sleep after its idle timeout in the middle of a collection run.
# The request lasts while this process lives; run_collection.sh and
# src/run_passes.py stop it at the end of a run, and -Seconds bounds it in
# case that never happens (run_passes.py sets it long enough for all passes).
#
# With -LogPath it also appends the power state every 30 s: AC (1 = plugged
# in), battery charge, and saver (1 = Windows energy saver on). On battery,
# energy saver switches on at 20% charge by default and slows every query, so
# the log shows where a run crossed that point.
#
# With -Status it prints the power state once, in the same form
# (ac=1 charge=85% saver=0), and exits. This is the power check that both
# run_collection.sh and src/run_passes.py use before measuring.
param([int]$Seconds = 14400, [string]$LogPath = "", [switch]$Status)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public struct PowerStatus {
    public byte AC; public byte BatteryFlag; public byte Charge; public byte Saver;
    public int LifeTime; public int FullLifeTime;
}
public static class KeepAwake {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
    [DllImport("kernel32.dll")] public static extern bool GetSystemPowerStatus(out PowerStatus status);
}
"@

function Get-PowerLine {
    $s = New-Object PowerStatus
    [void][KeepAwake]::GetSystemPowerStatus([ref]$s)
    "ac=$($s.AC) charge=$($s.Charge)% saver=$($s.Saver)"
}

if ($Status) {
    Write-Output (Get-PowerLine)
    exit 0
}

# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
$previous = [KeepAwake]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) {
    Write-Error "SetThreadExecutionState failed"
    exit 1
}
Write-Output "keep-awake on for up to $Seconds s"

$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
    if ($LogPath) {
        # Windows-side CPU load (percent, averaged over the processors): load1
        # inside WSL cannot see other Windows programs competing for the CPU.
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK') $(Get-PowerLine) cpu=$([int]$cpu)%"
        Add-Content -Path $LogPath -Value $line -Encoding utf8
    }
    Start-Sleep -Seconds 30
}
