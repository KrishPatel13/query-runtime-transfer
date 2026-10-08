#!/usr/bin/env bash
# Generate the query workload (once) and collect plans and runtimes for it.
#
# One pass, as in the pilot. Extra arguments are passed to src/collect.py,
# e.g. --resume, --overwrite, --seed 1, or --limit 5.
#
# Several passes on one machine, as in the full study: with MACHINE set, the
# extra arguments go to src/run_passes.py instead, which measures the passes
# into results/runs/$MACHINE, checks each against pass 1, and on the laptop
# checks the power state and keeps Windows awake for the whole run itself.
# QUERY_DIR then defaults to queries/tpch_sf1_v2, which must already exist.
#
# Usage (after setup_postgres.sh and load_tpch.sh):
#   bash scripts/run_collection.sh                     # one pass into results/pilot
#   MACHINE=M1 bash scripts/run_collection.sh          # three passes into results/runs/M1
#   PILOT_HOME=/var/tmp/pgstudy MACHINE=M3 bash scripts/run_collection.sh
set -euo pipefail

PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATES="${TEMPLATES:-1-22}"
PER_TEMPLATE="${PER_TEMPLATE:-20}"
MACHINE="${MACHINE:-}"
if [ -n "$MACHINE" ]; then
  QUERY_DIR="$(realpath -m "${QUERY_DIR:-$HERE/queries/tpch_sf1_v2}")"
else
  QUERY_DIR="$(realpath -m "${QUERY_DIR:-$HERE/queries/tpch_sf1}")"
fi
OUT="$(realpath -m "${OUT:-$HERE/results/pilot/runtime_dataset.csv}")"

# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"
pg_ctl status > /dev/null 2>&1 || pg_ctl -l "$PILOT_HOME/postgres.log" -w start

KIT="$PILOT_HOME/tpch-kit/dbgen"
if [ -f "$QUERY_DIR/manifest.json" ]; then
  echo "using the existing workload in $QUERY_DIR (TEMPLATES and PER_TEMPLATE are ignored)"
elif [ -n "$MACHINE" ]; then
  # The full study's workload is generated once and measured on every
  # machine, so a missing one is not generated here.
  echo "no workload in $QUERY_DIR (manifest.json is missing); generate it first" >&2
  exit 1
else
  python3 "$HERE/scripts/generate_qgen_queries.py" --qgen "$KIT/qgen" \
    --query-template-dir "$KIT/queries" --templates "$TEMPLATES" \
    --per-template "$PER_TEMPLATE" --out "$QUERY_DIR"
fi

if [ -n "$MACHINE" ]; then
  cd "$HERE/src"
  exec python3 run_passes.py --machine "$MACHINE" --query-dir "$QUERY_DIR" "$@"
fi

# On a laptop the power source changes CPU speed, so log it (WSL only):
# ac=1 on AC power, ac=0 on battery, the charge, and saver=1 when Windows
# energy saver is on. keep_awake.ps1 -Status reads it; src/run_passes.py uses
# the same check.
power() {
  if command -v powershell.exe > /dev/null && command -v wslpath > /dev/null; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$(wslpath -w "$HERE/scripts/keep_awake.ps1")" -Status 2> /dev/null | tr -d '\r' \
      || echo unknown
  else
    echo unknown
  fi
}

mkdir -p "$(dirname "$OUT")"
log="${OUT%.csv}.power.log"
status="$(power)"
echo "$(date -Iseconds) start $status" >> "$log"
case "$status" in
  *ac=0*) echo "warning: on battery; Windows energy saver (at 20% charge) will slow the queries" >&2 ;;
  *saver=1*) echo "warning: Windows energy saver is on and will slow the queries" >&2 ;;
esac

# On battery, Windows sleeps after a few idle minutes even while a query is
# running, which stalls the measurement and changes CPU speed afterwards. Ask
# Windows to stay awake for the length of the run (it expires after 4 hours in
# case this script is killed before it can cancel the request). The helper
# also logs the power state every 30 s to the power log.
keep_awake_pid=""
if command -v powershell.exe > /dev/null && command -v wslpath > /dev/null; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$(wslpath -w "$HERE/scripts/keep_awake.ps1")" -Seconds 14400 \
    -LogPath "$(wslpath -w "$log")" > /dev/null 2>&1 &
  keep_awake_pid=$!
  trap '[ -n "$keep_awake_pid" ] && kill "$keep_awake_pid" 2> /dev/null || true' EXIT
fi

cd "$HERE/src"
python3 collect.py --query-dir "$QUERY_DIR" --out "$OUT" "$@"

echo "$(date -Iseconds) end $(power)" >> "$log"
