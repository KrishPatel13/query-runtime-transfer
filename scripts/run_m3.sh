#!/usr/bin/env bash
# Run the label passes on the UW server (M3) at a quiet hour.
#
# The server is shared, so the passes start after START (default 01:30, local
# time) once the 5-minute load average is below MAX_LOAD (default 10, of 64
# cores), and at the latest WAIT_MAX_S seconds after START. With START empty
# the load check begins at once. collect.py logs the 1-minute load with every
# query.
#
# Other workloads and server machines use the same script:
#   MACHINE      output directory name (default M3)
#   QUERY_DIR    query directory (default queries/tpch_sf1_v2)
#   PG_ENV       environment file of the cluster (default $PILOT_HOME/pg.env)
#   DATABASE_URL collect.py's database URL (default postgresql:///tpch)
# TPC-DS on M3, for example:
#   MACHINE=M3_tpcds QUERY_DIR=queries/tpcds_sf1_kept PG_ENV=/var/tmp/pgstudy/tpcds/pg.env \
#     DATABASE_URL=postgresql:///tpcds START= bash scripts/run_m3.sh
#
# Usage on the server, detached so that it survives logging out:
#   setsid nohup bash scripts/run_m3.sh > /var/tmp/pgstudy/run_M3.log 2>&1 < /dev/null &
set -euo pipefail

PILOT_HOME="${PILOT_HOME:-/var/tmp/pgstudy}"
START="${START-01:30}"
MAX_LOAD="${MAX_LOAD:-10}"
WAIT_MAX_S="${WAIT_MAX_S:-7200}"
MACHINE="${MACHINE:-M3}"
QUERY_DIR="${QUERY_DIR:-queries/tpch_sf1_v2}"
PG_ENV="${PG_ENV:-$PILOT_HOME/pg.env}"
DATABASE_URL="${DATABASE_URL:-postgresql:///tpch}"
cd "$(dirname "$0")/.."

if [ -n "$START" ]; then
  target="$(date -d "$START" +%s)"
  if [ "$target" -le "$(date +%s)" ]; then
    target="$(date -d "tomorrow $START" +%s)"
  fi
  echo "$(date -Is) waiting until $(date -d "@$target" -Is)"
  sleep $(( target - $(date +%s) ))
else
  target="$(date +%s)"
fi
deadline=$(( target + WAIT_MAX_S ))
while true; do
  load5="$(cut -d' ' -f2 /proc/loadavg)"
  if awk -v l="$load5" -v m="$MAX_LOAD" 'BEGIN { exit !(l < m) }'; then
    echo "$(date -Is) load5 $load5 is below $MAX_LOAD; starting"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date -Is) load5 $load5 is still at least $MAX_LOAD; starting anyway at the deadline"
    break
  fi
  sleep 300
done

# shellcheck disable=SC1090
source "$PG_ENV"
python3 src/run_passes.py --machine "$MACHINE" --query-dir "$QUERY_DIR" --out-root "$PILOT_HOME/runs" \
  --database-url "$DATABASE_URL"
echo "$(date -Is) done"
