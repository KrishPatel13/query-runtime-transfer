#!/usr/bin/env bash
# Measure M5 on the laptop: M1 with PostgreSQL's default buffer pool
# (shared_buffers = 128MB instead of M1's 2GB), TPC-H and then TPC-DS.
#
# shared_buffers does not enter the planner's cost model, so the plans stay
# M1's (checked here with scripts/explain_all.py against M1's plans before
# any pass); the tables no longer fit in PostgreSQL's own buffers and every
# miss is read through the operating system's page cache instead. The
# setting is given on the pg_ctl command line (-o "-c shared_buffers=128MB"),
# so postgresql.conf stays as it is, and every pass records the server's
# settings in run.json.
#
# Waits until no run_passes.py and no node_timing_chain.sh of this user runs.
# At the end the TPC-H cluster runs as M1 again and TPC-DS is stopped.
#
# Usage (in WSL, from project_proposal):
#   bash scripts/run_m5_chain.sh > ~/cs680-pilot/run_M5_chain.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."
SB="${SB:-128MB}"
H_ENV=~/cs680-pilot/pg.env; DS_ENV=~/cs680-pilot/pg-tpcds.env
H_LOG=~/cs680-pilot/postgres.log; DS_LOG=~/cs680-pilot/postgres-tpcds.log
PLANS=~/cs680-pilot

echo "$(date -Is) waiting for other measurements to finish"
while pgrep -u "$(id -u)" -f "[r]un_passes.py|[n]ode_timing_chain.sh" > /dev/null; do
  sleep 60
done
echo "$(date -Is) starting M5 (shared_buffers=$SB)"

stop_cluster() {
  # shellcheck disable=SC1090
  (source "$1" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}

measure() {   # env, log, machine, query dir, database URL, M1 plan file
  local env="$1" log="$2" machine="$3" qdir="$4" url="$5" m1plans="$6"
  stop_cluster "$H_ENV"; stop_cluster "$DS_ENV"
  if [ ! -f "$m1plans" ]; then
    # shellcheck disable=SC1090
    (source "$env" && pg_ctl start -w -l "$log" && python3 scripts/explain_all.py --query-dir "$qdir" --out "$m1plans")
    stop_cluster "$env"
  fi
  # shellcheck disable=SC1090
  (source "$env" && pg_ctl start -w -l "$log" -o "-c shared_buffers=$SB" \
     && psql -XAtc "show shared_buffers" \
     && python3 scripts/explain_all.py --query-dir "$qdir" --out "$PLANS/plans_$machine.json" \
     && python3 scripts/explain_all.py --compare "$m1plans" "$PLANS/plans_$machine.json")
  # shellcheck disable=SC1090
  (source "$env" && python3 src/run_passes.py --machine "$machine" --query-dir "$qdir" --database-url "$url")
  stop_cluster "$env"
}

measure "$H_ENV" "$H_LOG" M5 queries/tpch_sf1_v2 "postgresql:///tpch" "$PLANS/plans_M1.json"
measure "$DS_ENV" "$DS_LOG" M5_tpcds queries/tpcds_sf1_kept "postgresql:///tpcds?host=/tmp&port=5433" \
  "$PLANS/plans_M1_tpcds.json"
# shellcheck disable=SC1090
(source "$H_ENV" && pg_ctl start -w -l "$H_LOG")
echo "$(date -Is) M5 done"
