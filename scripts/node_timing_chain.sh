#!/usr/bin/env bash
# Per-operator timings (scripts/node_timing.py) on every machine of one host,
# after that host's label passes have finished.
#
# HOST=laptop (in WSL): M2_tpcds, M2, M1, M1_tpcds into results/runs/<machine>/,
#   then the TPC-H cluster is left running unpinned and TPC-DS stopped, as before.
# HOST=server (UW server): M4, M4_tpcds, M3, M3_tpcds into $BASE/runs/<machine>/,
#   every cluster stopped at the end.
# Each machine is a cluster started pinned to one processor (start_pinned.sh) or
# unpinned. The script first waits until no run_passes.py and no
# run_m4_chain.sh of this user is running.
#
# Usage: HOST=laptop setsid nohup bash scripts/node_timing_chain.sh > <log> 2>&1 < /dev/null &
set -euo pipefail

HOST="${HOST:?set HOST to laptop or server}"
cd "$(dirname "$0")/.."
HERE="$(pwd)"

echo "$(date -Is) waiting for label passes to finish"
while pgrep -u "$(id -u)" -f "[r]un_passes.py|[r]un_m4_chain.sh" > /dev/null; do
  sleep 60
done
echo "$(date -Is) starting"

stop_cluster() {
  # shellcheck disable=SC1090
  (source "$1" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}
start_cluster() {   # env file, cpu ("" for unpinned), log file, test query
  local env="$1" cpu="$2" log="$3" test_query="$4"
  if [ -n "$cpu" ]; then
    PG_ENV="$env" CPU="$cpu" OUT="$(dirname "$log")/affinity-node-timing.txt" LOG_FILE="$log" \
      TEST_QUERY="$test_query" bash scripts/start_pinned.sh
  else
    # shellcheck disable=SC1090
    (source "$env" && pg_ctl start -w -l "$log")
  fi
}
timing() {   # env file, machine, query dir, database URL, out root
  local env="$1" machine="$2" qdir="$3" url="$4" root="$5"
  mkdir -p "$root/$machine"
  # shellcheck disable=SC1090
  (source "$env" && python3 scripts/node_timing.py --query-dir "$qdir" --database-url "$url" \
     --out "$root/$machine/node_times.jsonl" | tail -1)
}

H_Q="select count(*) from lineitem where l_quantity > 0"
DS_Q="select count(*) from store_sales where ss_quantity > 0"
if [ "$HOST" = laptop ]; then
  H_ENV=~/cs680-pilot/pg.env; DS_ENV=~/cs680-pilot/pg-tpcds.env; ROOT="$HERE/results/runs"; CPU=19
  H_LOG=~/cs680-pilot/postgres.log; DS_LOG=~/cs680-pilot/postgres-tpcds.log
  DS_URL="postgresql:///tpcds?host=/tmp&port=5433"; H_URL="postgresql:///tpch"
  # The pinned TPC-DS cluster of the M2_tpcds passes may still run.
  stop_cluster "$H_ENV"; stop_cluster "$DS_ENV"
  start_cluster "$DS_ENV" "$CPU" "$DS_LOG" "$DS_Q"; timing "$DS_ENV" M2_tpcds queries/tpcds_sf1_kept "$DS_URL" "$ROOT"; stop_cluster "$DS_ENV"
  start_cluster "$H_ENV" "$CPU" "$H_LOG" "$H_Q";    timing "$H_ENV" M2 queries/tpch_sf1_v2 "$H_URL" "$ROOT";             stop_cluster "$H_ENV"
  start_cluster "$H_ENV" "" "$H_LOG" "$H_Q";        timing "$H_ENV" M1 queries/tpch_sf1_v2 "$H_URL" "$ROOT";             stop_cluster "$H_ENV"
  start_cluster "$DS_ENV" "" "$DS_LOG" "$DS_Q";     timing "$DS_ENV" M1_tpcds queries/tpcds_sf1_kept "$DS_URL" "$ROOT"; stop_cluster "$DS_ENV"
  start_cluster "$H_ENV" "" "$H_LOG" "$H_Q"
elif [ "$HOST" = server ]; then
  BASE="${BASE:-/var/tmp/pgstudy}"
  H_ENV="$BASE/pg.env"; DS_ENV="$BASE/tpcds/pg.env"; ROOT="$BASE/runs"; CPU="${CPU:-63}"
  H_LOG="$BASE/postgres.log"; DS_LOG="$BASE/tpcds/postgres.log"
  H_URL="postgresql:///tpch"; DS_URL="postgresql:///tpcds"
  stop_cluster "$H_ENV"; stop_cluster "$DS_ENV"
  start_cluster "$H_ENV" "$CPU" "$H_LOG" "$H_Q";    timing "$H_ENV" M4 queries/tpch_sf1_v2 "$H_URL" "$ROOT";             stop_cluster "$H_ENV"
  start_cluster "$DS_ENV" "$CPU" "$DS_LOG" "$DS_Q"; timing "$DS_ENV" M4_tpcds queries/tpcds_sf1_kept "$DS_URL" "$ROOT"; stop_cluster "$DS_ENV"
  start_cluster "$H_ENV" "" "$H_LOG" "$H_Q";        timing "$H_ENV" M3 queries/tpch_sf1_v2 "$H_URL" "$ROOT";             stop_cluster "$H_ENV"
  start_cluster "$DS_ENV" "" "$DS_LOG" "$DS_Q";     timing "$DS_ENV" M3_tpcds queries/tpcds_sf1_kept "$DS_URL" "$ROOT"; stop_cluster "$DS_ENV"
else
  echo "HOST must be laptop or server" >&2
  exit 1
fi
echo "$(date -Is) node timing done"
