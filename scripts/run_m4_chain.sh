#!/usr/bin/env bash
# Measure M4 (the UW server with PostgreSQL pinned to one processor) after
# the M3 TPC-DS passes, on the same server: TPC-H first, then TPC-DS.
#
# Waits until no run_passes.py of this user runs, stops the TPC-DS cluster,
# starts the TPC-H cluster pinned (scripts/start_pinned.sh), runs the three
# M4 passes with scripts/run_m3.sh, stops it, and does the same for TPC-DS
# (M4_tpcds). Each cluster is stopped again at the end, unpinned.
#
# Usage on the server, detached:
#   setsid nohup bash scripts/run_m4_chain.sh > /var/tmp/pgstudy/run_M4_chain.log 2>&1 < /dev/null &
set -euo pipefail

BASE="${BASE:-/var/tmp/pgstudy}"
CPU="${CPU:-63}"
MAX_LOAD="${MAX_LOAD:-16}"
cd "$(dirname "$0")/.."

echo "$(date -Is) waiting for other passes to finish"
while pgrep -u "$(id -u)" -f "[r]un_passes.py" > /dev/null; do
  sleep 60
done
echo "$(date -Is) no passes running"

stop_cluster() {
  # shellcheck disable=SC1090
  (source "$1" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}

measure() {   # env file, machine, query dir, database URL, test query
  local env="$1" machine="$2" qdir="$3" url="$4" test_query="$5"
  stop_cluster "$BASE/pg.env"
  stop_cluster "$BASE/tpcds/pg.env"
  PG_ENV="$env" CPU="$CPU" OUT="$BASE/runs/$machine/affinity.txt" TEST_QUERY="$test_query" \
    LOG_FILE="$(dirname "$env")/postgres-$machine.log" bash scripts/start_pinned.sh
  MACHINE="$machine" QUERY_DIR="$qdir" PG_ENV="$env" DATABASE_URL="$url" START= MAX_LOAD="$MAX_LOAD" \
    bash scripts/run_m3.sh
  stop_cluster "$env"
}

measure "$BASE/pg.env" M4 queries/tpch_sf1_v2 postgresql:///tpch \
  "select count(*) from lineitem where l_quantity > 0"
measure "$BASE/tpcds/pg.env" M4_tpcds queries/tpcds_sf1_kept postgresql:///tpcds \
  "select count(*) from store_sales where ss_quantity > 0"
echo "$(date -Is) chain done"
