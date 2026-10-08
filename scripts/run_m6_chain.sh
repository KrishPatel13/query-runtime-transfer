#!/usr/bin/env bash
# Measure M6 on the UW server: M3 with the PostgreSQL server pinned to two
# virtual processors (CPUS, default 62,63), TPC-H and then TPC-DS.
#
# M4 pins PostgreSQL to one processor, so a plan's leader and its two
# parallel workers share one; M6 is the moderate cut, three processes on
# two processors. As for M4, the postmaster starts under taskset, every
# process it forks inherits the affinity, the planner does not look at the
# processor count, and the client keeps all processors. The affinity of the
# postmaster, its children and a test query's parallel workers is written to
# $BASE/runs/<machine>/affinity.txt.
#
# (A first attempt ran on the laptop on 2 October and was stopped after 96
# queries, because other programs were using the laptop's processors.)
#
# Waits until no run_passes.py, run_m4_chain.sh or node_timing_chain.sh of
# this user runs; every cluster is stopped at the end.
#
# Usage on the server, detached:
#   setsid nohup bash scripts/run_m6_chain.sh > /var/tmp/pgstudy/run_M6_chain.log 2>&1 < /dev/null &
set -euo pipefail

BASE="${BASE:-/var/tmp/pgstudy}"
CPUS="${CPUS:-62,63}"
MAX_LOAD="${MAX_LOAD:-24}"
cd "$(dirname "$0")/.."
H_ENV="$BASE/pg.env"; DS_ENV="$BASE/tpcds/pg.env"

echo "$(date -Is) waiting for other measurements to finish"
while pgrep -u "$(id -u)" -f "[r]un_passes.py|[r]un_m4_chain.sh|[n]ode_timing_chain.sh" > /dev/null; do
  sleep 60
done

stop_cluster() {
  # shellcheck disable=SC1090
  (source "$1" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}

measure() {   # env, machine, query dir, database URL, test query
  local env="$1" machine="$2" qdir="$3" url="$4" test_query="$5"
  stop_cluster "$H_ENV"; stop_cluster "$DS_ENV"
  local out="$BASE/runs/$machine/affinity.txt"
  mkdir -p "$(dirname "$out")"
  # shellcheck disable=SC1090
  (
    source "$env"
    taskset -c "$CPUS" pg_ctl start -w -l "$(dirname "$env")/postgres-$machine.log"
    pm=$(head -1 "$PGDATA/postmaster.pid")
    {
      echo "# $machine: PostgreSQL pinned to CPUs $CPUS of $(nproc --all) on $(hostname) ($(date -Iseconds))"
      echo "postmaster $pm: $(taskset -cp "$pm" | sed 's/.*: //')"
      for p in $(pgrep -P "$pm"); do
        echo "child $p ($(ps -o args= -p "$p" | cut -c1-40)): $(taskset -cp "$p" | sed 's/.*: //')"
      done
      psql -XAtq -c "set max_parallel_workers_per_gather = 2; set parallel_setup_cost = 0;
                     set parallel_tuple_cost = 0; $test_query" > /dev/null &
      q=$!
      seen=0
      for _ in $(seq 1 50); do
        for w in $(pgrep -u "$(id -u)" -f "postgres: parallel worker" || true); do
          echo "parallel worker $w: $(taskset -cp "$w" 2>/dev/null | sed 's/.*: //')"
          seen=1
        done
        [ "$seen" = 1 ] && break
        sleep 0.05
      done
      wait "$q"
    } | tee "$out"
    if grep -E "^(postmaster|child|parallel worker)" "$out" | grep -vq ": $CPUS\$"; then
      echo "a server process is not pinned to CPUs $CPUS" >&2
      exit 1
    fi
  )
  MACHINE="$machine" QUERY_DIR="$qdir" PG_ENV="$env" DATABASE_URL="$url" START= MAX_LOAD="$MAX_LOAD" \
    bash scripts/run_m3.sh
  stop_cluster "$env"
}

measure "$H_ENV" M6 queries/tpch_sf1_v2 "postgresql:///tpch" \
  "select count(*) from lineitem where l_quantity > 0"
measure "$DS_ENV" M6_tpcds queries/tpcds_sf1_kept "postgresql:///tpcds" \
  "select count(*) from store_sales where ss_quantity > 0"
echo "$(date -Is) M6 done"
