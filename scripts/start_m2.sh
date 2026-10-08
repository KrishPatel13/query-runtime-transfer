#!/usr/bin/env bash
# Start a laptop cluster as M2: M1's cluster with the PostgreSQL server
# pinned to one virtual processor of WSL.
#
# The postmaster is started under taskset, and every process it forks
# (backends, parallel workers, background workers) inherits its CPU
# affinity, so a plan's parallel workers share one processor. The planner
# does not look at the processor count, so the plans stay M1's. The rest of
# WSL (the collector, the operating system, other programs) keeps all
# processors, as for M1. STUDY.md explains why M2 is made this way rather
# than by limiting WSL to one processor.
#
# CLUSTER=tpch (default) starts the TPC-H cluster and stops the TPC-DS one;
# CLUSTER=tpcds does the opposite. The script starts the chosen cluster
# pinned to CPU (default 19, a virtual processor with few interrupts),
# checks the affinity of the postmaster, its children and a parallel worker
# of a test query, and writes the check to results/runs/M2/affinity.txt
# (results/runs/M2_tpcds/affinity.txt for TPC-DS).
#
# Usage (in WSL, from project_proposal): bash scripts/start_m2.sh
# Afterwards: python3 src/run_passes.py --machine M2 --query-dir queries/tpch_sf1_v2
# TPC-DS:     CLUSTER=tpcds bash scripts/start_m2.sh
#             python3 src/run_passes.py --machine M2_tpcds --query-dir queries/tpcds_sf1_kept \
#               --database-url 'postgresql:///tpcds?host=/tmp&port=5433'
# To end M2: source ~/cs680-pilot/pg.env && pg_ctl stop -m fast && pg_ctl start -w -l ~/cs680-pilot/postgres.log
# (pg-tpcds.env and postgres-tpcds.log for TPC-DS).
set -euo pipefail

CPU="${CPU:-19}"
CLUSTER="${CLUSTER:-tpch}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
case "$CLUSTER" in
  tpch)
    ENV=~/cs680-pilot/pg.env; OTHER_ENV=~/cs680-pilot/pg-tpcds.env; LOG=~/cs680-pilot/postgres.log
    OUT="$HERE/results/runs/M2/affinity.txt"
    TEST_QUERY="select count(*) from lineitem where l_quantity > 0" ;;
  tpcds)
    ENV=~/cs680-pilot/pg-tpcds.env; OTHER_ENV=~/cs680-pilot/pg.env; LOG=~/cs680-pilot/postgres-tpcds.log
    OUT="$HERE/results/runs/M2_tpcds/affinity.txt"
    TEST_QUERY="select count(*) from store_sales where ss_quantity > 0" ;;
  *)
    echo "CLUSTER must be tpch or tpcds, not $CLUSTER" >&2
    exit 1 ;;
esac

# shellcheck disable=SC1090
source "$OTHER_ENV"
if pg_ctl status >/dev/null 2>&1; then
    pg_ctl stop -m fast -w
fi
# shellcheck disable=SC1090
source "$ENV"
if pg_ctl status >/dev/null 2>&1; then
    echo "the $CLUSTER cluster is already running; stop it first so that it starts pinned" >&2
    exit 1
fi
nproc_all=$(nproc --all)
if [ "$CPU" -ge "$nproc_all" ]; then
    echo "CPU $CPU does not exist ($nproc_all processors)" >&2
    exit 1
fi

taskset -c "$CPU" pg_ctl start -w -l "$LOG"
pm=$(head -1 "$PGDATA/postmaster.pid")

mkdir -p "$(dirname "$OUT")"
{
    echo "# M2 ($CLUSTER): PostgreSQL pinned to CPU $CPU of $nproc_all ($(date -Iseconds))"
    echo "# command: taskset -c $CPU pg_ctl start -w -l $LOG"
    echo "postmaster $pm: $(taskset -cp "$pm" | sed 's/.*: //')"
    for p in $(pgrep -P "$pm"); do
        echo "child $p ($(ps -o args= -p "$p" | cut -c1-40)): $(taskset -cp "$p" | sed 's/.*: //')"
    done
    # A query with parallel workers; its workers must be pinned too.
    psql -XAtq -c "set max_parallel_workers_per_gather = 2; set parallel_setup_cost = 0; set parallel_tuple_cost = 0;
                   $TEST_QUERY" >/dev/null &
    q=$!
    seen=0
    for _ in $(seq 1 50); do
        for w in $(pgrep -f "postgres: parallel worker" || true); do
            echo "parallel worker $w: $(taskset -cp "$w" 2>/dev/null | sed 's/.*: //')"
            seen=1
        done
        [ "$seen" = 1 ] && break
        sleep 0.05
    done
    wait "$q"
    [ "$seen" = 1 ] || echo "no parallel worker seen during the test query"
} | tee "$OUT"

if grep -E "^(postmaster|child|parallel worker)" "$OUT" | grep -vq ": $CPU\$"; then
    echo "a server process is not pinned to CPU $CPU" >&2
    exit 1
fi
echo "M2 $CLUSTER cluster running, pinned to CPU $CPU"
