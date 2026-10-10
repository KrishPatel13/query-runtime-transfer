#!/usr/bin/env bash
# Start a stopped cluster with its PostgreSQL server pinned to one processor.
#
# The general form of scripts/start_m2.sh, for any machine and cluster: the
# postmaster starts under taskset, so every backend and parallel worker it
# forks shares one processor while the planner (which does not look at the
# processor count) keeps the plans. The script checks the affinity of the
# postmaster, its children and the parallel workers of a test query, writes
# the check to OUT, and exits with status 1 if any of them is not pinned.
#
# Settings (environment):
#   PG_ENV      the cluster's environment file (PGDATA, PGHOST, PGPORT, PATH)
#   CPU         the processor (default: the last one)
#   LOG_FILE    the server log (default: postgres-pinned.log next to PG_ENV)
#   OUT         where the affinity check goes (required)
#   TEST_QUERY  a query that gets parallel workers (default: TPC-H lineitem)
#   TEST_DB     the database of the test query (default: PGDATABASE of PG_ENV)
#
# M4 (the UW server with one processor), TPC-H:
#   PG_ENV=/var/tmp/pgstudy/pg.env OUT=/var/tmp/pgstudy/runs/M4/affinity.txt \
#     bash scripts/start_pinned.sh
# TPC-DS:
#   PG_ENV=/var/tmp/pgstudy/tpcds/pg.env OUT=/var/tmp/pgstudy/runs/M4_tpcds/affinity.txt \
#     TEST_QUERY="select count(*) from store_sales where ss_quantity > 0" bash scripts/start_pinned.sh
# To end it: source <PG_ENV> && pg_ctl stop -m fast
set -euo pipefail

: "${PG_ENV:?set PG_ENV to the environment file of the cluster}"
: "${OUT:?set OUT to the file for the affinity check}"
# shellcheck disable=SC1090
source "$PG_ENV"
nproc_all=$(nproc --all)
CPU="${CPU:-$((nproc_all - 1))}"
LOG_FILE="${LOG_FILE:-$(dirname "$PG_ENV")/postgres-pinned.log}"
TEST_QUERY="${TEST_QUERY:-select count(*) from lineitem where l_quantity > 0}"
TEST_DB="${TEST_DB:-$PGDATABASE}"

if pg_ctl status >/dev/null 2>&1; then
    echo "the cluster in $PGDATA is running; stop it first so that it starts pinned" >&2
    exit 1
fi
if [ "$CPU" -ge "$nproc_all" ]; then
    echo "CPU $CPU does not exist ($nproc_all processors)" >&2
    exit 1
fi

taskset -c "$CPU" pg_ctl start -w -l "$LOG_FILE"
pm=$(head -1 "$PGDATA/postmaster.pid")

mkdir -p "$(dirname "$OUT")"
{
    echo "# PostgreSQL in $PGDATA pinned to CPU $CPU of $nproc_all on $(hostname) ($(date -Iseconds))"
    echo "# command: taskset -c $CPU pg_ctl start -w -l $LOG_FILE"
    echo "postmaster $pm: $(taskset -cp "$pm" | sed 's/.*: //')"
    for p in $(pgrep -P "$pm"); do
        echo "child $p ($(ps -o args= -p "$p" | cut -c1-40)): $(taskset -cp "$p" | sed 's/.*: //')"
    done
    psql -XAtq -d "$TEST_DB" -c "set max_parallel_workers_per_gather = 2; set parallel_setup_cost = 0; set parallel_tuple_cost = 0;
                   $TEST_QUERY" >/dev/null &
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
    [ "$seen" = 1 ] || echo "no parallel worker seen during the test query"
} | tee "$OUT"

if grep -E "^(postmaster|child|parallel worker)" "$OUT" | grep -vq ": $CPU\$"; then
    echo "a server process is not pinned to CPU $CPU" >&2
    exit 1
fi
echo "cluster running, pinned to CPU $CPU"
