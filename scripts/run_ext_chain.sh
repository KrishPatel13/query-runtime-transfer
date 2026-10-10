#!/usr/bin/env bash
# Measure the extension's server environments (paper/PLAN.md, E1 and E2):
# TPC-H at scale factor 10 (database tpch10, queries/tpch_sf10_sub) in the
# cluster of BASE, made by scripts/setup_postgres.sh and scripts/load_tpch.sh,
# and the Join Order Benchmark (database imdb, queries/job_kept) in the
# cluster of JOB_BASE, a copy of the laptop's JOB cluster.
#
# The JOB cluster is a copy because a fresh load is not physically the same:
# the server's own load of the same IMDB snapshot (in BASE, used for the JOB
# screen) ended up with a few more or fewer pages in two tables than the
# laptop's, and the planner scales its estimates by the actual size, so 40 of
# 113 plans differed in their costs. As with M1's TPC-H and TPC-DS clusters,
# the laptop's stopped cluster is archived (scripts/export_cluster.sh) and
# started here (scripts/import_cluster.sh), which the chain does when JOB_BASE
# has no cluster yet, between measurements. Before the first JOB pass it
# compares every plan with the laptop's (LAPTOP_PLANS, scripts/explain_all.py)
# and stops if one differs.
#
# The JOB screen (every query of queries/job_all once, without a warm-up, on
# the unpinned server; scripts/job_queries.py keep) runs only when
# queries/job_kept is missing. Then, for each environment, both workloads,
# three passes each, through scripts/run_m3.sh:
#   M3_*  not pinned, as M3; passes start once the 5-minute load is below
#         MAX_LOAD_FREE (default 8)
#   M6_*  pinned to CPUS2 (default 62,63), as M6
#   M4_*  pinned to CPU1 (default 63), as M4; both start below MAX_LOAD_PINNED
#         (default 24), as for M4 and M6
# A pinned start checks the affinity of the postmaster, its children and a
# test query's parallel workers and writes it to runs/<machine>/affinity.txt.
# Only one cluster runs at a time. An environment whose three passes are
# already in BASE/runs is skipped, so the chain can be started again after an
# interruption.
#
# Usage on the server, detached, from a copy of project_proposal:
#   BASE=/var/tmp/<dir> JOB_BASE=/var/tmp/<dir2> JOB_ARCHIVE=<pgdata-*.tar.gz> LAPTOP_PLANS=<plans_M1_job.json> \
#     setsid nohup bash scripts/run_ext_chain.sh > <log> 2>&1 < /dev/null &
set -euo pipefail

BASE="${BASE:?set BASE to the PILOT_HOME of the extension cluster}"
JOB_BASE="${JOB_BASE:?set JOB_BASE to the PILOT_HOME of the copy of the laptop JOB cluster}"
LAPTOP_PLANS="${LAPTOP_PLANS:?set LAPTOP_PLANS to explain_all.py output of the laptop for queries/job_kept}"
JOB_PORT="${JOB_PORT:-5471}"
CPUS2="${CPUS2:-62,63}"
CPU1="${CPU1:-63}"
MAX_LOAD_FREE="${MAX_LOAD_FREE:-8}"
MAX_LOAD_PINNED="${MAX_LOAD_PINNED:-24}"
ENV="$BASE/pg.env"
JOB_ENV="$JOB_BASE/pg.env"
cd "$(dirname "$0")/.."

echo "$(date -Is) waiting for other measurements of this user to finish"
while pgrep -u "$(id -u)" -f "[r]un_passes.py|[r]un_m4_chain.sh|[r]un_m6_chain.sh|[n]ode_timing_chain.sh" > /dev/null; do
  sleep 60
done

stop_cluster() {   # env file
  # shellcheck disable=SC1090
  (source "$1" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}

start_cluster() {   # env file, machine, cpus (empty: not pinned), database, test query
  local env="$1" machine="$2" cpus="$3" db="$4" test_query="$5"
  stop_cluster "$ENV"; stop_cluster "$JOB_ENV"
  # shellcheck disable=SC1090
  (
    source "$env"
    local dir
    dir="$(dirname "$env")"
    if [ -z "$cpus" ]; then
      pg_ctl start -w -l "$dir/postgres-$machine.log"
      exit 0
    fi
    local out="$BASE/runs/$machine/affinity.txt"
    mkdir -p "$(dirname "$out")"
    taskset -c "$cpus" pg_ctl start -w -l "$dir/postgres-$machine.log"
    pm=$(head -1 "$PGDATA/postmaster.pid")
    {
      echo "# $machine: PostgreSQL pinned to CPUs $cpus of $(nproc --all) on $(hostname) ($(date -Iseconds))"
      echo "postmaster $pm: $(taskset -cp "$pm" | sed 's/.*: //')"
      for p in $(pgrep -P "$pm"); do
        echo "child $p ($(ps -o args= -p "$p" | cut -c1-40)): $(taskset -cp "$p" | sed 's/.*: //')"
      done
      psql -XAtq -d "$db" -c "set max_parallel_workers_per_gather = 2; set parallel_setup_cost = 0;
                     set parallel_tuple_cost = 0; $test_query" > /dev/null &
      q=$!
      seen=0
      for _ in $(seq 1 100); do
        for w in $(pgrep -u "$(id -u)" -f "postgres: parallel worker" || true); do
          echo "parallel worker $w: $(taskset -cp "$w" 2>/dev/null | sed 's/.*: //')"
          seen=1
        done
        [ "$seen" = 1 ] && break
        sleep 0.05
      done
      wait "$q"
      [ "$seen" = 1 ] || echo "no parallel worker seen during the test query"
    } | tee "$out"
    if grep -E "^(postmaster|child|parallel worker)" "$out" | grep -vq ": $cpus\$"; then
      echo "a server process is not pinned to CPUs $cpus" >&2
      exit 1
    fi
  )
}

done_passes() {   # machine: true when its three passes finished
  local d="$BASE/runs/$1"
  [ -f "$d/pass3/runtime_dataset.meta.json" ] && grep -q '"finished_at": "' "$d/pass3/runtime_dataset.meta.json"
}

passes() {   # machine, query dir, env file, database, max load
  local machine="$1"
  if done_passes "$machine"; then
    echo "$(date -Is) $machine already has three passes; skipped"
    return
  fi
  MACHINE="$machine" QUERY_DIR="$2" PG_ENV="$3" DATABASE_URL="postgresql:///$4" START= MAX_LOAD="$5" \
    PILOT_HOME="$BASE" bash scripts/run_m3.sh
}

# The JOB screen, once, in the extension cluster.
if [ ! -f queries/job_kept/split.csv ]; then
  start_cluster "$ENV" M3_job "" imdb ""
  # shellcheck disable=SC1090
  (source "$ENV" && python3 src/collect.py --query-dir queries/job_all --database-url postgresql:///imdb \
     --warmups 0 --repeats 1 --timeout-s 120 --machine M3_job --out "$BASE/screen_job/runtime_dataset.csv")
  "$BASE/../venv/bin/python" scripts/job_queries.py keep --screen "$BASE/screen_job/runtime_dataset.csv"
fi

# The copy of the laptop's JOB cluster, once, and the plan check.
if [ ! -f "$JOB_BASE/pgdata/PG_VERSION" ]; then
  : "${JOB_ARCHIVE:?set JOB_ARCHIVE to the laptop JOB cluster archive of scripts/export_cluster.sh}"
  stop_cluster "$ENV"
  PILOT_HOME="$JOB_BASE" PG_PREFIX="${PG_PREFIX:-$(dirname "$BASE")/pg-18.6}" \
    PG_SECURE=1 PGPORT="$JOB_PORT" PG_INITDB=0 bash scripts/setup_postgres.sh
  PILOT_HOME="$JOB_BASE" bash scripts/import_cluster.sh "$JOB_ARCHIVE"
fi
start_cluster "$JOB_ENV" M3_job "" imdb ""
# shellcheck disable=SC1090
(source "$JOB_ENV" && PGDATABASE=imdb python3 scripts/explain_all.py --query-dir queries/job_kept \
   --out "$JOB_BASE/plans_M3_job.json")
check=$(python3 scripts/explain_all.py --compare "$LAPTOP_PLANS" "$JOB_BASE/plans_M3_job.json" || true)
echo "plan check against the laptop: $check"
if ! echo "$check" | grep -q '"differ": 0'; then
  echo "some JOB plans differ from the laptop's; stopping before any JOB pass" >&2
  stop_cluster "$JOB_ENV"
  exit 1
fi
stop_cluster "$JOB_ENV"

for spec in "M3::$MAX_LOAD_FREE" "M6:$CPUS2:$MAX_LOAD_PINNED" "M4:$CPU1:$MAX_LOAD_PINNED"; do
  IFS=: read -r env cpus load <<< "$spec"
  if ! done_passes "${env}_sf10"; then
    start_cluster "$ENV" "${env}_sf10" "$cpus" tpch10 "select count(*) from lineitem where l_quantity > 0"
    passes "${env}_sf10" queries/tpch_sf10_sub "$ENV" tpch10 "$load"
    stop_cluster "$ENV"
  fi
  if ! done_passes "${env}_job"; then
    start_cluster "$JOB_ENV" "${env}_job" "$cpus" imdb "select count(*) from cast_info where id > 0"
    passes "${env}_job" queries/job_kept "$JOB_ENV" imdb "$load"
    stop_cluster "$JOB_ENV"
  fi
done
echo "$(date -Is) extension chain done"
