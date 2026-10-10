#!/usr/bin/env bash
# Measure JOB on the laptop (paper/PLAN.md, E2): M1_job, then M2_job with
# PostgreSQL pinned to one virtual processor (CPU, default 19, as M2).
#
# The laptop's JOB cluster is the original that the server's JOB cluster is
# copied from (scripts/export_cluster.sh), so the plans of every query are
# saved first (scripts/explain_all.py, JOB_HOME/plans_M1_job.json) for the
# server's check. With SERVER_PLANS set, they are also compared with that
# file and the script stops if a plan differs. The passes go through
# src/run_passes.py, whose laptop guard refuses to run on battery, with
# energy saver on or with the WSL clock not set up, and keeps Windows awake.
# Run nothing else on the laptop meanwhile.
#
# Usage in WSL, from project_proposal, after scripts/setup_job_laptop.sh:
#   bash scripts/run_job_laptop.sh
set -euo pipefail
SERVER_PLANS="${SERVER_PLANS:-}"
JOB_HOME="${JOB_HOME:-$HOME/cs680-pilot/job}"
CPU="${CPU:-19}"
ENV="$JOB_HOME/pg.env"
URL="postgresql:///imdb?host=/tmp&port=$(sed -n 's/^export PGPORT=//p' "$ENV")"
cd "$(dirname "$0")/.."

stop_cluster() {
  # shellcheck disable=SC1090
  (source "$ENV" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
}
for other in "$HOME/cs680-pilot/pg.env" "$HOME/cs680-pilot/pg-tpcds.env"; do
  # shellcheck disable=SC1090
  [ -f "$other" ] && (source "$other" && if pg_ctl status > /dev/null 2>&1; then pg_ctl stop -m fast -w; fi)
done

stop_cluster
# shellcheck disable=SC1090
(source "$ENV" && pg_ctl start -w -l "$JOB_HOME/postgres-M1_job.log" &&
 PGDATABASE=imdb python3 scripts/explain_all.py --query-dir queries/job_kept --out "$JOB_HOME/plans_M1_job.json")
if [ -n "$SERVER_PLANS" ]; then
  result=$(python3 scripts/explain_all.py --compare "$SERVER_PLANS" "$JOB_HOME/plans_M1_job.json" || true)
  echo "plan check against the server: $result"
  if ! echo "$result" | grep -q '"differ": 0'; then
    echo "some plans differ from the server's; no pass is run" >&2
    stop_cluster
    exit 1
  fi
fi

if [ ! -f results/runs/M1_job/pass3/runtime_dataset.meta.json ]; then
  # shellcheck disable=SC1090
  (source "$ENV" && python3 src/run_passes.py --machine M1_job --query-dir queries/job_kept \
     --database-url "$URL" --keep-awake-hours 8)
fi

stop_cluster
PG_ENV="$ENV" CPU="$CPU" TEST_DB=imdb OUT=results/runs/M2_job/affinity.txt LOG_FILE="$JOB_HOME/postgres-M2_job.log" \
  TEST_QUERY="select count(*) from cast_info where id > 0" bash scripts/start_pinned.sh
# shellcheck disable=SC1090
(source "$ENV" && python3 src/run_passes.py --machine M2_job --query-dir queries/job_kept \
   --database-url "$URL" --keep-awake-hours 8)
stop_cluster
echo "$(date -Is) laptop JOB done"
