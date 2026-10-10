#!/usr/bin/env bash
# The laptop's JOB cluster (paper/PLAN.md, E2: M1_job and M2_job).
#
# The study needs the same plans in every environment. The TPC-H and TPC-DS
# clusters were copied from the laptop to the server; the server's extension
# cluster cannot be copied while it measures, so the laptop loads the same
# IMDB snapshot (scripts/load_job.sh) and then takes over the server's
# planner statistics, dumped there with pg_dump --statistics-only (a
# PostgreSQL 18 option; pg_restore_relation_stats and
# pg_restore_attribute_stats set relpages, reltuples, relallvisible and the
# column statistics). The plans are then compared with
# scripts/explain_all.py before any pass.
#
# Usage in WSL, from project_proposal:
#   bash scripts/setup_job_laptop.sh <statistics dump of the server's imdb database>
set -euo pipefail
STATS="${1:?give the statistics dump of the imdb database on the server (pg_dump --statistics-only)}"
export PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot/job}"
PG_PREFIX="${PG_PREFIX:-$HOME/cs680-pilot/pg-18.6}" PGPORT="${PGPORT:-5434}" bash scripts/setup_postgres.sh
# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"
pg_ctl status > /dev/null 2>&1 || pg_ctl -l "$PILOT_HOME/postgres.log" -w start
PGDATABASE=imdb bash scripts/load_job.sh
echo "taking over the server's statistics from $STATS"
psql -X -q -v ON_ERROR_STOP=1 -d imdb -f "$STATS" > /dev/null
psql -XAt -d imdb -c "select count(*) from pg_stats where schemaname = 'public'"
pg_ctl stop -m fast -w
echo "$(date -Is) laptop JOB cluster ready"
