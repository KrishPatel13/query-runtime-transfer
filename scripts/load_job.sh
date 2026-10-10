#!/usr/bin/env bash
# Load the IMDB data of the Join Order Benchmark (paper/PLAN.md, E2) into the
# database named by PGDATABASE (default imdb).
#
# Downloads the JOB authors' IMDB snapshot (imdb.tgz, CSV files) once into
# $PILOT_HOME/imdb and records its SHA-256, creates the tables of
# sql/job_schema.sql (written by scripts/job_queries.py fetch), copies every
# CSV file in, adds the foreign-key indexes of sql/job_fkindexes.sql, and
# freezes and analyzes the database as scripts/load_tpch.sh does.
#
# Usage (after scripts/setup_postgres.sh):
#   PILOT_HOME=/var/tmp/<dir> bash scripts/load_job.sh
set -euo pipefail
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
URL="https://event.cwi.nl/da/job/imdb.tgz"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
caller_db="${PGDATABASE:-}"
# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"
export PGDATABASE="${caller_db:-imdb}"
case "$PGDATABASE" in
  tpch|tpcds|tpch10) echo "PGDATABASE=$PGDATABASE is a benchmark database of the study; choose another name" >&2; exit 1 ;;
esac
pg_ctl status > /dev/null 2>&1 || pg_ctl -l "$PILOT_HOME/postgres.log" -w start

DATA="$PILOT_HOME/imdb"
mkdir -p "$DATA"
if [ ! -f "$DATA/imdb.tgz" ]; then
  echo "downloading $URL"
  curl -fsSL -o "$DATA/imdb.tgz.part" "$URL"
  mv "$DATA/imdb.tgz.part" "$DATA/imdb.tgz"
fi
sha256sum "$DATA/imdb.tgz" | tee "$DATA/imdb.tgz.sha256"
if [ ! -f "$DATA/csv/title.csv" ]; then
  mkdir -p "$DATA/csv"
  tar -xzf "$DATA/imdb.tgz" -C "$DATA/csv"
fi

tables=$(grep -oP '^CREATE TABLE \K[a-z_]+' "$HERE/sql/job_schema.sql")
for t in $tables; do
  [ -f "$DATA/csv/$t.csv" ] || { echo "no $t.csv in $DATA/csv" >&2; exit 1; }
done
echo "creating database $PGDATABASE"
dropdb --if-exists "$PGDATABASE"
createdb "$PGDATABASE"
psql -q -v ON_ERROR_STOP=1 -f "$HERE/sql/job_schema.sql"
for t in $tables; do
  start=$(date +%s)
  # The snapshot's CSV files escape quotes with a backslash.
  psql -q -v ON_ERROR_STOP=1 -c "\\copy $t from '$DATA/csv/$t.csv' with (format csv, escape '\\')"
  echo "  loaded $t in $(( $(date +%s) - start )) s"
done
echo "adding the foreign-key indexes"
start=$(date +%s)
psql -q -v ON_ERROR_STOP=1 -f "$HERE/sql/job_fkindexes.sql"
echo "  done in $(( $(date +%s) - start )) s"
psql -q -v ON_ERROR_STOP=1 -c "VACUUM (FREEZE, ANALYZE)" -c "CHECKPOINT"

psql -At -c "select relname, reltuples::bigint, pg_size_pretty(pg_total_relation_size(oid))
             from pg_class where relkind = 'r' and relnamespace = 'public'::regnamespace
             order by pg_total_relation_size(oid) desc"
psql -At -c "select pg_size_pretty(pg_database_size(current_database()))"
