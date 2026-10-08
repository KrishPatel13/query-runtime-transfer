#!/usr/bin/env bash
# Build the TPC-H tools (dbgen, qgen), generate data at scale factor SF, and
# load it into the database named by PGDATABASE (default tpch).
#
# Usage (after scripts/setup_postgres.sh):
#   bash scripts/load_tpch.sh                            # SF=1 into tpch
#   SF=0.1 PGDATABASE=tpch_small bash scripts/load_tpch.sh  # small smoke test
set -euo pipefail

SF="${SF:-1}"
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
KIT_URL="https://github.com/gregrahn/tpch-kit.git"
KIT_COMMIT="852ad0a5ee31ebefeed884cea4188781dd9613a3"   # TPC-H tools 2.17.3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

caller_db="${PGDATABASE:-}"
# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"
export PGDATABASE="${caller_db:-$PGDATABASE}"   # pg.env sets tpch; keep a name given on the command line
# The database is dropped and recreated below, so a smoke test must not reuse
# the name of the SF=1 database the measurements depend on.
if [ "$SF" != "1" ] && [ "$PGDATABASE" = "tpch" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "SF=$SF would replace the SF=1 database 'tpch'; set PGDATABASE to another name (or FORCE=1)" >&2
  exit 1
fi
pg_ctl status > /dev/null 2>&1 || pg_ctl -l "$PILOT_HOME/postgres.log" -w start

cd "$PILOT_HOME"
[ -d tpch-kit ] || git clone -q "$KIT_URL" tpch-kit
git -C tpch-kit checkout -q "$KIT_COMMIT"
if [ ! -x tpch-kit/dbgen/qgen ]; then
  make -C tpch-kit/dbgen -s MACHINE=LINUX DATABASE=POSTGRESQL > "$PILOT_HOME/tpch-build.log" 2>&1
fi

DATA="$PILOT_HOME/data-sf$SF"
mkdir -p "$DATA"
if [ ! -f "$DATA/lineitem.tbl" ]; then
  (cd tpch-kit/dbgen && DSS_PATH="$DATA" ./dbgen -q -f -s "$SF")
fi

echo "creating database $PGDATABASE"
dropdb --if-exists "$PGDATABASE"
createdb "$PGDATABASE"
psql -q -v ON_ERROR_STOP=1 -f tpch-kit/dbgen/dss.ddl

for t in region nation part supplier partsupp customer orders lineitem; do
  # dbgen ends every row with a trailing '|', which COPY would read as an
  # extra empty column.
  start=$(date +%s)
  sed 's/|$//' "$DATA/$t.tbl" | psql -q -v ON_ERROR_STOP=1 -c "COPY $t FROM STDIN WITH (DELIMITER '|')"
  echo "  loaded $t in $(( $(date +%s) - start )) s"
done

echo "adding keys and indexes"
start=$(date +%s)
psql -q -v ON_ERROR_STOP=1 -f "$HERE/sql/tpch_keys_indexes.sql"
echo "  done in $(( $(date +%s) - start )) s"

echo "vacuum analyze"
psql -q -v ON_ERROR_STOP=1 -c "VACUUM (FREEZE, ANALYZE)" -c "CHECKPOINT"

psql -At -c "select relname, reltuples::bigint, pg_size_pretty(pg_total_relation_size(oid))
             from pg_class where relkind = 'r' and relnamespace = 'public'::regnamespace
             order by pg_total_relation_size(oid) desc"
psql -At -c "select pg_size_pretty(pg_database_size(current_database()))"
