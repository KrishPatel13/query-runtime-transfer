#!/usr/bin/env bash
# Generate TPC-DS data at scale factor SF with dsdgen and load it into the
# TPC-DS cluster made by scripts/tpcds_setup.sh (database tpcds by default).
#
# Steps, each timed: dsdgen; the tables of the kit's tpcds.sql with the
# primary keys defined there; COPY of every table; the kit's foreign keys
# (tpcds_ri.sql); the indexes in sql/tpcds_indexes.sql (the file explains
# which indexes the TPC-DS specification allows); VACUUM (FREEZE, ANALYZE)
# once, then a checkpoint. The data never changes afterwards.
#
# The foreign keys are declared for the same reason as in
# sql/tpch_keys_indexes.sql: PostgreSQL uses them when it estimates the size
# of a join. Clause 2.5.4.1 of the specification allows primary-key,
# foreign-key and NOT NULL constraints. tpcds_ri.sql declares 102 of them;
# two more are commented out in it because their columns (cp_promo_id,
# cr_ship_date_sk) are not in the kit's schema.
#
# dsdgen's files need three things before COPY takes them:
#   * every row ends with the field delimiter '|', which COPY would read as
#     an extra empty column, so it is removed on the way in, as load_tpch.sh
#     does for dbgen;
#   * a NULL is an empty field, so COPY runs with NULL '' (by default an
#     empty field would be an empty string, which integer columns reject);
#   * COPY rejects bytes that are not valid UTF-8, and some dsdgen builds
#     (the official kit's, for one) write a few names in Latin-1. Every file
#     is checked with iconv, and a file with invalid bytes goes through
#     scripts/tpcds_fix_utf8.py, which reads each invalid byte as Latin-1.
#     The record lists the lines it changed. The build from gregrahn's kit
#     writes these names (CÔTE D'IVOIRE and RÉUNION in customer.dat) as valid
#     UTF-8, so at SF=1 no file needs the filter. COPY's text format would
#     also read a backslash as an escape, so the script stops if any file
#     contains one.
#
# Writes the load record (row counts, sizes, times) with
# scripts/tpcds_load_record.py: for SF=1 in database tpcds to
# queries/tpcds_sf1/load_record.json and load_numbers.tex, otherwise to
# $PILOT_HOME/tpcds-load-sf<SF>-<database>.json.
#
# Usage (after scripts/tpcds_setup.sh):
#   bash scripts/tpcds_load.sh                                # SF=1 into tpcds
#   SF=0.1 TPCDS_DB=tpcds_small bash scripts/tpcds_load.sh    # small smoke test
set -euo pipefail

SF="${SF:-1}"
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="$PILOT_HOME/tpcds-kit/tools"
TPCDS_PGDATA="$PILOT_HOME/pgdata-tpcds"

# shellcheck disable=SC1091
source "$PILOT_HOME/pg-tpcds.env"
DB="${TPCDS_DB:-$PGDATABASE}"
# The database is dropped and recreated below, so a smoke test must not reuse
# the name of the SF=1 database the measurements depend on.
if [ "$SF" != "1" ] && [ "$DB" = "tpcds" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "SF=$SF would replace the SF=1 database 'tpcds'; set TPCDS_DB to another name (or FORCE=1)" >&2
  exit 1
fi
[ -x "$KIT/dsdgen" ] || { echo "$KIT/dsdgen is missing; run scripts/tpcds_setup.sh first" >&2; exit 1; }
pg_ctl -D "$TPCDS_PGDATA" status > /dev/null 2>&1 || pg_ctl -D "$TPCDS_PGDATA" -l "$PILOT_HOME/postgres-tpcds.log" -w start
# Make sure the connection settings reach the TPC-DS cluster and not the
# TPC-H one before anything is dropped.
where="$(psql -X -d postgres -Atc "select current_setting('data_directory')")"
[ "$where" = "$TPCDS_PGDATA" ] || { echo "PGHOST/PGPORT reach $where, not $TPCDS_PGDATA; stopping" >&2; exit 1; }
export PGDATABASE="$DB"
URL="postgresql:///$DB?host=$PGHOST&port=$PGPORT"

now() { date +%s.%N; }
since() { awk -v s="$1" -v e="$(date +%s.%N)" 'BEGIN { printf "%.1f", e - s }'; }

# dsdgen writes a marker with its run time only after it succeeds, so an
# interrupted run is generated again rather than loaded half-written.
DATA="$PILOT_HOME/tpcds-data-sf$SF"
if [ ! -f "$DATA/.dsdgen_seconds" ]; then
  mkdir -p "$DATA"
  s=$(now)
  # dsdgen reads its distributions (tpcds.idx) from the current directory.
  (cd "$KIT" && ./dsdgen -SCALE "$SF" -DIR "$DATA" -FORCE Y -QUIET Y)
  since "$s" > "$DATA/.dsdgen_seconds"
  echo "dsdgen: $(cat "$DATA/.dsdgen_seconds") s"
fi
gen_s="$(cat "$DATA/.dsdgen_seconds")"

TABLES="$(sed -n 's/^create table \([a-z_]*\).*/\1/p' "$KIT/tpcds.sql")"
declare -A needs_fix=()
for t in $TABLES; do
  f="$DATA/$t.dat"
  [ -f "$f" ] || { echo "$f is missing" >&2; exit 1; }
  if grep -q '[\]' "$f"; then
    echo "$f contains a backslash, which COPY's text format reads as an escape" >&2
    exit 1
  fi
  iconv -f UTF-8 -t UTF-8 "$f" > /dev/null 2>&1 || needs_fix[$t]=1
done

load_start=$(now)
echo "creating database $DB"
dropdb --if-exists "$DB"
createdb "$DB"
psql -q -v ON_ERROR_STOP=1 -f "$KIT/tpcds.sql"

copy_times=()
reencoded=()
copy_start=$(now)
for t in $TABLES; do
  s=$(now)
  copy="COPY $t FROM STDIN WITH (DELIMITER '|', NULL '')"
  if [ -n "${needs_fix[$t]:-}" ]; then
    python3 "$HERE/scripts/tpcds_fix_utf8.py" --count-file "$DATA/$t.reencoded" < "$DATA/$t.dat" \
      | sed 's/|$//' | psql -q -v ON_ERROR_STOP=1 -c "$copy"
    reencoded+=("$t=$(cat "$DATA/$t.reencoded")")
  else
    sed 's/|$//' "$DATA/$t.dat" | psql -q -v ON_ERROR_STOP=1 -c "$copy"
  fi
  copy_times+=("$t=$(since "$s")")
  echo "  loaded $t in ${copy_times[-1]#*=} s"
done
copy_s=$(since "$copy_start")

echo "adding foreign keys"
s=$(now)
psql -q -v ON_ERROR_STOP=1 -f "$KIT/tpcds_ri.sql"
fk_s=$(since "$s")
echo "  done in $fk_s s"

echo "adding indexes"
s=$(now)
psql -q -v ON_ERROR_STOP=1 -f "$HERE/sql/tpcds_indexes.sql"
index_s=$(since "$s")
echo "  done in $index_s s"

echo "vacuum (freeze, analyze)"
s=$(now)
psql -q -v ON_ERROR_STOP=1 -c "VACUUM (FREEZE, ANALYZE)" -c "CHECKPOINT"
vacuum_s=$(since "$s")
echo "  done in $vacuum_s s"
load_s=$(since "$load_start")

if [ "$SF" = "1" ] && [ "$DB" = "tpcds" ]; then
  record="$HERE/queries/tpcds_sf1/load_record.json"
  numbers=(--out-tex "$HERE/queries/tpcds_sf1/load_numbers.tex")
else
  record="$PILOT_HOME/tpcds-load-sf$SF-$DB.json"
  numbers=()
fi
usage="$("$KIT/dsdgen" -help 2>&1 || true)"
python3 "$HERE/scripts/tpcds_load_record.py" --database-url "$URL" --out-json "$record" "${numbers[@]}" \
  --scale "$SF" --data-dir "$DATA" \
  --kit-url "$(git -C "$KIT" remote get-url origin)" --kit-commit "$(git -C "$KIT" rev-parse HEAD)" \
  --kit-version "${usage%%$'\n'*}" \
  --seconds "dsdgen=$gen_s" "load=$load_s" "copy=$copy_s" "foreign_keys=$fk_s" "indexes=$index_s" \
            "vacuum_freeze_analyze=$vacuum_s" \
  --copy-seconds "${copy_times[@]}" --reencoded "${reencoded[@]}"
echo "record written to $record"
