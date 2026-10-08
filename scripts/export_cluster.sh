#!/usr/bin/env bash
# Archive M1's database cluster so that another machine runs on the same
# database files.
#
# A fresh TPC-H load elsewhere would run ANALYZE on a new random sample, which
# can change estimates and plans. A copy of M1's stopped data directory keeps
# the data, the statistics and so the plans identical. The copy is started on
# the other machine with scripts/import_cluster.sh.
#
# The script stops the cluster cleanly (pg_ctl stop -m fast: a shutdown
# checkpoint is written), checks that pg_controldata reports a clean
# shutdown, archives PGDATA with tar and zstd (gzip when zstd is not
# installed), writes a SHA-256 file next to the archive, and starts the
# cluster again only if it was running before, also when archiving fails.
# A fast stop would also end other sessions, so it refuses while another
# client is connected (a measurement pass, for example) unless EXPORT_FORCE=1.
#
# Usage (in WSL on the laptop, with no measurement running):
#   bash scripts/export_cluster.sh [output directory]      # default $PILOT_HOME/export
# The TPC-DS cluster has its own environment file and log:
#   PG_ENV=~/cs680-pilot/pg-tpcds.env LOG_FILE=~/cs680-pilot/postgres-tpcds.log bash scripts/export_cluster.sh
# Then copy both files to the server, e.g.
#   scp <archive> <archive>.sha256 waterloo:/var/tmp/pgstudy/
# (naming the server by its address, so that the copy lands on the same
# server every time).
set -euo pipefail

PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
PG_ENV="${PG_ENV:-$PILOT_HOME/pg.env}"
LOG_FILE="${LOG_FILE:-$PILOT_HOME/postgres.log}"
# shellcheck disable=SC1090
source "$PG_ENV"
OUT_DIR="$(realpath -m "${1:-$PILOT_HOME/export}")"

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  echo "no database cluster in PGDATA=$PGDATA" >&2
  exit 1
fi

was_running=0
if pg_ctl status > /dev/null 2>&1; then
  was_running=1
fi
partial=""
finish() {
  [ -n "$partial" ] && rm -f "$partial"
  if [ "$was_running" = 1 ] && ! pg_ctl status > /dev/null 2>&1; then
    echo "starting the cluster again"
    pg_ctl -l "$LOG_FILE" -w start
  fi
}
trap finish EXIT

if [ "$was_running" = 1 ]; then
  # A fast stop ends every session, including a measurement pass that is
  # still running, so refuse while any other client is connected.
  others="$(psql -d postgres -XAtc "select pid || ' ' || coalesce(usename, '') || ' ' || coalesce(datname, '') ||
                                           ' ' || coalesce(state, '') || ' ' || left(query, 60)
                                    from pg_stat_activity
                                    where backend_type = 'client backend' and pid <> pg_backend_pid()")"
  if [ -n "$others" ] && [ "${EXPORT_FORCE:-0}" != 1 ]; then
    echo "other sessions are connected to the cluster; wait for them to finish (or set EXPORT_FORCE=1):" >&2
    echo "$others" >&2
    exit 1
  fi
  echo "stopping the cluster in $PGDATA"
  pg_ctl stop -m fast -w
fi
state="$(pg_controldata "$PGDATA" | sed -n 's/^Database cluster state: *//p')"
if [ "$state" != "shut down" ]; then
  echo "cluster state is '$state', not 'shut down'; start it and stop it cleanly first" >&2
  exit 1
fi
# import_cluster.sh prints the same identifier for the copy.
echo "system identifier: $(pg_controldata "$PGDATA" | sed -n 's/^Database system identifier: *//p')"

mkdir -p "$OUT_DIR"
parent="$(dirname "$PGDATA")"
base="$(basename "$PGDATA")"
name="$base-$(date +%Y%m%d-%H%M%S)"
echo "archiving $PGDATA ($(du -sh "$PGDATA" | cut -f1))"
if command -v zstd > /dev/null; then
  archive="$OUT_DIR/$name.tar.zst"
  partial="$archive.part"
  tar -C "$parent" -cf - "$base" | zstd -q -f -T0 -o "$partial"
  zstd -q -t "$partial"
else
  archive="$OUT_DIR/$name.tar.gz"
  partial="$archive.part"
  tar -C "$parent" -czf "$partial" "$base"
  gzip -t "$partial"
fi
mv "$partial" "$archive"
partial=""
(cd "$OUT_DIR" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")

echo "wrote $archive ($(du -h "$archive" | cut -f1)) and $archive.sha256:"
cat "$archive.sha256"
