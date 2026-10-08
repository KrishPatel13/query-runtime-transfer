#!/usr/bin/env bash
# Build the TPC-DS tools (dsdgen, dsqgen) and create a separate PostgreSQL
# cluster for TPC-DS next to the TPC-H cluster made by setup_postgres.sh.
#
# The TPC-DS cluster runs the same PostgreSQL binaries, was created with the
# same initdb options, and uses the same settings block as the TPC-H cluster;
# only the port differs. The block is copied from the TPC-H cluster's
# postgresql.conf instead of being typed again, and the script stops if the
# two blocks ever differ in anything but the port. The TPC-H cluster is only
# read: this script never starts, stops or changes it.
#
# The cluster trusts every local connection and listens on localhost, like
# M1's TPC-H cluster, so it is for the laptop (M1) only: on a shared server
# any logged-in user could connect as superuser. The script therefore stops
# unless it runs in WSL, or TPCDS_ALLOW_ANY_HOST=1 is set for a private
# machine. On a shared server use setup_postgres.sh's secure mode instead.
#
# Usage (inside WSL on the laptop, after scripts/setup_postgres.sh):
#   bash scripts/tpcds_setup.sh
# Then, in every later shell that works on TPC-DS:
#   source ~/cs680-pilot/pg-tpcds.env
# and to stop the TPC-DS cluster when it is not needed:
#   pg_ctl stop
set -euo pipefail

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && [ "${TPCDS_ALLOW_ANY_HOST:-0}" != 1 ]; then
  echo "this makes a cluster that trusts every local connection, meant for WSL on the laptop only;" >&2
  echo "set TPCDS_ALLOW_ANY_HOST=1 on a private machine (never on a shared server)" >&2
  exit 1
fi

PG_VERSION="${PG_VERSION:-18.6}"
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
PREFIX="$PILOT_HOME/pg-$PG_VERSION"
TPCH_CONF="$PILOT_HOME/pgdata/postgresql.conf"
PGDATA_DIR="$PILOT_HOME/pgdata-tpcds"
PORT="${TPCDS_PORT:-5433}"
MARKER="# cs680 pilot settings"
KIT_URL="https://github.com/gregrahn/tpcds-kit.git"
KIT_COMMIT="5a3a81796992b725c2a8b216767e142609966752"   # TPC-DS tools 2.10.0 as patched there; head, 2020-03-11

[ -x "$PREFIX/bin/postgres" ] || { echo "$PREFIX has no PostgreSQL build; run scripts/setup_postgres.sh first" >&2; exit 1; }
grep -qx "$MARKER" "$TPCH_CONF" || { echo "no '$MARKER' block in $TPCH_CONF" >&2; exit 1; }

# The kit's query parser is generated with yacc and lex. As in
# setup_postgres.sh, missing bison and flex are unpacked into user space.
if ! command -v bison > /dev/null || ! command -v flex > /dev/null; then
  TOOLS="$PILOT_HOME/tools"
  if [ ! -x "$TOOLS/usr/bin/bison" ]; then
    mkdir -p "$TOOLS/debs"
    (cd "$TOOLS/debs" && apt-get download bison flex m4 libfl2 libfl-dev)
    for deb in "$TOOLS"/debs/*.deb; do dpkg -x "$deb" "$TOOLS"; done
  fi
  export PATH="$TOOLS/usr/bin:$PATH"
  export BISON_PKGDATADIR="$TOOLS/usr/share/bison"
  export M4="$TOOLS/usr/bin/m4"
fi

cd "$PILOT_HOME"
# Only tools/ and query_templates/ are needed. A partial clone with a sparse
# checkout downloads about 2 MB instead of the whole repository (28 MB with
# the specification and the answer sets), which matters on a slow link.
if [ ! -d tpcds-kit/.git ]; then
  git clone -q --filter=blob:none --no-checkout "$KIT_URL" tpcds-kit
  git -C tpcds-kit sparse-checkout set --no-cone /tools/ /query_templates/
fi
git -C tpcds-kit checkout -q "$KIT_COMMIT"

KIT="$PILOT_HOME/tpcds-kit/tools"
if [ ! -x "$KIT/dsdgen" ] || [ ! -x "$KIT/dsqgen" ]; then
  # Several kit sources define the same global variable without "extern".
  # GCC 10 and later reject that at link time unless -fcommon restores the
  # old behaviour; "-g -Wall" are the kit's own Linux flags. The kit calls
  # yacc, which the bison package only installs as an alternative.
  make -C "$KIT" -s OS=LINUX YACC="bison -y" LINUX_CFLAGS="-g -Wall -fcommon" \
    > "$PILOT_HOME/tpcds-build.log" 2>&1
fi
usage="$("$KIT/dsdgen" -help 2>&1 || true)"
echo "${usage%%$'\n'*}"   # the version line

cat > "$PILOT_HOME/pg-tpcds.env" <<EOF
export PATH="$PREFIX/bin:\$PATH"
export PGDATA="$PGDATA_DIR"
export PGHOST=/tmp
export PGPORT=$PORT
export PGDATABASE=tpcds
EOF
# shellcheck disable=SC1091
source "$PILOT_HOME/pg-tpcds.env"

block() { sed -n "/^$MARKER\$/,\$p" "$1"; }
if [ ! -f "$PGDATA_DIR/PG_VERSION" ]; then
  # Same initdb options as setup_postgres.sh.
  initdb -D "$PGDATA_DIR" --no-locale -E UTF8 --auth=trust > "$PILOT_HOME/initdb-tpcds.log"
  { echo; block "$TPCH_CONF" | sed "s/^port = .*/port = $PORT/"; } >> "$PGDATA_DIR/postgresql.conf"
fi
if ! diff <(block "$TPCH_CONF" | grep -v '^port = ') <(block "$PGDATA_DIR/postgresql.conf" | grep -v '^port = '); then
  echo "the settings block in $PGDATA_DIR/postgresql.conf differs from $TPCH_CONF (lines above)" >&2
  exit 1
fi
grep -qx "port = $PORT" "$PGDATA_DIR/postgresql.conf" || { echo "port $PORT is not set in $PGDATA_DIR/postgresql.conf" >&2; exit 1; }

pg_ctl -D "$PGDATA_DIR" status > /dev/null 2>&1 || pg_ctl -D "$PGDATA_DIR" -l "$PILOT_HOME/postgres-tpcds.log" -w start
psql -d postgres -Atc "select version()"
psql -d postgres -Atc "select name || ' = ' || current_setting(name) from pg_settings
                       where name in ('port', 'data_directory', 'shared_buffers', 'work_mem',
                                      'maintenance_work_mem', 'effective_cache_size') order by name"
