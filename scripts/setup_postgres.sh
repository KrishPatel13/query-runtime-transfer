#!/usr/bin/env bash
# Build PostgreSQL from source in user space and create the database cluster
# used for all measurements. No root access is needed.
#
# Usage (inside WSL or any Linux shell):
#   bash scripts/setup_postgres.sh
# Then, in every later shell:
#   source ~/cs680-pilot/pg.env
#
# Environment variables, all optional:
#   PILOT_HOME     where the build, the cluster and pg.env go (default ~/cs680-pilot)
#   PG_PREFIX      an existing PostgreSQL installation to use instead of
#                  downloading and building one (e.g. ~/cs680-pilot/pg-18.6)
#   PGPORT         the server's port (default 5432)
#   PG_SECURE=1    settings for a shared server, see below
#   PG_SOCKET_DIR  secure mode: the private socket directory (default $PILOT_HOME/socket)
#   PG_INITDB=0    write pg.env but create no cluster, for a cluster copied
#                  from M1 with scripts/import_cluster.sh
#   PG_SHARED_BUFFERS, PG_WORK_MEM, PG_MAINTENANCE_WORK_MEM, PG_EFFECTIVE_CACHE_SIZE
#                  memory settings of a new cluster. The defaults (2GB, 64MB,
#                  1GB, 8GB) are the measurement settings; change them only
#                  for throwaway test clusters.
#
# Secure mode (PG_SECURE=1). The default cluster trusts every local
# connection, listens on localhost and puts its socket in /tmp. On a shared
# server any logged-in user could then connect as superuser and run shell
# commands as this account (COPY ... TO PROGRAM). Secure mode instead
#   - runs initdb --auth-local=peer --auth-host=reject, so only this OS user
#     can connect, and only through the socket (initdb --auth=peer fails,
#     because peer authentication exists only for local connections);
#   - sets listen_addresses = '' (no TCP socket at all), puts the socket in a
#     private directory created with mode 0700, and sets
#     unix_socket_permissions = 0700;
#   - writes that directory and the port into pg.env as PGHOST and PGPORT;
#   - makes PILOT_HOME mode 0700 and refuses a network file system there: the
#     data directory must be on a local disk, or spills would write temporary
#     files over the network inside the timed region.
#   Example for the UW server (a cluster copied from M1, see import_cluster.sh):
#     PILOT_HOME=/var/tmp/pgstudy PG_SECURE=1 PGPORT=5433 PG_INITDB=0 bash scripts/setup_postgres.sh
set -euo pipefail

PG_VERSION="${PG_VERSION:-18.6}"
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
PREFIX="${PG_PREFIX:-$PILOT_HOME/pg-$PG_VERSION}"
PGDATA_DIR="$PILOT_HOME/pgdata"
PORT="${PGPORT:-5432}"
SECURE="${PG_SECURE:-0}"
INITDB="${PG_INITDB:-1}"
SHARED_BUFFERS="${PG_SHARED_BUFFERS:-2GB}"
WORK_MEM="${PG_WORK_MEM:-64MB}"
MAINTENANCE_WORK_MEM="${PG_MAINTENANCE_WORK_MEM:-1GB}"
EFFECTIVE_CACHE_SIZE="${PG_EFFECTIVE_CACHE_SIZE:-8GB}"
SOCKET_DIR="${PG_SOCKET_DIR:-$PILOT_HOME/socket}"

if [ "$SECURE" = 1 ]; then
  # The path goes into postgresql.conf inside single quotes and must fit the
  # socket name limit, so keep it plain and absolute.
  if [[ ! "$SOCKET_DIR" =~ ^/[A-Za-z0-9/._-]+$ ]] || [ "$SOCKET_DIR" = /tmp ]; then
    echo "PG_SOCKET_DIR must be a private absolute path of letters, digits and / . _ - (got $SOCKET_DIR)" >&2
    exit 1
  fi
  if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
    echo "PGPORT must be a number (got $PORT)" >&2
    exit 1
  fi
  if [ -L "$PILOT_HOME" ] || [ -L "$SOCKET_DIR" ]; then
    echo "PILOT_HOME and PG_SOCKET_DIR must not be symbolic links; use the real directories" >&2
    exit 1
  fi
  # umask 077: the directories are private from the moment they exist.
  (umask 077 && mkdir -p "$PILOT_HOME")
  chmod 700 "$PILOT_HOME"
  fstype="$(stat -f -c %T "$PILOT_HOME")"
  case "$fstype" in
    nfs*|ceph|cifs|smb*|afs|lustre|gpfs|fuse*|9p|v9fs)
      echo "PILOT_HOME=$PILOT_HOME is on a network file system ($fstype); use a local disk such as /var/tmp" >&2
      exit 1 ;;
  esac
  (umask 077 && mkdir -p "$SOCKET_DIR")
  chmod 700 "$SOCKET_DIR"
  if [ "$(stat -c %u "$PILOT_HOME")" != "$(id -u)" ] || [ "$(stat -c %u "$SOCKET_DIR")" != "$(id -u)" ]; then
    echo "$PILOT_HOME and $SOCKET_DIR must belong to $(id -un)" >&2
    exit 1
  fi
fi

if [ -n "${PG_PREFIX:-}" ]; then
  if [ ! -x "$PREFIX/bin/postgres" ]; then
    echo "PG_PREFIX=$PREFIX has no bin/postgres" >&2
    exit 1
  fi
  echo "using the PostgreSQL build in $PREFIX: $("$PREFIX/bin/postgres" --version)"
else
  mkdir -p "$PILOT_HOME/src"
  cd "$PILOT_HOME/src"

  # Since PostgreSQL 17 the release tarballs no longer ship pre-generated parser
  # files, so the build needs bison and flex (and m4, which both call). If they
  # are missing and there is no root access, unpack the distribution packages
  # into user space instead of installing them system-wide.
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

  tarball="postgresql-$PG_VERSION.tar.gz"
  url="https://ftp.postgresql.org/pub/source/v$PG_VERSION/$tarball"
  if [ ! -f "$tarball" ]; then
    curl -fsSLO "$url"
    curl -fsSLO "$url.sha256"
  fi
  sha256sum -c "$tarball.sha256"

  if [ ! -x "$PREFIX/bin/postgres" ]; then
    rm -rf "postgresql-$PG_VERSION"
    tar xzf "$tarball"
    cd "postgresql-$PG_VERSION"
    # readline, zlib and ICU headers are not installed and none of them matter
    # for timing queries. Building without LLVM also means JIT is unavailable,
    # so JIT compile time never shows up in the measured runtimes.
    ./configure --prefix="$PREFIX" --without-readline --without-zlib --without-icu > configure.log
    make -j"$(nproc)" > make.log
    make install > install.log
    cd "$PILOT_HOME/src"
  fi
fi

if [ "$SECURE" = 1 ]; then
  pghost="\"$SOCKET_DIR\""
else
  pghost=/tmp
fi
mkdir -p "$PILOT_HOME"
cat > "$PILOT_HOME/pg.env" <<EOF
export PATH="$PREFIX/bin:\$PATH"
export PGDATA="$PGDATA_DIR"
export PGHOST=$pghost
export PGPORT=$PORT
export PGDATABASE=tpch
EOF
# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"

if [ "$INITDB" != 1 ]; then
  echo "wrote $PILOT_HOME/pg.env and created no cluster (PG_INITDB=$INITDB); copy one in with scripts/import_cluster.sh"
  exit 0
fi

if [ ! -f "$PGDATA_DIR/PG_VERSION" ]; then
  if [ "$SECURE" = 1 ]; then
    initdb -D "$PGDATA_DIR" --no-locale -E UTF8 --auth-local=peer --auth-host=reject > "$PILOT_HOME/initdb.log"
    cat >> "$PGDATA_DIR/postgresql.conf" <<EOF

# cs680 settings for a shared server: no TCP socket, a private Unix socket
port = $PORT
listen_addresses = ''
unix_socket_directories = '$SOCKET_DIR'
unix_socket_permissions = 0700
shared_buffers = $SHARED_BUFFERS
work_mem = $WORK_MEM
maintenance_work_mem = $MAINTENANCE_WORK_MEM
effective_cache_size = $EFFECTIVE_CACHE_SIZE
EOF
  else
    initdb -D "$PGDATA_DIR" --no-locale -E UTF8 --auth=trust > "$PILOT_HOME/initdb.log"
    # Memory settings sized so the SF=1 database stays cached. Planner cost
    # constants are left at their defaults, since PostgreSQL's own cost estimate
    # is the baseline being studied.
    cat >> "$PGDATA_DIR/postgresql.conf" <<EOF

# cs680 pilot settings
port = $PORT
listen_addresses = 'localhost'
unix_socket_directories = '/tmp'
shared_buffers = $SHARED_BUFFERS
work_mem = $WORK_MEM
maintenance_work_mem = $MAINTENANCE_WORK_MEM
effective_cache_size = $EFFECTIVE_CACHE_SIZE
EOF
  fi
fi

if [ "$SECURE" = 1 ]; then
  # Check the settings the server will use (postgres -C reads every
  # configuration file) before it starts, so that an existing cluster made
  # without secure mode is never started here. postgres -C prints
  # unix_socket_permissions in decimal: 448 is 0700.
  for pair in "listen_addresses=" "unix_socket_directories=$SOCKET_DIR" "unix_socket_permissions=448" \
              "port=$PORT"; do
    name="${pair%%=*}"
    want="${pair#*=}"
    got="$(postgres -D "$PGDATA_DIR" -C "$name")"
    if [ "$got" != "$want" ]; then
      echo "$name is '$got' in $PGDATA_DIR, expected '$want'; not starting it" >&2
      exit 1
    fi
  done
  # Every rule must be a peer rule for the socket or a reject rule.
  loose="$(awk '!/^[[:space:]]*(#|$)/ && !(($1 == "local" && $NF == "peer") || $NF == "reject")' \
    "$PGDATA_DIR/pg_hba.conf")"
  if [ -n "$loose" ]; then
    echo "$PGDATA_DIR/pg_hba.conf allows more than peer connections through the socket; not starting it:" >&2
    echo "$loose" >&2
    exit 1
  fi
fi

pg_ctl status > /dev/null 2>&1 || pg_ctl -l "$PILOT_HOME/postgres.log" -w start
psql -d postgres -Atc "select version()"
if [ "$SECURE" = 1 ]; then
  psql -d postgres -Atc "select 'hba: ' || type || ' ' || array_to_string(database, ',') || ' ' ||
                                array_to_string(user_name, ',') || ' ' || coalesce(address, '') || ' ' || auth_method
                         from pg_hba_file_rules order by rule_number"
fi
