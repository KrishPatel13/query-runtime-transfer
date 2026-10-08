#!/usr/bin/env bash
# Start a copy of M1's database cluster on a shared server (M3), so that it
# runs on the same database files as M1 and no other user can connect.
#
# The copy comes from scripts/export_cluster.sh on M1: the stopped data
# directory as pgdata-<date>.tar.zst (or .tar.gz) with a .sha256 file next to
# it. A fresh TPC-H load would run ANALYZE on a new random sample, which can
# change estimates and plans; the copy keeps the data, the statistics and so
# the plans identical.
#
# Usage on the UW server, with the archive and its .sha256 copied to local disk:
#   PILOT_HOME=/var/tmp/pgstudy PG_SECURE=1 PGPORT=5433 PG_INITDB=0 bash scripts/setup_postgres.sh
#   PILOT_HOME=/var/tmp/pgstudy bash scripts/import_cluster.sh /var/tmp/pgstudy/pgdata-<date>.tar.zst
# The first command builds PostgreSQL and writes a secure pg.env (private
# socket directory as PGHOST, the port as PGPORT) without creating a cluster.
# This script then
#   - reads PGDATA, PGHOST and PGPORT from $PILOT_HOME/pg.env and checks that
#     PGDATA is $PILOT_HOME/pgdata, that PILOT_HOME is on a local disk (spills
#     would otherwise write temporary files over the network inside the timed
#     region), and that PILOT_HOME and the socket directory have mode 0700
#     and belong to this user;
#   - verifies the archive against its SHA-256 file;
#   - extracts it into a temporary directory in PILOT_HOME and checks that it
#     holds one data directory of the PostgreSQL major version on PATH, shut
#     down cleanly;
#   - replaces pg_hba.conf with the single line "local all all peer": only
#     through the socket, and only as the database role named like the OS
#     user (M1's file trusts every local connection);
#   - appends port, listen_addresses = '' (no TCP socket), the socket
#     directory and unix_socket_permissions = 0700 to postgresql.conf.
#     PostgreSQL uses the last value of a setting, so these replace M1's
#     connection settings while every memory and planner setting stays as on
#     M1; postgres -C then confirms the values the server will use;
#   - adds a superuser role named like this OS user if the copy has none. M1's
#     cluster has only the role of the OS user that created it, and peer
#     authentication needs a role with the local user's name. The role is
#     created in single-user mode, which needs no connection; roles do not
#     change the data, the statistics or any setting;
#   - moves the copy to $PILOT_HOME/pgdata with mode 0700, starts it, and
#     prints the settings that matter as SHOW reports them.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: PILOT_HOME=<dir> bash scripts/import_cluster.sh <archive>.tar.zst|.tar.gz" >&2
  exit 2
fi
archive="$(realpath "$1")"
PILOT_HOME="${PILOT_HOME:-$HOME/cs680-pilot}"
if [ ! -f "$PILOT_HOME/pg.env" ]; then
  echo "no $PILOT_HOME/pg.env; write it first with" >&2
  echo "  PILOT_HOME=$PILOT_HOME PG_SECURE=1 PGPORT=<port> PG_INITDB=0 bash scripts/setup_postgres.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$PILOT_HOME/pg.env"

# ---------------------------------------------------------------- checks
if [ "$PGDATA" != "$PILOT_HOME/pgdata" ]; then
  echo "PGDATA is $PGDATA in $PILOT_HOME/pg.env, expected $PILOT_HOME/pgdata" >&2
  exit 1
fi
if [[ ! "$PGHOST" =~ ^/[A-Za-z0-9/._-]+$ ]] || [ "$PGHOST" = /tmp ] || [ ! -d "$PGHOST" ] || [ -L "$PGHOST" ]; then
  echo "PGHOST=$PGHOST in $PILOT_HOME/pg.env is not a private socket directory; write pg.env with PG_SECURE=1" >&2
  exit 1
fi
if [[ ! "$PGPORT" =~ ^[0-9]+$ ]]; then
  echo "PGPORT=$PGPORT in $PILOT_HOME/pg.env is not a port number" >&2
  exit 1
fi
for d in "$PILOT_HOME" "$PGHOST"; do
  if [ -L "$d" ] || [ "$(stat -c %u "$d")" != "$(id -u)" ] || [ "$(stat -c %a "$d")" != 700 ]; then
    echo "$d must be a directory of $(id -un) with mode 700 (setup_postgres.sh with PG_SECURE=1 makes it so)" >&2
    exit 1
  fi
done
fstype="$(stat -f -c %T "$PILOT_HOME")"
case "$fstype" in
  nfs*|ceph|cifs|smb*|afs|lustre|gpfs|fuse*|9p|v9fs)
    echo "PILOT_HOME=$PILOT_HOME is on a network file system ($fstype); use a local disk such as /var/tmp" >&2
    exit 1 ;;
esac
if [ -e "$PGDATA" ]; then
  echo "$PGDATA already exists; stop that cluster and move it away first" >&2
  exit 1
fi
me="$(id -un)"
if [[ ! "$me" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
  echo "unexpected user name '$me'" >&2
  exit 1
fi

# -------------------------------------------------------------- checksum
sumfile="$archive.sha256"
if [ ! -f "$sumfile" ]; then
  echo "no $sumfile next to the archive; copy it along with the archive" >&2
  exit 1
fi
read -r expected listed < "$sumfile"
if [ "${listed#\*}" != "$(basename "$archive")" ]; then
  echo "$sumfile is the checksum of ${listed#\*}, not of $(basename "$archive")" >&2
  exit 1
fi
echo "checking $(basename "$archive") against $(basename "$sumfile")"
actual="$(sha256sum "$archive" | cut -d' ' -f1)"
if [ "$actual" != "$expected" ]; then
  echo "checksum mismatch: the archive has $actual, $sumfile says $expected" >&2
  exit 1
fi
echo "checksum OK: $actual"

# --------------------------------------------------------------- extract
umask 077
tmp="$(mktemp -d "$PILOT_HOME/import.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
case "$archive" in
  *.tar.zst)
    if ! command -v zstd > /dev/null; then
      echo "zstd is needed to unpack $archive" >&2
      exit 1
    fi
    zstd -dc "$archive" | tar -C "$tmp" -xf - ;;
  *.tar.gz|*.tgz)
    tar -C "$tmp" -xzf "$archive" ;;
  *)
    echo "unknown archive type: $archive (expected .tar.zst or .tar.gz)" >&2
    exit 1 ;;
esac
mapfile -t top < <(ls -A "$tmp")
if [ "${#top[@]}" -ne 1 ] || [ ! -f "$tmp/${top[0]}/PG_VERSION" ]; then
  echo "the archive must hold exactly one data directory; it holds: ${top[*]}" >&2
  exit 1
fi
new="$tmp/${top[0]}"
chmod 700 "$new"

data_major="$(cat "$new/PG_VERSION")"
bin_major="$(postgres --version | sed -E 's/^[^0-9]*([0-9]+).*/\1/')"
if [ "$data_major" != "$bin_major" ]; then
  echo "the copy is a PostgreSQL $data_major cluster, but postgres on PATH is $(postgres --version)" >&2
  exit 1
fi
if [ -e "$new/postmaster.pid" ]; then
  echo "the copy has a postmaster.pid, so it was archived from a running server; export it again" >&2
  exit 1
fi
state="$(pg_controldata "$new" | sed -n 's/^Database cluster state: *//p')"
if [ "$state" != "shut down" ]; then
  echo "the copy's cluster state is '$state', not 'shut down'; export it again" >&2
  exit 1
fi
echo "system identifier: $(pg_controldata "$new" | sed -n 's/^Database system identifier: *//p')"

# ------------------------------------------------------ secure settings
printf 'local all all peer\n' > "$new/pg_hba.conf"
cat >> "$new/postgresql.conf" <<EOF

# cs680: settings for a shared server, appended by scripts/import_cluster.sh
# on $(date -Iseconds). The last value of a setting wins, so these replace
# the connection settings above; every other setting is M1's.
port = $PGPORT
listen_addresses = ''
unix_socket_directories = '$PGHOST'
unix_socket_permissions = 0700
EOF
# postgres -C reads every configuration file, postgresql.auto.conf included,
# and prints unix_socket_permissions in decimal: 448 is 0700.
for pair in "listen_addresses=" "unix_socket_directories=$PGHOST" "unix_socket_permissions=448" \
            "port=$PGPORT"; do
  name="${pair%%=*}"
  want="${pair#*=}"
  got="$(postgres -D "$new" -C "$name")"
  if [ "$got" != "$want" ]; then
    echo "$name would be '$got', expected '$want' (set in postgresql.auto.conf?); not starting the copy" >&2
    exit 1
  fi
done

# Peer authentication needs a role named like this OS user. Single-user mode
# runs as the bootstrap superuser, whatever the OS user is called.
# A failed statement does not change its exit status, so the log is checked;
# log_min_messages=notice lets the notice below into it.
log="$PILOT_HOME/import_single_user.log"
echo "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$me') THEN CREATE ROLE \"$me\" SUPERUSER LOGIN; RAISE NOTICE 'created role $me'; END IF; END \$\$;" \
  | postgres --single -c log_min_messages=notice -D "$new" postgres > "$log" 2>&1 \
  || { echo "adding role $me failed; see $log" >&2; exit 1; }
if grep -qE "ERROR|FATAL" "$log"; then
  echo "adding role $me failed; see $log" >&2
  exit 1
fi
if grep -q "created role $me" "$log"; then
  echo "added superuser role $me for peer authentication"
fi

mv "$new" "$PGDATA"
chmod 700 "$PGDATA"

# ----------------------------------------------------------------- start
pg_ctl -D "$PGDATA" -l "$PILOT_HOME/postgres.log" -w start
echo "settings (SHOW):"
psql -d postgres -XAt -F ' = ' -c "
  select name, current_setting(name)
  from unnest(array['server_version', 'data_directory', 'port', 'listen_addresses', 'unix_socket_directories',
                    'unix_socket_permissions', 'shared_buffers', 'work_mem', 'maintenance_work_mem',
                    'effective_cache_size', 'max_parallel_workers_per_gather', 'max_parallel_workers',
                    'max_worker_processes', 'jit', 'seq_page_cost', 'random_page_cost', 'cpu_tuple_cost',
                    'cpu_index_tuple_cost', 'cpu_operator_cost', 'parallel_setup_cost', 'parallel_tuple_cost',
                    'effective_io_concurrency', 'default_statistics_target']) with ordinality as s(name, i)
  order by i"
echo "authentication rules:"
psql -d postgres -XAtc "select type || ' ' || array_to_string(database, ',') || ' ' ||
                               array_to_string(user_name, ',') || ' ' || auth_method
                        from pg_hba_file_rules order by rule_number"
echo "databases:"
psql -d postgres -XAt -F ' ' -c "select datname, pg_size_pretty(pg_database_size(datname))
                                 from pg_database where not datistemplate order by datname"
