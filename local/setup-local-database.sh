#!/bin/sh
# KN-FAPI-07: one-time setup of Knowledge's database in a Postgres running on this machine (localhost:5432).
# Not needed with the shared container (infrastructure/local-postgres/): its first start creates the same database and roles.
# Use this only for a native Postgres on the host.
#
# Creates the `knowledge` database if missing and the login roles the app and the index command connect as
# (local/init/10-roles.sh). It does not run the migrations: `docker compose up -d` does, as the administrative user.
# Safe to run again: nothing is changed when it all exists. Passwords come from .env.local, never from this file.
#
# Needs: psql on PATH, the server running, and the pgvector extension (>= 0.8.0) installed in that server.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$HERE/../.env.local"
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE. Copy .env.local.example to .env.local and set the values." >&2; exit 1; }
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

: "${KNOWLEDGE_DB_ADMIN_USER:?KNOWLEDGE_DB_ADMIN_USER is not set in .env.local}"
: "${KNOWLEDGE_DB_RETRIEVE_PASSWORD:?KNOWLEDGE_DB_RETRIEVE_PASSWORD is not set in .env.local}"
: "${KNOWLEDGE_DB_INGEST_PASSWORD:?KNOWLEDGE_DB_INGEST_PASSWORD is not set in .env.local}"

export PGHOST="${KNOWLEDGE_DB_SETUP_HOST:-localhost}" PGPORT="${KNOWLEDGE_DB_SETUP_PORT:-5432}"
export PGUSER="$KNOWLEDGE_DB_ADMIN_USER" PGPASSWORD="${KNOWLEDGE_DB_ADMIN_PASSWORD:-}"
export POSTGRES_USER="$PGUSER" POSTGRES_DB=knowledge

command -v psql >/dev/null || { echo "psql is not on PATH." >&2; exit 1; }
pg_isready -q || { echo "No Postgres answers on $PGHOST:$PGPORT." >&2; exit 1; }

version="$(psql -d postgres -Atc "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")"
[ -n "$version" ] || { echo "The pgvector extension is not installed in this Postgres (see docs/instructions/knowledge_service_build_test_deploy.md)." >&2; exit 1; }
lowest="$(printf '%s\n%s\n' 0.8.0 "$version" | sort -V | head -n 1)"
[ "$lowest" = "0.8.0" ] || { echo "pgvector $version is too old: hnsw.iterative_scan needs 0.8.0 or newer." >&2; exit 1; }

if [ -z "$(psql -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname = 'knowledge'")" ]; then
  psql -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE knowledge"
  echo "Created database knowledge."
fi

sh "$HERE/init/10-roles.sh"
echo "Ready: database knowledge, roles kn_app_retrieve and kn_app_ingest. Next: docker compose up -d"
