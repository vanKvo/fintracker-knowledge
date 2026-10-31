#!/bin/sh
# KN-FAPI-07: the login roles the local app connects as, created on the database's first start.
#
# Each login role is a member of one NOLOGIN role from migration V3 (kn_retrieve, kn_ingest), so it inherits exactly
# that role's grants and the Row-Level Security policies written for it. They are never superusers and never BYPASSRLS:
# a superuser would skip RLS locally and fail in AWS. Passwords come from the environment (.env.local), never from this file.
#
# The NOLOGIN roles are created here only if missing; migration V3 creates them with the same IF NOT EXISTS check and
# grants them their table privileges later, so the order of this script and Flyway does not matter.
# Not `set -e`: the image sources scripts that are not executable. Each failure below stops through ON_ERROR_STOP.

: "${KNOWLEDGE_DB_RETRIEVE_PASSWORD:?KNOWLEDGE_DB_RETRIEVE_PASSWORD is not set}"
: "${KNOWLEDGE_DB_INGEST_PASSWORD:?KNOWLEDGE_DB_INGEST_PASSWORD is not set}"

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER:-postgres}" --dbname "${POSTGRES_DB:-knowledge}" \
  -v retrieve_password="$KNOWLEDGE_DB_RETRIEVE_PASSWORD" -v ingest_password="$KNOWLEDGE_DB_INGEST_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I NOLOGIN', r) FROM unnest(ARRAY['kn_retrieve', 'kn_ingest']) AS r
 WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) \gexec

SELECT format('CREATE ROLE kn_app_retrieve LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE IN ROLE kn_retrieve', :'retrieve_password')
 WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kn_app_retrieve') \gexec

SELECT format('CREATE ROLE kn_app_ingest LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE IN ROLE kn_ingest', :'ingest_password')
 WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kn_app_ingest') \gexec
SQL
