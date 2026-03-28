#!/bin/bash
set -e

PGHOST="192.168.1.212"
PGUSER="postgres"

echo "Dropping existing star_data database (if it exists)..."
psql -h "$PGHOST" -U "$PGUSER" -d postgres -c "DROP DATABASE IF EXISTS star_data;"

echo "Creating fresh star_data database..."
psql -h "$PGHOST" -U "$PGUSER" -d postgres -c "CREATE DATABASE star_data OWNER postgres;"

echo "Recreating public schema and applying privileges..."
psql -h "$PGHOST" -U "$PGUSER" -d star_data <<'EOF'
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public AUTHORIZATION postgres;

GRANT USAGE, CREATE ON SCHEMA public TO ingest_user;
GRANT CREATE ON DATABASE star_data TO ingest_user;
EOF

echo "Ensuring ingest_user exists and has correct privileges..."

psql -h "$PGHOST" -U "$PGUSER" -v ON_ERROR_STOP=1 <<'EOF'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_roles WHERE rolname = 'ingest_user'
    ) THEN
        CREATE ROLE ingest_user LOGIN PASSWORD 'ingest_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE star_data TO ingest_user;
GRANT USAGE ON SCHEMA public TO ingest_user;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO ingest_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO ingest_user;
EOF


echo "Creating readonly user..."

psql -h "$PGHOST" -U "$PGUSER" -v ON_ERROR_STOP=1 <<'EOF'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_roles WHERE rolname = 'readonly'
    ) THEN
        CREATE ROLE readonly LOGIN PASSWORD 'readonly_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE star_data TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO readonly;

ALTER ROLE readonly SET default_transaction_read_only = on;
EOF


echo "Database reset complete."
