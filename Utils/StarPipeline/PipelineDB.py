from RMS.Logger import LoggingManager, getLogger
import RMS.ConfigReader as cr
import os
import psycopg

import socket



# Constants


# urls
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"

# Paths and names
CALSTARS_DATA_DIR = "CALSTARS"
PLATEPARS_ALL_RECALIBRATED_JSON = "platepars_all_recalibrated.json"
DIRECTORY_INGESTED_MARKER = ".processed"
FILE_SYSTEM_MARKERS_ENABLED = False
CALSTAR_FILES_TABLE_NAME = "calstar_files"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
CHARTS = "charts"
PORT = 22

# Clamp to Postgres INTEGER storage range (magnitudes × 1e6)
PG_INT_MAG_LIMIT = 2147.483647
PG_BIGINT_MAG_LIMIT = 9.22e12


config = cr.parse(os.path.join(os.getcwd(), ".config"))

# Initialize the logger
log_manager = LoggingManager()
log_manager.initLogging(config)

# Get the logger handle
log = getLogger("rmslogger")


class Flags:
    BAD_AUTO_PP = 1 << 0
    BAD_MAD = 1 << 1
    FEW_STARS = 1 << 2
    MOON_IN_FOV = 1 << 3
    SKY_NOT_FULLY_DARK = 1 << 4
    SPLINE_NOT_BUILT = 1 << 5

    NAMES = {
        BAD_AUTO_PP: "bad auto platepar",
        BAD_MAD: "bad median absolute stellar difference",
        FEW_STARS: "too few stars",
        MOON_IN_FOV: "illuminated moon close to fov",
        SKY_NOT_FULLY_DARK: "dusk or dawn",
        SPLINE_NOT_BUILT: "could not build a spline on x, y, r"}

    @classmethod
    def decode(cls, value):
        """Return a list of flag names set in the given integer."""
        return [name for bit, name in cls.NAMES.items() if value & bit]



def createStationTable(conn):
    sql = """
          CREATE TABLE IF NOT EXISTS station \
          ( \
              station_name \
              CHAR \
          ( \
              6 \
          ) PRIMARY KEY,
              name TEXT,
              notes TEXT
              ); \
          """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createCalstarFilesTable(conn):
    sql = """
          CREATE TABLE IF NOT EXISTS calstar_files \
          ( \
              file_name \
              TEXT \
              PRIMARY \
              KEY, \
              ingestion_time \
              BIGINT
          ); \
          """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createSessionTable(conn):
    sql = """CREATE TABLE IF NOT EXISTS session \
             ( \
                 session_id \
                 SERIAL \
                 PRIMARY \
                 KEY, \
                 session_name \
                 TEXT \
                 NOT \
                 NULL \
                 UNIQUE, \
                 station_name \
                 TEXT \
                 NOT \
                 NULL, \

                 start_jd \
                 BIGINT, \
                 end_jd \
                 BIGINT, \

                 pixel_scale_h \
                 INTEGER, \
                 pixel_scale_v \
                 INTEGER, \

                 lat \
                 INTEGER, \
                 lon \
                 INTEGER, \
                 elevation \
                 INTEGER \
             );"""

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createFrameTable(conn):
    sql = """
          CREATE TABLE IF NOT EXISTS frame \
          ( \
              frame_name \
              TEXT \
              PRIMARY \
              KEY, \
              session_name \
              TEXT \
              REFERENCES \
              session \
          ( \
              session_name \
          ),
              jd_mid BIGINT,
              frame_index INTEGER,
              quality_flags SMALLINT,
              mad INTEGER
              ); \
          """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createStarTable(conn):
    sql = """
          CREATE TABLE IF NOT EXISTS star \
          ( \
              station_name \
              TEXT \
              NOT \
              NULL \
              REFERENCES \
              station \
          ( \
              station_name \
          ),
              star_name TEXT NOT NULL,
              ra INTEGER,
              dec INTEGER,
              mag INTEGER,
              catalog_source TEXT,
              canonical_name TEXT,
              PRIMARY KEY \
          ( \
              station_name, \
              star_name \
          )
              ); \
          """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createObservationTable(conn):
    sql = """
          CREATE TABLE IF NOT EXISTS observation \
          ( \
              obs_id \
              BIGSERIAL \
              PRIMARY \
              KEY, \
              jd_mid \
              BIGINT, \
              session_name \
              TEXT \
              REFERENCES \
              session \
          ( \
              session_name \
          ),
              station_name TEXT,

              frame_name TEXT REFERENCES frame \
          ( \
              frame_name \
          ),
              star_name TEXT,

              -- CALSTARS fields (scaled where needed)
              y INTEGER,
              x INTEGER,
              intens_sum INTEGER,
              ampltd INTEGER,
              fwhm INTEGER,
              bg_lvl INTEGER,
              snr INTEGER,
              nsatpx SMALLINT,

              -- Derived fields
              mag INTEGER,
              obs_mag_corrected INTEGER,
              cat_mag INTEGER,
              mag_err INTEGER,
              mag_cor INTEGER,
              sun_angle INTEGER,
              mean_curvature INTEGER,
              max_curvature INTEGER,

              -- Astrometric solution (scaled RA/Dec)
              ra INTEGER,
              dec INTEGER,

              -- Flags
              flags SMALLINT,
              
              
              -- Median absolute deviation
              mad INTEGER
              ); \
          """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


def createSpatialModelTable(conn):
    """
    Create the spatial_model table if it does not exist.
    This table stores multiple spatial correction models per frame.
    """

    sql_create = """
                      CREATE TABLE IF NOT EXISTS spatial_model (
                        frame_name      TEXT NOT NULL,
                        model_type      TEXT NOT NULL,
                        version         INTEGER NOT NULL DEFAULT 1,
                    
                        -- Model payload
                        grid_mag        JSONB,
                        xmid            JSONB,
                        ymid            JSONB,
                        params          JSONB,
                    
                        -- Diagnostics
                        n_points        INTEGER,
                        rms_mag         DOUBLE PRECISION,
                        median_resid    DOUBLE PRECISION,
                    
                        created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                    
                        PRIMARY KEY (frame_name, model_type, version)
                    );            """
    print("SQL to create spatial model")
    log.info(sql_create)

    # Optional: enforce known model types
    sql_check_exists = """
                       SELECT 1
                       FROM pg_constraint
                       WHERE conname = 'spatial_model_type_check'; \
                       """

    sql_add_check = """
                    ALTER TABLE spatial_model
                        ADD CONSTRAINT spatial_model_type_check
                            CHECK (model_type IN ('binned', 'gaussian', 'tps', 'none')); \
                    """

    with conn.cursor() as cur:
        cur.execute(sql_create)

        # Add CHECK constraint only if missing
        cur.execute(sql_check_exists)
        exists = cur.fetchone()

        if not exists:
            cur.execute(sql_add_check)

    conn.commit()


def createObservationIndexes(conn):
    with conn.cursor() as cur:
        # JD-only index (already have)
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_jd_mid
                        ON observation (jd_mid);
                    """)

        # JD + RA + DEC (already have)
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_jd_ra_dec
                        ON observation (jd_mid, ra, dec);
                    """)

        # Frame-level lookups (CRITICAL)
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_frame_name
                        ON observation (frame_name);
                    """)

        # Station-level lookups
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_station_name
                        ON observation (station_name);
                    """)

        # Star-level lookups (optional but useful)
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_star_name
                        ON observation (star_name);
                    """)

        conn.commit()



def createIngestWorkTable(conn):
        ddl = """
        CREATE TABLE IF NOT EXISTS ingest_work (
            remote_filename     TEXT PRIMARY KEY,
            jd_int          BIGINT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            claimed_by      TEXT,
            claimed_at      TIMESTAMPTZ,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

def markJobDone(conn, remote_filename):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_work
            SET status = 'done', updated_at = now()
            WHERE remote_filename = %s
            """,
            (remote_filename,)
        )
    conn.commit()


def markJobError(conn, remote_filename, msg):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_work
            SET status = 'error', updated_at = now()
            WHERE remote_filename = %s
            """,
            (remote_filename,)
        )
    conn.commit()

def jobsRemaining(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM ingest_work
            WHERE status = 'pending';
        """)
        (count,) = cur.fetchone()
        return count


def claimNextJob(conn):
    """
    Atomically claim the next pending job.
    Records hostname and timestamp.
    Returns remote_filename or None if no work is available.
    """
    sql = """
        UPDATE ingest_work AS w
        SET status = 'claimed',
            claimed_by = %s,
            claimed_at = now(),
            updated_at = now()
        FROM (
            SELECT remote_filename, jd_int
            FROM ingest_work
            WHERE status = 'pending'
            ORDER BY jd_int, remote_filename
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        ) AS sub
        WHERE w.remote_filename = sub.remote_filename
        RETURNING w.remote_filename, w.jd_int;

    """

    with conn.cursor() as cur:
        host_name = socket.gethostname()
        cur.execute(sql, (host_name, ))
        row = cur.fetchone()
        conn.commit()


    return row if row else None

def resetStalledJobs(conn, this_machine=False):
    """
    Reset any jobs that have been claimed for more than 30 minutes.
    These are considered stalled and returned to the pending queue.
    """

    params = []
    sql = """
          UPDATE ingest_work
          SET status     = 'pending',
              claimed_by = NULL,
              claimed_at = NULL,
              updated_at = now()
          WHERE status = 'claimed' """

    if this_machine:
        host_name = socket.gethostname()
        sql += f" AND claimed_by = %s;"
        params.append(host_name)
    else:
        sql +=       "AND claimed_at < now() - interval '30 minutes';"

    log.info("Resetting stalled jobs")
    log.info(sql)

    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def createRejectedFrameTable(conn):
    sql = """
            CREATE TABLE IF NOT EXISTS rejected_frame (
    frame_name      TEXT PRIMARY KEY,
    station_name    TEXT,
    timestamp       TIMESTAMPTZ,
    reason          TEXT,
    tps_error       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
    CREATE INDEX IF NOT EXISTS idx_rejected_frame_name
    ON rejected_frame(frame_name);


    

          """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()



def createAllTables(conn):
    createStationTable(conn)
    createSessionTable(conn)
    createFrameTable(conn)
    createStarTable(conn)
    createObservationTable(conn)
    #createCalstarFilesTable(conn)
    #createSpatialModelTable(conn)
    createIngestWorkTable(conn)
    createRejectedFrameTable(conn)


def createAllIndexes(conn):
    createObservationIndexes(conn)


def revokeCreatesIngestUser(conn):
    with conn.cursor() as cur:
        cur.execute("REVOKE CREATE ON DATABASE star_data FROM ingest_user;")
        cur.execute("REVOKE CREATE ON SCHEMA public FROM ingest_user;")
    conn.commit()


def createIngestUserIfMissing(conn):
    with conn.cursor() as cur:
        # Check if role exists
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname='ingest_user';")
        exists = cur.fetchone()

        if not exists:
            # Create the role WITHOUT a password
            # Operator sets the password manually once
            cur.execute("CREATE ROLE ingest_user LOGIN;")

    conn.commit()


def grantIngestUserPrivileges(conn):
    with conn.cursor() as cur:
        # Database + schema access
        cur.execute("GRANT CONNECT ON DATABASE star_data TO ingest_user;")
        cur.execute("GRANT USAGE ON SCHEMA public TO ingest_user;")

        # Table privileges
        cur.execute("""
            GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO ingest_user;
        """)

        # Future tables
        cur.execute("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE ON TABLES TO ingest_user;
        """)

    conn.commit()


def auditIngestUserPrivileges(conn):
    print("\n=== INGEST USER PRIVILEGE AUDIT ===")

    with conn.cursor() as cur:
        # 1. Who am I?
        cur.execute("SELECT current_user;")
        print("Current user:", cur.fetchone()[0])

        # 2. What server am I connected to?
        cur.execute("SELECT inet_server_addr(), inet_server_port();")
        print("Connected to:", cur.fetchone())

        # 3. What is my search_path?
        cur.execute("SHOW search_path;")
        print("search_path:", cur.fetchone()[0])

        # 4. What schemas exist?
        cur.execute("""
                    SELECT schema_name
                    FROM information_schema.schemata
                    ORDER BY schema_name;
                    """)
        schemas = [row[0] for row in cur.fetchall()]
        print("Schemas on server:", schemas)

        # 5. Does a schema named after the user exist?
        cur.execute("""
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name = current_user;
                    """)
        user_schema = cur.fetchone()
        print("User-named schema exists:", bool(user_schema))

        # 6. What privileges does ingest_user have on each table?
        print("\nTable privileges:")
        cur.execute("""
                    SELECT table_schema, table_name, privilege_type
                    FROM information_schema.table_privileges
                    WHERE grantee = current_user
                    ORDER BY table_schema, table_name, privilege_type;
                    """)
        rows = cur.fetchall()
        if not rows:
            print("  (No table privileges found!)")
        else:
            for schema, table, priv in rows:
                print(f"  {schema}.{table}: {priv}")

        # 7. Can ingest_user SELECT from calstar_files?
        print("\nTesting SELECT on public.calstar_files...")
        try:
            cur.execute("SELECT COUNT(*) FROM public.calstar_files;")
            print("  SELECT OK:", cur.fetchone()[0], "rows")
        except Exception as e:
            print("  SELECT FAILED:", e)

        # 8. Can ingest_user INSERT into calstar_files?
        print("\nTesting INSERT on public.calstar_files...")
        try:
            cur.execute("""
                        INSERT INTO public.calstar_files (file_name, ingestion_time)
                        VALUES ('audit_test', 0)
                        """)
            conn.rollback()  # Don't leave junk
            print("  INSERT OK")
        except Exception as e:
            print("  INSERT FAILED:", e)

        cur.execute("SELECT inet_server_addr(), inet_server_port();")
        log.warning("SERVER: %s", cur.fetchone())

    print("=== END AUDIT ===\n")


def ensureCalstarFilePrivileges(conn):
    """Ensure ingest_user has the privileges required for ON CONFLICT DO UPDATE."""
    with conn.cursor() as cur:
        cur.execute("GRANT INSERT ON public.calstar_files TO ingest_user;")
        cur.execute("GRANT SELECT ON public.calstar_files TO ingest_user;")
        cur.execute("GRANT UPDATE (ingestion_time) ON public.calstar_files TO ingest_user;")
    conn.commit()


def grantSequencePrivileges(conn):
    with conn.cursor() as cur:
        cur.execute("GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO ingest_user;")
    conn.commit()


def setIngestUserSearchPath(conn):
    with conn.cursor() as cur:
        cur.execute("ALTER ROLE ingest_user SET search_path = public;")
    conn.commit()


def createDatabaseIfMissing(conn):
    # Connect to the default database

    conn.autocommit = True  # REQUIRED for CREATE DATABASE

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname='star_data';")
        exists = cur.fetchone()

        if not exists:
            cur.execute("CREATE DATABASE star_data;")


def initialiseDatabase(conn):
    createIngestUserIfMissing(conn)
    setIngestUserSearchPath(conn)
    createAllTables(conn)
    createAllIndexes(conn)
    grantIngestUserPrivileges(conn)
    ensureCalstarFilePrivileges(conn)
    grantSequencePrivileges(conn)
    revokeCreatesIngestUser(conn)

    pass

def resetDatabaseForReingest(conn):
    """
    Reset all derived and metadata tables while preserving ingest_work entries.
    - Truncate observation, frame, session, rejected_frame.
    - Truncate station and star.
    - Reset ingest_work.status back to 'pending'.
    """

    # Tables we want to fully clear
    tables_to_truncate = [
        "observation",
        "frame",
        "session",
        "rejected_frame",
        "station",
        "star"
    ]

    with conn.cursor() as cur:

        # 1. Wipe all selected tables
        for tbl in tables_to_truncate:
            cur.execute(f"TRUNCATE TABLE {tbl} RESTART IDENTITY CASCADE;")

        # 2. Reset ingest_work statuses
        cur.execute("""
            UPDATE ingest_work
            SET status = 'pending',
                claimed_by = NULL,
                claimed_at = NULL,
                updated_at = now()
            WHERE status != 'pending';
        """)

    conn.commit()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Pipeline DB Utilities""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('postgresql_host', help="""PostgreSQL server host """)

    arg_parser.add_argument('-r', '--reset_ingestion', dest='reset_ingestion', default=False, action="store_true",
                            help="Reset ingestion")
    cml_args = arg_parser.parse_args()

    reset_ingestion = cml_args.reset_ingestion
    postgresql_host = cml_args.postgresql_host

    if cml_args.reset_ingestion:
        log.info("Resetting ingestion")
        with psycopg.connect(host=postgresql_host, dbname="star_data", user="postgres") as reset_ingestion_conn:
            createDatabaseIfMissing(reset_ingestion_conn)
            initialiseDatabase(reset_ingestion_conn)
            resetDatabaseForReingest(reset_ingestion_conn)