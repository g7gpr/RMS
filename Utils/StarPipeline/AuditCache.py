import argparse
import psycopg
from pathlib import Path


def getDbList(conn, daycode):
    """Return set of remote filenames from ingest_work for the given day."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT remote_filename
            FROM ingest_work
            WHERE remote_filename LIKE %s
        """, (f"%_{daycode}_%",))
        return {Path(row[0]).name for row in cur.fetchall()}


def getCacheList(cache_root, daycode):
    """Return set of cached archive filenames for the given day."""
    day_dir = Path(cache_root) / daycode
    if not day_dir.exists():
        return set()
    return {p.name for p in day_dir.glob("*_raw.tar.bz2")}


def compareSets(db_set, cache_set):
    print("\n=== Summary ===")
    print(f"DB entries:    {len(db_set)}")
    print(f"Cache entries: {len(cache_set)}")
    normalised_db_set = set()
    for name in db_set:
        if "_detected" in name:
            normalised = name.replace("_detected", "_raw")
        elif "_metadata" in name:
            normalised = name.replace("_metadata", "_raw")
        else:
            normalised = name
        normalised_db_set.add(normalised)

    print(f"\n=== DB but NOT in Cache {len(normalised_db_set - cache_set)} ===")
    for f in sorted(normalised_db_set - cache_set):
        print("  ", f)

    print(f"\n=== Cache but NOT in DB {len(cache_set - normalised_db_set)} ===")
    for f in sorted(cache_set - normalised_db_set):
        print("  ", f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare DB ingest_work entries against local cache for a given day."
    )
    parser.add_argument(
        "daycode",
        help="Day code in YYYYMMDD format"
    )
    parser.add_argument(
        "--cache-root", default="/mnt/rms/cache/RMS_data/CALSTARS",
        help="Local cache root directory"
    )
    parser.add_argument(
        "--db-conn", default="dbname=star_data user=ingest_user host=192.168.217.212",
        help="PostgreSQL connection string"
    )

    args = parser.parse_args()

    conn = psycopg.connect(args.db_conn)

    db_set = getDbList(conn, args.daycode)
    cache_set = getCacheList(args.cache_root, args.daycode)

    compareSets(db_set, cache_set)


if __name__ == "__main__":
    main()
