import argparse
import psycopg
from pathlib import Path
from PipelineDB import extractStub
from Ingest import sortFilesByTime

def getDbList(conn, daycode):
    """Return set of remote filenames from ingest_work for the given day."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT remote_filename
            FROM ingest_work
            WHERE status='done' AND remote_filename LIKE %s
        """, (f"%_{daycode}_%",))
        return {Path(row[0]).name for row in cur.fetchall()}


def getCacheList(cache_root, daycode):
    """Return set of cached archive filenames for the given day."""
    day_dir = Path(cache_root) / daycode
    if not day_dir.exists():
        return set()
    raw_set = {p.name for p in day_dir.glob("*_raw.tar.bz2")}
    dir_set = {p.name for p in day_dir.iterdir() if p.is_dir()}

    return raw_set | dir_set


def compareSets(db_set, cache_set):
    print("\n=== Summary ===")
    print(f"DB entries:    {len(db_set)}")
    print(f"Cache entries: {len(cache_set)}")



    print(f"\n=== DB but NOT in Cache {len(db_set - cache_set)} ===")
    missing_from_cache_list = sortFilesByTime(db_set - cache_set)
    for f in missing_from_cache_list:
        print("  ", f)

    print(f"\n=== Cache but NOT in DB {len(cache_set - db_set)} ===")
    missing_from_db_list = sortFilesByTime(cache_set - db_set)
    for f in missing_from_db_list:
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
    db_stubs_set = { extractStub(remoteFilename) for remoteFilename in db_set}
    cache_stubs_set = {extractStub(cacheFilename) for cacheFilename in cache_set}

    compareSets(db_stubs_set, cache_stubs_set)


if __name__ == "__main__":
    main()
