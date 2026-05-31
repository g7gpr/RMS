import argparse
import psycopg
from Utils.StarPipeline.Ingest import getRemoteFileList
from RMS.Logger import LoggingManager, getLogger
from Utils.StarPipeline.PipelineDB import repairMissingCacheEntries, resetStalledJobs
from Utils.StarPipeline.SortCALSTARCache import refileArchives, createDayArchives
from Utils.StarPipeline.Ingest import createDatabaseIfMissing, initialiseDatabase

if __name__ == "__main__":



    # Initialize the logger
    log_manager = LoggingManager()

    # Get the logger handle
    log = getLogger("rmslogger")

    parser = argparse.ArgumentParser(description="Undertake routine housekeeping on the star-pipeline")

    parser.add_argument("--cache-root", default="/mnt/rms/cache/RMS_data/CALSTARS", help="Local cache root directory")

    parser.add_argument(   "--db-conn", default="dbname=star_data user=postgres host=192.168.217.212",help="PostgreSQL connection string")

    cml_args = parser.parse_args()
    conn = psycopg.connect(cml_args.db_conn)
    cache_root = cml_args.cache_root


    createDatabaseIfMissing(conn)
    initialiseDatabase(conn)


    getRemoteFileList(log, "analysis", "gmn.uwo.ca", path_template="/home/stationID/files/processed")
    repairMissingCacheEntries(log, conn, cache_root)
    resetStalledJobs(log, conn)
    refileArchives(log, cache_root)
    createDayArchives(log, cache_root)






