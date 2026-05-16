#!/usr/bin/env python3
import os
import time
import datetime

from Utils.StarPipeline.PipelineMetrics.Sessions import latestSessions
from Utils.StarPipeline.PipelineMetrics.Frames import frameCounts
from Utils.StarPipeline.PipelineMetrics.Observations import observationCounts, totalObservations
from Utils.StarPipeline.PipelineMetrics.IngestionRate import calstarsPerDay, activeStationCount, ingestionStalled
from pathlib import Path

INTERVAL = datetime.timedelta(minutes=1)

BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def printSection(title):
    print(f"{BLUE}=== {title} ==={RESET}")

def showLatestSessions():
    lines = ["Latest Sessions"]
    for row in latestSessions():
        lines.append(str(f"{row[0]}\t{row[1]}\t{row[2]}"))
    return "\n".join(lines)

def showFrameCounts():
    lines = ["Frame Counts"]
    for row in frameCounts():
        lines.append(str(row[1]))
    return "\n".join(lines)

def showObservationCounts():

    lines = ["Observation Counts"]
    for row in observationCounts():
        lines.append(str(row))
    return "\n".join(lines)

def showTotalObservations():

    lines = ["Total Observations"]

    return f"=== Total Observations ===\n{totalObservations()}"

def jdRange():
    """
    Return earliest and latest JD from the observation table.
    """
    from Utils.StarPipeline.PipelineMetrics.db import getConn
    conn = getConn()

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(jd_mid), MAX(jd_mid) FROM frame;")
        row = cur.fetchone()

    if row is None:
        return None, None

    # Convert microdays → days
    jd_min = row[0] / 1e6 if row[0] is not None else None
    jd_max = row[1] / 1e6 if row[1] is not None else None

    return jd_min, jd_max

def showJdRange():

    jd_min, jd_max = jdRange()
    jd_range = jd_max - jd_min

    if jd_min is None:
        return "=== JD Range ===\nNo data"

    return (
        "=== JD Range ===\n"
        f"Earliest JD:  {jd_min:011.3f}\n"
        f"Range JD:     {jd_range:011.3f}\n"
        f"Latest JD:    {jd_max:011.3f}"
    )

def showActiveStations():

    n = activeStationCount(hours=48)
    return f"=== Active Stations (48h) ===\n{n}"

def showIngestionHealth():
    stalled = ingestionStalled()
    if stalled:
        return "=== Ingestion Health ===\nWARNING: Ingestion stalled"
    else:
        return "=== Ingestion Health ===\nOK"

def showIngestionRate():
    rate = calstarsPerDay(window_days=7)
    return (
        "=== Ingestion Rate (7‑day rolling) ===\n"
        f"{rate:.2f} calstars/day"
    )

def showWorkerLeaderboard():
    """
    Show workers (claimed_by) with number of claimed and completed jobs.
    """
    from Utils.StarPipeline.PipelineMetrics.db import getConn
    conn = getConn()

    sql = """
        SELECT
            claimed_by AS hostname,
            COUNT(*) FILTER (WHERE status = 'claimed') AS number_claimed,
            COUNT(*) FILTER (WHERE status = 'done') AS number_completed
        FROM ingest_work
        WHERE claimed_by IS NOT NULL
        GROUP BY claimed_by
        ORDER BY number_completed DESC;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        return "=== Worker Leaderboard ===\nNo workers have claimed jobs yet"

    lines = ["=== Worker Leaderboard ===", "hostname        claimed   completed"]
    for hostname, claimed, completed in rows:
        lines.append(f"{hostname:14} {claimed:7}   {completed:10}")

    return "\n".join(lines)



def dashboard():
    try:
        sections = [
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            #showIngestionHealth(),
            showQueueHealth(),
            showLatestSessions(),
            showFrameCounts(),
            #showObservationCounts(),
            #showTotalObservations(),
            showJdRange(),
            showActiveStations(),
            showWorkerLeaderboard()]

    except Exception as e:
        # Database not ready or query failed
        sections = [
        f"Database not ready at {datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}",
        f"Reason: {type(e).__name__}"
    ]

    output = "\n\n".join(sections)
    os.system("clear")
    print(output)


def showQueueHealth():
    """
    Show ingestion queue health:
    - total jobs
    - cache files
    - pending
    - done
    - error
    - first 5 error jobs (normalised)
    """
    from Utils.StarPipeline.PipelineMetrics.db import getConn
    conn = getConn()

    # --- Query DB job counts ---
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ingest_work;")
        total_jobs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM ingest_work WHERE status = 'pending';")
        pending = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM ingest_work WHERE status = 'done';")
        done = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM ingest_work WHERE status = 'error';")
        error = cur.fetchone()[0]

        # First 5 error jobs
        cur.execute("""
            SELECT remote_filename
            FROM ingest_work
            WHERE status = 'error'
            ORDER BY jd_int ASC
            LIMIT 5;
        """)
        results = cur.fetchall()
        error_rows = [r[0] for r in results]

    # --- Count cache files ---
    cache_root = "/mnt/rms/cache/RMS_data/CALSTARS"
    cache_count = 0
    for day_dir in Path(cache_root).iterdir():
        if day_dir.is_dir():
            cache_count += len(list(day_dir.glob("*_raw.tar.bz2")))

    # --- Normalise error filenames ---
    normalised_errors = []
    for name in error_rows:
        if "_detected" in name:
            normalised = name.replace("_detected", "_raw")
        elif "_metadata" in name:
            normalised = name.replace("_metadata", "_raw")
        else:
            normalised = name
        normalised_errors.append(normalised)

    # --- Build output ---
    lines = [
        "=== Queue Health ===",
        f"Total jobs:     {total_jobs}",
        f"Cache files:    {cache_count}",
        f"Pending:        {pending}",
        f"Completed:      {done}",
        f"Error:          {error}",
        "",
        "First 5 error jobs:"
    ]

    if normalised_errors:
        for f in normalised_errors:
            lines.append(f"  {f}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)



if __name__ == "__main__":



    next_iteration_start = datetime.datetime.now(tz=datetime.timezone.utc)

    while True:
        now = datetime.datetime.now(tz=datetime.timezone.utc)

        # Sleep until the scheduled time (if we're early)
        if now < next_iteration_start:
            time.sleep((next_iteration_start - now).total_seconds())

        # Run the dashboard
        dashboard()

        # Schedule the next iteration
        next_iteration_start += INTERVAL