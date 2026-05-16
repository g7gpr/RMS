#!/usr/bin/env python3
import os
import time
import datetime

from Utils.StarPipeline.PipelineMetrics.Sessions import latestSessions
from Utils.StarPipeline.PipelineMetrics.Frames import frameCounts
from Utils.StarPipeline.PipelineMetrics.Observations import observationCounts, totalObservations
from Utils.StarPipeline.PipelineMetrics.IngestionRate import activeStationCount, ingestionStalled
from Utils.StarPipeline.PipelineMetrics.db import getConn
from Utils.StarPipeline.PipelineDB import claimNextJob
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

    conn = getConn()

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(jd_mid), MAX(jd_mid) FROM frame;")
        row = cur.fetchone()

    if row is None or row[0] is None or row[1] is None:
        return None, None

    # Convert microdays → days
    jd_min = row[0] / 1e6 if row[0] is not None else None
    jd_max = row[1] / 1e6 if row[1] is not None else None

    return jd_min, jd_max

def showJdRange():

    jd_min, jd_max = jdRange()


    if jd_min is None or jd_max is None:
        return "=== JD Range ===\nNo data"

    jd_range = jd_max - jd_min

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
        rate = ingestionRate(days=7)
        conn = getConn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ingest_work WHERE status = 'pending';")
            pending = cur.fetchone()[0]
        if rate > 0:
            daysLeft = pending / rate
            finishDate = (datetime.date.today() +
                          datetime.timedelta(days=daysLeft))
            finishStr = finishDate.strftime("%Y-%m-%d")
            return (f"=== Ingestion Health ===\n"
                    f"OK Processing {rate:.0f} CALSTARS per day\n"
                    f"Estimated queue drain date: {finishStr}")
        else:
            return (f"=== Ingestion Health ===\n"
                    f"OK Processing 0 CALSTARS per day\n"
                    f"Estimated queue drain date: unknown")

def ingestionRate(days=7):
    """
    Return average jobs/day completed over the last N days,
    compensating for shorter ingestion histories.
    """

    conn = getConn()

    sql = """SELECT COUNT(*),
                   MIN(updated_at)
            FROM ingest_work
            WHERE status = 'done'
              AND updated_at >= now() - (%s * interval '1 day');"""

    with conn.cursor() as cur:
        # Count jobs done in the window
        cur.execute(sql, (days, ))
        count, earliest_ts = cur.fetchone()

    conn.close()

    if count == 0:
        return 0.0

    # Compute actual time span in days
    now = datetime.datetime.now(datetime.timezone.utc)

    if earliest_ts is None:
        # Should not happen if count > 0, but safe fallback
        return 0.0

    actual_days = (now - earliest_ts).total_seconds() / 86400.0

    # Clamp to at least a tiny positive number
    actual_days = max(actual_days, 1e-6)

    # If ingestion has been running longer than the window,
    # use the window size instead.
    actual_days = min(actual_days, days)

    return count / actual_days



def showWorkerLeaderboard():
    """
    Show workers (claimed_by) with number of claimed and completed jobs.
    """

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

    sections = [
        datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        showIngestionHealth(),
        showQueueHealth(),
        showNextJob(),
        showLatestSessions(),
        showFrameCounts(),
        showJdRange(),
        showActiveStations(),
        showWorkerLeaderboard()]
    """
    except Exception as e:
        # Database not ready or query failed
        sections = [
        f"Database not ready at {datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}",
        f"Reason: {type(e).__name__}"
    ]
    """
    output = "\n\n".join(sections)
    os.system("clear")
    print(output)

def showNextJob():
    """
    Show the next job that would be claimed by a worker (dry‑run).
    """



    conn = getConn()

    try:
        row = claimNextJob(conn, dry_run=True)
    finally:
        conn.close()

    if row is None:
        return "=== Next Job ===\nNo pending jobs"

    remote_filename, jd_int = row
    return (
        "=== Next Job ===\n"
        f"remote_filename: {remote_filename}\n"
        f"jd_int:          {jd_int}"
    )


def showQueueHealth():
    """
    Show ingestion queue health:
    - total jobs
    - cache files
    - pending
    - done
    - error
    - first 5 error jobs (normalised, with worker)
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

        # First 5 error jobs (with worker)
        cur.execute("""
            SELECT remote_filename, claimed_by
            FROM ingest_work
            WHERE status = 'error'
            ORDER BY jd_int ASC
            LIMIT 5;
        """)
        results = cur.fetchall()

    conn.close()

    # --- Count cache files ---
    cache_root = "/mnt/rms/cache/RMS_data/CALSTARS"
    cache_count = 0
    for day_dir in Path(cache_root).iterdir():
        if day_dir.is_dir():
            cache_count += len(list(day_dir.glob("*_raw.tar.bz2")))

    # --- Normalise error filenames ---
    normalised_errors = []
    for remote_filename, claimed_by in results:
        if "_detected" in remote_filename:
            normalised = remote_filename.replace("_detected", "_raw")
        elif "_metadata" in remote_filename:
            normalised = remote_filename.replace("_metadata", "_raw")
        else:
            normalised = remote_filename

        normalised_errors.append((normalised, claimed_by))

    # --- Build output ---
    lines = [
        "=== Queue Health ===",
        f"Total jobs:     {total_jobs}",
        f"Cache files:    {cache_count}",
        f"Pending:        {pending}",
        f"Completed:      {done}",
        "",
        f"First 5 error jobs of {error}:"
    ]

    if normalised_errors:
        for fname, worker in normalised_errors:
            worker_label = worker if worker is not None else "(unclaimed)"
            lines.append(f"  {fname}    [{worker_label}]")
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