#!/usr/bin/env python3
import os
import time
import datetime

from Utils.StarPipeline.PipelineMetrics.Sessions import latestSessions
from Utils.StarPipeline.PipelineMetrics.Frames import frameCounts
from Utils.StarPipeline.PipelineMetrics.Observations import observationCounts, totalObservations
from Utils.StarPipeline.PipelineMetrics.IngestionRate import calstarsPerDay, activeStationCount, ingestionStalled

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
        lines.append(str(row))
    return "\n".join(lines)

def showFrameCounts():
    lines = ["Frame Counts"]
    for row in frameCounts():
        lines.append(str(row))
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

def dashboard():
    try:
        sections = [
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            showIngestionHealth(),
            showLatestSessions(),
            showFrameCounts(),
            #showObservationCounts(),
            #showTotalObservations(),
            showJdRange(),
            showIngestionRate(),
            showActiveStations()]

    except Exception as e:
        # Database not ready or query failed
        sections = [
        f"Database not ready at {datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}",
        f"Reason: {type(e).__name__}"
    ]

    output = "\n\n".join(sections)
    os.system("clear")
    print(output)




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