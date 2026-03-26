#!/usr/bin/env python3
import os
import time
import datetime

from Utils.PipelineMetrics.Sessions import latestSessions
from Utils.PipelineMetrics.Frames import frameCounts
from Utils.PipelineMetrics.Observations import observationCounts, totalObservations
from Utils.PipelineMetrics.IngestionRate import calstarsPerDay

INTERVAL = datetime.timedelta(minutes=10)

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


def showIngestionRate():

    fps, stalled = calstarsPerDay()

    banner = ""
    if stalled:
        banner = f"{RED}!!! INGESTION STALLED — NO NEW CALSTARS !!!{RESET}\n"

    return banner + f"=== Ingestion Rate (calstars/day) ===\n{fps:.2f}"

def dashboard():

    sections = [
        showLatestSessions(),
        showFrameCounts(),
        showObservationCounts(),
        showTotalObservations(),
        showIngestionRate(),
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