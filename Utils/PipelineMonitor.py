#!/usr/bin/env python3
import os
import time

from Utils.PipelineMetrics.Sessions import latestSessions
from Utils.PipelineMetrics.Frames import frameCounts
from Utils.PipelineMetrics.Observations import observationCounts, totalObservations
from Utils.PipelineMetrics.IngestionRate import framesPerSecond

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

    lines = ["Ingestion Rate (frames/sec)"]
    return f"=== Ingestion Rate (frames/sec) ===\n{framesPerSecond():.2f}"

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
    while True:
        dashboard()
        time.sleep(120)
