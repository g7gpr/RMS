# RPi Meteor Station
# Copyright (C) 2026 David Rollinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

import os
import shutil
import datetime
import time

from Utils.StarPipeline.Cache import sortFilesByTime
from Utils.StarPipeline.Cache import LOCAL_PATH_TEMPLATE
from pathlib import Path


import uuid


# Constants



QUEUES = "/archive/dispatcher/lightcurve/queues/"
REGISTRATIONS = "/archive/dispatcher/lightcurve/registrations/"
PROCESSED_SESSIONS_FILE = "/archive/dispatcher/lightcurve/statusfiles/processed_sessions.txt"
LIGHT_CURVE_TEMPLATE_PATH = "/archive/lightcurves/data/stars/star"

def showQueueStatus():
    """Display the current status of all worker queues.

    Print the contents of each worker queue along with the age of
    queued and completed jobs.

    Return:
        None
    """
    print("\033[2J\033[H", end="")
    print("Queue Status")
    print("=" * 62)

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    if not os.path.exists(QUEUES):
        print("Queue Directory does not exist")
        return

    for queue_dir in sorted(Path(QUEUES).iterdir()):

        if not queue_dir.is_dir():
            continue

        jobs = sorted(queue_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"{queue_dir.name:40s}")

        for item in jobs:
            age = (now.timestamp() - item.stat().st_mtime)
            age_minutes = int(age / 60)
            if item.name.endswith(".completed"):
                print(f"    [DONE] {item.name} {age_minutes}m")
            else:
                print(f"    [JOB ] {item.name} {age_minutes}m")

        print()

def getArchiveList(cache_path):
    """Retrieve the list of archives available for processing.

    Arguments:
        cache_path: [str]
            Path to the cache directory.

    Return:
        work_list: [list]
            List of archive paths sorted by observation time.
    """

    paths_to_files = Path(cache_path).rglob("*.tar.bz2")



    return sortFilesByTime([str(f) for f in paths_to_files])


def getWorkerID():
    """Generate a unique worker identifier.

    Return:
        worker_id: [str]
            Unique identifier for a worker process.
    """
    return str(uuid.uuid4())


def registerWorker(worker_id):
    """Register a worker with the dispatcher.

    Arguments:
        worker_id: [str]
            Unique worker identifier.

    Return:
        None
    """

    registrations_dir = Path(REGISTRATIONS)

    registrations_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (registrations_dir / worker_id).touch()

def waitForQueue(worker_id):
    """Wait for a queue to be created for a worker.

    Arguments:
        worker_id: [str]
            Unique worker identifier.

    Return:
        queue_dir: [Path]
            Path to the worker queue directory.
    """

    queue_dir = Path(QUEUES) / worker_id

    while not queue_dir.exists():
        print(f"Waiting for queue for {worker_id}")
        time.sleep(10)

    return queue_dir

def getJob(worker_id):
    """Retrieve the next available job for a worker.

    Arguments:
        worker_id: [str]
            Unique worker identifier.

    Return:
        job_name: [str]
            Name of the next available job.

        None:
            Returned if no jobs are available.
    """

    queue_dir = Path(QUEUES) / worker_id

    jobs = sorted(
        (
            job for job in queue_dir.iterdir()
            if (
                job.is_file()
                and not job.name.endswith(".completed")
                and not (queue_dir / f"{job.name}.completed").exists()
            )
        ),
        key=lambda p: p.stat().st_mtime
    )

    if not jobs:
        return None

    return jobs[0].name

def createJobQueues():
    """Create queues for all registered workers.

    Ensure a queue directory exists for every registered worker.

    Return:
        None
    """

    registrations_dir = Path(REGISTRATIONS)
    queues_dir = Path(QUEUES)

    queues_dir.mkdir(parents=True, exist_ok=True)
    registrations_dir.mkdir(parents=True, exist_ok=True)

    for registration in registrations_dir.iterdir():

        if not registration.is_file():
            continue

        worker_id = registration.stem

        queue_dir = queues_dir / worker_id

        queue_dir.mkdir(parents=True, exist_ok=True)


def populateJobQueues(work_list, processed_sessions, queue_depth=3):
    """Populate worker queues with available jobs.

    Assign unprocessed archives to worker queues until each queue
    reaches the configured depth.

    Arguments:
        work_list: [list]
            List of available archive files.

        processed_sessions: [set]
            Set of previously completed archive filenames.

    Keyword arguments:
        queue_depth: [int]
            Target number of jobs to maintain in each worker queue.

    Return:
        None
    """

    queued_jobs = set()

    # Find every job currently queued
    for queue_dir in Path(QUEUES).iterdir():

        if not queue_dir.is_dir():
            continue

        for job in queue_dir.iterdir():

            if job.is_file():
                queued_jobs.add(job.name)

    # Filter out completed and already queued jobs
    available_jobs_list = [
        Path(job).name
        for job in work_list
        if Path(job).name not in processed_sessions
           and Path(job).name not in queued_jobs
    ]

    waiting_jobs_count = len(available_jobs_list)
    job_idx = 0

    jobs_added = 0
    for queue_dir in Path(QUEUES).iterdir():

        if not queue_dir.is_dir():
            continue

        actual_queue_depth = len(list(queue_dir.iterdir()))



        while actual_queue_depth < queue_depth and job_idx < len(available_jobs_list):
            (queue_dir / available_jobs_list[job_idx]).touch()
            print(f"Added {available_jobs_list[job_idx]} for worker {os.path.basename(queue_dir)}")
            actual_queue_depth += 1
            job_idx += 1
            jobs_added += 1

    if jobs_added > 0:
        waiting_jobs_count -= jobs_added

        print(f"{jobs_added} jobs dispatched to workers, {waiting_jobs_count} jobs waiting.")

    return waiting_jobs_count



def jobCompleted(worker_id, job):
    """Mark a job as completed.

    Arguments:
        worker_id: [str]
            Unique worker identifier.

        job: [str]
            Job filename.

    Return:
        None
    """

    queue_dir = Path(QUEUES) / worker_id
    (queue_dir / f"{job}.completed").touch()

def updateProcessedSessions(processed_sessions):
    """Update the list of completed sessions.

    Scan worker queues for completed jobs and record them in the
    processed sessions file.

    Arguments:
        processed_sessions: [set]
            Set of completed session filenames.

    Return:
        None
    """

    queues_dir = Path(QUEUES)

    for queue_dir in queues_dir.iterdir():

        if not queue_dir.is_dir():
            continue

        for completed_file in queue_dir.glob("*.completed"):

            job_name = completed_file.stem

            if job_name not in processed_sessions:

                processed_sessions.add(job_name)

                with open(PROCESSED_SESSIONS_FILE, "a") as f:
                    f.write(job_name + "\n")


            completed_file.unlink(missing_ok=True)


            job_file = queue_dir / job_name

            if job_file.exists():
                job_file.unlink()

    return

def cleanupQueues(last_queue_cleanup,
                  hours=12,
                  force=False):
    """Remove stale worker registrations and queues.

    Arguments:
        last_queue_cleanup: [datetime]
            Timestamp of the previous cleanup operation.

    Keyword arguments:
        hours: [float]
            Maximum worker inactivity period before cleanup.

        force: [bool]
            Force cleanup regardless of elapsed time.

    Return:
        cleanup_time: [datetime]
            Timestamp of the completed cleanup operation.
    """

    now = datetime.datetime.now(tz=datetime.timezone.utc)

    if (now - last_queue_cleanup) < datetime.timedelta(hours=hours) and not force:

        return last_queue_cleanup

    print("Running queue cleanup...")

    for registration in Path(REGISTRATIONS).iterdir():

        if not registration.is_file():
            continue

        age = now.timestamp() - registration.stat().st_mtime
        worker_id = registration.name

        if age > hours * 3600:



            queue_dir = Path(QUEUES) / worker_id

            print(f"Worker {worker_id} stale {age/3600:.1f} hours")

            if queue_dir.exists():
                shutil.rmtree(queue_dir)

            registration.unlink()

        else:

            print(f"Worker {worker_id} is active {age/3600:.1f} hours")

    return now


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Dispatch start curve processing jobs \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("-r", "--reverse", action="store_true", help="Process newest jobs first")

    cache_path = os.path.dirname(LOCAL_PATH_TEMPLATE)
    processed_sessions_file = PROCESSED_SESSIONS_FILE

    cml_args = arg_parser.parse_args()

    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(os.path.dirname(processed_sessions_file), exist_ok=True)


    open(processed_sessions_file, "a").close()

    with open(processed_sessions_file) as f:
        processed_sessions = {
            line.strip()
            for line in f
            if line.strip()
        }

    work_list = getArchiveList(cache_path)
    if cml_args.reverse:
        work_list.reverse()


    last_queue_cleanup = datetime.datetime.now(tz=datetime.timezone.utc)

    while True:
        createJobQueues()
        last_queue_cleanup = cleanupQueues(last_queue_cleanup)
        waiting_jobs = populateJobQueues(work_list, processed_sessions, queue_depth=5)
        updateProcessedSessions(processed_sessions)
        time.sleep(10)