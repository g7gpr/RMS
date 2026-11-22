# RPi Meteor Station
# Copyright (C) 2026 David Rollinson Kristen Felker
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
import tempfile
import tarfile
import numpy as np

import subprocess
import json
import requests
import RMS.ConfigReader as cr
import shutil
import sys
import datetime
import time
import random
import socket
import cv2
import ephem
import tqdm

from RMS.Astrometry.Conversions import datetime2JD, jd2Date
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Misc import mkdirP
from pathlib import Path
from Utils.StarPipeline.CacheBuilder import sortFilesByTime
from RMS.Formats.CALSTARS import readCALSTARS, renderStars
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalogQueries import Catalog, renderStarsGaussian
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP, raDec2AltAz
from Utils.Flux import detectMoon

from Utils.StarPipeline.CacheBuilder import ARCHIVE

import uuid

# Local directories
# Local directories
ARCHIVE = "/archive"
CALSTARS_DATA_DIR = "CALSTARS"
QUEUES = "/archive/despatcher/queues"
REGISTRATIONS = "/archive/despatcher/registrations"
PROCESSED_SESSIONS_FILE = "/archive/processed_sessions.txt"


def getWorkList(archive):


    print(f"Working through {archive}")
    paths_to_files = Path(archive).rglob("*.tar.bz2")



    return sortFilesByTime([str(f) for f in paths_to_files])


def getWorkerID():

    return str(uuid.uuid4())

from pathlib import Path

def registerWorker(worker_id):

    registrations_dir = Path(REGISTRATIONS)

    registrations_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (registrations_dir / worker_id).touch()

def waitForQueue(worker_id):

    queue_dir = Path(QUEUES) / worker_id

    while not queue_dir.exists():
        time.sleep(1)

    return queue_dir


def getJob(worker_id):

    queue_dir = Path(QUEUES) / worker_id

    jobs = sorted(
        job
        for job in queue_dir.iterdir()
        if job.is_file()
        and not job.name.endswith(".completed")
    )

    if not jobs:
        return None

    return jobs[0].name

def jobToPath(job):


    parts = job.split('_')

    date_str = parts[1]

    return Path(ARCHIVE) / Path(CALSTARS_DATA_DIR) / date_str / job

def createJobQueues():


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
    queued_jobs = set()

    # Find every job currently queued
    for queue_dir in Path(QUEUES).iterdir():

        if not queue_dir.is_dir():
            continue

        for job in queue_dir.iterdir():

            if job.is_file():
                queued_jobs.add(job.name)

    # Filter out completed and already queued jobs
    available_jobs = [
        Path(job).name
        for job in work_list
        if Path(job).name not in processed_sessions
           and Path(job).name not in queued_jobs
    ]

    job_idx = 0

    for queue_dir in Path(QUEUES).iterdir():

        if not queue_dir.is_dir():
            continue

        queue_depth = len(list(queue_dir.iterdir()))

        while queue_depth < 3 and job_idx < len(available_jobs):
            (queue_dir / available_jobs[job_idx]).touch()
            print(f"Added {available_jobs[job_idx]} for {queue_dir}")
            queue_depth += 1
            job_idx += 1

def jobCompleted(worker_id, job):

    queue_dir = Path(QUEUES) / worker_id
    (queue_dir / f"{job}.completed").touch()

def updateProcessedSessions(processed_sessions):

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

            completed_file.unlink()

            job_file = queue_dir / job_name

            if job_file.exists():
                job_file.unlink()

    return


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Despatch start curve processing jobs \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("-r", "--reverse", action="store_true", help="Process newest jobs first")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    os.makedirs(ARCHIVE, exist_ok=True)
    calstars_directory_path = os.path.join(ARCHIVE, CALSTARS_DATA_DIR)
    os.makedirs(calstars_directory_path, exist_ok=True)


    open("/archive/processed_sessions.txt", "a").close()

    with open("/archive/processed_sessions.txt") as f:
        processed_sessions = {
            line.strip()
            for line in f
            if line.strip()
        }

    work_list = getWorkList(calstars_directory_path)
    if cml_args.reverse:
        work_list.reverse()
    cat = Catalog(config)

    while True:
        createJobQueues()
        populateJobQueues(work_list, processed_sessions, queue_depth=5)
        updateProcessedSessions(processed_sessions)
        time.sleep(10)