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

import paramiko
import subprocess
import json
import requests
import sys
import datetime
import numpy as np
import time
import random
import socket
import psycopg


from RMS.Formats import FFfile, Platepar
from RMS.Astrometry.Conversions import J2000_JD, date2JD
from RMS.Logger import LoggingManager, getLogger


# Constants

# urls
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"

# Paths and names
CALSTARS_DATA_DIR = "CALSTARS"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
PORT = 22

def scale1e6(value):
    # Pass through None
    if value is None:
        return None

    # Pass through NaN or infinities
    try:
        if not np.isfinite(value):
            return None
    except Exception:
        # Non-numeric types → pass through unchanged
        return None

    # Normal numeric case
    return int(round(value * 1_000_000))

def connectionProblem(host, port=22, timeout=3):
    """Report connection problems.

    Arguments:
        host: [str] Hostname.

    Keyword arguments:
        port: [int] Port number, optional, default 22.
        timeout: [float] Optional, default 3, timeout before connection reported failed.
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)

        try:
            banner = sock.recv(1024)
            if not banner:
                # Connection closed before banner
                return True
            # Banner received  not banned
            return False

        except socket.timeout:
            # No banner within timeout also typical of Fail2ban
            return True

        finally:
            sock.close()

    except (socket.timeout, ConnectionRefusedError, OSError):
        # Can't connect at all return True
        return True

def lsRemote(host, username, port, remote_path):
    """Return: list of filenames in remote directory, or empty list if directory does not exist."""

    remote = "{}@{}:{}".format(username, host, remote_path)
    cmd = [
        "rsync",
        "--list-only",
        "-z",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p {}".format(port),
        remote + "/",
        "."
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Directory does not exist
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "no such file" in stderr or "failed to open" in stderr:
            log.info("Remote directory {} does not exist".format(remote_path))
            return []
        # Any other error is real
        raise RuntimeError("rsync failed: {}".format(result.stderr.strip()))

    files = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            files.append(parts[-1])
    remote_file_count = len(files)
    word = "file" if remote_file_count == 1 else "files"
    # log.info(f"Remote directory {remote_path} contained {remote_file_count} {word}")

    return files

def getStationList(url=STATION_COORDINATES_JSON, country_code=None):
    """Get a list of stations using the station coordinates json.

    Arguments:
        url: [str] Optional, default STATION_COORDINATES_JSON, url of the json of station coordinates.

    Returns:
        [list] station names.
    """

    log.info(f"Downloading station list from {url}")

    response = requests.get(url)
    response.raise_for_status()  # fail fast if the JSON is unreachable

    stations_dict = response.json()
    station_list = []

    for station in stations_dict.keys():
        station_lower = station.lower()

        if country_code is None:
            station_list.append(station)
        else:
            if station_lower.startswith(country_code.lower()):
                station_list.append(station)

    return sorted(station_list)

def filterByDate(files_list, earliest_date=None, latest_date=None, station=None, always_return_one=True):
    """
    Filter a list of bz2 files by date.
    Arguments:
        files_list: [list] list of bz2 files.

    Keyword arguments:
        earliest_date: [datetime] optional, default None, earliest date to pick, if None, 3 days before now.
        latest_date: [datetime] optional, default None, latest date to pick, if None, 3 days after now.
        always_return_one: [bool] Optional, default True, return at least one file, if any are available after earliest date
    Returns:
        filtered_files_list: [list] list of bz2 files filtered by date
    """


    if earliest_date is None:
        earliest_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)

    if latest_date is None:
        latest_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)

    files_list.sort()

    filtered_files_list = []
    for file in files_list:

        parts = file.split("_")
        if len(parts) != 5:
            continue

        if station is not None:
            if not file.startswith(station):
                log.info(f"Unexpected file {file}")
                continue

        try:
            date = parts[1]
            time = parts[2]
            year, month, day = int(date[0:4]), int(date[4:6]), int(date[6:8])
            hour, minute, second = int(time[0:2]), int(time[2:4]), int(time[4:6])

            file_date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second,
                                          tzinfo=datetime.timezone.utc)


        except Exception:
            log.warning(f"Skipping malformed filename: {file}")
            continue

        if earliest_date < file_date < latest_date:
            filtered_files_list.append(file)

    if always_return_one and not len(filtered_files_list):
        #If we have no files in the date range, then return the last file from the sorted list
        filtered_files_list.append(files_list[-1])

    return filtered_files_list

def getFileType(file_name):
    """Given a filename, get the extension.

    Args:
        file_name: [str] File name

    Returns:
        [str]: File extension
    """
    return file_name.split(".")[0].split("_")[4]


def getRemoteFiles(username, host, port, remote_dir, delay_retry=True):

    remote_files = []

    while not remote_files:
        try:
            remote_files = sorted(
                lsRemote(host, username, port, remote_dir),
                reverse=True
            )
        except Exception as e:
            log.warning(f"lsRemote failed for {host}:{remote_dir}: {e}")
            remote_files = []

        if remote_files:
            return remote_files

        if connectionProblem(host):
            if not delay_retry:
                log.info(f"Connection problem for {host}, not delaying")
                break
            delay = random.randint(600, 900)

            time.sleep(delay)
        else:
            log.info(f"No remote files found in {remote_dir} for {host}")
            break

    return remote_files


def downloadFile(host, username, local_path, remote_path, port=PORT,  silent=False, bw_limit=None):
    """Download a single file try compressed rsync first, then fall back to Paramiko.

    Arguments:
        host: [str] hostname of remote machine.
        username: [str] username for remote machine.
        local_path: [path] full path of local target.
        remote_path: [path] full path of remote target

    Keyword arguments:
        port: [str] Optional, default PORT constant.

        silent: [bool] optional, default False.

    Return:
        Nothing.
    """

    try:

        remote = "{}@{}:{}".format(username, host, remote_path)
        result = subprocess.run(['rsync', '-z', remote], capture_output=True, text=True)
        if "No such file or directory" in result.stderr :
            if not silent:
                print("Remote file {} was not found.".format(os.path.basename(remote)))
            return
        else:
            cmd = [
                'rsync',
                '--partial',
                '--partial-dir=.rsync-partial',
            ]

            if bw_limit is not None:
                cmd.append(f'--bwlimit={bw_limit}')
                log.info(f"Starting rsync with a band width limit of {bw_limit}")

            cmd.extend([remote, os.path.join(local_path, os.path.basename(remote_path))])

            result = subprocess.run(cmd, capture_output=True, text=True)

        if not os.path.exists(os.path.expanduser(local_path)):
            if not silent:
                print("Download of {} from {}@{} failed. You need to add your keys to remote using ssh-copy-id."
                                .format(remote_path, username,host))
            sys.exit(1)
        return
    except:
        pass

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
    try:
        ssh.connect(hostname=host, port=port, username=username)
    except:
        if not silent:
            print("Login to {}@{} failed. You may need to add your keys to remote using ssh-copy-id."
              .format(username,host))
        sys.exit()
    try:
        sftp = ssh.open_sftp()
        remote_file_list = sftp.listdir(os.path.dirname(remote_path))
        if remote_file_list:
            sftp.get(remote_path, local_path)

    finally:
        sftp.close()
        ssh.close()

    return



def downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=22, max_tries=3, verbose=False, bw_limit=None):

    remote_file = os.path.basename(full_remote_path_to_bz2)
    download_start_time = datetime.datetime.now(datetime.timezone.utc)
    download_count = 0
    target_path = os.path.join(t, os.path.basename(full_remote_path_to_bz2))
    if verbose:
        log.info(f"Downloading {full_remote_path_to_bz2} to {target_path}")

    while not os.path.exists(target_path) and download_count < max_tries:
        downloadFile(host, username, t, full_remote_path_to_bz2, port=port, bw_limit=bw_limit)
        download_count += 1
        # If the file is now present, break immediately
        if os.path.exists(target_path):
            download_end_time = datetime.datetime.now(datetime.timezone.utc)
            downloaded_size = os.path.getsize(os.path.join(t, remote_file)) / (1000 ** 2)
            rate_mb_s = downloaded_size / (download_end_time - download_start_time).total_seconds()
            if verbose:
             log.info(
                    f"Downloaded {remote_file} of size {downloaded_size:.2f}MB at {rate_mb_s:.2f} MB/s after {download_count} try")
            return True

        delay = random.randint(600, 900)
        resume_time = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=delay)).isoformat()
        log.info(f"Waiting {delay/60:.1f} minutes until {resume_time} for {os.path.basename(target_path)}")
        time.sleep(delay)



    log.warning(f"Failed to download {remote_file} after {download_count} tries")

    return False


def discoverRemoteFiles(stations, username, host, port,
                        remote_processed_dir_template,
                        min_interval_sec=1, target_interval_sec=3):

    filtered_files = []

    # Initialise cadence
    start_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    next_allowed = start_time
    for idx, station in enumerate(stations, start=1):
        # Start of this iteration is the scheduled cadence time


        iteration_start = next_allowed

        remote_dir = remote_processed_dir_template.replace(
            "stationID", station.lower()
        )

        retry = 3

        # Retry loop for Fail2ban-style blocks
        while retry > 0:
            retry -= 1
            try:
                station_files = lsRemote(host, username, port, remote_dir)
                break

            except Exception as e:
                msg = str(e).lower()

                # Detect Fail2ban / SSH refusal
                if "connection refused" in msg or "unexpectedly closed" in msg:
                    pause = random.uniform(600, 900)
                    log.warning(
                        f"Fail2ban likely active for {station}. "
                        f"Sleeping until {(iteration_start + datetime.timedelta(seconds=pause)).replace(microsecond=0).isoformat()} before retrying."
                    )
                    time.sleep(pause)
                    continue

                # Log and skip
                log.warning(f"Failed to list remote files for {station}: {e}")
                station_files = []
                break

        elapsed_time_seconds = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        seconds_per_download = elapsed_time_seconds / idx
        end_time = (start_time + datetime.timedelta(seconds = len(stations) * seconds_per_download)).replace(microsecond=0)
        plural = '' if len(stations) == 1 else 's'
        log.info(f"Processing station {idx}/{len(stations)}: {station} had {len(station_files)} file{plural}. Start / Completion {start_time.isoformat()} / {end_time.isoformat()} {seconds_per_download:.1f} sec/station")

        # Filter valid tars
        for file_name in station_files:
            if file_name.endswith("tar.bz2") and len(file_name.split("_")) == 5 and file_name.startswith(station.upper()) and "imgdata" not in file_name:
                filtered_files.append(file_name)

        next_allowed = iteration_start + datetime.timedelta(seconds=target_interval_sec)

        delay = (next_allowed - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        time.sleep(max(min_interval_sec, delay))

    return filtered_files

def parseServerFileTimestamp(file_name):
    parts = file_name.split("_")
    if len(parts) < 4:
        return None

    date_str = parts[1]
    time_str = parts[2]

    try:
        dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None

def sortFilesByTime(files):
    return sorted(files, key=parseServerFileTimestamp)

def saveRemoteFiles(remote_files, json_path):
    serialisable = [{"file_name": file_name} for file_name in remote_files]

    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)



def populateWorkQueue(conn, file_name_list):
    """
    Stream rows into a temp staging table using COPY, then merge into ingest_work
    with ON CONFLICT DO NOTHING.
    """

    with conn.cursor() as cur:

        # 1. Create temp staging table (auto-dropped at commit)
        cur.execute("""
            CREATE TEMP TABLE ingest_work_stage (
                remote_path TEXT,
                jd_int      BIGINT
            ) ON COMMIT DROP;
        """)

        # 2. COPY into the staging table
        with cur.copy("COPY ingest_work_stage (remote_path, jd_int) FROM STDIN") as copy:
            for file_name in file_name_list:
                dt = FFfile.getMiddleTimeFF(file_name, fps=25, ret_milliseconds=True, ff_frames=256)
                jd = date2JD(*dt)
                jd_int = scale1e6(jd)
                time_now = datetime.datetime.now(tz=datetime.timezone.utc)

                now_jd_int = scale1e6(date2JD(time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second, time_now.microsecond / 1000))

                if jd_int is None:
                    continue

                if jd_int > now_jd_int:
                    log.info(f"Rejecting file {file_name} as observation start time is in the future")
                    continue

                copy.write_row((file_name, jd_int))

        # 3. Merge into real table with ON CONFLICT DO NOTHING
        cur.execute("""
            INSERT INTO ingest_work (remote_filename, jd_int)
            SELECT remote_path, jd_int
            FROM ingest_work_stage
            ON CONFLICT (remote_path) DO NOTHING;
        """)

    # 4. Commit drops the temp table automatically
    conn.commit()

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Populate Work Queue""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('path_template', help="""Template to remote file stores i.e. user@host:/home/stationID/files/ """)

    arg_parser.add_argument('postgresql_host', help="""PostgreSQL server host """)

    arg_parser.add_argument('--country_code', metavar='COUNTRY_CODE', help="""Country code to work on""")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    country_code = cml_args.country_code

    path_template = cml_args.path_template
    if not path_template.endswith("/"):
        path_template = f"{path_template}/"

    user, _, remainder = path_template.partition("@")
    hostname, _, path_template = remainder.partition(":")

    # Initialize the logger
    log_manager = LoggingManager()

    # Get the logger handle
    log = getLogger("rmslogger")
    postgresql_host = cml_args.postgresql_host

    log.info("Populating the ingestion table")
    station_list = getStationList()
    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as conn:
        for station in station_list:
            log.info(f"Downloading files for {station}")
            remote_files = discoverRemoteFiles([station], user, hostname, 22, remote_processed_dir_template=path_template)
            remote_files_sorted = sortFilesByTime(remote_files)
            populateWorkQueue(conn, remote_files_sorted)
    log.info("Table populated")
