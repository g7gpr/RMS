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


import math
from RMS.Formats.FFfile import getMiddleTimeFF, filenameToDatetime
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import inspect
import traceback
import os
import tempfile
import tarfile

import paramiko
import subprocess
import json
import requests
from Cython.Compiler.ExprNodes import infer_sequence_item_type

import RMS.ConfigReader as cr
import shutil
import sys
import datetime
import numpy as np
import time
import random
import socket
import psycopg
import ephem


from RMS.Formats import FFfile, Platepar
from RMS.Astrometry.CheckFit import starListToDict
from RMS.Astrometry.Conversions import J2000_JD, date2JD
from RMS.Formats.StarCatalog import Catalog
from RMS.Astrometry.Conversions import latLonAlt2ECEF
from RMS.Routines.MaskImage import getMaskFile
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP, raDec2AltAz
from RMS.Formats.CALSTARS import readCALSTARS, maxCalstarsToPNG, calstarsToMP4
from RMS.Misc import mkdirP
from RMS.Logger import LoggingManager, getLogger
from pathlib import Path
from RMS.Astrometry.AutoPlatepar import autoFitPlatepar, loadCatalogStars
from Utils.Flux import detectMoon
from multiprocessing import Pool, Process, cpu_count
from collections import defaultdict
from Utils.StarPipeline.PipelineDB import createDatabaseIfMissing, initialiseDatabase, Flags, auditIngestUserPrivileges, claimNextJob, markJobDone, markJobError, resetStalledJobs, jobsRemaining, getNextErrorJob
from collections import Counter
from scipy.interpolate import SmoothBivariateSpline

from Utils.StarPipeline.PipelineDB import PG_INT_MAG_LIMIT, PG_BIGINT_MAG_LIMIT

# url
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"

# Paths and names
ARCHIVE = "/archive"
REMOTE_FILES_JSON = "remotefiles.json"
REMOTE_PATH_TEMPLATE = "/home/stationID/files/incoming/processed"
CALSTARS_DATA_DIR = "CALSTARS"
PLATEPARS_ALL_RECALIBRATED_JSON = "platepars_all_recalibrated.json"
PLATEPARS_FLUX_RECALIBRATED_JSON = "platepars_flux_recalibrated.json"
FTP_DETECT_INFO_TEMPLATE = "FTPdetectinfo_"
DIRECTORY_INGESTED_MARKER = ".processed"
FILE_SYSTEM_MARKERS_ENABLED = False
CALSTAR_FILES_TABLE_NAME = "calstar_files"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
CHARTS = "charts"
PORT = 22



import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



def waitForDownloadWindow(start_hour_utc: int, end_hour_utc: int):
    now = datetime.datetime.now(datetime.timezone.utc)
    current_hour = now.hour

    # Case 1: window does not wrap midnight (e.g., 02–06 UTC)
    if start_hour_utc < end_hour_utc:
        in_window = start_hour_utc <= current_hour < end_hour_utc
    else:
        # Case 2: window wraps midnight (e.g., 22–04 UTC)
        in_window = current_hour >= start_hour_utc or current_hour < end_hour_utc

    if in_window:
        return  # safe to download now

    # Compute next allowed time in UTC
    today = now.date()
    if start_hour_utc > current_hour:
        next_start = datetime.datetime.combine(
            today, datetime.time(start_hour_utc, 0), tzinfo=datetime.timezone.utc
        )
    else:
        tomorrow = today + datetime.timedelta(days=1)
        next_start = datetime.datetime.combine(
            tomorrow, datetime.time(start_hour_utc, 0), tzinfo=datetime.timezone.utc
        )

    sleep_seconds = (next_start - now.replace(tzinfo=datetime.timezone.utc)).total_seconds()

    # Add jitter (0–2 minutes)
    sleep_seconds += random.uniform(0, 120)

    log.info(f"Outside download window. Sleeping for {sleep_seconds/3600:.2f} hours")
    time.sleep(sleep_seconds)




def makeTarBz2(source_dir, output_file):
    """ Archive a directory to a tar.bz2 file.

    Arguments:
        source_dir: [str] Full path to be archived.
        output_file: [str] Full path and name of archive file.

    Returns:
        Nothing.
    """
    source_dir = source_dir.resolve()
    dir_name = source_dir.name

    with tarfile.open(output_file, "w:bz2") as tar:
        tar.add(source_dir, arcname=dir_name)

def extractCalstarArchives(root, archives_list, remove_archives=True):
    """Extract files from a tar archive, and optionally remove the archive. Lists or a string can be passed. If an
    archive does not exist, skip it.

    Arguments:
        root: [str] Path to a directory containing archives.
        archives_list: [list | str] List of archive files or a string.

    Keyword arguments:
        remove_archives: [bool] Optional, default True, remove the archive file after detection

    Returns:
        output_dir_list: [list] List of the paths to the extracted directories
    """

    root = Path(root)
    output_dir_list = []

    for archive_name in archives_list:
        arc_name = Path(archive_name)
        archive_full_path = root / arc_name
        if not archive_full_path.exists():
            continue

        if not archive_full_path.is_file():
            continue

        # Derive original directory name
        dir_name = arc_name.name.replace("_raw.tar.bz2", "")
        output_dir = root / dir_name

        # Ensure output directory does not already exist
        if output_dir.exists():
            log.warning(f"Output directory {output_dir} already exists, removing it first")
            shutil.rmtree(output_dir)

        # Extract archive (no filter argument!)
        with tarfile.open(archive_full_path, "r:*") as tar:
            try:
                top_levels = {m.name.split("/")[0] for m in tar.getmembers() if m.name}
                tar.extractall(root)
                if output_dir.exists():
                    # Make the directory world writeable, so that any worker can delete
                    os.chmod(output_dir, 0o777)
                    output_dir_list.append(output_dir)
                    for dirpath, dirnames, filenames in os.walk(output_dir):
                        os.chmod(dirpath, 0o777)
                        for f in filenames:
                            os.chmod(os.path.join(dirpath, f), 0o666)

                else:
                    log.error(f"Expected extracted directory {output_dir} but it does not exist")

            except Exception as e:
                msg = "Exception: {}".format(str(e))
                tb = traceback.format_exc().encode("ascii", "replace").decode("ascii")

                log.error(msg)
                log.error(tb)

                log.warning(f"{archive_full_path.name} was corrupted and could not be extracted - removing")
                archive_full_path.unlink()
                continue

        # Optionally remove the archive
        if remove_archives and archive_full_path.exists():
            #log.info(f"Removing archive {archive_path}")
            archive_full_path.unlink()

    return output_dir_list

def getDirectorySize(path):
    """Returns the size of all files and directories in a directory.

    Arguments:
        path: [path] Path to a directory.

    Returns:
        [int]: Size of contents in bytes.
    """

    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total

def archiveCalstarDirectories(conn, root, directories_list, ingested_only=True, verbose=False):

    """Given a list of directories, archive them named e.g. AU0004_20260317_111157_992974_raw.tar.bz2,
    and remove the source directory.

    Arguments:
        root: [str] Directory containing folders.
        directories_list: [list] List of names of directories to archive.

    Keyword Arguments:
        ingested_only: Optional, default true, if true then only arhive directories containing the ingested marker.

    Returns:
        Nothing.
    """

    root = Path(root)

    for d in sorted(directories_list):
        source_dir = root / os.path.basename(d).split('_')[1] / d
        if not os.path.isdir(source_dir):
            continue

        tar_file_name = f"{d}_raw.tar.bz2"
        output_file = root / os.path.basename(tar_file_name).split('_')[1]  / tar_file_name

        #log.info(f"Creating {os.path.basename(output_file)}")
        makeTarBz2(source_dir, output_file)
        os.chmod(output_file,0o666)

        # compute uncompressed size
        uncompressed_size = getDirectorySize(source_dir)  / (1024 ** 2)
        # compute compressed size
        compressed_size = output_file.stat().st_size / (1024 **2)
        if uncompressed_size > 0 and verbose:
            log.info(f"Removing {os.path.basename(source_dir)} of size {uncompressed_size:.1f} MB and replaced with archive of size {compressed_size:.1f} MB - ratio {compressed_size / uncompressed_size:.2f}")
        shutil.rmtree(source_dir, ignore_errors=True)

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

def dropTable(conn, table_name):
    """If a table exists, drop it.
    Arguments:
        conn: [object] Connection to database.
        table_name: [string] Name of the table.
    Returns:
        Nothing."""

    sql_command = f"DROP TABLE IF EXISTS {table_name};"
    #log.info(f"Dropping table {table_name}")
    #log.info(f"Executing sql command {sql_command}")

    with conn.cursor() as cur:
        cur.execute(sql_command)
    conn.commit()

def buildCalstarFilename(calstar_directory_path):
    base = os.path.basename(calstar_directory_path)
    parts = base.split("_")

    # Expecting: YYYYMMDD_HHMMSS_STATIONID_SEQUENCE
    # Produces:  CALSTARS_YYYYMMDD_HHMMSS_STATIONID_SEQUENCE
    return f"CALSTARS_{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"


def lsRemote(log, host, username, port, remote_path):
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

def extractBz2(input_directory, working_directory, host, username, local_target_list=None):
    """Extract BZ2 files from a directory.

    Arguments:
        input_directory: [str] directory containing bz2 files.
        working_directory: [str] directory to work in, possibly a /tmp/ directory.

    Keyword arguments:
        local_target_list: optional, default None, specify files to extract, if None, extract all ending .bz2.

    Returns:
        working_directory
    """


    bz2_list = []
    input_directory = os.path.expanduser(input_directory)
    if local_target_list is None:
        local_target_list = os.listdir(input_directory)
    for filename in local_target_list:
        if filename.endswith(".bz2"):
            bz2_list.append(filename)

    bz2_list.sort()
    mkdirP(working_directory)
    os.chmod(working_directory, 0o777)
    extractBz2Files(bz2_list, input_directory, working_directory, host=host, username=username)

    return working_directory

def extractBz2Files(bz2_list, input_directory, working_directory, silent=True, host=None, username=None, port=PORT):
    """Extract BZ2 files from a directory into a subdirectory of working_directory, if extraction fails, redownload.

    Arguments:
        bz2_list: list file names of bz2 files, paths will be stripped.
        input_directory: directory containing bz2 files.
        working_directory: directory path to hold the subdirectorie of extracted bz2 files.

    Keyword Arguments:
        silent: optional, default True.
        host: optional, default REMOTE_SERVER constant.
        username: optional, default USER_NAME constant.
        port: optional, default PORT constant.

    Return:
        Nothing.
    """

    for bz2 in bz2_list:
        basename_bz2 = str(os.path.basename(bz2))
        station_directory = str(os.path.join(working_directory, basename_bz2.split("_")[0]).lower())
        mkdirP(station_directory)
        bz2_directory = os.path.join(station_directory, basename_bz2.split(".")[0])
        if os.path.exists(bz2_directory):
            continue
        mkdirP(bz2_directory)
        if not silent:
            log.info("Extracting {}".format(bz2))

        try:
            with tarfile.open(os.path.join(input_directory, bz2), 'r:bz2') as tar:
                # todo: work out why this does not support filter keyword argument
                tar.extractall(path=bz2_directory)
        except:
            if not silent:
                log.info("Unable to extract".format(basename_bz2))

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

    return

def getStationList(log, url=STATION_COORDINATES_JSON, country_code=None):
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

def makePlatePar(captured_directory):

    print("No platepar found - this is a placeholder for automatic platepar creation")
    pass

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

def moveFiles(local_target, path_source_list, path_local_list):

    files_available = []
    mkdirP(local_target)
    log.info(f"Setting permissions on {local_target}")
    try:
        os.chmod(os.path.dirname(local_target), 0o777)
        os.chmod(local_target, 0o777)
    except:
        pass
    for p_source, p_local in zip(path_source_list, path_local_list):
        if os.path.exists(p_source):


            shutil.move(p_source, p_local)
            os.chmod(p_local, 0o666)
            files_available.append(os.path.basename(p_local))
    return files_available

def getFromRemote(conn, host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path, bw_limit=None):


    parts = remote_file.split("_")
    if len(parts) < 4:
        log.error(f"Unexpected remote filename format: {remote_file}")
        return None, None, None, None, 0

    local_dir_name = os.path.join("_".join(remote_file.split("_")[1:2]), "_".join(remote_file.split("_")[0:4]))
    calstars_name = f"CALSTARS_{os.path.basename(local_dir_name)}.txt"
    full_remote_path_to_bz2 = os.path.join(remote_dir, remote_file)
    local_target = os.path.join(calstars_data_full_path, local_dir_name)

    with tempfile.TemporaryDirectory() as t:

        # Download from remote
        if downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=port, bw_limit=bw_limit):
            pass
            #log.info(f"Downloaded {full_remote_path_to_bz2} to {local_target}")
            size_mb = os.path.getsize(os.path.join(t, remote_file)) / 1024 ** 2
        else:
            log.warning(f"Failed to download {full_remote_path_to_bz2} to {local_target}")
            return None, None, None, None, 0
        # Create a directory
        extraction_dir = os.path.join(t, "extracted")
        mkdirP(extraction_dir)

        # And extract
        extractBz2(t, extraction_dir, host, username)

        # Create the expected paths for all the extracted files
        extracted_files_path = os.path.join(extraction_dir, station_name.lower(), remote_file.split(".")[0])

        if not os.path.exists(extracted_files_path):
            log.error(f"Extraction failed or unexpected structure: {extracted_files_path}")
            return None, None, None, None, 0

        extracted_config_path = os.path.join(extracted_files_path, ".config")
        extracted_platepar_path = os.path.join(extracted_files_path, config.platepar_name)
        extracted_mask_path = os.path.join(extracted_files_path, config.mask_file)
        extracted_recalibrated_path = os.path.join(extracted_files_path, PLATEPARS_ALL_RECALIBRATED_JSON)
        extracted_flux_recalibrated_path = os.path.join(extracted_files_path, PLATEPARS_FLUX_RECALIBRATED_JSON)
        extracted_calstars_path = os.path.join(extracted_files_path, calstars_name)
        ftp_detect_info_name = f"{FTP_DETECT_INFO_TEMPLATE}{os.path.basename(local_dir_name)}.txt"
        extracted_ftp_detect_info_path = os.path.join(extracted_files_path,ftp_detect_info_name)

        # Place in a list
        path_source_list = [extracted_config_path, extracted_platepar_path,
                            extracted_mask_path,   extracted_calstars_path,
                            extracted_recalibrated_path, extracted_flux_recalibrated_path,
                            extracted_ftp_detect_info_path]

        # Create the expected paths for all the files in the data directory
        local_config_path = os.path.join(local_target, os.path.basename(config.config_file_name))
        local_platepar_path = os.path.join(local_target, config.platepar_name)
        local_mask_path = os.path.join(local_target, config.mask_file)
        local_calstars_path = os.path.join(local_target, calstars_name)
        local_recalibrated_path = os.path.join(local_target, PLATEPARS_ALL_RECALIBRATED_JSON)
        local_ftp_detect_info_path = os.path.join(local_target, ftp_detect_info_name)

        # Place in a list
        path_local_list = [local_config_path, local_platepar_path, local_mask_path, local_calstars_path,
                           local_recalibrated_path, local_ftp_detect_info_path]

        # Move the files from the tempdir to the target dir
        files_available = moveFiles(local_target, path_source_list, path_local_list)

    return local_config_path, local_platepar_path, local_recalibrated_path, calstars_name, size_mb

def extractSessionNameFromCalstar(calstars_path):
    """
    Extract the RMS session name from a CALSTARS filename.

    Example:
        CALSTARS_AU000C_20260323_104736_740403.txt
    returns:
        AU000C_20260323_104736
    """
    base = os.path.basename(calstars_path)

    # Strip .txt
    if base.endswith(".txt"):
        base = base[:-4]

    parts = base.split("_")

    # Expected:
    # ["CALSTARS", "AU000C", "20260323", "104736", "740403"]
    station_id = parts[1]
    date = parts[2]
    time = parts[3]

    return f"{station_id}_{date}_{time}"

def ingestWorker(remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db=True, catalog_stars=None, bw_limit=None, break_on_exception=True, force_job=None, force_error=False):

    # Each worker must open its own DB connection
    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as worker_conn:

        if force_error:
            job_to_force = getNextErrorJob(worker_conn)
            if job_to_force is None:
                log.info("No error jobs found - returning")
                return
        else:
            job_to_force = force_job


        # Claim the job (forced or normal)
        row = claimNextJob(worker_conn, force_job=job_to_force)

        # If no job was returned, exit cleanly
        if row is None:
            log.info("No job available for this worker")
            return

        remote_file, jd_scaled = row

        if remote_file is None:
            return


        if break_on_exception:
            processServerFile(worker_conn, remote_file, remote_station_processed_dir, username, host, port,
                              calstars_data_full_path, write_db, catalog_stars, bw_limit=bw_limit)
            markJobDone(worker_conn, remote_file)

        else:

            try:
                processServerFile(worker_conn, remote_file, remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db, catalog_stars, bw_limit=bw_limit)
                markJobDone(worker_conn, remote_file)

            except Exception as e:
                tb = traceback.format_exc()
                error_msg = f"{type(e).__name__}: {e}\n{tb}"
                log.error(error_msg)
                # Put back in an archive in all cases
                local_dir_name = "_".join(remote_file.split("_")[0:4])
                archiveCalstarDirectories(worker_conn, calstars_data_full_path, [local_dir_name], ingested_only=True)
                markJobError(worker_conn, remote_file, str(e))

def processServerFile(conn=None, remote_file=None, remote_station_processed_dir=None, username=None, host=None, port=None,
                      calstars_data_full_path=None, bw_limit=None):


    processing_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    station_name = remote_file.split("_")[0]
    remote_dir = remote_station_processed_dir.replace("stationID", station_name.lower())

    file_type = getFileType(remote_file)
    if file_type != "metadata" and file_type != "detected":
        log.info(f"File of type {file_type} not required")
        return

    local_dir_name = "_".join(remote_file.split("_")[0:4])
    local_target = os.path.join(calstars_data_full_path, f"{local_dir_name.split('_')[1]}", local_dir_name)

    # If we already have a .bz2 file, extract it so we can work on it
    local_calstars_archive_path = f"{local_target}_raw.tar.bz2"
    extractCalstarArchives(os.path.dirname(local_calstars_archive_path), [os.path.basename(local_calstars_archive_path)], remove_archives=True)
    calstars_name = f"CALSTARS_{local_dir_name}.txt"

    # If we don't have a directory, then get from remote working in a temporary directory
    if not os.path.exists(os.path.join(calstars_data_full_path, local_dir_name.split("_")[1], local_dir_name)):
        log.info(f"Retrieving {remote_file} from {username}@{host}:/{remote_dir}")
        local_config_path, local_platepar_path, local_recalibrated_path, calstars_name, size_mb = getFromRemote(conn, host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path, bw_limit=bw_limit)
    else:
        log.info(f"Cache hit for {remote_file}")

    archiveCalstarDirectories(conn, calstars_data_full_path, [local_dir_name], ingested_only=True)

    return


def discoverRemoteFiles(log, stations, username, host, port,
                        remote_processed_dir_template, postgresql_host="192.168.217.212",
                        min_interval_sec=1, target_interval_sec=3):


    filtered_files = []

    # Initialise cadence
    start_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    next_allowed = start_time
    for idx, station in enumerate(stations, start=1):

        # Start of this iteration is the scheduled cadence time
        iteration_start = next_allowed
        remote_dir = remote_processed_dir_template.replace("stationID", station.lower())

        retry = 3

        # Retry loop for Fail2ban-style blocks
        while retry > 0:
            retry -= 1
            try:
                station_files = lsRemote(log, host, username, port, remote_dir)
                break

            except Exception as e:
                msg = str(e).lower()

                # Detect Fail2ban / SSH refusal
                if "connection refused" in msg or "unexpectedly closed" in msg:
                    pause = random.uniform(600, 900)
                    log.warning(
                        f"Fail2ban likely active for {station}. "
                        f"Sleeping until {(datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) + datetime.timedelta(seconds=pause)).replace(microsecond=0).isoformat()} before retrying."
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
        valid_tars = []
        for file_name in station_files:
            if file_name.endswith("tar.bz2") and len(file_name.split("_")) == 5 and file_name.startswith(station.upper()) and "imgdata" not in file_name:
                valid_tars.append(file_name)
                filtered_files.append(file_name)

        next_allowed = iteration_start + datetime.timedelta(seconds=target_interval_sec)

        delay = (next_allowed - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        time.sleep(max(min_interval_sec, delay))

    filtered_files = sortFilesByTime(filtered_files)

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

def saveRemoteFiles(log, remote_files, json_path):

    serialisable = [{"file_name": file_name} for file_name in remote_files]


    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)

def loadRemoteFiles(json_path, country_code=""):

    if not os.path.exists(json_path):
        getRemoteFileList(country_code=country_code)

    with open(json_path, "r") as f:
        data = json.load(f)
    return [item["file_name"] for item in data]

def extractMedianPixelScale(observation_dict):
    pixel_scale_h_values = []
    pixel_scale_v_values = []

    for frame in observation_dict.values():
        for obs in frame:
            pixel_scale_h_values.append(obs["pixel_scale_h"])
            pixel_scale_v_values.append(obs["pixel_scale_v"])

    median_h = float(np.median(pixel_scale_h_values))
    median_v = float(np.median(pixel_scale_v_values))

    return median_h, median_v



def createSavePathName(file_path, file_name, extension='png'):



    save_path_name = os.path.join(file_path, f"{file_name.split('.')[0]}.{extension}")

    return save_path_name

def fmt(val, fmt_str):
    return fmt_str.format(val) if val is not None else "—"


def resetIngestion(conn, local_calstars_path, ingestion_marker):

    dir_contents = sorted(os.listdir(local_calstars_path))
    for object in dir_contents:
        object_full_path = os.path.join(local_calstars_path, object)
        if os.path.isdir(object_full_path):
            calstar_dir_contents = os.listdir(object_full_path)
            if ingestion_marker in calstar_dir_contents:
                os.unlink(os.path.join(object_full_path, ingestion_marker))


        elif os.path.isfile(object_full_path):
            extractCalstarArchives(local_calstars_path, object)
            dir_name = object.replace("_CALSTAR.tar.bz2", "")

            calstar_path = os.path.join(local_calstars_path, dir_name)

            log.info(f"Extracted {object} to {os.path.basename(calstar_path)}")
            calstar_dir_contents = os.listdir(calstar_path)
            if ingestion_marker in calstar_dir_contents:
                os.unlink(os.path.join(calstar_path, ingestion_marker))
            archiveCalstarDirectories(conn, local_calstars_path, [os.path.basename(calstar_path)], ingested_only=False)
            pass

def populateWorkQueue(conn, file_name_list, log):
    """
    Insert each remote filename into ingest_work with ON CONFLICT DO NOTHING.
    This is the correct simplified version for station-by-station housekeeping.
    """

    file_count = len(file_name_list)
    log.info(f"Sorting {file_count} files by time")
    file_name_list = sortFilesByTime(file_name_list)


    now = datetime.datetime.now(tz=datetime.timezone.utc)
    now_jd_int = scale1e6(date2JD(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second,
        now.microsecond / 1000
    ))

    log.info("Starting write")

    with conn.cursor() as cur:
        i, step = 0, 1000
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        for file_name in file_name_list:
            i += 1

            if i % step == 0:
                now = datetime.datetime.now(tz=datetime.timezone.utc)
                time_elapsed_seconds = (now - start_time).total_seconds()
                jobs_remaining = file_count - i
                completion_time = (now + datetime.timedelta(seconds = jobs_remaining * (time_elapsed_seconds / i))).replace(microsecond=0)
                step *= 2
                log.info(f"Prepared {i} of {file_count} files completion time is {completion_time}")
            # Compute JD for this file
            dt = FFfile.getMiddleTimeFF(
                file_name, fps=25,
                ret_milliseconds=True, ff_frames=256
            )
            jd = date2JD(*dt)
            jd_int = scale1e6(jd)

            if jd_int is None:
                continue

            # Reject future-dated files
            if jd_int > now_jd_int:
                log.info(f"Rejecting file {file_name} as observation start time is in the future")
                continue

            # Insert directly into ingest_work
            cur.execute("""
                INSERT INTO ingest_work (remote_filename, jd_int)
                VALUES (%s, %s)
                ON CONFLICT (remote_filename) DO NOTHING;
            """, (file_name, jd_int))

        log.info("Starting commit")
        conn.commit()
        log.info("Commit completed")


def archiveWholeDir(full_path_to_dir, verbose=False):

    dirs_to_archive = []
    conn = None
    local_dir_list =  os.listdir(full_path_to_dir)
    for object in local_dir_list:
        full_path_to_object = os.path.join(full_path_to_dir, object)
        if os.path.isdir(full_path_to_object):
            if verbose:
                log.info(f"Adding directory {object} to archive list")
            dirs_to_archive.append(full_path_to_object)

    archiveCalstarDirectories(conn, full_path_to_dir, dirs_to_archive, ingested_only=False)

def buildCache(config, remote_files_sorted, calstars_data_dir, history_days=21, username=None, host=None, remote_station_processed_dir=None, port=22, bw_limit=None):

    calstars_data_full_path = os.path.join(config.data_dir, calstars_data_dir)

    conn = None

    archiveWholeDir(calstars_data_full_path)

    remote_files_sorted = filterByDate(remote_files_sorted, earliest_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(history_days))
    remote_files_sorted = sortFilesByTime(remote_files_sorted)

    cache_build_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    files_downloaded = 0
    downloaded_mb = 0
    files_to_download = len(remote_files_sorted)
    for remote_file in remote_files_sorted:
        station_name = remote_file.split("_")[0]
        remote_dir = remote_station_processed_dir.replace("stationID", station_name.lower())

        file_type = getFileType(remote_file)
        if file_type != "metadata" and file_type != "detected":
            log.info(f"File of type {file_type} not required")
            continue

        local_dir_name = "_".join(remote_file.split("_")[0:4])

        local_target = os.path.join(calstars_data_full_path, local_dir_name)

        # If we already have a .bz2 file then skip this
        local_calstars_archive_path = f"{local_target}_CALSTAR.tar.bz2"
        if os.path.exists(local_calstars_archive_path):
            files_to_download -= 1
            continue

        calstars_name = f"CALSTARS_{local_dir_name}.txt"
        local_config_path = os.path.join(local_target, os.path.basename(config.config_file_name))
        if os.path.basename(config.config_file_name) != ".config":
            log.warning(f"Unusual .config filename {config.config_file_name}")
            pass

        full_path_to_downloaded_file = os.path.join(calstars_data_full_path, f"{local_dir_name}_CALSTAR.tar.bz2")
        # If we don't have an archive, then get from remote working in a temporary directory
        if not os.path.exists(full_path_to_downloaded_file):
            _, _, _, _, size_mb = getFromRemote(conn, host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path, bw_limit=bw_limit)
            downloaded_mb += size_mb
        archiveWholeDir(calstars_data_full_path)
        if os.path.exists(full_path_to_downloaded_file):
            files_downloaded += 1

        time_elapsed = datetime.datetime.now(tz=datetime.timezone.utc) - cache_build_start_time
        if files_downloaded > 0:
            file_download_time_seconds = time_elapsed.total_seconds() / files_downloaded
            total_download_time_seconds = file_download_time_seconds * files_to_download
            cumulative_mb_s = downloaded_mb / time_elapsed.total_seconds()
            completion_time_str = (datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=total_download_time_seconds)).replace(microsecond=0).isoformat()
            log.info(f"{remote_file} {size_mb:.1f}MB rate {cumulative_mb_s:.1f}MB/s estimated cache download completion time {completion_time_str} {files_downloaded}/{files_to_download}")
    return






def getRemoteFileList(log, user, hostname, port=22, path_template="/home/stationID/files/incoming/processed", country_code=""):

    station_list = getStationList(log, country_code=country_code)
    remote_files = discoverRemoteFiles(log, station_list, user, hostname, 22, remote_processed_dir_template=path_template)
    remote_files = sortFilesByTime(remote_files)
    saveRemoteFiles(log, remote_files, os.path.join(ARCHIVE,REMOTE_FILES_JSON))


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Ingest CALSTAR data \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('path_template', help="""Template to remote file stores i.e. user@host:/home/stationID/files/ """)

    arg_parser.add_argument('-t', '--threads', type=int,  default=2, help="""Number of concurrent threads to run """)


    arg_parser.add_argument('--write_db', dest='write_db', default=False, action="store_true",
                            help="Write to db")

    arg_parser.add_argument('-r', '--reset_ingestion', dest='reset_ingestion', default=False, action="store_true",
                            help="Reset all ingestion markers")

    arg_parser.add_argument('-p', '--populate_ingestion_table', dest='populate_ingestion_table', default=False, action="store_true",
                            help="Populate ingestion table and then quit immediately")

    arg_parser.add_argument('-s', '--reset_stalled', dest='reset_stalled', default=False, action="store_true",
                            help="Reset stalled jobs and then quit immediately")

    arg_parser.add_argument('-b', '--build_cache', dest='build_cache', default=False, action="store_true",
                            help="Build local files cache")

    arg_parser.add_argument('-g', '--get_remote_file_list', dest='get_remote_file_list', default=False, action="store_true",
                            help="Build the list of remote files")

    arg_parser.add_argument('--create_db', dest='create_db', default=False,
                            action="store_true",
                            help="Create the database")

    arg_parser.add_argument('--country_code', metavar='COUNTRY_CODE', help="""Country code to work on""")

    arg_parser.add_argument('--bw_limit', metavar='BANDWIDTH_LIMIT', help="""Bandwidth limit per thread""")

    arg_parser.add_argument('--calstars_data_dir', metavar='CALSTARS_DATA_DIR', help="""Calstars data dir""")

    arg_parser.add_argument('--force_error', dest='force_error', default=False,
                            action="store_true",
                            help="Sisyphus mode - only work on the jobs which show error with exception handling disabled. This is to assist debugging.")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))
    country_code = cml_args.country_code
    create_db = cml_args.create_db
    build_cache = cml_args.build_cache
    write_db = cml_args.write_db
    get_remote_file_list = cml_args.get_remote_file_list
    bw_limit = cml_args.bw_limit
    force_error = cml_args.force_error

    calstars_data_dir = os.path.join(config.data_dir, CALSTARS_DATA_DIR) if cml_args.calstars_data_dir is None else cml_args.calstars_data_dir

    user, _, remainder = cml_args.path_template.partition("@")
    hostname, _, path_template = remainder.partition(":")

    calstars_directory_path = calstars_data_dir

    print(f"Checking for {calstars_directory_path}")
    if not os.path.exists(calstars_directory_path):
        Path(calstars_directory_path).mkdir(parents=True, exist_ok=True)

    directories_list = os.listdir(calstars_directory_path)


    # Initialize the logger
    log_manager = LoggingManager()


    # Get the logger handle
    log = getLogger("rmslogger")

    if get_remote_file_list:
        getRemoteFileList(log, user, hostname, path_template=path_template, country_code=country_code)

    if build_cache:
        remote_files = loadRemoteFile(os.path.join(ARCHIVE, REMOTE_FILES_JSON), country_code=country_code)
        remote_files_sorted = sortFilesByTime(remote_files)
        buildCache(config, remote_files_sorted, calstars_data_dir, username=user, host=hostname, remote_station_processed_dir=path_template)
        sys.exit()

