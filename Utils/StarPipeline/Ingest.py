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

import logging
import math

from RMS.Formats.FFfile import getMiddleTimeFF

"""
Database configuration instructions

1. Optional - drop the whole database, for a fresh start

psql -h 192.168.1.174 -U postgres -d postgres

DROP DATABASE IF EXISTS star_data;

\q

2. Create the database

psql -h 192.168.1.174 -U postgres -d postgres

CREATE DATABASE star_data OWNER postgres;

\q

3. Connect to the new database

psql -h 192.168.1.174 -U postgres -d star_data

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public AUTHORIZATION postgres;

GRANT CREATE ON SCHEMA public TO ingest_user;
GRANT CREATE ON DATABASE star_data TO ingest_user;
GRANT USAGE  ON SCHEMA public TO ingest_user;

\q
"""

"""
CALSTARS Database Schema (PostgreSQL)
-------------------------------------

All scaled numeric fields in this schema are scaled by 1e6.


station
-------
station_id        CHAR(6) PRIMARY KEY
name              TEXT
notes             TEXT


session - one session per calstars file
---------------------------------------
session_name      TEXT PRIMARY KEY   -- reduced form of the CALSTARS file name
station_id        CHAR(6) REFERENCES station(station_id)

start_jd          BIGINT             -- scaled x1e6
end_jd            BIGINT             -- scaled x1e6

pixel_scale_h     INTEGER            -- scaled x1e6 (arcsec per pixel)
pixel_scale_v     INTEGER            -- scaled x1e6 (arcsec per pixel)

lat               INTEGER            -- latitude scaled x1e6
lon               INTEGER            -- longitude scaled x1e6
elevation         INTEGER            -- elevation (meters) scaled x1e6

config_hash       CHAR(32)           -- pipeline/configuration hash
comment           TEXT               -- free-form session notes


frame - one frame per FITS file
-------------------------------
frame_name        TEXT PRIMARY KEY   -- the FITS file name
session_name      TEXT REFERENCES session(session_name)

jd_mid            BIGINT             -- mid-exposure JD scaled x1e6
frame_index       INTEGER
quality_flags     SMALLINT           -- bitwise quality flags


star
----
star_name         TEXT PRIMARY KEY   -- taken from the GMN star catalog
ra                INTEGER            -- scaled x1e6
dec               INTEGER            -- scaled x1e6
mag               INTEGER            -- scaled x1e6 (catalog magnitude)
catalog_source    TEXT
canonical_name    TEXT


observation
-----------
obs_id            BIGSERIAL PRIMARY KEY

frame_name        TEXT REFERENCES frame(frame_name)
star_name         TEXT REFERENCES star(star_name)

y                 INTEGER            -- pixel Y
x                 INTEGER            -- pixel X

intens_sum        INTEGER            -- scaled x1e6
ampltd            INTEGER            -- scaled x1e6
fwhm              INTEGER            -- scaled x1e6
bg_lvl            INTEGER            -- scaled x1e6
snr               INTEGER            -- scaled x1e6
nsatpx            SMALLINT

mag               INTEGER            -- scaled x1e6 (instrumental magnitude)
mag_err           INTEGER            -- scaled x1e6

flags             SMALLINT           -- bitwise flags


Notes
-----
- Only obs_id is a surrogate key; all other relationships use natural keys.
- Ingestion uses ON CONFLICT DO NOTHING for idempotency.
- Schema is intentionally minimal and integer-heavy for performance and scale.
- Session-level metadata (pixel scale, lat, lon, elevation) ensures reproducibility.


"""






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
from multiprocessing import Pool
from collections import defaultdict
from Utils.StarPipeline.PipelineDB import createDatabaseIfMissing, initialiseDatabase, Flags, auditIngestUserPrivileges

JD_OFFSET = J2000_JD
DEBUG_CALSTAR_INSERT = False

#Most floats are multiplied by this scale factor and stored as INTEGER
DB_SCALE_FACTOR = 1e6

#List and types for any columns which are not INTEGER
TYPE_MAP = {
    "stationID": "TEXT",
    "jd": "BIGINT",
}

# Constants

# urls
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"

# Paths and names
CALSTARS_DATA_DIR = "CALSTARS"
PLATEPARS_ALL_RECALIBRATED_JSON = "platepars_all_recalibrated.json"
DIRECTORY_INGESTED_MARKER = ".processed"
FILE_SYSTEM_MARKERS_ENABLED = False
CALSTAR_FILES_TABLE_NAME = "calstar_files"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
CHARTS = "charts"
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

def scale1e3(value):
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
    return int(round(value * 1000))

def buildFrameRows(observation_dict, session_name):
    frame_rows = []

    for fits_file, frame_list in observation_dict.items():
        frame_name = extractFrameName(fits_file)
        frame_index = extractFrameIndex(fits_file)

        # If there are no stars, then do no more work here
        if not frame_list:
            log.info(f"{fits_file} had no stars, skipping")
            continue

        # Get JD from any star entry (all stars in frame share same JD)
        first_obs = frame_list[0]
        jd_mid = scale1e6(first_obs["jd"])

        quality_flags = observation_dict[fits_file][0]['flag']
        median_absolute_deviation = scale1e6(observation_dict[fits_file][0]['mad'])

        frame_rows.append((
            frame_name,
            session_name,
            jd_mid,
                frame_index,
            quality_flags,
            median_absolute_deviation
        ))

    return frame_rows

def buildStarRows(observation_dict):
    star_set = set()

    for frame_list in observation_dict.values():
        for obs in frame_list:
            if obs["name"] is None:
                continue

            star_set.add((
                obs["name"],
                obs["station_name"],
                scale1e6(obs["cat_ra"]),
                scale1e6(obs["cat_dec"]),
                scale1e6(obs["cat_mag"]),
                "RMS",
                None
            ))

    return list(star_set)

def buildObservationRows(observation_dict, session_name, station_name):
    observation_rows = []

    for fits_file, frame_list in observation_dict.items():
        frame_name = extractFrameName(fits_file)
        frame_jd_mid = scale1e6(frame_list[0]["jd"])

        # frame_list is now a list of observation dicts
        for obs in frame_list:
            observation_rows.append((
                frame_name,
                obs["name"],
                scale1e6(obs["obs_y"]),
                scale1e6(obs["obs_x"]),
                obs["intens_sum"],
                obs["ampltd"],
                scale1e6(obs["fwhm"]),
                obs["bg_lvl"],
                scale1e6(obs["snr"]),
                obs["nsatpx"],
                scale1e6(obs["obs_mag"]),
                scale1e6(obs["cat_mag"]),
                scale1e6(obs["err_mag"]),
                scale1e6(obs["obs_ra"]),
                scale1e6(obs["obs_dec"]),
                session_name,
                station_name,
                frame_jd_mid,
                obs["flag"],
                scale1e6(obs["mad"]),
                scale1e3(obs["sun_angle"])
            ))

    return observation_rows

def buildAllRows(observation_dict, session_name):
    frame_rows = buildFrameRows(observation_dict, session_name)
    star_rows = buildStarRows(observation_dict)
    station_name = session_name[:6]
    observation_rows = buildObservationRows(observation_dict, station_name=station_name, session_name=session_name)

    return frame_rows, star_rows, observation_rows

def extractFrameName(fits_file):
    """
    Convert an RMS FITS filename into the canonical frame_name key.
    Example:
        FF_AU000A_20260319_111559_925_0009216.fits
    becomes:
        AU000A_20260319_111559
    """
    base = fits_file.rsplit("/", 1)[-1]          # strip path
    base = base.replace(".fits", "")             # strip extension

    parts = base.split("_")

    # Expected: ["FF", "AU000A", "20260319", "111559", "925", "0009216"]
    if len(parts) < 4:
        raise ValueError(f"Unexpected FITS filename format: {fits_file}")

    # Drop the leading "FF"
    frame_name = "_".join(parts[1:4])
    return frame_name

def extractFrameIndex(fits_file):
    """
    Extract the frame index from an RMS FITS filename.
    Example:
        FF_AU000A_20260319_111559_925_0009216.fits
    returns:
        925
    """
    base = fits_file.rsplit("/", 1)[-1]
    base = base.replace(".fits", "")

    parts = base.split("_")

    # Expected: ["FF", "AU000A", "20260319", "111559", "925", "0009216"]
    if len(parts) < 5:
        raise ValueError(f"Unexpected FITS filename format: {fits_file}")

    return int(parts[4])

def writeSessionBatch(conn, session_name, station_id, start_jd, end_jd, pixel_scale_h, pixel_scale_v,
                      frame_rows, star_rows, observation_rows, session_config=None):
    """
    Write one full CALSTARS session to the database in a single transaction.
    """

    observation_count = len(observation_rows)
    log.info(f"Starting write for {session_name} with {observation_count} entries.")

    if session_config is None:
        lat, lon, elevation = None, None, None
    else:
        lat = session_config.latitude
        lon = session_config.longitude
        ele = session_config.elevation

    if not observation_count:
        return observation_count

    try:
        with conn.cursor() as cur:

            # Ensure station exists
            cur.execute(
                "INSERT INTO station (station_name) VALUES (%s) ON CONFLICT DO NOTHING",
                (station_id,)
            )

            # Insert session
            cur.execute("""
                        INSERT INTO session (session_name,
                                             station_name,
                                             start_jd,
                                             end_jd,
                                             pixel_scale_h,
                                             pixel_scale_v,
                                             lat,
                                             lon,
                                             elevation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                        """,
                        (
                            session_name,
                            station_id,
                            scale1e6(start_jd),
                            scale1e6(end_jd),
                            scale1e6(pixel_scale_h),
                            scale1e6(pixel_scale_v),
                            scale1e6(lat),
                            scale1e6(lon),
                            scale1e3(ele)
                        ))

            # Insert frames
            cur.executemany("""
                INSERT INTO frame (frame_name, session_name, jd_mid, frame_index, quality_flags, mad)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, frame_rows)

            # Insert stars
            cur.executemany("""
                INSERT INTO star (star_name, station_name, ra, dec, mag, catalog_source, canonical_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, star_rows)

            # Insert observations
            cur.executemany("""
                INSERT INTO observation (
                    frame_name,
                    star_name,
                    y, x,
                    intens_sum, ampltd, fwhm, bg_lvl, snr, nsatpx,
                    mag, cat_mag, mag_err, ra, dec,
                    session_name,
                    station_name,
                    jd_mid,
                    flags, 
                    mad,
                    sun_angle
                )
                VALUES (%s, %s,
                        %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s,        -- session_name
                        %s,        -- station_name
                        %s,
                        %s,
                        %s,
                        %s)
                ON CONFLICT DO NOTHING;
                """, observation_rows)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    return observation_count

def ensureList(value):
    """Return: list containing value, or value itself if already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    raise TypeError(f"Expected str or list, got {type(value).__name__}")

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
    archives_list = ensureList(archives_list)

    for archive_name in archives_list:
        arc_name = Path(archive_name)
        archive_full_path = root / arc_name
        if not archive_full_path.exists():
            continue
        log.info(f"Extracting {archive_name}")


        if not archive_full_path.is_file():
            continue

        # Derive original directory name
        dir_name = arc_name.name.replace("_CALSTAR.tar.bz2", "")
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
                    output_dir_list.append(output_dir)
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

def archiveCalstarDirectories(conn, root, directories_list, ingested_only=True):
    """Given a list of directories, archive them named e.g. AU0004_20260317_111157_992974_CALSTAR.tar.bz2,
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
        source_dir = root / d
        if not os.path.isdir(source_dir):
            continue
        # Check for ingestion marker inside the directory
        if not isIngested(conn, source_dir) and ingested_only:
            log.info(f"Not archiving {os.path.basename(source_dir)} not yet ingested")
            continue



        tar_file_name = f"{d}_CALSTAR.tar.bz2"
        output_file = root / tar_file_name

        #log.info(f"Creating {os.path.basename(output_file)}")
        makeTarBz2(source_dir, output_file)

        # compute uncompressed size
        uncompressed_size = getDirectorySize(source_dir)  / (1024 ** 2)
        # compute compressed size
        compressed_size = output_file.stat().st_size / (1024 **2)

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

def recordCalstarFileIngested(conn, file_name):
    ingestion_time = int(time.time() * 1_000_000)

    sql = """
          INSERT INTO calstar_files (file_name, ingestion_time)
          VALUES (%s, %s) ON CONFLICT (file_name)
            DO \
          UPDATE SET ingestion_time = EXCLUDED.ingestion_time; \
          """

    with conn.cursor() as cur:

        # postgresql debugging code
        if DEBUG_CALSTAR_INSERT:
            cur.execute("SELECT current_user, current_database(), current_schema();")
            user, db, schema = cur.fetchone()
            log.warning("DB CONTEXT BEFORE INSERT: user=%s db=%s schema=%s", user, db, schema)
            cur.execute("SELECT table_schema FROM information_schema.tables WHERE table_name = 'calstar_files';")
            log.warning("SCHEMAS WHERE calstar_files EXISTS: %s", [row[0] for row in cur.fetchall()])
            cur.execute("SHOW search_path;")
            log.warning("SEARCH PATH BEFORE INSERT: %s", cur.fetchone()[0])

            log.warning(f"About to execute {sql}")
        cur.execute(sql, (file_name, ingestion_time))

        if DEBUG_CALSTAR_INSERT:
            log.warning(f"Executed {sql}")
    conn.commit()

def markIngested(conn, directory_path):
    """Save the ingested marker file into the folder_path.

    Arguments:
        directory_path: [str] Path to folder.

    Returns:
        Nothing.
    """

    calstar_filename = buildCalstarFilename(directory_path)
    recordCalstarFileIngested(conn, calstar_filename)
    directory_path = Path(directory_path)
    marker_file = directory_path / ".ingested"
    log.info(f"Marked {os.path.basename(directory_path)} as ingested")
    marker_file.touch()

def isIngestedFromFileSystem(directory_path):
    """If the folder_path contains the ingested file marker, return True.

    Arguments:
        directory_path: [str] Path to folder.

    Returns:
        [bool]: True if ingested file marker in directory, else false.
    """
    directory_path = Path(directory_path)
    marker_file = directory_path / ".ingested"
    if marker_file.exists():
        log.info(f"{os.path.basename(directory_path)} was already ingested")
        return True
    else:
        return False

def isIngestedFromDB(conn,file_name):

    sql = "SELECT 1 FROM calstar_files WHERE file_name = %s;"

    with conn.cursor() as cur:
        cur.execute(sql, (file_name,))
        return cur.fetchone() is not None

def buildCalstarFilename(calstar_directory_path):
    base = os.path.basename(calstar_directory_path)
    parts = base.split("_")

    # Expecting: YYYYMMDD_HHMMSS_STATIONID_SEQUENCE
    # Produces:  CALSTARS_YYYYMMDD_HHMMSS_STATIONID_SEQUENCE
    return f"CALSTARS_{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"

def isIngested(conn, calstar_directory_path):
    calstar_filename = buildCalstarFilename(calstar_directory_path)
    log.info(f"Checking ingestion status for {calstar_filename}")

    # Primary guard: database
    if isIngestedFromDB(conn, calstar_filename):
        log.info(f"{calstar_filename} is recorded as ingested in the database")
        return True

    # Secondary guard: filesystem marker
    if isIngestedFromFileSystem(calstar_directory_path) and FILE_SYSTEM_MARKERS_ENABLED:
        log.warning(f"Ingested in filesystem marker but not in DB: {calstar_filename}")
        recordCalstarFileIngested(conn, calstar_filename)
        return True

    return False

def dictInvert(d):
    """Given a nested dictionary with keys [a][b] return the nested dictionary with keys [b][a].

    Arguments:
        d:[dict] Nested dictionary.

    Returns:
        [dict] Nested dictionary.
    """
    out = {}

    for key, subdict in d.items():
        for subkey, value in subdict.items():
            if subkey not in out:
                out[subkey] = {}
            out[subkey][key] = value

    return out

def lsRemote(host, username, port, remote_path):
    """Return: list of filenames in remote directory, or empty list if directory does not exist."""

    remote = "{}@{}:{}".format(username, host, remote_path)
    cmd = [
        "rsync",
        "--list-only",
        "-z",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -p {}".format(port),
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
    log.info(f"Remote directory {remote_path} contained {remote_file_count} {word}")

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

def downloadFile(host, username, local_path, remote_path, port=PORT,  silent=False):
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
            result = subprocess.run(['rsync', '-z', remote, local_path], capture_output=True, text=True)
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

def makePlatePar(captured_directory):

    print("No platepar found - this is a placeholder for automatic platepar creation")
    pass

def createColumns(conn, table, columns):
    """Create columns in a database if they do not already exist, use types from TYPEMAP.

    Arguments:
        conn: [object] Database connection.
        table: [str] Table name.
        columns: [list] List of columns to add.

    Returns:
        Nothing
    """

    with conn.cursor() as cur:
        # Fetch existing columns
        cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    """, (table,))
        existing_columns = {row[0] for row in cur.fetchall()}

        for col in columns:
            if col in existing_columns:
                continue

            coltype = TYPE_MAP.get(col, "INTEGER")

            cur.execute(
                f"ALTER TABLE {table} "
                f"ADD COLUMN IF NOT EXISTS {col} {coltype}"
            )

def getLeaves(data_dict):
    """Return the keys of the inside dict in a nested dict.

    Args:
        data_dict: [dict] Dictionary.

    Returns:
        [list] Leaves.
    """

    leaves = set()
    for ff_dict in data_dict.values():
        for bottom_dict in ff_dict.values():
            leaves.update(bottom_dict.keys())
    return leaves

def buildUpsertSQL(table, leaves):
    """Given a table name and a list of leaves build the SQL values to update or insert into the table.

    Arguments:
        table: [str] Table name.
        leaves: [list] List of leaves.

    Returns:
        sql: [string] SQL value.
        cols: [list] List of columns.

    """
    cols = ["catalogue_id", "ff_name"] + list(leaves)
    placeholders = ", ".join("%s" for _ in cols)

    update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in leaves)

    sql = f"""
        INSERT INTO {table} ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (catalogue_id, ff_name)
        DO UPDATE SET {update_clause};
    """

    return sql, cols

def dbScaleIn(v):
    """If possible, scale values to be inserted into the database.

    Args:
        v: [str | float] Value

    Returns:
        [str | float] Scaled value
    """

    try:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if np.isinf(v):
            return None


        return int(v * DB_SCALE_FACTOR)

    except Exception as e:
        log.error(f"dbScaleIn failed for value {v!r} of type {type(v)}: {e}")
        pass  # <-- your breakpoint goes here
        raise

def dbScaleOut(v):
    """Invert the database scaling
    Arguments:
        v: [float] Value

    Returns:
        value: [float] Scaled value"""
    if v is None:
        return None

    return float(v / DB_SCALE_FACTOR)

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

def downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=22, max_tries=3):

    remote_file = os.path.basename(full_remote_path_to_bz2)
    download_start_time = datetime.datetime.now(datetime.timezone.utc)
    log.info(f"Downloading {remote_file}")
    download_count = 0
    target_path = os.path.join(t, os.path.basename(full_remote_path_to_bz2))

    while not os.path.exists(target_path) and download_count < max_tries:
        downloadFile(host, username, t, full_remote_path_to_bz2, port=port)
        download_count += 1
        # If the file is now present, break immediately
        if os.path.exists(target_path):
            download_end_time = datetime.datetime.now(datetime.timezone.utc)
            downloaded_size = os.path.getsize(os.path.join(t, remote_file)) / (1000 ** 2)
            rate_mb_s = downloaded_size / (download_end_time - download_start_time).total_seconds()
            log.info(
                f"Downloaded {remote_file} of size {downloaded_size:.2f}MB at {rate_mb_s:.2f} MB/s after {download_count} try")
            return True
        delay = random.randint(600, 900)
        log.info(f"Waiting {delay/60:.1f} minutes for {target_path}")
        time.sleep(delay)



    log.warning(f"Failed to download {remote_file} after {download_count} tries")

    return False

def moveFiles(local_target, path_source_list, path_local_list):

    files_available = []
    for p_source, p_local in zip(path_source_list, path_local_list):
        if os.path.exists(p_source):
            mkdirP(local_target)
            shutil.move(p_source, p_local)
            files_available.append(os.path.basename(p_local))
    return files_available

def markIngestedIfFilesMissing(conn, path_local_list, files_available, local_target):

    for f in path_local_list:
        if os.path.basename(f) not in files_available and f != "mask.bmp":
            # Mark this folder as ingested so we don't waste time on it in future
            log.warning(f"Missing files for {os.path.basename(local_target)}: {f}")
            markIngested(conn, local_target)
            continue

def getFromRemote(conn, host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path):


    parts = remote_file.split("_")
    if len(parts) < 4:
        log.error(f"Unexpected remote filename format: {remote_file}")
        return None, None, None, None

    local_dir_name = "_".join(remote_file.split("_")[0:4])
    calstars_name = f"CALSTARS_{local_dir_name}.txt"
    full_remote_path_to_bz2 = os.path.join(remote_dir, remote_file)
    local_target = os.path.join(calstars_data_full_path, local_dir_name)

    with tempfile.TemporaryDirectory() as t:

        # Download from remote
        if downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=port):
            log.info(f"Downloaded {full_remote_path_to_bz2} to {local_target}")
        else:
            log.warning(f"Failed to download {full_remote_path_to_bz2} to {local_target}")
            return None, None, None, None
        # Create a directory
        extraction_dir = os.path.join(t, "extracted")
        mkdirP(extraction_dir)

        # And extract
        extractBz2(t, extraction_dir, host, username)

        # Create the expected paths for all the extracted files
        extracted_files_path = os.path.join(extraction_dir, station_name.lower(), remote_file.split(".")[0])

        if not os.path.exists(extracted_files_path):
            log.error(f"Extraction failed or unexpected structure: {extracted_files_path}")
            return None, None, None, None

        extracted_config_path = os.path.join(extracted_files_path, ".config")
        extracted_platepar_path = os.path.join(extracted_files_path, config.platepar_name)
        extracted_mask_path = os.path.join(extracted_files_path, config.mask_file)
        extracted_recalibrated_path = os.path.join(extracted_files_path, PLATEPARS_ALL_RECALIBRATED_JSON)
        extracted_calstars_path = os.path.join(extracted_files_path, calstars_name)

        # Place in a list
        path_source_list = [extracted_config_path, extracted_platepar_path, extracted_mask_path,
                            extracted_calstars_path, extracted_recalibrated_path]

        # Create the expected paths for all the files in the data directory
        local_config_path = os.path.join(local_target, os.path.basename(config.config_file_name))
        local_platepar_path = os.path.join(local_target, config.platepar_name)
        local_mask_path = os.path.join(local_target, config.mask_file)
        local_calstars_path = os.path.join(local_target, calstars_name)
        local_recalibrated_path = os.path.join(local_target, PLATEPARS_ALL_RECALIBRATED_JSON)

        # Place in a list
        path_local_list = [local_config_path, local_platepar_path, local_mask_path, local_calstars_path,
                           local_recalibrated_path]

        # Move the files from the tempdir to the target dir
        files_available = moveFiles(local_target, path_source_list, path_local_list)

        # If we are missing key files, then mark ingested - this is not beautiful
        markIngestedIfFilesMissing(conn, path_local_list, files_available, local_target)


    return local_config_path, local_platepar_path, local_recalibrated_path, calstars_name

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

def claimNextJob(conn):
    sql = """
    UPDATE ingest_work
    SET status = 'claimed',
        updated_at = now()
    WHERE remote_path = (
        SELECT remote_path
        FROM ingest_work
        WHERE status = 'pending'
        ORDER BY jd_int ASC, remote_path ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING remote_path, jd_int;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        conn.commit()

    return row if row else None


def markJobDone(conn, remote_path):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_work
            SET status = 'done', updated_at = now()
            WHERE remote_path = %s
            """,
            (remote_path,)
        )
    conn.commit()


def markJobError(conn, remote_path, msg):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_work
            SET status = 'error', updated_at = now()
            WHERE remote_path = %s
            """,
            (remote_path,)
        )
    conn.commit()



def worker(remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db=True, catalog_stars=None):

    # Each worker must open its own DB connection
    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as worker_conn:
        while True:
            remote_file, jd_scaled = claimNextJob(worker_conn)
            if remote_file is None:
                time.sleep(120)
                continue

            try:
                processServerFile(worker_conn, remote_file, remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db, catalog_stars)
                markJobDone(worker_conn, remote_file)
            except Exception as e:
                markJobError(worker_conn, remote_file, str(e))

def chunkByHour(file_list, day_divider=24):

    days = defaultdict(list)
    for f in file_list:
        dt = FFfile.getMiddleTimeFF(f, fps=25, ret_milliseconds=True, ff_frames=256)
        jd = date2JD(*dt)
        day = int(jd * day_divider)
        days[day].append(f)
    return dict(days)


def runParallel(remote_station_processed_dir=None, username=None, host=None, port=None, calstars_data_full_path=None, write_db=True, catalog_stars=None, concurrent_threads=2):


    with Pool(concurrent_threads) as pool:
        args_list = []

        args_list.append(
            (
                remote_station_processed_dir,
                username,
                host,
                port,
                calstars_data_full_path,
                write_db,
                catalog_stars
            )
        )

        results = pool.starmap(worker, args_list)
    return results



def processServerFile(conn=None, remote_file=None, remote_station_processed_dir=None, username=None, host=None, port=None,
                      calstars_data_full_path=None, write_db=True, catalog_stars=None):

    print(f"Entering Process Server File with {remote_file}")
    station_name = remote_file.split("_")[0]
    remote_dir = remote_station_processed_dir.replace("stationID", station_name.lower())

    file_type = getFileType(remote_file)
    if file_type != "metadata" and file_type != "detected":
        log.info(f"File of type {file_type} not required")
        return

    local_dir_name = "_".join(remote_file.split("_")[0:4])

    local_target = os.path.join(calstars_data_full_path, local_dir_name)

    if isIngested(conn, local_target):
        log.info(f"{local_dir_name} already processed")
        return

    log.info(f"Working on {local_dir_name}")
    # If we already have a .bz2 file, extract it so we can work on it
    local_calstars_archive_path = f"{local_target}_CALSTAR.tar.bz2"
    extractCalstarArchives(calstars_data_full_path, [os.path.basename(local_calstars_archive_path)], remove_archives=True)
    calstars_name = f"CALSTARS_{local_dir_name}.txt"
    local_config_path = os.path.join(local_target, os.path.basename(config.config_file_name))
    if os.path.basename(config.config_file_name) != ".config":
        log.warning(f"Unusual .config filename {config.config_file_name}")
        pass

    local_platepar_path = os.path.join(local_target, config.platepar_name)
    local_calstars_path = os.path.join(local_target, calstars_name)
    local_recalibrated_path = os.path.join(local_target, PLATEPARS_ALL_RECALIBRATED_JSON)

    # If we don't have a directory, then get from remote working in a temporary directory
    if not os.path.exists(os.path.join(calstars_data_full_path, local_dir_name)):
        log.info(f"Retrieving {remote_file} from {username}@{host}:/{remote_dir}")
        local_config_path, local_platepar_path, local_recalibrated_path, calstars_name = getFromRemote(conn, host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path)

    if local_config_path is None:
        log.info(f"Skipping {remote_file} because config file not available")
        # Mark ingested, because we don't want to look at this again
        markIngested(conn, local_target)
        return


    if not write_db:
        log.info(f"Data from {local_dir_name} not being written to database as writes not enabled.")
        return

    if write_db:
        log.info(f"Ingesting {calstars_name}")

        observation_session_config = cr.parse(local_config_path)
        observation_session_dict, start_jd, end_jd = calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recalibrated_path, local_calstars_path, catalog_stars=catalog_stars)

        pixel_scale_h, pixel_scale_v = extractMedianPixelScale(observation_session_dict)
        session_name = extractSessionNameFromCalstar(local_calstars_path)
        frame_rows, star_rows, observation_rows = buildAllRows(observation_session_dict, session_name)

        writeSessionBatch(
            conn,
            session_name=session_name,
            station_id=station_name,
            start_jd=start_jd,
            end_jd=end_jd,
            pixel_scale_h=pixel_scale_h,
            pixel_scale_v=pixel_scale_v,
            frame_rows=frame_rows,
            star_rows=star_rows,
            observation_rows=observation_rows,
            session_config=observation_session_config
        )

        markIngested(conn, local_target)


    # Put back in an archive in all cases
    archiveCalstarDirectories(conn, calstars_data_full_path, [local_dir_name], ingested_only=True)

    return

def ingest(config, file_list, conn, calstars_data_dir=CALSTARS_DATA_DIR,
           remote_station_processed_dir=None, write_db=True,
           host=None, username=None, port=PORT, concurrent_threads=2):

    """
    In a subdirectoy of station_data_dir create a directory for each station containing mask
    platepar and config file.

    Arguments:
        config: [config] RMS config instance - used to get data_dir.
        file_list: [list] list of files to retrieve and ingest.
        conn: [object] database connection object.

    Keyword arguments:
        country_code: [str] Country code to work on.
        calstars_data_dir: [str] target name in RMS_data, optional, default STATIONS_DATA_DIR.
        remote_station_processed_dir: [str] path on remote server, optional, default REMOTE_STATION_PROCESSED_DIR.
        host: [str] host name of remote machine, optional, default REMOTE_SERVER.
        username: [str] username for remote machine, optional, default USER_NAME.
        port: [int] optional, default PORT constant, optional, default PORT.
        history_days: [float] optional, default None, number of days to go back when building database.
        write_db: [Bool] optional, default True, write into the database
    Return:
        Nothing.
    """

    with conn.cursor() as cur:
        cur.execute("SELECT current_user;")
        log.info(f"Python is connecting as:{cur.fetchone()[0]}")
        cur.execute("SELECT inet_server_addr(), inet_server_port();")
        log.info(f"Python is connected to:{cur.fetchone()}")

    calstars_data_full_path = os.path.join(config.data_dir, calstars_data_dir)

    catalog_stars = loadCatalogStars(config, config.catalog_mag_limit)


    runParallel(remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db=write_db, catalog_stars=catalog_stars, concurrent_threads=concurrent_threads)

    #Single threading approach - not in use
    """
    log.info("Starting to download files")

    hour_chunks = chunkByHour(file_list, day_divider=48)
    sorted_hours = sorted(hour_chunks.keys())

    for hour in sorted_hours:
        hour_files = sorted(hour_chunks[hour])

        log.info(f"Working on jd {hour} - following files to be processed")
        for f in sortFilesByTime(hour_files):
            log.info(f"\t{f}")
    
    #for f in file_list:
    #    processServerFile(conn, f, remote_station_processed_dir, username, host, port, calstars_data_full_path, write_db=write_db, catalog_stars=catalog_stars)
    """


def getLatestCalstarFile(conn, station_id):
    sql = """
        SELECT file_name
        FROM calstar_files
        WHERE file_name LIKE %s
        ORDER BY ingestion_time DESC
        LIMIT 1;
    """
    pattern = f"CALSTARS_{station_id}_%"

    with conn.cursor() as cur:
        cur.execute(sql, (pattern,))
        row = cur.fetchone()

    return row

def discoverRemoteFiles(stations, username, host, port,
                        remote_processed_dir_template,
                        min_interval_sec=1, target_interval_sec=10):

    filtered_files = []

    # Initialise cadence
    next_allowed = datetime.datetime.now(datetime.timezone.utc)

    for idx, station in enumerate(stations, start=1):
        # Start of this iteration is the scheduled cadence time
        log.info(f"Processing station {idx}/{len(stations)}: {station}")
        iteration_start = next_allowed

        remote_dir = remote_processed_dir_template.replace(
            "stationID", station.lower()
        )

        retry = 3

        # --- Retry loop for Fail2ban-style blocks ---
        while retry > 0:
            retry -= 1
            try:
                station_files = lsRemote(host, username, port, remote_dir)
                break   # success → exit retry loop

            except Exception as e:
                msg = str(e).lower()

                # Detect Fail2ban / SSH refusal
                if "connection refused" in msg or "unexpectedly closed" in msg:
                    pause = random.uniform(600, 900)
                    log.warning(
                        f"Fail2ban likely active for {station}. "
                        f"Sleeping {pause:.1f} seconds before retrying."
                    )
                    time.sleep(pause)
                    continue

                # Other errors → log and skip this station
                log.warning(f"Failed to list remote files for {station}: {e}")
                station_files = []
                break

        # --- Filter valid tarballs ---
        for file_name in station_files:
            if (
                file_name.endswith("tar.bz2")
                and len(file_name.split("_")) == 5
                and file_name.startswith(station.upper())
                and "imgdata" not in file_name
            ):
                filtered_files.append(file_name)

        # --- Advance cadence anchor ---
        next_allowed = iteration_start + datetime.timedelta(seconds=target_interval_sec)

        # --- Sleep until the next scheduled time ---
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

def loadRemoteFiles(json_path):
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

def minSunBelowHorizon(fits_file_list, c, sun_angle=-18, chunk_size=1):

    if not len(fits_file_list):
        return [], np.array([])

    log.info(f"First/last fits file was {fits_file_list[0]}/{fits_file_list[-1]}")
    # Initialize observer
    o = ephem.Observer()
    o.lat, o.lon, o.elevation  = str(c.latitude), str(c.longitude), float(c.elevation)
    sun = ephem.Sun()
    sun.compute(o)
    sun_alt_deg = math.degrees(float(sun.alt))
    last_sun_alt_deg = sun_alt_deg
    angle_list, astronomical_night_list = [], []
    setting_count, rising_count = 0, 0
    for i, fits_file in enumerate(fits_file_list):

        # Recompute Sun altitude every chunk_size frames
        if i % chunk_size == 0:
            o.date = getMiddleTimeFF(fits_file, c.fps, dt_obj=True)
            sun.compute(o)
            sun_alt_deg = math.degrees(float(sun.alt))
            if last_sun_alt < sun_alt_deg:
                setting_count += 1
            elif last_sun_alt > sun_alt_deg:
                rising_count += 1
            else:
                pass
            last_sun_alt = sun_alt_deg
        angle_list.append(sun_alt_deg)

        if sun_alt_deg < sun_angle:
            astronomical_night_list.append(fits_file)
        else:
            pass

    return astronomical_night_list, np.array(angle_list), setting_count, rising_count



def calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recal_path, local_calstars_path, catalog_stars=None):
    """
      Parses a calstar data structures in archived directories path,
      converts to RaDec, corrects magnitude data and writes newer data to database
    """

    ob_flag = Flags()
    obs_con = cr.parse(local_config_path)
    calstars_name = os.path.basename(local_calstars_path)


    if os.path.exists(local_recal_path):
        with open(local_recal_path, 'r') as fh:
            pp_recal_json = json.load(fh)
            log.info(f"Read {os.path.basename(local_recal_path)}")
    else:
        log.info(f"No file {os.path.basename(local_recal_path)}`")
        pp_recal_json = None

    pp = Platepar()

    pp.read(local_platepar_path)

    calstar, chunk = readCALSTARS(os.path.dirname(local_calstars_path), calstars_name)
    star_dict = starListToDict(obs_con, [calstar, chunk])

    if not len(star_dict):
        return {}, None, None

    fits_files_from_calstar_list = [calstar_entry[0] for calstar_entry in calstar]
    total_calstar_fits = len(fits_files_from_calstar_list)

    # Find out which fits_files do not have the illuminated moon in view
    fits_files_without_moon_list = detectMoon(fits_files_from_calstar_list, pp, obs_con)
    dropped_files_count = len(fits_files_from_calstar_list) - len(fits_files_without_moon_list)
    plural = "" if dropped_files_count == 1 else "s"
    log.info(f"Flagging {dropped_files_count} fits file{plural} as disrupted by moon approx {100*dropped_files_count/total_calstar_fits:3.2f}%")



    # Find out which fits files are not in astronomical night
    astronomical_night_list, sun_below_horizon_angle_list, setting_count, rising_count = minSunBelowHorizon(fits_files_from_calstar_list, obs_con, sun_angle=-18)
    dropped_files_count = len(fits_files_from_calstar_list) - len(astronomical_night_list)
    plural = "" if dropped_files_count == 1 else "s"
    log.info(f"Flagging {dropped_files_count}  setting/rising {setting_count}/{rising_count} fits file{plural} as in astronomical dusk or dawn approx {100*dropped_files_count/total_calstar_fits:3.2f}%")

    # Next take the intersection

    observation_dict = {}

    fits_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    dt = FFfile.getMiddleTimeFF(calstar[0][0], obs_con.fps, ret_milliseconds=True, ff_frames=256)
    start_jd = date2JD(*dt)

    dt = FFfile.getMiddleTimeFF(calstar[-1][0], obs_con.fps, ret_milliseconds=True, ff_frames=256)
    end_jd = date2JD(*dt)

    dir_path = os.path.dirname(local_calstars_path)


    auto_pp, matched_star_pairs, used_ff = autoFitPlatepar(dir_path, obs_con, catalog_stars=catalog_stars,
                                                           platepar_template=pp, verbose=False)

    flags = 0
    if auto_pp is None:
        # if autoFitPlatepar can't make sense of the best observation, then set the bad autoFitPlatepar bit and never unset it for this calstar
        flags |= ob_flag.BAD_AUTO_PP
    else:
        flags &= ~ob_flag.BAD_AUTO_PP
        pp = auto_pp


    for fits_file, star_list in calstar:

        if fits_file not in fits_files_without_moon_list:
            # Moon in field of view
            flags |= ob_flag.MOON_IN_FOV
        else:
            flags &= ~ob_flag.MOON_IN_FOV

        if fits_file not in astronomical_night_list:
            # Set sky not fully dark
            flags |= ob_flag.SKY_NOT_FULLY_DARK
        else:
            flags &= ~ob_flag.SKY_NOT_FULLY_DARK

        if len(star_list) >= 40:
            flags &= ~ob_flag.FEW_STARS
        else:
            flags |= ob_flag.FEW_STARS


        fits_station_id = fits_file.split('_')[1]
        dt = FFfile.getMiddleTimeFF(fits_file, obs_con.fps, ret_milliseconds=True, ff_frames=256)
        jd = date2JD(*dt)

        if pp.station_code != obs_con.stationID:
            log.warning("Platepar mismatch")

        if pp_recal_json is not None:
            if fits_file in pp_recal_json:
                pp.loadFromDict(pp_recal_json[fits_file])

        pixel_scale_h = pp.fov_h / pp.X_res
        pixel_scale_v = pp.fov_v / pp.Y_res
        pixel_scale = max(pixel_scale_h, pixel_scale_v)
        radius_deg = pixel_scale * 3
        if jd in star_dict:
            stars_list = star_dict[jd]
            stars_list = np.array(stars_list)
        else:
            continue

        if not (stars_list.dtype == np.float64):
            continue

        stars = np.array(stars_list)

        arr_obs_y, arr_obs_x, arr_obs_intensity_sum = stars[:,0], stars[:,1], stars[:,2]
        arr_ampltd, arr_fwhm, arr_bg_lvl, arr_snr, arr_nsatpx = stars[:,3], stars[:,4], stars[:,5], stars[:,6], stars[:,7]

        arr_jd = np.full_like(arr_obs_x, jd, dtype=float)

        _arr_jd, arr_obs_ra, arr_obs_dec, arr_obs_mag = xyToRaDecPP(
            arr_jd, arr_obs_x, arr_obs_y, arr_obs_intensity_sum, pp,
            jd_time=True, measurement=True, precompute_pointing_corr=True, extinction_correction=True
        )

        results_list = cat.queryRaDec(arr_obs_ra, arr_obs_dec, n_brightest=1, radius_deg=radius_deg)

        arr_obs_az, arr_obs_alt = raDec2AltAz(arr_obs_ra, arr_obs_dec, arr_jd, obs_con.latitude, obs_con.longitude)
        frame_list = []

        cat_mag_list = []

        for res, obs_mag in zip(results_list, arr_obs_mag):
            if res is not None:
                cat_mag_list.append(res[0][3])
            else:
                cat_mag_list.append(obs_mag)


        arr_cat_mag = np.array(cat_mag_list)
        mean_absolute_deviation = np.median(np.abs(arr_cat_mag - arr_obs_mag))

        if mean_absolute_deviation > 0.2:
            flags |= ob_flag.BAD_MAD
        else:
            flags &= ~ob_flag.BAD_MAD

        for i, (query_results, o_ra, o_dec, o_mag, o_x, o_y, o_intens_sum,
                o_az, o_alt, o_ampltd, o_fwhm, o_bg_lvl, o_snr, o_nsatpx, sun_below_horizon_angle) in enumerate(zip(
                results_list,
            arr_obs_ra,arr_obs_dec, arr_obs_mag,
            arr_obs_x, arr_obs_y,arr_obs_intensity_sum,
            arr_obs_az, arr_obs_alt,
            arr_ampltd, arr_fwhm, arr_bg_lvl, arr_snr, arr_nsatpx,
            sun_below_horizon_angle_list)):

            if o_intens_sum <= 0:
                log.info(
                    f"Observation from session {calstars_name} on {fits_file} "
                    f"at {o_x:.2f} {o_y:.2f} had an unrealistic intensity sum."
                )
                continue

            # Matched vs unmatched
            if query_results is None:
                name = None
                c_ra = None
                c_dec = None
                c_mag = None
                theta = None
            else:
                name, c_ra, c_dec, c_mag, theta = query_results[0]

            mag_err = None if c_mag is None else (o_mag - c_mag)

            frame_list.append({
                "name": name,
                "station_name": obs_con.stationID.upper(),
                "jd": float(jd),
                "stationID": fits_station_id.upper(),

                "cat_ra": c_ra,
                "cat_dec": c_dec,
                "cat_mag": c_mag,

                "obs_ra": o_ra,
                "obs_dec": o_dec,
                "theta": theta,
                "obs_az": o_az,
                "obs_alt": o_alt,

                "obs_mag": o_mag,
                "err_mag": mag_err,

                "obs_x": o_x,
                "obs_y": o_y,

                "intens_sum": o_intens_sum,
                "ampltd": o_ampltd,
                "fwhm": o_fwhm,
                "bg_lvl": o_bg_lvl,
                "snr": o_snr,
                "nsatpx": o_nsatpx,

                "pixel_scale_h": pixel_scale_h,
                "pixel_scale_v": pixel_scale_v,
                "mad": mean_absolute_deviation,
                "sun_angle": sun_below_horizon_angle,
                "flag": flags
            })

        observation_dict[fits_file] = frame_list

    fits_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    elapsed_seconds = (fits_end_time - fits_start_time).total_seconds()
    fits_count = len(observation_dict)
    log.info(f"Read {calstars_name} at {fits_count / elapsed_seconds:.1f} fits / second")

    return observation_dict, start_jd, end_jd

def resetIngestion(local_calstars_path, ingestion_marker):

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

def populateWorkQueue(conn, file_name_list):

    with conn.cursor() as cur:
        with cur.copy("COPY ingest_work (remote_path, jd_int) FROM STDIN") as copy:

            for file_name in file_name_list:
                dt = FFfile.getMiddleTimeFF(file_name, fps=25, ret_milliseconds=True, ff_frames=256)
                jd = date2JD(*dt)
                jd_int = scale1e6(jd)

                if jd_int is None:
                    continue
                copy.write_row((file_name, jd_int))

        conn.commit()



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Ingest CALSTAR data \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('path_template', help="""Template to remote file stores i.e. user@host:/home/stationID/files/ """)

    arg_parser.add_argument('postgresql_host', help="""PostgreSQL server host """)

    arg_parser.add_argument('-t', '--threads', type=int,  default=2, help="""Number of concurrent threads to run """)


    arg_parser.add_argument('--write_db', dest='write_db', default=False, action="store_true",
                            help="Write to db")

    arg_parser.add_argument('-r', '--reset_ingestion', dest='reset_ingestion', default=False, action="store_true",
                            help="Reset all ingestion markers")

    arg_parser.add_argument('-p', '--populate_ingestion_table', dest='populate_ingestion_table', default=False, action="store_true",
                            help="Populate ingestion table and then quit immediately")

    arg_parser.add_argument('--create_database', dest='create_database', default=False,
                            action="store_true",
                            help="Populate ingestion table and then quit immediately")

    arg_parser.add_argument('--country', metavar='COUNTRY', help="""Country code to work on""")

    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))
    country_code = cml_args.country
    create_database = cml_args.create_database

    calstars_directory_path = os.path.join(config.data_dir, CALSTARS_DATA_DIR)

    if not os.path.exists(calstars_directory_path):
        Path(calstars_directory_path).mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    log_manager = LoggingManager()


    # Get the logger handle
    log = getLogger("rmslogger")

    user, _, remainder = cml_args.path_template.partition("@")
    hostname, _, path_template = remainder.partition(":")

    postgresql_host = cml_args.postgresql_host
    concurrent_threads = cml_args.threads

    log.info(f"Starting ingestion from {user}@{hostname} with path template {path_template} and {concurrent_threads} concurrent threads")
    log.info(f"Postgresql host {postgresql_host}")



    cwd = os.getcwd()




    local_calstars_path = Path(os.path.expanduser(config.data_dir)) / CALSTARS_DATA_DIR
    if cml_args.reset_ingestion:

        log.info(f"Removing all ingestion markers ({DIRECTORY_INGESTED_MARKER}) from {local_calstars_path}")
        resetIngestion(local_calstars_path, DIRECTORY_INGESTED_MARKER)


    write_db = cml_args.write_db

    directories_list = os.listdir(calstars_directory_path)

    station_list = getStationList(country_code=country_code)
    #remote_files = discoverRemoteFiles(station_list, user, hostname, 22, remote_processed_dir_template=path_template)
    #saveRemoteFiles(remote_files, os.path.expanduser("~/RMS_data/remotefiles.json"))
    remote_files = loadRemoteFiles(os.path.expanduser("~/RMS_data/remotefiles.json"))
    remote_files_sorted = sortFilesByTime(remote_files)

    print(len(remote_files))


    if create_database:
        with psycopg.connect(host=postgresql_host, dbname="star_data", user="postgres") as conn:

            createDatabaseIfMissing(conn)
            initialiseDatabase(conn)


    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as conn:

        auditIngestUserPrivileges(conn)
        log.info("Loading star catalog")
        cat = Catalog(config, lim_mag=10)
        log.info(f"Loaded catalog of {cat.entry_count} entries")

        if cml_args.populate_ingestion_table:
            log.info("Populating the ingestion table")
            populateWorkQueue(conn, remote_files_sorted)
            print("Returned from populate work queue")
            log.info("Table populated")
        else:

            ingest(config, remote_files_sorted, conn, username=user, host=hostname, remote_station_processed_dir=path_template, write_db=write_db, concurrent_threads=concurrent_threads)




    pass