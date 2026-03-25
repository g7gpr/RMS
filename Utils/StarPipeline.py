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



from __future__ import print_function, division, absolute_import


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

JD_OFFSET = J2000_JD

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
DIRECTORY_INGESTED_MARKER = ".ingested"
CALSTAR_FILES_TABLE_NAME = "calstar_files"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
CHARTS = "charts"
PORT = 22

def createStationTable(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS station (
        station_id      CHAR(6) PRIMARY KEY,
        name            TEXT,
        notes           TEXT
    );
    """
    with conn.cursor() as cur:

        cur.execute(sql)
    conn.commit()


def createSessionTable(conn):



    sql = """CREATE TABLE IF NOT EXISTS session (
                session_id      SERIAL PRIMARY KEY,
                session_name    TEXT NOT NULL UNIQUE,
                station_id      TEXT NOT NULL,
        
                start_jd        BIGINT,
                end_jd          BIGINT,
        
                pixel_scale_h   INTEGER,
                pixel_scale_v   INTEGER,
        
                lat             INTEGER,
                lon             INTEGER,
                elevation       INTEGER);"""

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createFrameTable(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS frame (
        frame_name      TEXT PRIMARY KEY,
        session_name    TEXT REFERENCES session(session_name),
        jd_mid          BIGINT,
        frame_index     INTEGER,
        quality_flags   SMALLINT
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createStarTable(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS star (
        star_name       TEXT PRIMARY KEY,
        ra              INTEGER,
        dec             INTEGER,
        mag             INTEGER,
        catalog_source  TEXT,
        canonical_name  TEXT
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createObservationTable(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS observation (
        obs_id          BIGSERIAL PRIMARY KEY,

        frame_name      TEXT REFERENCES frame(frame_name),
        star_name       TEXT REFERENCES star(star_name),

        -- CALSTARS fields (scaled where needed)
        y               INTEGER,
        x               INTEGER,
        intens_sum      INTEGER,
        ampltd          INTEGER,
        fwhm            INTEGER,
        bg_lvl          INTEGER,
        snr             INTEGER,
        nsatpx          SMALLINT,

        -- Derived fields
        mag             INTEGER,
        mag_err         INTEGER,

        flags           SMALLINT
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def createAllTables(conn):
    createStationTable(conn)
    createSessionTable(conn)
    createFrameTable(conn)
    createStarTable(conn)
    createObservationTable(conn)


def scale1e6(value):
    return int(round(value * 1_000_000))

def buildFrameRows(observation_dict, session_name):
    frame_rows = []

    for fits_file, frame_dict in observation_dict.items():
        frame_name = extractFrameName(fits_file)
        frame_index = extractFrameIndex(fits_file)

        # If there are no stars, then do no more work here
        if not frame_dict:
            log.info(f"{fits_file} had no stars, skipping")
            continue

        # Get JD from any star entry (all stars in frame share same JD)
        first_star = next(iter(frame_dict.values()))
        jd_mid = scale1e6(first_star["jd"])

        quality_flags = 0  # placeholder for now

        frame_rows.append((
            frame_name,
            session_name,
            jd_mid,
            frame_index,
            quality_flags
        ))

    return frame_rows

def buildStarRows(observation_dict):
    star_set = set()

    for frame_dict in observation_dict.values():
        for star_name, d in frame_dict.items():
            star_set.add((
                star_name,
                scale1e6(d["cat_ra"]),
                scale1e6(d["cat_deg"]),
                scale1e6(d["cat_mag"]),
                "RMS",
                None
            ))

    return list(star_set)

def buildObservationRows(observation_dict):
    observation_rows = []

    for fits_file, frame_dict in observation_dict.items():
        frame_name = extractFrameName(fits_file)

        for star_name, d in frame_dict.items():
            observation_rows.append((
                frame_name,
                star_name,
                scale1e6(d["obs_y"]),
                scale1e6(d["obs_x"]),
                d["intens_sum"],
                d["ampltd"],
                scale1e6(d["fwhm"]),
                d["bg_lvl"],
                scale1e6(d["snr"]),
                d["nsatpx"],
                scale1e6(d["obs_mag"]),
                scale1e6(d["err_mag"]),
                0  # flags
            ))

    return observation_rows

def buildAllRows(observation_dict, session_name):
    frame_rows = buildFrameRows(observation_dict, session_name)
    star_rows = buildStarRows(observation_dict)
    observation_rows = buildObservationRows(observation_dict)

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
                "INSERT INTO station (station_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (station_id,)
            )

            # Insert session
            cur.execute("""
                        INSERT INTO session (session_name,
                                             station_id,
                                             start_jd,
                                             end_jd,
                                             pixel_scale_h,
                                             pixel_scale_v,
                                             lat,
                                             lon,
                                             elevation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;
                        """, (
                            session_name,
                            station_id,
                            scale1e6(start_jd),
                            scale1e6(end_jd),
                            scale1e6(pixel_scale_h),
                            scale1e6(pixel_scale_v),
                            scale1e6(lat),
                            scale1e6(lon),
                            scale1e6(ele)
                        ))

            # Insert frames
            cur.executemany("""
                INSERT INTO frame (frame_name, session_name, jd_mid, frame_index, quality_flags)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, frame_rows)

            # Insert stars
            cur.executemany("""
                INSERT INTO star (star_name, ra, dec, mag, catalog_source, canonical_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, star_rows)

            # Insert observations
            cur.executemany("""
                INSERT INTO observation (
                    frame_name,
                    star_name,
                    y, x,
                    intens_sum, ampltd, fwhm, bg_lvl, snr, nsatpx,
                    mag, mag_err,
                    flags
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, observation_rows)

        # If everything succeeded, commit once
        conn.commit()

    except Exception as e:
        # Roll back the entire session atomically
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

def archiveCalstarDirectories(root, directories_list, ingested_only=True):
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
        if ingested_only:

            dir_list = os.listdir(source_dir)
            if DIRECTORY_INGESTED_MARKER not in dir_list:
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

def createCalstarFilesTable(conn):
    """Create the calstar files table.

    Args:
        conn: [object] Database connection.

    Returns:
        Nothing.
    """

    sql_command = ""
    sql_command += f"CREATE TABLE IF NOT EXISTS {CALSTAR_FILES_TABLE_NAME}\n"
    sql_command += "(file_name TEXT PRIMARY KEY, ingestion_time TIMESTAMPTZ NOT NULL);"

    with conn.cursor() as cur:
        cur.execute(sql_command)
    conn.commit()

def createTableStarObservations(conn):
    """If the star_observations table does not exist, then create.
    Arguments:
        conn: [object] Connection to database.

    Returns:
        Nothing
    """

    # If a table does not exist, create with a composite primary key of catalogue_id and ff_name.
    # A single observation (i.e. ff_file) should never have the same catalogue id twice
    command_list = []

    sql_command = ""
    sql_command += f"CREATE TABLE IF NOT EXISTS {STAR_OBSERVATIONS_TABLE_NAME}\n"
    sql_command += f"        (catalogue_id TEXT, ff_name TEXT, PRIMARY KEY(catalogue_id, ff_name));\n"

    command_list.append(sql_command)

    # Add an index on catalogue_id
    sql_command = ""
    sql_command += f"CREATE INDEX IF NOT EXISTS idx_star_obs_catalogue_id\n"
    sql_command += f"           ON {STAR_OBSERVATIONS_TABLE_NAME} (catalogue_id);\n"

    command_list.append(sql_command)

    #log.info("Executing...")
    #log.info(f"{sql_command}")

    with conn.cursor() as cur:
        for c in command_list:
            cur.execute(c)
    conn.commit()

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
    """Into CALSTAR_FILES_TABLE_NAME insert file_name and the time of ingestion

    Arguemnts:
        conn: [object] Connection to database.
        file_name: [string] Name of the file.

    Returns:
        Nothing.
    """

    ingestion_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sql_command = ""
    sql_command += f"INSERT INTO {CALSTAR_FILES_TABLE_NAME} (file_name, ingestion_time)\n"
    sql_command += "VALUES (%s, %s)\n"
    sql_command += "ON CONFLICT (file_name)\n"
    sql_command += "DO UPDATE SET ingestion_time = EXCLUDED.ingestion_time;"

    #log.info(f"Recording {file_name} as ingested in database")
    #log.info(f"Executing sql command {sql_command}")

    with conn.cursor() as cur:
        cur.execute(sql_command, (file_name, ingestion_time))
        conn.commit()

def markIngested(directory_path):
    """Save the ingested marker file into the folder_path.

    Arguments:
        directory_path: [str] Path to folder.

    Returns:
        Nothing.
    """
    directory_path = Path(directory_path)
    marker_file = directory_path / ".ingested"
    log.info(f"Marked {os.path.basename(directory_path)} as ingested")
    marker_file.touch()

def isIngested(directory_path):
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

def filterByDate(files_list, earliest_date=None, latest_date=None, station=None):
    """
    Filter a list of bz2 files by date.
    Arguments:
        files_list: [list] list of bz2 files.

    Keyword arguments:
        earliest_date: [datetime] optional, default None, earliest date to pick, if None, 3 days before now.
        latest_date: [datetime] optional, default None, latest date to pick, if None, 3 days after now.

    Returns:
        filtered_files_list: [list] list of bz2 files filtered by date
    """


    if earliest_date is None:
        earliest_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)

    if latest_date is None:
        latest_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)



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

def initialiseDatabase(postgresql_host):
    """
    Get the connection to the stellar magnitude database, if tables do not exist, then create.

    Arguments:
        host: [str] Hostname

    Returns:
        conn:[object] connection object instance.
    """
    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as conn:
        createTableStarObservations(conn)
        createCalstarFilesTable(conn)
    return

def getRemoteFiles(username, host, port, remote_dir):

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
            delay = random.randint(600, 900)
            log.info(f"Connection problem for {host}, retrying in {delay/60:.1f} minutes")
            time.sleep(delay)
        else:
            log.info(f"No remote files found in {remote_dir} for {host}")
            break

    return remote_files

def downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=22, max_tries=3):

    remote_file = os.path.basename(full_remote_path_to_bz2)
    download_start_time = datetime.datetime.now(datetime.timezone.utc)
    # log.info(f"Downloading {remote_file}")
    download_count = 0
    while not os.path.exists(
            os.path.join(t, os.path.basename(full_remote_path_to_bz2))) and download_count < max_tries:
        downloadFile(host, username, t, full_remote_path_to_bz2, port=port)
        download_count += 1

    if os.path.exists(os.path.join(t, os.path.basename(full_remote_path_to_bz2))):
        download_end_time = datetime.datetime.now(datetime.timezone.utc)
        downloaded_size = os.path.getsize(os.path.join(t, remote_file)) / (1000 ** 2)
        rate_mb_s = downloaded_size / (download_end_time - download_start_time).total_seconds()
        log.info(f"Downloaded {remote_file} of size {downloaded_size:.2f}MB at {rate_mb_s:.2f} MB/s after {download_count} try")
    else:
        log.warning(f"Failed to download {remote_file}")


def moveFiles(local_target, path_source_list, path_local_list):

    files_available = []
    for p_source, p_local in zip(path_source_list, path_local_list):
        if os.path.exists(p_source):
            mkdirP(local_target)
            shutil.move(p_source, p_local)
            files_available.append(os.path.basename(p_local))
    return files_available

def markIngestedIfFilesMissing(path_local_list, files_available, local_target):

    for f in path_local_list:
        if os.path.basename(f) not in files_available and f != "mask.bmp":
            # Mark this folder as ingested so we don't waste time on it in future
            log.warning(f"Missing files for {os.path.basename(local_target)}: {f}")
            markIngested(local_target)
            continue

def getFromRemote(host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path):


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
        downloadWithRetries(t, host, username, full_remote_path_to_bz2, port=port)

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
        markIngestedIfFilesMissing(path_local_list, files_available, local_target)


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


def processStation(station, remote_station_processed_dir, username, host, port, history_days, calstars_data_full_path, write_db=True):


    remote_dir = remote_station_processed_dir.replace("stationID", station.lower())
    remote_files = getRemoteFiles(username, host, port, remote_dir)
    remote_files = filterByDate(remote_files, earliest_date=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=history_days), station=station)
    fits_files_processed_this_station, star_observations_processed_this_station = 0, 0
    log.info(f"For station:{station} {len(remote_files)} files to process")

    # This loop processes files held locally - the remote files is used to create the local file names
    for remote_file in remote_files:

        fits_in_this_session = 0

        remote_file_start_time = time.perf_counter()
        file_type = getFileType(remote_file)
        if file_type != "metadata" and file_type != "detected":
            continue


        station_name = remote_file.split("_")[0]
        local_dir_name = "_".join(remote_file.split("_")[0:4])
        log.info(f"Working on {local_dir_name}")
        local_target = os.path.join(calstars_data_full_path, local_dir_name)

        # If we have a .bz2 file, extract it so we can work on it
        local_calstars_archive_path = f"{local_target}_CALSTAR.tar.bz2"
        extractCalstarArchives(calstars_data_full_path, [os.path.basename(local_calstars_archive_path)], remove_archives=True)
        calstars_name = f"CALSTARS_{local_dir_name}.txt"
        local_config_path = os.path.join(local_target, os.path.basename(config.config_file_name))
        local_platepar_path = os.path.join(local_target, config.platepar_name)
        local_calstars_path = os.path.join(local_target, calstars_name)
        local_recalibrated_path = os.path.join(local_target, PLATEPARS_ALL_RECALIBRATED_JSON)

        # If we don't have a directory, then get from remote working in a temporary directory
        if not os.path.exists(os.path.join(calstars_data_full_path, local_dir_name)):
            log.info(f"Retrieving {remote_file} from {host}")
            local_config_path, local_platepar_path, local_recalibrated_path, calstars_name = getFromRemote(host, username, port, station_name, remote_dir, remote_file, calstars_data_full_path)

        if not isIngested(local_target):
            star_observations_processed = 0
            observation_session_dict = {}

            if not write_db:
                log.info(f"Data from {local_dir_name} not being written to database as writes not enabled.")
                continue

            if write_db:
                log.info(f"Ingesting {calstars_name}")
                observation_session_config = cr.parse(local_config_path)
                observation_session_dict, start_jd, end_jd = calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recalibrated_path, local_calstars_path)

                pixel_scale_h, pixel_scale_v = extractMedianPixelScale(observation_session_dict)
                session_name = extractSessionNameFromCalstar(local_calstars_path)
                frame_rows, star_rows, observation_rows = buildAllRows(observation_session_dict, session_name)
                fits_in_this_session = len(frame_rows)

                star_observations_processed = writeSessionBatch(
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
                star_observations_processed_this_station += star_observations_processed

                markIngested(local_target)

            remote_file_end_time = time.perf_counter()
            time_elapsed = remote_file_end_time - remote_file_start_time


            if star_observations_processed != 0 and fits_in_this_session != 0:
                stars_observations_second = star_observations_processed / time_elapsed

                fits_files_processed_this_station += fits_in_this_session

                fits_processed_per_second = fits_in_this_session / time_elapsed
                # About one fits every 10 seconds at  - only observing for half of 24 hours so one every 20 seconds
                fits_generated_per_second = 0.05

                log.info(f"For {remote_file} processed {stars_observations_second:.0f} obs/sec {fits_processed_per_second:.0f} fits/sec")

                no_of_cameras = fits_processed_per_second / fits_generated_per_second
                log.info(f"From this iteration pipeline can support up to {no_of_cameras:.0f} cameras")

        # Put back in an archive if it has been ingested
        archiveCalstarDirectories(calstars_data_full_path, [local_dir_name], ingested_only=True)

    return star_observations_processed_this_station, fits_files_processed_this_station

def ingest(config, station_list, conn, country_code=None, calstars_data_dir=CALSTARS_DATA_DIR,
           remote_station_processed_dir=None, write_db=True,
           host=None, username=None, port=PORT, history_days=None):

    """
    In a subdirectoy of station_data_dir create a directory for each station containing mask
    platepar and config file.

    Arguments:
        config: [config] RMS config instance - used to get data_dir.
        station_list: [list] list of stations.
        conn: [object] database connetion object.

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

    if country_code is None:
        country_code = ""

    if history_days is None:
        history_days = 365

    calstars_data_full_path = os.path.join(config.data_dir, calstars_data_dir)

    log.info("Starting to download files")
    total_fits_processed = 0
    routine_start_time = time.perf_counter()


    for station in station_list:

        star_observations_processed_this_station, fits_files_processed_this_station = \
            processStation(station, remote_station_processed_dir, username, host, port, history_days, calstars_data_full_path, write_db=write_db)
        total_fits_processed += fits_files_processed_this_station
        routine_elapsed_time = time.perf_counter() - routine_start_time
        total_fits_processed_per_second = total_fits_processed / routine_elapsed_time
        fits_generated_per_second = 0.06
        log.info(f"Cumulative rate is {total_fits_processed_per_second:.0f} fits per second")

        faster_than_real_time = total_fits_processed_per_second / fits_generated_per_second
        log.info(f"Pipeline can support up to {faster_than_real_time:.0f} cameras")



def extractMedianPixelScale(observation_dict):
    pixel_scale_h_values = []
    pixel_scale_v_values = []

    for frame in observation_dict.values():
        for obs in frame.values():
            pixel_scale_h_values.append(obs["pixel_scale_h"])
            pixel_scale_v_values.append(obs["pixel_scale_v"])

    median_h = float(np.median(pixel_scale_h_values))
    median_v = float(np.median(pixel_scale_v_values))

    return median_h, median_v



def calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recal_path, local_calstars_path):
    """
      Parses a calstar data structures in archived directories path,
      converts to RaDec, corrects magnitude data and writes newer data to database

      """

    observation_config = cr.parse(local_config_path)
    calstars_name = os.path.basename(local_calstars_path)

    calstar, chunk = readCALSTARS(os.path.dirname(local_calstars_path), calstars_name)

    if os.path.exists(local_recal_path):

        with open(local_recal_path, 'r') as fh:
            pp_recal_json = json.load(fh)
            log.info(f"Read {os.path.basename(local_recal_path)}")
    else:
        log.info(f"No file {os.path.basename(local_recal_path)}`")

    pp = Platepar()
    star_dict = starListToDict(observation_config, [calstar, chunk])


    # If the star dict is empty then this was a poor observation session
    if not len(star_dict):
        return {}, None, None

    pp.read(local_platepar_path)

    observation_dict = {}

    fits_start_time = datetime.datetime.now(tz=datetime.timezone.utc)



    dt = FFfile.getMiddleTimeFF(calstar[0][0], observation_config.fps, ret_milliseconds=True, ff_frames=256)
    start_jd = date2JD(*dt)

    dt = FFfile.getMiddleTimeFF(calstar[-1][0], observation_config.fps, ret_milliseconds=True, ff_frames=256)
    end_jd = date2JD(*dt)

    for fits_file, star_list in calstar:
        fits_station_id = fits_file.split('_')[1]
        frame_dict = {}

        dt = FFfile.getMiddleTimeFF(fits_file, observation_config.fps, ret_milliseconds=True, ff_frames=256)
        jd = date2JD(*dt)

        if pp.station_code != observation_config.stationID:
            log.warning("Platepar mismatch")

        if fits_file in pp_recal_json:
            # log.info(f"Reading in new platepar for {fits_file}")
            # If we have a platepar in pp_recal then use it, else just use the uncalibrated platepar
            pp.loadFromDict(pp_recal_json[fits_file])

        pixel_scale_h = pp.fov_h / pp.X_res
        pixel_scale_v = pp.fov_v / pp.Y_res
        pixel_scale = max(pixel_scale_h, pixel_scale_v)
        radius_deg = pixel_scale * 3

        # Extract stars for the given Julian date
        if jd in star_dict:
            stars_list = star_dict[jd]
            stars_list = np.array(stars_list)
        else:
            continue

        # If the type is not float, it means something went wrong, so skip this
        if not (stars_list.dtype == np.float64):
            continue


        stars = np.array(stars_list)

        arr_obs_y, arr_obs_x, arr_obs_intensity_sum = stars[:,0], stars[:,1], stars[:,2]
        arr_ampltd, arr_fwhm, arr_bg_lvl, arr_snr, arr_nsatpx = stars[:,3], stars[:,4], stars[:,5], stars[:,6], stars[:,7]

        arr_jd = np.full_like(arr_obs_x, jd, dtype=float)

        _arr_jd, arr_obs_ra, arr_obs_dec, arr_obs_mag = \
                                xyToRaDecPP(arr_jd, arr_obs_x, arr_obs_y, arr_obs_intensity_sum, pp,
                                            jd_time=True,  measurement=True, precompute_pointing_corr=True, extinction_correction=True)

        results_list = cat.queryRaDec(arr_obs_ra, arr_obs_dec, n_brightest=1, radius_deg=radius_deg)


        arr_obs_az, arr_obs_alt = raDec2AltAz(arr_obs_ra, arr_obs_dec, arr_jd, observation_config.latitude, observation_config.longitude)
        for r in zip(results_list, arr_obs_ra, arr_obs_dec, arr_obs_mag, arr_obs_x, arr_obs_y, arr_obs_intensity_sum, arr_obs_az, arr_obs_alt,
                                            arr_ampltd, arr_fwhm, arr_bg_lvl, arr_snr, arr_nsatpx):

            query_results, o_ra, o_dec, o_mag, o_x, o_y, o_intens_sum, o_az, o_alt, o_ampltd, o_fwhm, o_bg_lvl, o_snr, o_nsatpx = r
            name, c_ra, c_deg, c_mag, theta = query_results

            if o_intens_sum <= 0:
                log.info(f"Observation from session {calstars_name} on {fits_file} at {o_x:.2f} {o_y:.2f} had an unrealistic intensity sum.")
                continue

            if query_results == []:
                continue

            # Detect the same star appearing in two places
            duplicate_counter = 1
            while name in frame_dict:
                #log.error(f"Duplicate catalogue star {name} in {fits_file} at image coordinates x:{o_x:.1f}, r:{o_y:.1f} sky coordinates RA:{o_ra:.2f} DEC:{o_dec:.2f}")
                #log.error(f"Initial / this observations {frame_dict[name]['obs_mag']:.2f} / {o_mag:.2f}")
                name = f"{name}_duplicate_{duplicate_counter:03d}"
                #log.error(f"Storing as {name}")
                duplicate_counter += 1

            # Compute magnitude error
            mag_err = o_mag - c_mag

            frame_dict[name] = {
                "jd": float(jd),
                "stationID": fits_station_id.upper(),

                # catalogue
                "cat_ra": c_ra,
                "cat_deg": c_deg,
                "cat_mag": c_mag,

                # observed astrometry (not stored in DB)
                "obs_ra": o_ra,
                "obs_dec": o_dec,
                "theta": theta,
                "obs_az": o_az,
                "obs_alt": o_alt,

                # observed photometry
                "obs_mag": o_mag,
                "err_mag": mag_err,

                # pixel coordinates
                "obs_x": o_x,
                "obs_y": o_y,

                # CALSTARS raw fields (renamed to match DB schema)
                "intens_sum": o_intens_sum,
                "ampltd": o_ampltd,
                "fwhm": o_fwhm,
                "bg_lvl": o_bg_lvl,
                "snr": o_snr,
                "nsatpx": o_nsatpx,

                # Pixel scaling
                "pixel_scale_h": pixel_scale_h,
                "pixel_scale_v": pixel_scale_v
            }

        observation_dict[fits_file] = frame_dict
        pass




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
            archiveCalstarDirectories(local_calstars_path, [os.path.basename(calstar_path)], ingested_only=False)
            pass

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Ingest CALSTAR data \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('path_template', help="""Template to remote file stores i.e. user@host:/home/stationID/files/ """)

    arg_parser.add_argument('postgresql_host', help="""PostgreSQL server host """)

    arg_parser.add_argument('-d', '--days_history', type=int,  default=7, help="""Number of days of history """)


    arg_parser.add_argument('--write_db', dest='write_db', default=False, action="store_true",
                            help="Write to db")

    arg_parser.add_argument('-r', '--reset_ingestion', dest='reset_ingestion', default=False, action="store_true",
                            help="Reset all ingestion markers")



    arg_parser.add_argument('--country', metavar='COUNTRY', help="""Country code to work on""")

    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))
    country_code = cml_args.country

    calstars_directory_path = os.path.join(config.data_dir, CALSTARS_DATA_DIR)

    if not os.path.exists(calstars_directory_path):
        Path(calstars_directory_path).mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("rmslogger")

    user, _, remainder = cml_args.path_template.partition("@")
    hostname, _, path_template = remainder.partition(":")

    postgresql_host = cml_args.postgresql_host

    log.info(f"Starting ingestion from {user}@{hostname} with path template {path_template}")
    log.info(f"Postgresql host {postgresql_host}")

    days_history = cml_args.days_history
    log.info(f"Using a history of {days_history} days")

    cwd = os.getcwd()



    station_list = getStationList(country_code=country_code)

    log.info("Loading star catalog")
    cat = Catalog(config, lim_mag=10)
    log.info(f"Loaded catalog of {cat.entry_count} entries")



    local_calstars_path = Path(os.path.expanduser(config.data_dir)) / CALSTARS_DATA_DIR
    if cml_args.reset_ingestion:

        log.info(f"Removing all ingestion markers ({DIRECTORY_INGESTED_MARKER}) from {local_calstars_path}")
        resetIngestion(local_calstars_path, DIRECTORY_INGESTED_MARKER)


    write_db = cml_args.write_db

    # initialiseDatabase(postgresql_host = postgresql_host)


    directories_list = os.listdir(calstars_directory_path)


    archiveCalstarDirectories(os.path.join(config.data_dir, CALSTARS_DATA_DIR), directories_list, ingested_only=True)

    with psycopg.connect(host=postgresql_host, dbname="star_data", user="ingest_user") as conn:

        with conn.cursor() as cur:

            cur.execute("SHOW search_path;")
            log.info("PYTHON search_path:", cur.fetchone())
            cur.execute("SELECT current_database();")
            log.info("PYTHON database:", cur.fetchone())
        createAllTables(conn)
        ingest(config, station_list, conn, username=user, host=hostname, country_code=country_code, remote_station_processed_dir=path_template, history_days=days_history, write_db=write_db)




    pass