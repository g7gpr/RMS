# RPi Meteor Station
# Copyright (C) 2025 David Rollinson
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

from datetime import tzinfo

import RMS.Math
import os
import pickle
import tempfile
import tarfile
import paramiko
import subprocess
import json
import requests
import tqdm
import RMS.ConfigReader as cr
import gc
import shutil
import sys
import datetime
import sqlite3

from RMS.Astrometry.MatchStars import matchStars
from RMS.Formats import CALSTARS, FFfile, Platepar, StarCatalog
from RMS.Astrometry.CheckFit import starListToDict
from RMS.Formats.StarCatalog import readStarCatalog
from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting, photometryFitRobust, getFOVSelectionRadius
from RMS.Astrometry.CyFunctions import subsetCatalog
from RMS.Formats.StarCatalog import Catalog

import numpy as np



from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, ECEF2AltAz, altAz2RADec, raDec2AltAz, AER2ECEF
from RMS.Math import angularSeparationDeg, vectNorm
from RMS.Routines.MaskImage import getMaskFile
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP
from scipy.spatial import cKDTree
from RMS.Formats.CALSTARS import readCALSTARS, maxCalstarsToPNG, calstarsToMP4
from scipy.spatial import cKDTree


from RMS.Misc import mkdirP
import matplotlib.pyplot as plt



REMOTE_SERVER = 'gmn.uwo.ca'
USER_NAME = "analysis"
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"
CALSTARS_DATA_DIR = "CALSTARS"
REMOTE_STATION_PROCESSED_DIR = "/home/$STATION/files/processed"
WORKING_DIRECTORY = os.path.expanduser("~/RMS_data/Coverage")
PLATEPARS_ALL_RECALIBRATED_JSON = "platepars_all_recalibrated.json"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"
STAR_OBSERVATION_DB_PATH = os.path.expanduser("~/RMS_data/magnitudes.db")

CHARTS = "charts"
PORT = 22

from RMS.Logger import LoggingManager, getLogger



def buildSphericalTree(raDeg, decDeg):
    """
    Build a cKDTree for spherical coordinates (RA, Dec in degrees).
    Returns (tree, xyz) where xyz is the Nx3 unit-vector array.
    """

    ra  = np.deg2rad(raDeg)
    dec = np.deg2rad(decDeg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    xyz = np.column_stack((x, y, z))
    tree = cKDTree(xyz)

    return tree, xyz

def querySphericalTree(tree, raDeg, decDeg, radiusDeg):
    """
    Query a spherical KD-tree for all catalog entries within an angular radius.
    Returns an array of indices (possibly empty).
    """

    ra  = np.deg2rad(raDeg)
    dec = np.deg2rad(decDeg)

    qx = np.cos(dec) * np.cos(ra)
    qy = np.cos(dec) * np.sin(ra)
    qz = np.sin(dec)

    queryVec = np.array([qx, qy, qz])

    # Convert angular radius to Euclidean chord distance
    theta = np.deg2rad(radiusDeg)
    euclidR = 2 * np.sin(theta / 2)

    return np.array(tree.query_ball_point(queryVec, euclidR), dtype=int)


def selectBrightest(indices, catalog, magCol=2):
    """
    Given an array of catalog indices, return a 1-element array containing
    the index of the brightest star (lowest magnitude).
    Returns an empty array if no matches.
    """

    if len(indices) == 0:
        return np.empty((0,), dtype=int)

    mags = catalog[indices, magCol]
    i = np.argmin(mags)

    return np.array([indices[i]], dtype=int)





def createTableStarObservations(conn):

    """
    If the star_observations table does not exist, then create
    Args:
        conn (): connection to database

    Returns:

    """

    # Returns true if the table exists in the database
    try:
        tables = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' and name = '{}';".format(STAR_OBSERVATIONS_TABLE_NAME)).fetchall()

        if len(tables) > 0:
            return conn
    except:

        return None

    # If a table does not exist, create with a composite primary key of catalogue_id and ff_name.
    # A single observation (i.e. ff_file) should never have the same catalogue id twice
    sql_command = ""
    sql_command += f"CREATE TABLE IF NOT EXISTS {STAR_OBSERVATIONS_TABLE_NAME}\n"
    sql_command += f"        (catalogue_id INT, ff_name TEXT, PRIMARY KEY(catalogue_id, ff_name))\n"

    print(sql_command)
    conn.execute(sql_command)

    return conn

def createTableCatalogue(conn):

    """
    Creates the catalogue table if it does not exist
    Args:
        conn (): connection to database

    Returns:
        connection to database
    """
    table_name = "catalogue"
    # Returns true if the table exists in the database
    try:
        tables = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' and name = '{}';".format(table_name)).fetchall()

        if len(tables) > 0:
            return conn
    except:

        return None

    sql_command = ""
    sql_command += "CREATE TABLE {} \n".format(table_name)
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    sql_command += "r FLOAT NOT NULL, \n"
    sql_command += "d FLOAT NOT NULL, \n"
    sql_command += "mag FLOAT NOT NULL \n"

    sql_command += ") \n"
    conn.execute(sql_command)

    catalogueToDB(conn)

    return conn

def catalogueToDB(conn):
    """
    Read catalogue into database
    Args:
        conn (): connection instance

    Returns:
        Nothing
    """
    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)
    log.info("\nInserting catalogue data\n")
    for star in tqdm.tqdm(catalogue):
        sql_command = "INSERT INTO catalogue (r , d, mag) \n"
        sql_command += "Values ({} , {}, {})".format(star[0], star[1], star[2])
        conn.execute(sql_command)
    conn.commit()

def loadGaiaCatalog(dir_path, file_name, lim_mag=None):
    """ Read star data from the GAIA catalog in the .npy format.
        This function copied here to avoid reading in whole of SkyFit2

    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        lim_mag: [float] Faintest magnitude to return. None by default, which will return all stars.

    Return:
        results: [2d ndarray] Rows of (ra, dec, mag), angular values are in degrees.
    """

    file_path = os.path.expanduser(os.path.join(dir_path, file_name))

    # Read the catalog
    results = np.load(str(file_path), allow_pickle=False)

    # Filter by limiting magnitude
    if lim_mag is not None:
        results = results[results[:, 2] <= lim_mag]

    # Sort stars by descending declination
    results = results[results[:, 1].argsort()[::-1]]

    return results

def dictInvert(d):

    out = {}

    for key, subdict in d.items():
        for subkey, value in subdict.items():
            if subkey not in out:
                out[subkey] = {}
            out[subkey][key] = value

    return out


def lsRemote(host, username, port, remote_path):
    """Return the files in a remote directory, prefer rsync if available

    Arguments:
        host: [str] remote host.
        username: [str] user account to use.
        port: [int] remote port number.
        remote_path: [str] path of remote directory to list.

    Return:
        files: [list of strings] Names of remote files.
    """

    try:

        remote = "{}@{}:{}".format(username, host, os.path.join(remote_path))
        result_lines = subprocess.run(['rsync', '-z', "{}/".format(remote)], capture_output=True, text=True).stdout.splitlines()

        file_list = []
        for line in result_lines:
            file_list.append(line.split()[-1])

        return file_list
    except:
        pass

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys




def extractBz2(input_directory, working_directory, local_target_list=None):

    """
    Extract BZ2 files from a directory.

    Arguments:
        input_directory: [str] directory containing bz2 files.
        working_directory: [str] directory to work in, possibly a /tmp/ directory.

    Keyword arguments:
        local_target_list: optional, default None, specify files to extract, if None, extract all ending .bz2

    Returns:

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
    extractBz2Files(bz2_list, input_directory, working_directory)

    return working_directory

def extractBz2Files(bz2_list, input_directory, working_directory, silent=True, host=REMOTE_SERVER, username=USER_NAME, port=PORT):
    """
    Extract BZ2 files from a directory into a subdirectory of working_directory, if extraction fails, redownload.

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
                tar.extractall(path=bz2_directory)
        except:
            if not silent:
                log.info("Redownloading {}".format(basename_bz2))
            remote_path = REMOTE_STATION_PROCESSED_DIR.replace("$STATION", basename_bz2.split("_")[0].lower())
            remote_path = os.path.join(remote_path, basename_bz2)
            downloadFile(host, username, remote_path, port)
            with tarfile.open(os.path.join(input_directory, basename_bz2), 'r:bz2') as tar:
                tar.extractall(path=bz2_directory)




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

    """
    Get a list of stations.

    Arguments:
        url: [str] Optional, default STATION_COORDINATES_JSON, url of the json of station coordinates

    Returns:
        [list] station names
    """

    print("Downloading station list from {}".format(url))
    station_list, stations_dict = [], json.loads(requests.get(url).content.decode('utf-8'))

    for station in stations_dict:
        if country_code is None:
            station_list.append(station)
        else:
            if station.lower().startswith(country_code.lower()):
                station_list.append(station)
    return sorted(station_list)

def filterByDate(files_list, earliest_date=None, latest_date=None):
    """
    Filter a list of bz2 files by date.
    Arguments:
        files_list: [list] list of bz2 files

    Keyword arguments:
        earliest_date: optional, default None, earliest date to pick, if None, 3 days before now
        latest_date: optional, default None, latest date to pick, if None, 3 days after now

    Returns:
        filtered_files_list: [list] list of bz2 files filtered by date
    """


    if earliest_date is None:
        earliest_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)

    if latest_date is None:
        latest_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)

    filtered_files_list = []
    for file in files_list:

        if len(file.split("_")) != 5:
            continue

        date = file.split("_")[1]
        time = file.split("_")[2]
        year, month, day = int(date[0:4]), int(date[4:6]), int(date[6:8])
        hour, minute, second = int(time[0:2]), int(time[2:4]), int(time[4:6])
        file_date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, tzinfo=datetime.timezone.utc)
        if earliest_date < file_date < latest_date:
            filtered_files_list.append(file)

    return filtered_files_list


def getFileType(file_name):

    return file_name.split(".")[0].split("_")[4]

def makePlatePar(captured_directory):

    print("No platepar found - this is a placeholder for automatic platepar creation")
    pass


def createColumns(conn, table, columns):
    cur = conn.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cur.fetchall()}

    for col in columns:
        if col not in existing_columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")


def upsertRow(conn, table, catalogue_id, ff_name, values):


    # Build column list
    cols = ["catalogue_id", "ff_name"] + list(values.keys())
    placeholders = ", ".join("?" for _ in cols)

    # Build update clause for dynamic fields
    update_clause = ", ".join(f"{col}=excluded.{col}" for col in values.keys())

    sql = f"""
        INSERT INTO {table} ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(catalogue_id, ff_name)
        DO UPDATE SET {update_clause}
    """

    params = [catalogue_id, ff_name] + list(values.values())
    conn.execute(sql, params)


def getLeaves(data_dict):

    leaves = set()
    for ff_dict in data_dict.values():
        for bottom_dict in ff_dict.values():
            leaves.update(bottom_dict.keys())
    return leaves


def buildUpsertSQL(table, bottom_keys):
    cols = ["catalogue_id", "ff_name"] + list(bottom_keys)
    placeholders = ", ".join("?" for _ in cols)
    update_clause = ", ".join(f"{col}=excluded.{col}" for col in bottom_keys)

    sql = f"""
        INSERT INTO {table} ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(catalogue_id, ff_name)
        DO UPDATE SET {update_clause}
    """

    return sql, cols

def buildParamList(data, bottom_keys):
    params = []
    for catalogue_id, ff_dict in data.items():
        for ff_name, bottom_dict in ff_dict.items():
            row = [catalogue_id, ff_name] + [bottom_dict[k] for k in bottom_keys]
            params.append(row)
    return params


def writeStarObservationsToDB(data_dict):


    leaves_keys = getLeaves(data_dict)
    if not len(leaves_keys):
        return
    conn = getStationStarDBConn(STAR_OBSERVATION_DB_PATH)
    conn.execute("BEGIN")
    createColumns(conn, STAR_OBSERVATIONS_TABLE_NAME, leaves_keys)


    sql, cols = buildUpsertSQL(STAR_OBSERVATIONS_TABLE_NAME, leaves_keys)
    param_list = buildParamList(data_dict, leaves_keys)
    conn.executemany(sql, param_list)

    conn.execute("COMMIT")
    conn.close()
    return

def getStationStarDBConn(db_path, force_delete=False):
    """
    Get the connection to the stellar magnitude database, if it does not exist, then create
    Args:
        db_path (): full path to database
        force_delete (): optional, default false, delete and create

    Returns:
        conn (): connection object instance
    """
    # Create the station star database

    if force_delete:
        os.unlink(db_path)

    if not os.path.exists(os.path.dirname(db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(db_path))



    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode = WAL")
    createTableStarObservations(conn)
    createTableCatalogue(conn)
    return conn


def makeConfigPlateParCalstarsLib(config, station_list, cat, country_code=None, calstars_data_dir=CALSTARS_DATA_DIR,
                                  remote_station_processed_dir=REMOTE_STATION_PROCESSED_DIR,
                                  host=REMOTE_SERVER, username=USER_NAME, port=PORT):

    """
    In a subdirectoy of station_data_dir create a directory for each station containing mask
    platepar and config file.

    Arguments:
        config: [config] RMS config instance - used to get data_dir.
        station_list: [list] list of stations.

    Keyword arguments:
        calstars_data_dir: [str] target name in RMS_data, optional, default STATIONS_DATA_DIR.
        remote_station_processed_dir: [str] path on remote server, optional, default REMOTE_STATION_PROCESSED_DIR.
        host: [str] host name of remote machine, optional, default REMOTE_SERVER.
        username: [str] username for remote machine, optional, default USER_NAME.
        port: [int] optional, default PORT constant, optional, default PORT.

    Return:
        Nothing.
    """

    if country_code is None:
        country_code = ""

    calstars_data_full_path = os.path.join(config.data_dir, calstars_data_dir)

    log.info("Starting to download files.")
    for station in station_list:

        remote_dir = remote_station_processed_dir.replace("$STATION", station.lower())
        remote_files = sorted(lsRemote(host, username, port, remote_dir), reverse=True)
        remote_files = filterByDate(remote_files, earliest_date=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21))
        log.info(f"For station:{station} {len(remote_files)} files to process")
        if not len(remote_files):
            pass
        for remote_file in remote_files:

            file_type = getFileType(remote_file)
            if file_type != "metadata" and file_type != "detected":
                continue
            station_name = remote_file.split("_")[0]

            local_dir_name = "_".join(remote_file.split("_")[0:4])
            local_target = os.path.join(calstars_data_full_path, local_dir_name)



            files_already_downloaded = []
            if os.path.exists(calstars_data_full_path):
                if os.path.isdir(calstars_data_full_path):
                    files_already_downloaded = os.listdir(calstars_data_full_path)



            with tempfile.TemporaryDirectory() as t:


                # Create paths up front to reduce clutter
                extraction_dir = os.path.join(t, "extracted")
                calstars_name = f"CALSTARS_{local_dir_name}.txt"
                local_target_full_path = os.path.join(local_target)
                local_config_path = os.path.join(local_target_full_path, os.path.basename(config.config_file_name))
                local_platepar_path = os.path.join(local_target_full_path, config.platepar_name)
                local_mask_path = os.path.join(local_target_full_path, config.mask_file)
                local_calstars_path = os.path.join(local_target_full_path, calstars_name)
                local_recalibrated_path = os.path.join(local_target_full_path, PLATEPARS_ALL_RECALIBRATED_JSON)
                local_json_path = os.path.join(local_target_full_path, f"{local_dir_name}_star_observations.json")


                extracted_files_path = os.path.join(extraction_dir, station_name.lower(), remote_file.split(".")[0])
                extracted_config_path = os.path.join(extracted_files_path, ".config")
                extracted_platepar_path = os.path.join(extracted_files_path, config.platepar_name)
                extracted_mask_path = os.path.join(extracted_files_path, config.mask_file)
                extracted_recalibrated_path = os.path.join(extracted_files_path, PLATEPARS_ALL_RECALIBRATED_JSON)
                extracted_calstars_path = os.path.join(extracted_files_path, calstars_name)
                full_remote_path_to_bz2 = os.path.join(remote_dir, remote_file)



                path_source_list = [extracted_config_path, extracted_platepar_path, extracted_mask_path, extracted_calstars_path, extracted_recalibrated_path]
                path_local_list = [local_config_path, local_platepar_path, local_mask_path, local_calstars_path, local_recalibrated_path]


                log.info(f"\tWorking on {local_dir_name}")
                # Download, and extract the file into a subdir
                if local_dir_name not in files_already_downloaded:
                    log.info(f"\tDownloading from {full_remote_path_to_bz2}")
                    downloadFile(host, username, t, full_remote_path_to_bz2)
                    log.info(f"\tExtracting to {extraction_dir}")
                    mkdirP(extraction_dir)
                    extractBz2(t, extraction_dir)

                    for p_source, p_local in zip(path_source_list, path_local_list):
                        if os.path.exists(p_source):
                            mkdirP(local_target_full_path)
                            shutil.move(p_source, p_local)
                        else:
                            missing_at_least_one_file = True

                missing_at_least_one_file = False
                for f in path_local_list:
                    if not os.path.exists(f) and os.path.basename(f) != "mask.bmp":
                        missing_at_least_one_file = True
                        break

                if os.path.exists(local_json_path):
                    try:
                        with open(local_json_path, "r") as f:
                            dict_from_calstar = json.load(f)
                            log.info(f"\tWriting to database")
                            writeStarObservationsToDB(dict_from_calstar)
                            log.info(f"\tDatabase write completed")

                    except:
                        if os.path.exists(local_json_path):
                            os.unlink(local_json_path)

                if not missing_at_least_one_file and not os.path.exists(local_json_path):
                    log.info(f"\tIngesting {calstars_name}")
                    dict_from_calstar = calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recalibrated_path, local_calstars_path)

                    #with open(local_json_path, "w") as f:
                    #    json.dump(dict_from_calstar, f, indent=4, sort_keys=True)
                    #    f.flush()
                    #    os.fsync(f.fileno())
                    log.info(f"\tWriting to database")
                    writeStarObservationsToDB(dict_from_calstar)

                if os.path.exists(local_json_path):

                    log.info(f"\tMaking image")
                    maxCalstarsToPNG(os.path.dirname(local_calstars_path), os.path.basename(local_calstars_path))
                    #print(f"\tMaking MP4")
                    #calstarsToMP4(os.path.dirname(local_calstars_path), os.path.basename(local_calstars_path))






def makeGeoJson(names, lats, lons, output_file_path=None):
    # Example input lists

    # Build GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name , "icon": "Binoculars"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            }
            for name, lat, lon in zip(names, lats, lons)
        ]
    }

    if not output_file_path is None:
        with open(os.path.expanduser(output_file_path), "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


def makeStationsInfoDict(c, stations_data_dir=CALSTARS_DATA_DIR, country_code=None):
    """
    Make a dictionary, keyed by station name including location, geo (rads) and ecef, platepar and mask.

    Arguments:
        c: [config] RMS config instance.

    Keyword arguments:
        stations_data_dir: [str] target name in RMS_data, optional, default STATIONS_DATA_DIR.

    Return:
        stations_info_dict: [dict] dictionary with station name as key.
    """

    # Initialise
    stations_info_dict = {}
    names_list, lats_list, lons_list = [], [], []
    stations_data_full_path = os.path.join(c.data_dir, stations_data_dir)

    # Get the stations from the directory names in data_dit/STATIONS_DATA_DIR
    stations_list = sorted(os.listdir(stations_data_full_path))

    # Iterate and populate if all the expected fies are present
    for station in tqdm.tqdm(stations_list):

        if country_code is not None:
            if not station.lower().startswith(country_code.lower()):
                continue

        # Create paths
        station_info_path = os.path.join(stations_data_full_path, station)
        config_path = os.path.join(station_info_path,".config")
        pp_full_path = os.path.join(station_info_path, c.platepar_name)

        if os.path.exists(config_path):
            c = cr.parse(os.path.join(station_info_path, ".config"))
        else:
            continue

        # Locations
        lat_rads, lon_rads, ele_m = np.radians(c.latitude), np.radians(c.longitude), c.elevation
        x, y, z = latLonAlt2ECEF(lat_rads, lon_rads, ele_m)

        # Masks
        mask_struct = getMaskFile(station_info_path, c, silent=True)

        # Platepar
        pp = Platepar()
        if os.path.exists(pp_full_path):
            pp.read(pp_full_path)
        else:
            continue

        # Write dict
        stations_info_dict[station.lower()] =    {
                                                    'ecef' : (x, y, z),
                                                    'geo':
                                                        {
                                                            'lat_rads': lat_rads,
                                                            'lon_rads': lon_rads,
                                                            'ele_m': ele_m
                                                            },
                                                    'pp': pp,
                                                    'mask': mask_struct
                                                        }

        # Update lists
        names_list.append(station.lower())
        lats_list.append(np.degrees(lat_rads))
        lons_list.append(np.degrees(lon_rads))



    makeGeoJson(names_list, lats_list, lons_list,"~/RMS_data/stations_geo_json.json")

    return stations_info_dict

def calstarRaDecToDict(config, local_config_path, local_platepar_path, local_recal_path, local_calstars_path):
    """
      Parses a calstar data structures in archived directories path,
      converts to RaDec, corrects magnitude data and writes newer data to database

      Args:
          calstar (): calstar data structure for one observation session
          conn (): connection to database
          archived_directory_path ():
          latest_jd (): optional, default 0, latest jd for this station in the database

      Returns:
          calstar_radec (): list of stellar magnitude data in radec format
      """

    observation_config = cr.parse(local_config_path)
    calstar, chunk = readCALSTARS(os.path.dirname(local_calstars_path), os.path.basename(local_calstars_path))

    with open(local_recal_path, 'r') as fh:
        pp_recal_json = json.load(fh)

    pp = Platepar()
    star_dict = starListToDict(observation_config, [calstar, chunk])


    # If the star dict is empty then this was a poor observation session
    if not len(star_dict):
        return {}

    pp.read(local_platepar_path)

    observation_dict = {}

    fits_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    for fits_file, star_list in calstar:

        frame_dict = {}

        dt = FFfile.getMiddleTimeFF(fits_file, observation_config.fps, ret_milliseconds=True, ff_frames=256)
        jd = date2JD(*dt)

        if fits_file in pp_recal_json:
            # If we have a platepar in pp_recal then use it, else just use the uncalibrated platepar
            pp.loadFromDict(pp_recal_json[fits_file])

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

        obs_x, obs_y, intensity = stars[:,0], stars[:,1], stars[:,2]
        jd = np.full_like(obs_x, jd, dtype=float)

        _, obs_ra_arr, obs_dec_arr, obs_mag_arr = xyToRaDecPP(jd, obs_x, obs_y, intensity, pp, jd_time=True,
                                                  extinction_correction=True, measurement=True, precompute_pointing_corr=True)
        results = cat.queryRaDec(obs_ra_arr, obs_dec_arr)


        for r in zip(results, obs_ra_arr, obs_dec_arr, obs_mag_arr):
            r, o_ra, o_dec, o_mag = r
            name, cat_ra, cat_deg, cat_mag, theta = r

            frame_dict[name] = { "jd": jd,  "stationID": config.stationID.upper(),
                                            "cat_ra": cat_ra, "cat_deg": cat_deg, "cat_mag": cat_mag,
                                            "obs_ra": o_ra, "obs_dec": o_dec, "obs_mag": o_mag, "theta": theta}

        pass



            #print("Observed")
            #print(f"Ra:{obs_ra:6.3f}, Dec:{obs_dec:6.3f}, Mag:{obs_mag:6.3f}")

            #print("From catalog")
            #print(f"Ra:{cat_ra:6.3f}, Dec:{cat_dec:6.3f}, Mag:{cat_mag:6.3f}")

        observation_dict[fits_file] = frame_dict
        pass




    fits_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    elapsed_seconds = (fits_end_time - fits_start_time).total_seconds()
    fits_count = len(observation_dict)
    log.info(f"Processing at {fits_count / elapsed_seconds:.1f} fits / second")

    return dictInvert(observation_dict)



if __name__ == "__main__":

    import argparse



    arg_parser = argparse.ArgumentParser(description="""Conduct analysis of variable stars \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-p', '--plot', dest='plot_charts', default=False, action="store_true",
                            help="Plot chart for debugging purposes.")




    arg_parser.add_argument('--country', metavar='radec', help="""Country code to work on""")

    cml_args = arg_parser.parse_args()


    config = cr.parse(os.path.join(os.getcwd(),".config"))
    country_code = cml_args.country
    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("rmslogger")

    conn = getStationStarDBConn(STAR_OBSERVATION_DB_PATH)

    cwd = os.getcwd()



    station_list = getStationList(country_code=country_code)

    log.info("Loading star catalog")
    cat = Catalog(config)
    log.info("Loaded catalog")
    makeConfigPlateParCalstarsLib(config, station_list, cat, country_code=country_code)


    pass