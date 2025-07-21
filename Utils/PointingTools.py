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

import pickle
import json
from itertools import product
import os
import bz2
import pyvista as pv
from curses.ascii import isalnum
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyvista import active_scalars_algorithm

from RMS.Routines.MaskImage import loadMask
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, vectNorm, altAz2RADec, raDec2AltAz, J2000_JD, ECEF2AltAz, raDec2Vector, datetime2JD, geo2Cartesian, JULIAN_EPOCH, cartesian2Geo
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.EventMonitor import angularSeparationVectDeg, platepar2AltAz

import cv2
import RMS.Formats.FFfits as FFfits
import sqlite3
import time
import datetime
import numpy as np
import tempfile
import tarfile
import paramiko
import subprocess
import RMS.ConfigReader as ConfigReader
from RMS.Formats.Platepar import Platepar

from RMS.Misc import mkdirP
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo

HOST = "192.168.1.241"
USERNAME = "gmn"
PORT = "22"

def computeENUCoordinates(origin, config_platepar_dict):

    lat_origin_deg, lon_origin_deg, ele_origin_m, ecef_origin_e, ecef_origin_c, ecef_origin_u = origin
    station_list_enu = []
    for station in config_platepar_dict:
        lat_deg = config_platepar_dict[station]['config'].latitude
        lon_deg = config_platepar_dict[station]['config'].longitude
        ele_m = config_platepar_dict[station]['config'].elevation
        # print(station)
        #print("Lat: {} Lon: {} Ele: {}".format(lat_deg, lon_deg, ele_m))
        station_e, station_n, station_u =  \
                    latLonAlt2ENUDeg(lat_deg, lon_deg, ele_m, lat_origin_deg, lon_origin_deg, ele_origin_m)

        station_list_enu.append([station, (station_e, station_n, station_u)])

        #lat_deg, lo_deg, ele_m = enu2LatLonAlt(station_e, station_n, station_u, lat_origin_deg, lon_origin_deg, ele_origin_m)
        #print("Lat: {} Lon: {} Ele: {}".format(lat_deg, lon_deg, ele_m))

    return station_list_enu


def latLonAlt2ENUDeg(lat_deg, lon_deg, ele_m, lat_origin_deg, lon_origin_deg, ele_origin_m):

    lat_origin_rads, lon_origin_rads = np.radians(lat_origin_deg), np.radians(lon_origin_deg)
    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    ecef_origin_m = latLonAlt2ECEF(lat_origin_rads, lon_origin_rads, ele_origin_m)
    ecef_point_m = np.array(latLonAlt2ECEF(lat_rad, lon_rad, ele_m))
    translation = ecef_point_m - ecef_origin_m
    r = np.array([
        [-np.sin(lon_origin_rads), np.cos(lon_origin_rads), 0],
        [-np.sin(lat_origin_rads) * np.cos(lon_origin_rads), -np.sin(lat_origin_rads) * np.sin(lon_origin_rads),
         np.cos(lat_origin_rads)],
        [np.cos(lat_origin_rads) * np.cos(lon_origin_rads), np.cos(lat_origin_rads) * np.sin(lon_origin_rads),
         np.sin(lat_origin_rads)]
    ])
    e, n, u = r @ translation
    return e, n, u

def computeENUPointingUnitVectors(origin, config_platepar_dict):

    lat_origin_deg, lon_origin_deg, ele_origin_m, ecef_origin_x, ecef_origin_y, ecef_origin_z = origin
    lat_origin_rads, lon_origin_rads = np.radians(lat_origin_deg), np.radians(lon_origin_deg)
    ecef_origin_m = latLonAlt2ECEF(lat_origin_rads, lon_origin_rads, ele_origin_m)

    enu_pointing = []
    for station in config_platepar_dict:
        az_deg = config_platepar_dict[station]['pp'].az_centre
        ele_deg = config_platepar_dict[station]['pp'].alt_centre
        az_rads, ele_rads = np.radians(az_deg), np.radians(ele_deg)

        east = np.cos(ele_rads) * np.sin(az_rads)
        north = np.cos(ele_rads) * np.cos(az_rads)
        up = np.sin(ele_rads)


        enu_pointing.append([station, [east, north, up]])

    return enu_pointing

def retrieveBz2File(file_name, server):


    return

def lsRemote(host, username, port, remote_path):
    """Return the files in a remote directory.

    Arguments:
        host: [str] remote host.
        username: [str] user account to use.
        port: [int] remote port number.
        remote_pat: [str] path of remote directory to list.

    Return:)
        files: [list of strings] Names of remote files.
    """

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
    ssh.connect(hostname=host, port=port, username=username)

    try:
        sftp = ssh.open_sftp()
        files = sftp.listdir(remote_path)
        return files
    finally:
        sftp.close()
        ssh.close()

def parseTrajectoryReport(trajectory_report_path):

    with open(os.path.expanduser(trajectory_report_path), 'rb') as f:
        report = f.read().decode('utf-8').splitlines()

        trajectory_report_dict = {}
        timing_offsets_dict = {}
        section = None
        for line in report:
            if len(line):
                if line.startswith('Timing offsets (from input data):'):
                    section = "TimingOffsets"
                elif line.startswith('Reference point on the trajectory:'):
                    section = "ReferencePointOnTheTrajectory"

            if section == "TimingOffsets":
                if ": " in line:
                    key_value = line.split(":")
                    key = key_value[0].strip()
                    value = float(key_value[1].strip().split()[0])
                    timing_offsets_dict[key] = value


    trajectory_report_dict['TimingOffsets'] = timing_offsets_dict
    return trajectory_report_dict

def createTemporaryWorkArea(temp_dir=None):

    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory().name
    else:
        temp_dir = os.path.expanduser(temp_dir)
        mkdirP(temp_dir)
        temp_dir = os.path.expanduser(temp_dir)

    return temp_dir

def extractBz2(input_directory, working_directory, station_list, epoch):

    bz2_list = []
    input_directory = os.path.expanduser(input_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".bz2") and filename.split("_")[0].lower() in station_list:
            bz2_list.append(filename)

    bz2_list.sort(reverse=True)



    mkdirP(working_directory)

    bz2_directory_list = extractBz2Files(bz2_list, input_directory, working_directory, station_list, epoch)



    return bz2_directory_list

def extractBz2Files(bz2_list, input_directory, working_directory, station_list, epoch_utc):


    bz2_directory_list = []
    for bz2 in bz2_list:
        station = bz2.split("_")[0]
        file_date = os.path.basename(bz2).split("_")[1]
        file_time = os.path.basename(bz2).split("_")[2]
        year, month, day = file_date[0:4], file_date[4:6], file_date[6:8]
        hour, minute, second = file_time[0:2], file_time[2:4], file_time[4:6]
        directory_time_object = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        directory_time_object_utc = directory_time_object.replace(tzinfo=datetime.timezone.utc)
        if directory_time_object_utc < epoch_utc and station.lower() in station_list:
            station_directory = os.path.join(working_directory, bz2.split("_")[0]).lower()
            mkdirP(station_directory)
            bz2_directory = os.path.join(station_directory, bz2.split(".")[0])
            bz2_directory_list.append(bz2_directory)
            if os.path.exists(bz2_directory):
                continue
            mkdirP(bz2_directory)
            print("Extracting {}".format(bz2))
            with tarfile.open(os.path.join(input_directory, bz2), 'r:bz2') as tar:
                tar.extractall(path=bz2_directory)

    return bz2_directory_list

def readInFTPDetectInfoFiles(working_directory, station_list=None, local_available_directories=None, event_time=None):


    archived_directory_list, station_directories = getArchivedDirectories(working_directory, event_time=event_time, station_list=station_list)
    if local_available_directories is not None:
        archived_directory_list = local_available_directories
    ftp_dict = getFTPFileDictionary(archived_directory_list, station_directories, working_directory, station_list=station_list, event_time=event_time)
    return ftp_dict

def getFTPFileDictionary(archived_directory_list, station_directories, working_directory, station_list=None, event_time=None):

    ftp_dict = {}

    for archived_directory in sorted(archived_directory_list, reverse=True):
        station = os.path.basename(archived_directory).split("_")[0].lower()
        print("Working on station {}".format(station))
        if station_list is not None:
            if not station in station_list:
                continue
            if not event_time is None:
                directory_date = os.path.basename(archived_directory).split("_")[1]
                directory_time = os.path.basename(archived_directory).split("_")[2]
                year, month, day = directory_date[0:4], directory_date[4:6], directory_date[6:8]
                hour, minute, second = directory_time[0:2], directory_time[2:4], directory_time[4:6]
                directory_time_object = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                if directory_time_object < event_time:

                    ftp_file_name = getFTPFileName(archived_directory, station, working_directory)
                    print("Loading {} from {}".format(ftp_file_name, archived_directory))

                    ftp_path = os.path.join(working_directory, station, archived_directory)
                    print("For station {} reading {}".format(station, ftp_file_name))
                    ftp_dict[station] = readFTPdetectinfo(ftp_path, ftp_file_name)

    return ftp_dict

def getFTPFileName(archived_directory, station, working_directory):
    ar_date, ar_time = archived_directory.split("_")[1], archived_directory.split("_")[2]
    ar_milliseconds = archived_directory.split("_")[3]
    ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
    print("Preparing to use {}".format(ftp_file_name))
    if not os.path.exists(os.path.join(working_directory, station, archived_directory, ftp_file_name)):
        directory_containing_ftp = os.path.join(working_directory, station.lower(), archived_directory)
        ftp_file_name = None

        for file_name in os.listdir(directory_containing_ftp):
            if file_name.startswith("FTPdetectinfo") and file_name.endswith(".txt") and "manual" in file_name:
                ftp_file_name = file_name
                return ftp_file_name

        for file_name in os.listdir(directory_containing_ftp):

            if file_name.startswith("FTPdetectinfo") and file_name.endswith(".txt") and file_name.split("_")[1] == station.upper():

                ftp_file_name = file_name
                return ftp_file_name

    return ftp_file_name

def getArchivedDirectories(working_directory, event_time=None, station_list=None):

    if station_list is None:
        station_directories = sorted(os.listdir((working_directory)))
    else:
        station_directories = station_list
    archived_directory_list = []
    for station_directory in station_directories:
        target_path = os.path.join(working_directory, station_directory)
        if not os.path.exists(target_path):
            print("No files found for {}".format(target_path))
            continue
        extracted_directories_directory_list = os.listdir(os.path.join(working_directory, station_directory))
        if extracted_directories_directory_list is not None:
            extracted_directories_directory_list.sort(reverse=True)
            for extracted_directory in extracted_directories_directory_list:
                fits_list = []
                extracted_directory_date = extracted_directory.split("_")[1]
                extracted_directory_time = extracted_directory.split("_")[2]
                year, month, day = extracted_directory_date[0:4], extracted_directory_date[4:6], extracted_directory_date[6:8]
                hour, minute, second = extracted_directory_time[0:2], extracted_directory_time[2:4], extracted_directory_time[4:6]
                extracted_directory_time_object = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                if extracted_directory_time_object < event_time:
                    extracted_directory_files = os.listdir(os.path.join(working_directory, station_directory, extracted_directory))
                    for file_name in extracted_directory_files:
                        if file_name.startswith("FF_{}".format(station_directory.upper())) and file_name.endswith(".fits"):
                            fits_list.append(file_name)
                    for fits_file in fits_list:
                        file_time = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(fits_file), "%Y-%m-%d %H:%M:%S.%f")
                        if abs(event_time - file_time).total_seconds() < 20:
                            archived_directory_list.append(os.path.join(working_directory, station_directory, extracted_directory))


    return archived_directory_list, station_directories

def clusterByTime(ftp_dict, station_list, event_time, duration):
    # Rearrange into time

    observations = getObservations(ftp_dict, station_list=station_list, event_time=event_time,duration=duration)
    events = groupObservationsIntoEvents(observations)

    return events

def getPathsOfFilesToRetrieve(station_list, epoch_utc=None):

    files_to_retrieve = []
    print("Getting paths to retrieve for stations {} at time {}".format(station_list, epoch_utc))

    for station in station_list:
        remote_path = os.path.join("/home", station.lower(), "files", "incoming")
        bz2_files = []
        while bz2_files == []:
            try:
                bz2_files = lsRemote("192.168.1.241", "gmn", 22, remote_path)
            except:
                time.sleep(120)

        bz2_files.sort(reverse=True)
        for file_name in bz2_files:
            if file_name.endswith("_detected.tar.bz2"):
                file_name_time_naive = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(file_name), "%Y-%m-%d %H:%M:%S.%f")
                file_name_time_utc = file_name_time_naive.replace(tzinfo=datetime.timezone.utc)
                if file_name_time_utc < epoch_utc:
                    files_to_retrieve.append(os.path.join(remote_path, file_name))
                    break

    if len(files_to_retrieve):
        print("Following paths to be retrieved")
        for file_path in files_to_retrieve:
            print("\t{}".format(file_path))

    return files_to_retrieve

def downloadFile(host, username, port, remote_path, local_path):
    """Download a single file try compressed rsync first, then fall back to Paramiko

    Arguments:
        host: [str] hostname of remote machine.
        username: [str] username for remote machine.
        port: [str] port.
        remote_path: [path] full path to destination.
        local_path: [path] full path of local target.

    Return:
        Nothing.
    """

    try:

        remote = "{}@{}:{}".format(username, host, remote_path)
        result = subprocess.run(['rsync', '-z', remote], capture_output=True, text=True)
        if "No such file or directory" in result.stderr :
            print("Remote file {} was not found.".format(os.path.basename(remote)))
            return
        else:
            result = subprocess.run(['rsync', '-z', remote, local_path], capture_output=True, text=True)
        if not os.path.exists(os.path.expanduser(local_path)):
            print("Download of {} from {}@{} failed. You need to add your keys to remote using ssh-copy-id.".format(remote_path, username,host))
            quit()
        return
    except:
        pass

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
    try:
        ssh.connect(hostname=host, port=port, username=username)
    except:
        print("Login to {}@{} failed. You need to add your keys to remote using ssh-copy-id.".format(username,host))
        quit()
    try:
        sftp = ssh.open_sftp()
        remote_file_list = sftp.listdir(os.path.dirname(remote_path))
        if remote_file_list:
            sftp.get(remote_path, local_path)

    finally:
        sftp.close()
        ssh.close()

    return

def filesNotAvailableLocally(station_list, epoch_utc):

    station_files_to_retrieve = []
    local_dirs_to_use = []
    print("Station list {}".format(station_list))
    for station in station_list:
        file_present_locally = False
        local_station_path = os.path.expanduser(os.path.join("~/tmp/pointing_tools_working_area/", station.lower()))
        if not os.path.exists(local_station_path):
            station_files_to_retrieve.append(station)
            print("\tMust retrieve files for {}".format(station))
            continue
        station_detected_dir_list = os.listdir(local_station_path)
        station_detected_dir_list.sort(reverse=True)

        for detected_dir in station_detected_dir_list:
            detected_dir_date = detected_dir.split("_")[1]
            detected_dir_time = detected_dir.split("_")[2]
            year, month, day = detected_dir_date[0:4], detected_dir_date[4:6], detected_dir_date[6:8]
            hour, minute, second = detected_dir_time[0:2], detected_dir_time[2:4], detected_dir_time[4:6]
            detected_dir_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            detected_dir_time = detected_dir_time.replace(tzinfo=datetime.timezone.utc)
            if (detected_dir_time - epoch_utc).total_seconds() < 7 * 24 * 60 * 60:
                detected_dir_full_path = os.path.join("~/tmp/pointing_tools_working_area", station.lower(), detected_dir)
                detected_dir_full_path = os.path.expanduser(detected_dir_full_path)
                local_dirs_to_use.append(detected_dir_full_path)
                file_present_locally = True
                break

        if not file_present_locally:
            print("No file present locally for station {}, adding to retrieve list".format(station.lower()))
            station_files_to_retrieve.append(station)

    return station_files_to_retrieve, local_dirs_to_use

def bz2Valid(file_path):


    try:
        with bz2.open(file_path, 'rb') as f:
            while f.read(1024):  # Read in chunks
                pass
        return True
    except (OSError, EOFError):
        return False


def downloadFiles(station_list, epoch_utc=None):


    if epoch_utc == None:
        epoch_utc = datetime.datetime.now(tz=datetime.timezone.utc)


    if station_list is not None and epoch_utc is not None:
        station_list_to_get, local_available_directories = filesNotAvailableLocally(station_list, epoch)
        remote_path_list = getPathsOfFilesToRetrieve(station_list_to_get, epoch_utc)
        if len(remote_path_list):
            print("Retrieving from remote:")
            for d in remote_path_list:
                print("\t{}".format(d))
        if len(local_available_directories):
            print("These directories already available:")
            for d in local_available_directories:
                print("\t{}".format(d))
        for path in remote_path_list:
            basename = os.path.basename(path)
            local_target = os.path.join(os.path.expanduser("~/RMS_data/bz2files/"), basename)
            if not os.path.exists(local_target):
                print("Downloading {} to {}".format(basename, local_target))
                downloadFile("192.168.1.241", "gmn", 22, path, local_target )
            elif not bz2Valid(local_target):
                print("Downloading {} to {}".format(basename, local_target))
                downloadFile("192.168.1.241", "gmn", 22, path, local_target)

    working_area = createTemporaryWorkArea("~/tmp/pointing_tools_working_area")
    bz2_directory_list = extractBz2("~/RMS_data/bz2files", working_area, station_list, epoch)


    return bz2_directory_list

def processDatabase(database_path, country_code):
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if country_code is None:
        cursor.execute(
        'SELECT "Unique trajectory identifier", "Beginning UTC Time", "Duration sec", "Participating Stations", "Peak AbsMag" FROM Trajectories Order by "Peak AbsMag" ASC')
    else:
        cursor.execute(
        'SELECT "Unique trajectory identifier", "Beginning UTC Time", "Duration sec", "Participating Stations", "Peak AbsMag" FROM Trajectories WHERE "Participating Stations" LIKE "% {}%" Order by "Peak AbsMag" ASC'.format(country_code))
    row_list = []
    for row in cursor:
        row_list.append(row)
    cursor.close()
    conn.close()
    for row in row_list:
        uti, utc_time, duration_sec, stations, magnitude = row[0], row[1].strip(), row[2], row[3].lower(), row[4]
        if datetime.datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S.%f") < datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, microsecond=0):
            continue

        stations = stations.replace('\n','').replace(" ","")
        station_list = stations.split(",")
        event_time = datetime.datetime.strptime(utc_time.strip(), "%Y-%m-%d %H:%M:%S.%f")
        output_file = os.path.join(os.path.expanduser("~/RMS_data/trajectory_images"), uti)
        output_file = "{}.png".format(output_file)
        if not os.path.exists(output_file):
            print("Producing chart for uti  {}".format(uti))
            print("Involving stations       {}".format(station_list))
            print("At time                  {}".format(utc_time))
            print("Magnitude                {}".format(magnitude))

            produceCollatedChart(input_directory,
                             station_list=station_list,
                             event_time=event_time,
                             duration=duration_sec, target_file_name=output_file, magnitude=magnitude)
        else:
            print("Chart for uti {} already exists".format(uti))
    pass



# WGS84 ellipsoid constants
a = 6378137.0          # semi-major axis (meters)
f = 1 / 298.257223563  # flattening
e2 = f * (2 - f)       # eccentricity squared

def geo2ECEF(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    return np.array([x, y, z])

def enu2ECEFMatrix(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.array([
        [-np.sin(lon),            np.cos(lon),           0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])

def altAz2enuVectorDegrees(alt_deg, az_deg):

    alt_rads, az_rads = np.radians(alt_deg), np.radians(az_deg)

    e = np.cos(alt_rads) * np.sin(az_rads)  # East
    n = np.cos(alt_rads) * np.cos(az_rads)  # North
    u = np.sin(alt_rads)  # Up

    return np.array([e, n, u])


def enu2Ecef(e, n, u, lat_ref, lon_ref, alt_ref):
    # Constants
    a = 6378137.0  # WGS-84 semi-major axis
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lat = np.radians(lat_ref)
    lon = np.radians(lon_ref)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x0 = (N + alt_ref) * np.cos(lat) * np.cos(lon)
    y0 = (N + alt_ref) * np.cos(lat) * np.sin(lon)
    z0 = ((1 - e2) * N + alt_ref) * np.sin(lat)

    # ENU to ECEF rotation
    R = np.array([
        [-np.sin(lon), -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
        [ np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
        [          0,            np.cos(lat),            np.sin(lat)]
    ])
    enu_vec = np.array([e, n, u])
    ecef_offset = R @ enu_vec
    return np.array([x0, y0, z0]) + ecef_offset



def enu2AltAz(vec):
    # Convert to unit vector
    #vec = np.array([e, n, u])
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-length vector has undefined alt/az.")

    unit = vec / norm

    # Altitude: angle above horizon (Up component)
    alt_rad = np.arcsin(unit[2])
    alt_deg = np.degrees(alt_rad)

    # Azimuth: angle clockwise from North
    az_rad = np.arctan2(unit[0], unit[1])  # East over North
    az_deg = (np.degrees(az_rad)) % 360    # Normalize to [0, 360)

    return alt_deg, az_deg

def computePointCoverage(origin, config_mask_platepar_dict, point_e,point_n,point_u):

    lat_origin_deg, lon_origin_deg, ele_origin_m, ecef_origin_x, ecef_origin_y, ecef_origin_z = origin

    point_lat_rads, point_lon_rads, point_alt_m = enu2LatLonAlt(point_e, point_n, point_u, lat_origin_deg, lon_origin_deg, ele_origin_m)
    point_ecef_x, point_ecef_y, point_ecef_z = latLonAlt2ECEFDeg(point_lat_rads, point_lon_rads, point_alt_m)
    enu_coordinates = computeENUCoordinates(origin, config_mask_platepar_dict)
    cameras_covering_point = 0
    cameras_list = []
    for camera in enu_coordinates:

        camera_found = False
        for s in config_mask_platepar_dict:

            if camera[0] == s.lower():
                camera_found = True
                station_e, station_n, station_u = camera[1]
                pp = config_mask_platepar_dict[s]['pp']
                diagonal_fov = np.sqrt(pp.fov_v ** 2 + pp.fov_h ** 2)
                break


        if not camera_found:
            continue

        # compute vector from station to point in enu
        v_e = point_e - station_e
        v_n = point_n - station_n
        v_u = point_u - station_u
        station_to_point_enu_norm = vectNorm((v_e, v_n, v_u))
        fov_vec_enu = vectNorm(altAz2enuVectorDegrees(pp.alt_centre, pp.az_centre))

        ang_sep = angularSeparationVectDeg(station_to_point_enu_norm, fov_vec_enu)


        if ang_sep < diagonal_fov / 2:
            # there is a good chance this will be in the fov
            alt_point, az_point = enu2AltAz(station_to_point_enu_norm)
            ra, dec = altAz2RADec(az_point, alt_point, 2460878 , pp.lat, pp.lon)
            x, y = raDecToXYPP(np.array([ra]), np.array([dec]), 2460878, pp)
            x, y = round(x[0]), round(y[0])
            # now check the mask

            if 0 < x < pp.X_res and 0 < y < pp.Y_res:
                mask_arr = config_mask_platepar_dict[s]['mask'].img
                mask_val = mask_arr[y, x]
                if mask_val != 255:
                    #print("Masked for camera {} at alt, az {},{}".format(s,az_point,alt_point))
                    continue

                cameras_covering_point += 1
                cameras_list.append(camera)
            else:
                #print("Out of sensor range camera:{} x:{} y:{}".format(s, x,y))
                pass
    return cameras_covering_point, cameras_list



def computeCoverageQuality(origin, config_mask_platepar_dict, e_range, n_range, u_range, step_size=10000):

    e_range_list = range(e_range[0], e_range[1], step_size)
    n_range_list = range(n_range[0], n_range[1], step_size)
    u_range_list = range(u_range[0], u_range[1], step_size)

    cartesian_product_list = list(product(e_range_list, n_range_list, u_range_list))
    coverage_list = []
    for e, n, u in cartesian_product_list:
        coverage_list.append([e, n, u, computePointCoverage(origin, config_mask_platepar_dict, e,n,u)])

    return coverage_list

def plot(coordinates_list, pointing_list, single_station_coverage, zero_station_coverage, good_coverage):


    # Unpack coordinates
    name_list, e_list, n_list, u_list, vector_start_list, vector_end_list  = [], [], [], [], [], []
    for coordinates, pointing    in zip(coordinates_list, pointing_list):
        name, coordinate_enu = coordinates
        name, pointing_enu = pointing
        e, n, u = coordinate_enu
        _e, _n, _u = pointing_enu
        e_list.append(e)
        n_list.append(n)
        u_list.append(u)
        start_vector = np.array((e, n, u))
        end_vector = start_vector + np.array((_e, _n, _u))
        vector_start_list.append(start_vector)
        vector_end_list.append(end_vector)
        name_list.append(name)

    # Create plot
    fig = plt.figure(figsize=(24,18))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim(-200000, +400000)
    #ax.set_ylim(-200000, +400000)
    ax.set_zlim(0, +600000)
    ax.set_box_aspect([1, 1, 1])

    # Plot the camera directions
    pointing_data_enu = zip(name_list, vector_start_list, vector_end_list)

    for pointing_enu in pointing_data_enu:
        name = pointing_enu[0]
        x, _x = pointing_enu[1][0], pointing_enu[2][0]
        y, _y = pointing_enu[1][1], pointing_enu[2][1]
        z, _z = pointing_enu[1][2], pointing_enu[2][2]
        vx, vy, vz = (_x - x), (_y - y), (_z - z)
        ax.text(-2500 + x + 90000 * vx, y + 90000 * vy, z + 90000 * vz, name, fontsize=6)
        ax.quiver(x, y, z, vx, vy, vz, length=80000)


    # Plot the stations
    ax.scatter(e_list, n_list, u_list, c='red', s=40)

    # Plot the coverage
    e_list, n_list, u_list = [], [], []
    for point in single_station_coverage:

        e, n, u = point[0], point[1], point[2]
        e_list.append(e)
        n_list.append(n)
        u_list.append(u)

    ax.scatter(e_list, n_list, u_list, c='green', s=100, alpha=0.2, edgecolor='none')

    e_list, n_list, u_list = [], [], []
    for point in zero_station_coverage:

        e, n, u = point[0], point[1], point[2]
        e_list.append(e)
        n_list.append(n)
        u_list.append(u)

    #ax.scatter(e_list, n_list, u_list, c='black', s=100, alpha=0.05, edgecolor='none')

    e_list, n_list, u_list = [], [], []
    for point in good_coverage:
        e, n, u = point[0], point[1], point[2]
        e_list.append(e)
        n_list.append(n)
        u_list.append(u)

    #ax.scatter(e_list, n_list, u_list, c='blue', s=100, alpha=0.05, edgecolor='none')

    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_zlabel('U')
    plt.title('Single station coverage, plotted in ENU metres', fontsize=30)
    plt.show()



def getConfigMaskPlateparDict(bz2_directory_list):

    bz2_directory_list = sorted(bz2_directory_list)
    config_pp_dict = {}
    for directory in bz2_directory_list:
        station = os.path.basename(directory).split("_")[0].lower()
        station_dict = {}
        config_path = os.path.join(directory, ".config")
        config = ConfigReader.parse(config_path)
        station_dict['config'] = config
        pp = Platepar()
        pp_name = config.platepar_name
        pp_path = os.path.join(directory, pp_name)
        pp.read(pp_path)
        station_dict['pp'] = pp
        mask_path = os.path.join(directory, config.mask_file)
        station_dict['mask'] = loadMask(mask_path)
        config_pp_dict[station] = station_dict



    return config_pp_dict

def getStationCoordinatesList(dict):

    x_list, y_list, z_list, name_list = [], [], [], []
    for station in dict:
        config = dict[station]['config']
        lat = np.radians(config.latitude)
        lon = np.radians(config.longitude)
        ele = config.elevation
        x, y, z = latLonAlt2ECEF(lat, lon, ele)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        name_list.append(station)

    return x_list, y_list, z_list, name_list

def computeAverageLocation(dict):

    x_list, y_list, z_list, name_list = getStationCoordinatesList(dict)
    x = np.mean(x_list)
    y = np.mean(y_list)
    z = np.mean(z_list)

    lat_rads, lon_rads, ele_m = ecef2LatLonAlt(x, y, z)
    lat_deg, lon_deg = np.degrees(lat_rads), np.degrees(lon_rads)

    return lat_deg, lon_deg, ele_m, x, y, z

def latLonAlt2ECEFDeg(lat_deg, lon_deg, elevation_m):

    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(lat_rad, lon_rad, elevation_m)

    return ecef_x, ecef_y, ecef_z

def ecef2LatLonAltDeg(ecef_x, ecef_y, ecef_z):

    lat_rad, lon_rad, elevation_m = ecef2LatLonAlt(ecef_x, ecef_y, ecef_z)
    lat_deg, lon_deg = np.degrees(lat_rad), np.degrees(lon_rad)

    return lat_deg, lon_deg, elevation_m

def enu2LatLonAlt(e, n, u, lat_origin_deg, lon_origin_deg, ele_origin_deg):

    x, y, z = enu2Ecef(e, n, u, lat_origin_deg, lon_origin_deg, ele_origin_deg)
    lat_deg, lon_deg, ele_m = ecef2LatLonAltDeg(x, y, z)

    return lat_deg, lon_deg, ele_m

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('station_list', help='Stations to Plot')

    arg_parser.add_argument('-e', '--epoch', nargs=1, metavar='EPOCH', type=str,
                            help="Epoch of the map")

    arg_parser.add_argument('-u', '--country_code', nargs=1, metavar='COUNTRY_CODE', type=str,
                            help="Country code to process.")

    arg_parser.add_argument('-o', '--oneshot', dest='one_shot', default=False, action="store_true",
                            help="Run once, and terminate.")

    arg_parser.add_argument('-d', '--deletedb', dest='delete_db', default=False, action="store_true",
                            help="Delete the event_monitor database at initialisation.")

    arg_parser.add_argument('-k', '--keepfiles', dest='keepfiles', default=False, action="store_true",
                            help="Keep working files")

    arg_parser.add_argument('-n', '--noupload', dest='noupload', default=False, action="store_true",
                            help="Do not upload")

    cml_args = arg_parser.parse_args()
    mkdirP(os.path.expanduser("~/RMS_data"))
    mkdirP(os.path.expanduser("~/RMS_data/bz2files"))
    mkdirP(os.path.expanduser("~/RMS_data/trajectory_images"))

    perth_observatory_lat = -32.0076728
    perth_observatory_lon = +116.132912
    perth_observatory_ele_m = 382


    print(perth_observatory_lat, perth_observatory_lon, perth_observatory_ele_m)
    ecef_x, ecef_y, ecef_z = latLonAlt2ECEFDeg(perth_observatory_lat, perth_observatory_lon, perth_observatory_ele_m)
    print(ecef_x, ecef_y, ecef_z)
    lat_deg, lon_deg, ele_m = ecef2LatLonAltDeg(ecef_x, ecef_y, ecef_z)
    print(lat_deg, lon_deg, ele_m)

    e, n, u = latLonAlt2ENUDeg(perth_observatory_lat, perth_observatory_lon, perth_observatory_ele_m, perth_observatory_lat-1, perth_observatory_lon, perth_observatory_ele_m)
    print(e, n, u)
    print(enu2LatLonAlt(e, n, u, perth_observatory_lat-1, perth_observatory_lon, perth_observatory_ele_m))
    x, y, z = enu2Ecef(e, n, u, perth_observatory_lat - 1, perth_observatory_lon, perth_observatory_ele_m)
    print(x, y, z)
    lat, lon, alt = ecef2LatLonAltDeg(x, y, z)
    print(lat, lon, alt)
    lat, lon, alt = enu2LatLonAlt(e, n, u, perth_observatory_lat-1, perth_observatory_lon, perth_observatory_ele_m)
    print(lat, lon, alt)



    station_list_raw = cml_args.station_list

    if station_list_raw is not None:
        station_list_raw = cml_args.station_list.split(",")
        station_list = []
        for station in station_list_raw:
            station_list.append(station.strip())

    epoch=cml_args.epoch
    if epoch is None:
        epoch = datetime.datetime.now(tz=datetime.timezone.utc)
    else:
        epoch_date = os.path.basename(epoch).split("_")[0]
        epoch_time = os.path.basename(epoch).split("_")[1]
        year, month, day = epoch_date[0:4], epoch_date[4:6], epoch_date[6:8]
        hour, minute, second = epoch_time[0:2], epoch_time[2:4], epoch_time[4:6]
        epoch_utc = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

    bz2_directory_list = downloadFiles(station_list, epoch)
    config_mask_platepar_dict = getConfigMaskPlateparDict(bz2_directory_list)

    origin = computeAverageLocation(config_mask_platepar_dict)

    enu_coordinates = computeENUCoordinates(origin, config_mask_platepar_dict)
    enu_pointing_unit_vectors = computeENUPointingUnitVectors(origin, config_mask_platepar_dict)

    coverage_quality = computeCoverageQuality(origin, config_mask_platepar_dict, [-200000, +200000], [-200000, +200000], [0, 400000], step_size=20000)

    good_coverage=[]
    single_station_coverage = []
    for point in coverage_quality:
        cameras_covering_point = point[3][0]
        if cameras_covering_point > 2:
            good_coverage.append(point)
        pass

    single_station_coverage = []
    for point in coverage_quality:
        cameras_covering_point = point[3][0]
        if cameras_covering_point == 1:
            single_station_coverage.append(point)
        pass

    zero_station_coverage = []
    for point in coverage_quality:
        cameras_covering_point = point[3][0]
        if cameras_covering_point == 0:
            zero_station_coverage.append(point)
        pass


    with open('single_station_coverage.pkl', 'wb') as f:
        pickle.dump(single_station_coverage, f)
    


    with open('single_station_coverage.pkl', 'rb') as f:
        single_station_coverage = pickle.load(f)



    plot(enu_coordinates, enu_pointing_unit_vectors, single_station_coverage, zero_station_coverage, good_coverage)