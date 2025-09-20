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


from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, ECEF2AltAz, altAz2RADec, raDec2AltAz, AER2ECEF
from RMS.Math import angularSeparationDeg, vectNorm
from RMS.Routines.MaskImage import getMaskFile
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP
from scipy.spatial import cKDTree


from RMS.Misc import mkdirP
import matplotlib.pyplot as plt



REMOTE_SERVER = 'gmn.uwo.ca'
USER_NAME = "analysis"
STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"
STATIONS_DATA_DIR = "StationData"
REMOTE_STATION_PROCESSED_DIR = "/home/$STATION/files/processed"
WORKING_DIRECTORY = os.path.expanduser("~/RMS_data/Coverage")
CHARTS = "charts"
PORT = 22

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
    ssh.connect(hostname=host, port=port, username=username, sock=sock)

    try:
        sftp = ssh.open_sftp()
        file_list = sftp.listdir(remote_path)
        return file_list
    finally:
        sftp.close()
        ssh.close()

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
            print("Extracting {}".format(bz2))

        try:
            with tarfile.open(os.path.join(input_directory, bz2), 'r:bz2') as tar:
                tar.extractall(path=bz2_directory)
        except:
            if not silent:
                print("Redownloading {}".format(basename_bz2))
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
        earliest_date = datetime.datetime.now() - datetime.timedelta(days=3)

    if latest_date is None:
        latest_date = datetime.datetime.now() + datetime.timedelta(days=3)

    filtered_files_list = []
    for file in files_list:

        if len(file.split("_")) != 5:
            continue

        date = file.split("_")[1]
        time = file.split("_")[2]
        year, month, day = int(date[0:4]), int(date[4:6]), int(date[6:8])
        hour, minute, second = int(time[0:2]), int(time[2:4]), int(time[4:6])
        file_date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
        if earliest_date < file_date < latest_date:
            filtered_files_list.append(file)

    return filtered_files_list


def makeConfigPlateParMaskLib(config, station_list, stations_data_dir=STATIONS_DATA_DIR,
                              remote_station_processed_dir=REMOTE_STATION_PROCESSED_DIR,
                              host=REMOTE_SERVER, username=USER_NAME, port=PORT):

    """
    In a subdirectoy of station_data_dir create a directory for each station containing mask
    platepar and config file.

    Arguments:
        config: [config] RMS config instance - used to get data_dir.
        station_list: [list] list of stations.

    Keyword arguments:
        stations_data_dir: [str] target name in RMS_data, optional, default STATIONS_DATA_DIR.
        remote_station_processed_dir: [str] path on remote server, optional, default REMOTE_STATION_PROCESSED_DIR.
        host: [str] host name of remote machine, optional, default REMOTE_SERVER.
        username: [str] username for remote machine, optional, default USER_NAME.
        port: [int] optional, default PORT constant, optional, default PORT.

    Return:
        Nothing.
    """

    stations_data_full_path = os.path.join(config.data_dir, stations_data_dir)

    print("Starting to download files.")
    for station in tqdm.tqdm(station_list):
        local_target = os.path.join(stations_data_full_path, station.lower())

        with tempfile.TemporaryDirectory() as t:
            remote_dir = remote_station_processed_dir.replace("$STATION", station.lower())

            # Create paths up front to reduce clutter
            extraction_dir = os.path.join(t, "extracted")
            local_target_full_path = os.path.join(local_target)
            local_config_path = os.path.join(local_target_full_path, os.path.basename(config.config_file_name))
            local_platepar_path = os.path.join(local_target_full_path, config.platepar_name)
            local_mask_path = os.path.join(local_target_full_path, config.mask_file)

            # If data already exists, then continue to next station
            if os.path.exists(local_config_path) and \
                    os.path.exists(local_platepar_path) and \
                        os.path.exists(local_mask_path):
                continue


            # Get the list of the files, newest at the top
            remote_files = sorted(lsRemote(host, username, port, remote_dir), reverse=True)

            remote_files = filterByDate(remote_files, earliest_date=datetime.datetime(year=2024, month=1, day=1))

            if not len(remote_files):
                continue


            # Pick the newest file, or continue to next station if no files returned
            if len(remote_files):
                latest_remote_file = remote_files[0]
            else:
                continue

            extracted_files_path = os.path.join(extraction_dir, station.lower(), latest_remote_file.split(".")[0])
            extracted_config_path = os.path.join(extracted_files_path, ".config")
            extracted_platepar_path = os.path.join(extracted_files_path, config.platepar_name)
            extracted_mask_path = os.path.join(extracted_files_path, config.mask_file)
            full_remote_path_to_bz2 = os.path.join(remote_dir, latest_remote_file)


            # Download, and extract the file into a subdir
            downloadFile(host, username, t, full_remote_path_to_bz2)
            mkdirP(extraction_dir)
            extractBz2(t, extraction_dir)

            # If the source files exist in this tempdir copy everything to the target
            if os.path.exists(extracted_config_path) and \
               os.path.exists(extracted_platepar_path) and \
               os.path.exists(extracted_mask_path):

                mkdirP(local_target_full_path)
                shutil.move(extracted_config_path, local_config_path)
                shutil.move(extracted_platepar_path, local_platepar_path)
                shutil.move(extracted_mask_path, local_mask_path)


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


def makeStationsInfoDict(c, stations_data_dir=STATIONS_DATA_DIR, country_code=None):
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

def filterPointsByElevation(points_list, min_ele, max_ele):

    """
    Filter points based on elevation, excludes the limits.

    Args:
        points_list: [list] list of ECEF points (x,y,z).
        min_ele: [float] minimum elevation in metres.
        max_ele: [float] maximum elevation in metres.

    Returns:
        [list] filtered list of points based on elevation.

    """

    points_filtered_by_elevation = []
    for point in points_list:
        lat_rads, lon_rads, alt = ecef2LatLonAlt(point[0], point[1], point[2])
        if min_ele < alt < max_ele:
            points_filtered_by_elevation.append(point)

    return points_filtered_by_elevation

def roundList(list, resolution_m):
    """
    Round a list of lists of numbers to the specified resolution.

    Args:
        list: [[list]] list of lists of numbers.
        resolution_m: [float] resolution.

    Returns:
        [[list]] rounded list of lists of numbers, rounded to resolution.
    """


    output_list = []

    for point in list:
        output_coord = []
        for coord in point:
            output_coord.append(round(coord / resolution_m) * resolution_m)
        output_list.append(output_coord)

    return output_list

def makeECEFPointListAroundStations(station_info_dict, max_dist_m, resolution_m, min_ele_m=20000, max_ele_m=100000):
    """

    Args:
        station_info_dict: [dict] station info dict.
        max_dist_m: [float] maximum distance to the station in metres.
        resolution_m: [float] resolution in metres.
        min_ele_m: [float] minimum elevation in metres.
        max_ele_m: [float] maximum elevation in metres.

    Returns:
        combined_points_array: [array] Unique points in ECEF snapped to resolution around each station.
    """

    if not len(station_info_dict):
        return []

    # Compute a list of offsets to apply to each station - do this only once and reuse
    offsets_list = np.arange(0 - max_dist_m, 0 + max_dist_m + resolution_m, resolution_m)
    # Make the vertices
    x_list, y_list, z_list = np.meshgrid(offsets_list, offsets_list, offsets_list, indexing='ij')

    # Fill the cube
    local_points_cube = np.vstack([x_list.ravel(), y_list.ravel(), z_list.ravel()]).T

    # Trim away anything outside the sphere of max distance - this will leave some points with negative elevations for
    # some stations - expected.
    points_template = local_points_cube[np.linalg.norm(local_points_cube, axis=1) <= max_dist_m]

    # Free up some memory
    del x_list, y_list, z_list, offsets_list, local_points_cube
    gc.collect()

    # Initialise an array for points within elevation range for all stations, without duplicates
    combined_points_array = np.empty((0,3))

    # Iterate over all the stations
    for station in tqdm.tqdm(station_info_dict):

        # Get the ecef information for this station
        station_ecef = station_info_dict[station]['ecef']

        # create local points by shifting template by station origin
        local_points = points_template + station_ecef

        # Combine the points within the elevation limit for this station with points found so far
        combined_points_array = np.vstack([ np.array(filterPointsByElevation(local_points, min_ele_m, max_ele_m)), combined_points_array])

        # Round to resolution and force to integers for speed (not sure if this applies in python)
        indices = (combined_points_array / resolution_m).round().astype(int)

        # Remove duplicates better for memory to do this each iteration - worse for time
        combined_points_array = np.array((list(set([tuple(row) for row in indices])))) * resolution_m

    return combined_points_array


def makeECEFPointList(station_info_dict, min_ele_m=20000, max_ele_m=100000, resolution_m = 200000, max_distance_to_station_m=500000):
    """
    Given the station info dict, make a list of ECEF points around the stations in the list within the local WGS84
    elevation limits and at the specified resolution.

    Args:
        station_info_dict: [dict] station info dict.

    Keyword arguments:
        min_ele_m: [float] minimum elevation in metres.
        max_ele_m: [float] maximum elevation in metres.
        resolution_m: [float] resolution in metres.
        max_distance_to_station_m: [float] maximum distance to station in metres.

    Returns:
        [dict] list of ECEF points around the station.
    """

    print("Making array of coordinates, radius {:.0f}km at resolution {:.1f}km around {} stations"
          .format(max_distance_to_station_m / 1000, resolution_m / 1000, len(station_info_dict)))

    ecef_point_array_around_stations = makeECEFPointListAroundStations(station_info_dict,
                                                                       max_distance_to_station_m, resolution_m,
                                                                       min_ele_m=min_ele_m, max_ele_m=max_ele_m)

    return ecef_point_array_around_stations

def addStationsToECEFArray(ecef_point_array, station_info_dict, radius=500000):
    """
    Given an array of ECEF points, and the station_info_dict, return a list of mappings per ECEF point to station
    positions and names

    Arguments:
        ecef_point_array: [array] array of ECEF points.
        station_info_dict: [dict] station info dict.

    Keyword arguments:
        radius: [float] radius in metres around station to permit association.

    Returns:
        mapping_list: [list] [[ecef_point:(x, y, z), [station_ecef:(x, y, z)], [station_names]]
    """

    station_position_ecef_list, station_name_list = [], []

    for station in station_info_dict:
        station_ecef = station_info_dict[station]['ecef']
        station_position_ecef_list.append(station_ecef)
        station_name_list.append(station)

    station_name_array = np.array((station_name_list))
    station_position_ecef_array = np.array((station_position_ecef_list))
    tree = cKDTree(station_position_ecef_array)
    cameras_in_range = tree.query_ball_point(ecef_point_array, r=radius)

    mapping_list = []
    for i, indices in enumerate(cameras_in_range):
        mapping_list.append((ecef_point_array[i], station_position_ecef_array[indices], station_name_array[indices]))

    return mapping_list

def checkVisible(station_info_dict, vecs_normalised_array, station_name_list):
    """
    Are the vectors in an array of vectors visible from stations in a list

    Arguments:
        station_info_dict: [dict] station info dict.
        vecs_normalised_array: [array] array of normalized vectors with the pointing information from station to pont
        station_name_list: [list] list of stations names.

    Return:
        visible_mask: [array of bool] for each item in vecs_normalised_array and station_name_list True if visible
        x_list: [list] array of x coordinates, or np.nan if not visible
        y_list: [list] array of y coordinates, or np.nan if not visible
    """

    # Initialise lists to hold points
    x_list, y_list = [], []

    # Initialise an array for a mask for whether stations can see a point, and a counter
    mask_visibla, i = np.zeros(len(vecs_normalised_array), dtype=bool), 0
    for vec_norm, station in zip(vecs_normalised_array, station_name_list):

        # Get the station info
        station_info  = station_info_dict[station]
        pp, mask_struct = station_info['pp'], station_info['mask']
        lat_degs, lon_degs = np.degrees(station_info['geo']['lat_rads']), np.degrees(station_info['geo']['lon_rads'])
        station_ecef = station_info['ecef']

        # From the station ECEF, and the normalised vector to the point compute az and alt
        check_point_az_deg, check_point_alt_deg = ECEF2AltAz(station_ecef, station_ecef + vec_norm)

        # And ra dec - working at platpar reference time
        check_point_ra_deg, check_point_dec_deg = altAz2RADec(check_point_az_deg, check_point_alt_deg, pp.JD, lat_degs, lon_degs)

        # Now get the pp ra dec at platepar reference time
        fov_ra, fov_dec = altAz2RADec(pp.az_centre, pp.alt_centre, pp.JD, lat_degs, lon_degs)

        # Compute angular separation
        angular_separation = angularSeparationDeg(fov_ra, fov_dec, check_point_ra_deg, check_point_dec_deg)

        # Initialse the image coordinates
        x_float, y_float = np.nan, np.nan

        # Is it within FoV
        if angular_separation < np.hypot(pp.fov_h, pp.fov_v ) / 2:

            # Get a notional point which is in the correct direction 1000m meters away
            point = station_ecef + vec_norm * 1000

            # Find the lat and lon and elevation
            point_lat_rads, point_lon_rads, point_alt_m  = ecef2LatLonAlt(point[0], point[1], point[2])
            point_lat_degs, point_lon_degs = np.degrees(point_lat_rads), np.degrees(point_lon_rads)

            # Get the image coordinates for this point
            x_float_arr, y_float_arr = geoHt2XY(pp, point_lat_degs, point_lon_degs, point_alt_m)
            x_float, y_float = x_float_arr[0], y_float_arr[0]

            # Take ints - pixels are always integer values
            x, y = int(x_float), int(y_float)

            # Check against the image mask
            if mask_struct is None:
                visible = True
            else:
                y_res, x_res = (np.array((mask_struct.img)).shape)
                if 0 < x < x_res and 0 < y < y_res:
                    if mask_struct.img[y, x] == 255:
                        visible = True
                    else:
                        visible = False
                else:
                    visible = False
        else:
            visible = False

        # Update the mask
        mask_visibla[i] = visible

        # Increment the counter
        i += 1

        # Always append to the coordinates list, so that the mask will work, store at best precision
        x_list.append(x_float)
        y_list.append(y_float)

    return mask_visibla, np.array(x_list), np.array(y_list)


def ray_intersection_point(c1, d1, c2, d2):
    """
    Find closest point between two rays (least-squares intersection).

    Parameters:
        c1, c2: (3,) camera positions
        d1, d2: (3,) unit direction vectors

    Returns:
        p: (3,) estimated intersection point
        d1_len, d2_len: distances from each camera to p
    """
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    w0 = c1 - c2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b
    if denom == 0:
        # Parallel rays — return midpoint between origins
        p = (c1 + c2) / 2
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        p1 = c1 + s * d1
        p2 = c2 + t * d2
        p = (p1 + p2) / 2  # midpoint of closest approach

    return p, np.linalg.norm(p - c1), np.linalg.norm(p - c2)

def asymmetric_cone_volume(r, h):
    """Volume of a narrow cone: V = (π/3) * r² * h"""
    return (np.pi / 3) * r**2 * h


def pairwise_cone_volumes_actual(camera_positions, pixel_directions, angular_fovs, resolutions):
    """
    Compute pairwise cone volumes using actual ray intersection geometry.

    Parameters:
        camera_positions: (N, 3)
        pixel_directions: (N, 3)
        angular_fovs: (N,) in radians
        resolutions: (N,) pixel counts

    Returns:
        volume_matrix: (N, N) symmetric matrix of combined cone volumes
    """
    N = len(camera_positions)
    dirs = pixel_directions / np.linalg.norm(pixel_directions, axis=1, keepdims=True)
    cone_angles = angular_fovs / resolutions

    volume_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            p, d_i, d_j = ray_intersection_point(camera_positions[i], dirs[i],
                                                 camera_positions[j], dirs[j])

            r_i = d_i * np.tan(cone_angles[i])
            r_j = d_j * np.tan(cone_angles[j])

            v_i = asymmetric_cone_volume(r_i, d_i)
            v_j = asymmetric_cone_volume(r_j, d_j)

            volume = min(v_i, v_j)  # conservative estimate
            volume_matrix[i, j] = volume
            volume_matrix[j, i] = volume

    return volume_matrix


import numpy as np

def intersectionOfRays(o1, d1, o2, d2):

    """
    Compute the closest point between two rays defined by origins o1, o2 and unit directions d1, d2.
    Returns the midpoint of the shortest segment connecting the two rays.
    """
    # Form unit vectors
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    # Cross product to get normal vector
    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-8:
        # Rays are nearly parallel; return midpoint between origins
        return (o1 + o2) / 2

    # Matrix system to solve for scalars along d1 and d2
    A = np.stack([d1, -d2, n], axis=1)
    b = o2 - o1
    try:
        t = np.linalg.lstsq(A, b, rcond=None)[0]
        p1 = o1 + t[0] * d1
        p2 = o2 + t[1] * d2
        return (p1 + p2) / 2
    except np.linalg.LinAlgError:
        return (o1 + o2) / 2

def pairwiseTriangulationError(origins_ecef, direction_unit_vectors, expected_result):
    """
    Given arrays of origins and unit directions, compute closest intersection points for all unique pairs.
    Returns a list of (i, j, point) tuples.
    """
    n = len(origins_ecef)
    calculated_position_list = []
    error_arr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                calculated_position = np.array(intersectionOfRays(origins_ecef[i], direction_unit_vectors[i], origins_ecef[j], direction_unit_vectors[j]))
                calculated_position_list.append([i, j, calculated_position])
                error_arr[i,j] = RMS.Math.dimHypot(calculated_position, expected_result)
    return calculated_position_list, error_arr


def getCalculatedPositionErrorFromImageCoordinates(station_info_dict, station_name_list, x_list, y_list, vecs_normalised_array, observed_point_array, iterations=100):

    error_matrix_list = []
    for i in range(iterations):

        origin_list, vector_list = [], []
        for station_name, x, y, v in zip(station_name_list, x_list, y_list, vecs_normalised_array):
            # Populate the origin list entry
            station_info = station_info_dict[station_name]
            origin_list.append(np.array(station_info['ecef']))


            pp = station_info["pp"]

            # Add some noise

            # Error in station location
            random_ecef = np.array((np.random.random() * 100 - 50, np.random.random() * 100 - 50, np.random.random() * 100 - 50))

            # Angle error - atmospheric effects
            random_angle_x = np.random.random() * 0.5 - 0.25
            random_angle_y = np.random.random() * 0.5 - 0.25
            random_angle_x_in_pixels = pp.X_res * random_angle_x / pp.fov_h
            random_angle_y_in_pixels = pp.Y_res * random_angle_y / pp.fov_v

            # Pixel offset, and the angle error
            random_pixel_x = np.random.random() * 3 - 1.5 + random_angle_x_in_pixels
            random_pixel_y = np.random.random() * 3 - 1.5 + random_angle_y_in_pixels


            # Compute the normal vectors from stations to the point

            # Get the distance to the point
            distance_to_point_m = RMS.Math.dimHypot(observed_point_array, np.array(station_info['ecef']))

            # Express is ra dec and add noise
            jd_array, ra_deg_array,  dec_deg_array, _ = xyToRaDecPP(np.array([pp.JD]), np.array([x + random_pixel_x]), np.array([y + random_pixel_y]), np.array([1]), pp, jd_time=True, extinction_correction=False, measurement=False)

            # Convert to local az alt
            az, alt = raDec2AltAz(ra_deg_array, dec_deg_array, pp.JD, pp.lat, pp.lon)

            # Get the point in ecef
            ecef_point = AER2ECEF(az, alt, distance_to_point_m, pp.lat, pp.lon, pp.elev)
            ecef_point = (ecef_point[0][0], ecef_point[1][0], ecef_point[2][0])

            # Compute normal vectors from the station to the point, with noise
            ecef_sta_pt = vectNorm(ecef_point - np.array(station_info['ecef'] + random_ecef))

            # Add to the list of vectors
            vector_list.append(ecef_sta_pt)


        # For ech pair of stations compute performance data for this point
        calculated_position_list, error_matrix = pairwiseTriangulationError(origin_list, vector_list, observed_point_array)

        # Add the error matrices to a list
        error_matrix_list.append(error_matrix)

    # And stack ths list of matrices - one matrix per iteratiojn
    error_matrix_over_iterations = np.stack(error_matrix_list)

    # Compute mean error over all iterations per combination
    error_matrix = np.mean(error_matrix_over_iterations, axis=0)

    # Exclude errors of station against same station
    non_reflexive_errors = error_matrix[~np.eye(error_matrix.shape[0], dtype=bool)]

    # Compute stats
    mean_error, std_dev = np.mean(non_reflexive_errors), np.std(non_reflexive_errors)

    # Init a variable to hold the best pair of stations
    best_pair = []


    if len(non_reflexive_errors) > 2:

        # Find the lowest error between different stations
        min_error = np.min(non_reflexive_errors)

        # Identfy where it is in the matrix, and return the pair of station names
        station_pair_indices = np.where(error_matrix == min_error)

        best_pair.append(station_name_list[station_pair_indices[0][0]])
        best_pair.append(station_name_list[station_pair_indices[1][0]])
    else:
        min_error = None

    return calculated_position_list, error_matrix, min_error, best_pair, mean_error, std_dev

def computeOrigin(points_list):

    """
    Compute the origin - by taking the mid points of East and North and the minimum of Up
    Arguments:
        points_list: [list] List os points in ECEF

    Returns:
        lat_deg:[float] Latitude in degrees
        lon_deg:[float] Longitude in degrees
        ele_deg:[float] Elevation in meters
        x: [float] x coordinate
        y: [float] y coordinate
        z: [float] z coordinate
    """


    # Transpose list of ECEF into three lists
    x_list, y_list, z_list = np.array(points_list).T

    # Initialise working lists of east, north, up
    e_list, n_list, u_list = [], [], []

    # Take the mid point of ECEF as our first origin
    lat_o_rad, lon_o_rad, ele_m_o =  ecef2LatLonAlt(np.mean(x_list), np.mean(y_list), np.mean(z_list))
    lat_o_deg, lon_o_deg = np.degrees(lat_o_rad), np.degrees(lon_o_rad)

    # Convert all the ECEF points into ENU using this estimate of an origin
    for x, y, z, in zip(x_list, y_list, z_list):
        lat_rads, lon_rads, ele_m = ecef2LatLonAlt(x, y, z)
        e, n, u = latLonAlt2ENUDeg(np.degrees(lat_rads), np.degrees(lon_rads), ele_m, lat_o_deg, lon_o_deg, ele_m_o)
        e_list.append(e)
        n_list.append(n)
        u_list.append(n)

    # Take the mid point of east and north and the minimum of up
    e_mid_m, n_mid_m, u_min = np.mean((np.min(e_list), np.max(e_list))), np.mean((np.min(n_list), np.max(n_list))), np.min(u_list)

    # Convert that into lat and lon using out temporary origin
    lat_deg, lon_deg, ele_m = enu2LatLonAltDeg(e_mid_m, n_mid_m, u_min, lat_o_deg, lon_o_deg, ele_m_o)

    # And ECEF
    x, y, z = enu2Ecef(lat_deg, lon_deg, ele_m, lat_o_deg, lon_o_deg, ele_m_o)

    return lat_deg, lon_deg, ele_m, x, y, z


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

def enu2LatLonAltDeg(e, n, u, lat_origin_deg, lon_origin_deg, ele_origin_deg):

    x, y, z = enu2Ecef(e, n, u, lat_origin_deg, lon_origin_deg, ele_origin_deg)
    lat_deg, lon_deg, ele_m = ecef2LatLonAltDeg(x, y, z)

    return lat_deg, lon_deg, ele_m


def computeAnglesPerPoint(station_info_dict, mapping_list, plot_charts=False):

    mapping_list_with_angles=[]

    output_mapping_list = []
    for observed_point_array, station_ecef_array, station_name_list in tqdm.tqdm(mapping_list):
        # Vectors from stations to reference_point

        point_lat, point_lon, point_ele = ecef2LatLonAlt(observed_point_array[0], observed_point_array[1], observed_point_array[2])


        vectors_array = observed_point_array - station_ecef_array  # shape (N, 3)
        normalisation_array = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        vecs_normalized_array = vectors_array / normalisation_array  # shape (N, 3)
        visible_mask, x_list, y_list = checkVisible(station_info_dict, vecs_normalized_array, station_name_list)

        # Create masked copies
        station_ecef_array_visible = station_ecef_array[visible_mask]
        station_name_list_visible = station_name_list[visible_mask]
        vecs_normalized_array_visible = vecs_normalized_array[visible_mask]
        x_list_visible, y_list_visible = x_list[visible_mask], y_list[visible_mask]


        if not len(station_name_list):
            continue

        # Dot product matrix
        dot_matrix = np.dot(vecs_normalized_array_visible, vecs_normalized_array_visible.T)  # shape (N, N)
        dot_matrix = np.clip(dot_matrix, -1.0, 1.0)

        # Angle matrix degrees
        angle_matrix = np.degrees(np.arccos(dot_matrix))

        # Score matrix in sin(angle)
        sin_score_matrix = np.sin(np.arccos(dot_matrix))  # shape (N, N)

        # Calculate errors in positions
        calculated_position_list, error_matrix, min_error, best_pair, mean_error, sd = \
            getCalculatedPositionErrorFromImageCoordinates(station_info_dict, station_name_list_visible,
                                                           x_list_visible, y_list_visible,
                                                           vecs_normalized_array_visible,
                                                           observed_point_array, iterations = 50)

        # If more than 6 stations saw the point, plot and save a chart
        if len(station_name_list_visible) > 6:

            if plot_charts:
                # Create scatter plot

                ecef_locations = []
                for station_name in station_name_list_visible:
                    station = station_info_dict[station_name]
                    ecef_locations.append(station['ecef'])
                    s_x, s_y, s_z = station['ecef']
                    lat_rads , lon_rads ,ele_m = ecef2LatLonAlt(s_x, s_y, s_z)
                    print("Station {} at lat, lon ({}, {})".format(station_name, np.rad2deg(lat_rads), np.rad2deg(lon_rads)))

                ecef_locations.append([observed_point_array[0], observed_point_array[1], observed_point_array[2]])

                lat_origin_deg, lon_origin_deg, ele_origin_m, x, y, z = computeOrigin(ecef_locations)
                point_lat_rads, point_lon_rads, point_ht = ecef2LatLonAlt(observed_point_array[0], observed_point_array[1], observed_point_array[2])
                e_list, n_list, u_list, labs_list = [], [], [], []

                lat_deg, lon_deg, ele_m = np.degrees(point_lat_rads), np.degrees(point_lon_rads), point_ht
                e_point, n_point, u_point = latLonAlt2ENUDeg(lat_deg, lon_deg, ele_m, lat_origin_deg, lon_origin_deg, ele_origin_m)

                e_list.append(e_point / 1000)
                n_list.append(n_point / 1000)
                u_list.append(u_point / 1000)

                labs_list.append("Point alt {:.1f}km".format(point_ht/1000))

                coordinates_of_good_station = []
                for station in station_name_list:
                    station_info = station_info_dict[station]
                    lat_deg = np.degrees(station_info['geo']['lat_rads'])
                    lon_deg = np.degrees(station_info['geo']['lon_rads'])
                    ele_m = station_info['geo']['ele_m']

                    labs_list.append("{} {}".format(len(labs_list) - 1, station))
                    e, n, u =   latLonAlt2ENUDeg(lat_deg, lon_deg, ele_m, lat_origin_deg, lon_origin_deg, ele_origin_m)
                    if station in best_pair:
                        coordinates_of_good_station.append([e, n, u])
                    e_list.append(e / 1000)
                    n_list.append(n / 1000)
                    u_list.append(u / 1000)

                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(e_list, n_list , u_list, c='blue', marker='o')
                for e, n, u, label in zip(n_list, e_list, u_list, labs_list):
                    ax.text(n+5, e+5, u +5 , label, fontsize=9)  # Offset to avoid overlap

                ax.set_aspect('equal')

                for coordinates in coordinates_of_good_station:
                    x_points, y_points, u_points = [], [], []
                    x_points.append(coordinates[0] / 1000)
                    y_points.append(coordinates[1] / 1000)
                    u_points.append(coordinates[2] / 1000)
                    x_points.append(e_point / 1000)
                    y_points.append(n_point / 1000)
                    u_points.append(u_point / 1000)

                    ax.plot(x_points, y_points, u_points, color='red', label='Connecting Line')

                # Label axes
                ax.set_xlabel("North km")
                ax.set_ylabel("East km")
                ax.set_zlabel("Up km")
                i_1, i_2 = np.where(station_name_list_visible == best_pair[0]), np.where(station_name_list_visible == best_pair[1])
                convergence_angle = angle_matrix[i_1, i_2][0][0]

                plot_title = "Stations and reference point minimum error {:.0f}m with stations {} {} angle {:.1f} degrees".format(min_error, best_pair[0], best_pair[1], convergence_angle)

                plt.title(plot_title)

                plot_file_name = "{}_{}_angle_{}_deg.png".format(best_pair[0], best_pair[1], convergence_angle)
                plt.grid(True)
                plot_dir = os.path.join(WORKING_DIRECTORY, CHARTS)
                mkdirP(plot_dir)
                plot_full_path =  os.path.join(plot_dir, plot_file_name)
                plt.savefig(plot_full_path)
                plt.close()
                pass

        # Add information to the mapping
        output_mapping_list.append([observed_point_array, station_ecef_array_visible, station_name_list_visible,
                                    angle_matrix, sin_score_matrix,
                                    calculated_position_list,  error_matrix, mean_error, sd])

    return output_mapping_list

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Compute coverage quality of the GMN \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-p', '--plot', dest='plot_charts', default=False, action="store_true",
                            help="Plot chart for debugging purposes.")


    cml_args = arg_parser.parse_args()

    cwd = os.getcwd()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    mkdirP(WORKING_DIRECTORY)

    station_info_dict_path = os.path.join(WORKING_DIRECTORY, "station_info_dict.pkl")
    ecef_array_path = os.path.join(WORKING_DIRECTORY, "ecef_point_array_around_stations.npy")
    ecef_point_to_camera_mapping_path = os.path.join(WORKING_DIRECTORY, "ecef_point_to_camera_mapping.pkl")
    station_point_angle_score_mapping_list_path = os.path.join(WORKING_DIRECTORY, "station_point_angle_score_mapping_list.pkl")

    if False:
        station_list = getStationList(country_code = "au")
        makeConfigPlateParMaskLib(config, station_list)

    if not os.path.exists(station_info_dict_path):
        station_info_dict = makeStationsInfoDict(config, country_code="au")
        with open(station_info_dict_path, 'wb') as f:
            pickle.dump(station_info_dict, f)

    station_info_dict = pickle.load(open(station_info_dict_path, 'rb'))

    if not os.path.exists(ecef_array_path):
        ecef_point_array = makeECEFPointList(station_info_dict, min_ele_m=20000, max_ele_m=100000, resolution_m=10000)
        np.save(ecef_array_path, ecef_point_array)

    if not os.path.exists(ecef_point_to_camera_mapping_path):
        ecef_point_array = np.load(ecef_array_path)
        ecef_point_to_camera_mapping = addStationsToECEFArray(ecef_point_array, station_info_dict, radius=500000)


        with open(ecef_point_to_camera_mapping_path, 'wb') as f:
            pickle.dump(ecef_point_to_camera_mapping, f)

    ecef_point_to_camera_mapping = pickle.load(open(ecef_point_to_camera_mapping_path, 'rb'))

    if True:
        station_point_angle_score_mapping_list = computeAnglesPerPoint(station_info_dict, ecef_point_to_camera_mapping, cml_args.plot_charts)

        with open(station_point_angle_score_mapping_list_path, 'wb') as f:
            pickle.dump(station_point_angle_score_mapping_list, f)

    station_point_angle_score_mapping_list = pickle.load(open(station_point_angle_score_mapping_list_path, 'rb'))

    pass