# RPi Meteor Station
# Copyright (C) 2025 David Rollinson Kristen Felker
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

from __future__ import print_function

import os
import sys
import pickle
import argparse
import subprocess
from datetime import tzinfo
import moviepy

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


import RMS.ConfigReader as cr
import datetime
import pathlib
import json

import imageio
import tqdm

from RMS.Astrometry.Conversions import altAz2RADec, raDec2AltAz, jd2Date, date2JD, J2000_JD
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar, stationData
from RMS.Math import angularSeparationDeg
from RMS.Misc import mkdirP
from RMS.Routines.MaskImage import loadMask
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession


def startOfPreviousDay():

    this_time_yesterday = datetime.datetime.now(datetime.timezone.utc)  - datetime.timedelta(days=1)
    yesterday_start_time = datetime.datetime.combine(this_time_yesterday, datetime.datetime.min.time())

    return yesterday_start_time.replace(tzinfo=datetime.timezone.utc)

def startOfThisDay(time_object):

    return datetime.datetime.combine(time_object, datetime.datetime.min.time()).replace(tzinfo=datetime.timezone.utc)



def azimuthalProjectionToAltAz(x, y, dimension_x_min, dimension_x_max, dimension_y_min, dimension_y_max, minimum_elevation_deg):

    """
    Convert Cartesian coordinates (x, y) on a polar plot to azimuth and altitude angles.

    Arguments:
        x: [int] x coordinate
        y: [int] y coordinate
        dimension_x_min: [int] minimum x value
        dimension_x_max: [int] maximum x value
        dimension_y_min: [int] minimum y value
        dimension_y_max: [int] maximum y value
        minimum_elevation_deg: [float] minimum elevation degrees

    Return:
        alt_deg   : Altitude angle in degrees (from horizon up)
        az_deg    : Azimuth angle in degrees (0° = up, 90° = right)

    """
    # Compute origin
    x0, y0 = (dimension_x_min + dimension_x_max) / 2, (dimension_y_min + dimension_y_max) / 2

    # Normalise coordinates to centre
    dx, dy = x - x0, (dimension_y_max - y) - y0

    # Compute azimuth (angle around center)
    az_deg = np.degrees(np.arctan2(dy, dx)) % 360

    # Compute radial distance from center - limit is edge of image, not corner to corner
    r, rmax = np.sqrt(dx ** 2 + dy ** 2), min(dimension_x_max - x0, dimension_y_max - y0)

    # Map radius to altitude (center = 90°, edge = min_elev_deg)
    alt_deg = 90 - (90 - minimum_elevation_deg) * (r / rmax)

    return alt_deg, (az_deg - 90) % 360

def altAzToAzimuthalProjection(az_deg, alt_deg, dimension_x_min, dimension_x_max, dimension_y_min, dimension_y_max, minimum_elevaton_deg):
    """
    Convert azimuth and altitude angles to Cartesian coordinates on a polar plot.

    Arguments:
        alt_deg   : Altitude angle in degrees (from horizon up)
        az_deg    : Azimuth angle in degrees (0° = right, 90° = up)
        dimension_x_min: [int] minimum x value
        dimension_x_max: [int] maximum x value
        dimension_y_min: [int] minimum y value
        dimension_y_max: [int] maximum y value
        minimum_elevation_deg: [float] minimum elevation degrees

    Return:
        x: [int] x coordinate
        y: [int] y coordinate
    """

    az_deg += 90

    # Center of the plot
    x0 = (dimension_x_min + dimension_x_max) / 2
    y0 = (dimension_y_min + dimension_y_max) / 2

    # Max radius from center to edge
    rmax = min(dimension_x_max - x0, dimension_y_max - y0)

    # Convert altitude to radial distance
    r = rmax * (90 - alt_deg) / (90 - minimum_elevaton_deg)

    # Convert azimuth to angle in radians
    az_rad = np.radians(az_deg)

    # Compute Cartesian coordinates
    x = x0 + r * np.cos(az_rad)
    y = y0 - r * np.sin(az_rad)



    return x, y

def getStationsInfoDict(path_list=None, print_activity=False):

    """
    Either load configs from a given path_list, or look for configs in a multi_cam linux
    or single camera per username style architecture.

    Keyword arguments:
        path_list: [list] List of paths to config files.
        print_activity: [bool] Optional, default false, print activity for debugging.

    Return:
        stations_info_dict: [dict] dictionary with station name as key and station config as value.
    """


    # Initialise an empty dict
    stations_info_dict = {}


    # If we have been given paths
    if len(path_list):
        if print_activity:
            print("Command line gave path lists  :")
        for p in path_list:
            if print_activity:
                print("                                 {}".format(p))
            station_info = {}
            if p.endswith('.config'):
                station_full_path = os.path.dirname(p)
                c = cr.parse(os.path.expanduser(p))
            else:
                station_full_path = p
                c = cr.parse(os.path.expanduser(os.path.join(p,".config")))

            platepar_full_path = os.path.join(station_full_path, c.platepar_name)
            mask_full_path = os.path.join(station_full_path, c.mask_file)

            if os.path.exists(platepar_full_path):
                pp = Platepar()
                pp.read(platepar_full_path)
            else:
                pp = None
                continue

            if os.path.exists(mask_full_path):
                m = loadMask(mask_full_path).img
            else:
                m = None

            data_dir_sections = pathlib.Path(c.data_dir).parts
            config_path_section = pathlib.Path(p).parts
            if "home" in data_dir_sections:
                user_name_index = data_dir_sections.index("home") + 1

                i = 0
                for section in data_dir_sections:

                    if i == user_name_index:
                        c.data_dir = os.path.join(c.data_dir, config_path_section[user_name_index])
                    else:
                        c.data_dir = os.path.join(c.data_dir, section)
                    i += 1

            station_info['mask'] = m
            station_info['pp'] = pp
            station_info['config'] = c
            stations_info_dict[c.stationID.lower()] = station_info

        return stations_info_dict


    else:
        # Test if this is a multicam linux station, i.e. ~/source/Stations/XX0001
        stations_base_directory = os.path.expanduser("~/source/Stations/")
        if os.path.exists(stations_base_directory):
            if os.path.isdir(stations_base_directory):
                candidate_stations_list = sorted(os.listdir(stations_base_directory))
                if print_activity:
                    print("Searching for multicam linux configs found :")
                for station in candidate_stations_list:
                    if print_activity:
                        print("                                             {}".format(station))
                    station_info = {}
                    if len(station) == 6:
                        station_full_path = os.path.join(stations_base_directory, station)
                        config_path = os.path.join(station_full_path, ".config")
                        if os.path.exists(config_path):
                            c = cr.parse(config_path)
                            if c.stationID.lower() != user_name.lower():
                                continue
                            platepar_full_path = os.path.join(station_full_path, c.platepar_name)
                            mask_full_path = os.path.join(station_full_path, c.mask_file)
                            if os.path.exists(platepar_full_path):
                                pp = Platepar()
                                pp.read(platepar_full_path)
                            else:
                                pp = None
                                continue

                            if os.path.exists(mask_full_path):
                                m = loadMask(mask_full_path).img
                            else:
                                m = None

                            station_info['mask'] = m
                            station_info['pp'] = pp
                            station_info['config'] = c
                            stations_info_dict[c.stationID.lower()] = station_info

        # If we found configs, then return what we have found
        if len(stations_info_dict):
            return stations_info_dict

    # Test if this is a one camera per user system
    stations_info_dict = {}
    if print_activity:
        print("Looking for a one camera per username style architecture")
    if not len(path_list):
        stations_base_directory = ("/home")
        if os.path.exists(stations_base_directory):
            if os.path.isdir(stations_base_directory):
                candidate_stations_list = sorted(os.listdir(stations_base_directory))
                for user_name in candidate_stations_list:
                    station_info = {}
                    if len(user_name) == 6:
                        if print_activity:
                            print("Testing {} to see if it a camera account".format(user_name))
                        if user_name[0:2].isalpha():
                            station_full_path = os.path.join(stations_base_directory, user_name)
                            station_config_dir = os.path.join(station_full_path, "source","RMS")
                            config_path = os.path.join(station_config_dir, ".config")
                            if print_activity:
                                print("Looking for {}".format(config_path))

                            if os.path.exists(config_path):
                                if print_activity:
                                    print("Found {}".format(config_path))
                                c = cr.parse(config_path)
                                if c.stationID.lower() == user_name:
                                    platepar_full_path = os.path.join(station_config_dir, c.platepar_name)
                                    mask_full_path = os.path.join(station_config_dir, c.mask_file)
                                    data_dir_sections = pathlib.Path(c.data_dir).parts
                                    if "home" in data_dir_sections:
                                        user_name_index = data_dir_sections.index("home") + 1

                                        i = 0
                                        for section in data_dir_sections:

                                            if i == user_name_index:
                                                c.data_dir = os.path.join(c.data_dir, user_name)
                                            else:
                                                c.data_dir = os.path.join(c.data_dir, section)
                                            i += 1
                                        if os.path.exists(platepar_full_path):
                                            pp = Platepar()
                                            pp.read(platepar_full_path)
                                            if pp.station_code == c.stationID:
                                                station_info['pp'] = pp
                                                if print_activity:
                                                    print("Loaded platepar for {}".format(user_name))
                                            else:
                                                station_info['pp'] = None
                                                if print_activity:
                                                    print("No platepar for {}".format(user_name))
                                        else:
                                            if print_activity:
                                                print("No platepar for {}".format(user_name))
                                            station_info['pp'] = None
                                    if os.path.exists(mask_full_path):
                                        m = loadMask(mask_full_path).img
                                        if print_activity:
                                            print("Loaded mask for {}".format(user_name))
                                    station_info['mask'] = m
                                    station_info['config'] = c
                                    stations_info_dict[c.stationID.lower()] = station_info
                            else:
                                if print_activity:
                                    print("No config found at {}".format(config_path))

        if len(stations_info_dict):
            return stations_info_dict

    return stations_info_dict

def getCommaSeparatedListofStations(stations_info_dict):


    stations_as_text = ""
    for s in stations_info_dict.keys():
        stations_as_text = "{}, {}".format(stations_as_text, s.strip())

    if len(stations_as_text):
        stations_as_text = stations_as_text[2:]


    return stations_as_text

def getFitsFiles(transformation_layer_list, stations_info_dict, target_jd, print_activity=False):
    """
    Get the paths to fits files, in the same order as stations_list using info from stations_info_dict around target_image time.

    Arguments:
        transformation_layer_list: [[list]] list of [stations, time offsets]
        stations_info_dict: [dict] dictionary of station information.
        target_jd: [float] target time for image as a jd float

    Return:
        station_files_list:[[list]] list of [station, path to fits file]
    """


    target_image_time = jdToPyTime(target_jd)
    stations_files_list = []


    for s, time_offset_seconds in transformation_layer_list:
        if print_activity:
            print("Looking for fits files in station {} time offset {} from {}".format(s, time_offset_seconds, target_image_time))
        c = stations_info_dict[s]['config']
        captured_dir_path = os.path.join(c.data_dir, c.captured_dir)
        captured_dirs = sorted(os.listdir(captured_dir_path), reverse=True)
        if not len(captured_dirs):
            stations_files_list.append([s, None])
            continue

        for captured_dir in captured_dirs:
            dir_date, dir_time = captured_dir.split('_')[1], captured_dir.split('_')[2]
            year, month, day = int(dir_date[0:4]), int(dir_date[4:6]), int(dir_date[6:8])
            hour, minute, second = int(dir_time[0:2]), int(dir_time[2:4]), int(dir_time[4:6])
            dir_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second).replace(tzinfo=datetime.timezone.utc)
            dir_time += datetime.timedelta(seconds=time_offset_seconds)
            if dir_time < target_image_time:
                break

        if print_activity:
            print("Using {}".format(captured_dir))

        dir_files = sorted(os.listdir(os.path.join(captured_dir_path, captured_dir)))

        min_time_delta = np.inf

        closest_fits_file_full_path = None
        for file in dir_files:
            if file.startswith('FF_{}'.format(c.stationID)) and file.endswith('.fits'):

                file_date, file_time = file.split('_')[2], file.split('_')[3]
                year, month, day = int(file_date[0:4]), int(file_date[4:6]), int(file_date[6:8])
                hour, minute, second = int(file_time[0:2]), int(file_time[2:4]), int(file_time[4:6])
                file_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second).replace(tzinfo=datetime.timezone.utc)
                time_delta = abs(target_image_time + datetime.timedelta(seconds=time_offset_seconds) - file_time).total_seconds()
                if time_delta < min_time_delta:
                    closest_fits_file_full_path = os.path.join(c.data_dir, c.captured_dir, captured_dir, file)
                    min_time_delta = time_delta

        stations_files_list.append([s, closest_fits_file_full_path])
        if print_activity:
            if closest_fits_file_full_path is None:
                print("Could not find a file for {} for stations {}".format(target_image_time + datetime.timedelta(seconds=time_offset_seconds), s))
            else:
                print("Added {} with a time delta of {} seconds".format(os.path.basename(closest_fits_file_full_path), min_time_delta))

    return stations_files_list


def getFramesFilesPaths(stations_info_dict, earliest_time=None, latest_time=None, print_activity=False):


    frames_file_list = []
    for station in sorted(stations_info_dict.keys()):
        station_file_list = []
        config = stations_info_dict[station]['config']
        frames_dir_full_path = os.path.join(config.data_dir, config.frame_dir)

        if not os.path.exists(frames_dir_full_path):
            frames_file_list.append(station_file_list)
            continue

        for obj in sorted(os.listdir(frames_dir_full_path)):
            obj_full_path = os.path.join(frames_dir_full_path, obj)
            if os.path.isdir(obj_full_path):
                years_dir_list = sorted(os.listdir(obj_full_path))
                for day_dir in years_dir_list:
                    day_dir_full_path = os.path.join(obj_full_path, day_dir)
                    if not os.path.isdir(day_dir_full_path):
                        continue
                    for hour_dir in sorted(os.listdir(day_dir_full_path)):
                        hour_dir_full_path = os.path.join(day_dir_full_path, hour_dir)

                        for image_file in sorted(os.listdir(hour_dir_full_path)):
                            image_file_full_path = os.path.join(hour_dir_full_path, image_file)
                            file_time_object = getFileTime(image_file)
                            if earliest_time is None or latest_time is None:
                                station_file_list.append([file_time_object, image_file_full_path])
                            elif earliest_time is not None and latest_time is not None:
                                if earliest_time < file_time_object < latest_time:
                                    station_file_list.append([file_time_object, image_file_full_path])
                            elif earliest_time is None and latest_time is not None:
                                if file_time_object < latest_time:
                                    station_file_list.append([file_time_object, image_file_full_path])
                            elif earliest_time is not None and latest_time is None:
                                if earliest_time < file_time_object:
                                    station_file_list.append([file_time_object, image_file_full_path])


        frames_file_list.append(station_file_list)
    return frames_file_list



def getClosestTimeIndex(time_path_list, target_time):


    diffs = [abs((t[0] - target_time).total_seconds()) for t in time_path_list]


    if len(diffs):
        min_diffs = min(diffs)
        return diffs.index(min_diffs), min_diffs
    else:
        return None, np.inf


def getFramesAsList(stations_files_list, stations_info_dict, print_activity=False, compensation=[50, 80, 80, 99.75]):
    """
    Given a list of lists of stations and paths to fits files, return a list of images from
    the fits compensated to an average intensity of zero.

    Arguments:
        stations_files_list: [[list]] list of [station, path to fits file].
        stations_info_dict: [dict] dictionary of station information keyed by stationID.

    Keyword Arguments:
        print_activity: [bool] Optional, default False.

    Return:
        fits_list: [list] list of compensated fits images as arrays.
    """

    fits_dict, frames_list = {}, []
    for s, f in stations_files_list:
        if print_activity:
            print("Load frame {}".format(f))
        if f is None:
            pp = stations_info_dict[s]['pp']
            frames_list.append(np.array(np.zeros((pp.Y_res, pp.X_res))))
        else:
            try:
                frame = imageio.v3.imread(f, pilmode='L')
            except:
                print("Unable to load frame {}".format(f))
                frames_list.append(np.array(np.zeros((pp.Y_res, pp.X_res))))

            #plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
            #$plt.axis("off")  # optional: hides axes for cleaner display
            #plt.show()

            min_threshold, max_threshold = np.percentile(frame, compensation[0]), np.percentile(frame, compensation[1])
            if min_threshold == max_threshold:
                compensated_frame =  np.full_like(frame, 128)
            else:
                compensated_frame = (2 ** 16 * (frame - min_threshold) / (max_threshold - min_threshold)) - 2 ** 15

            frames_list.append(compensated_frame)

    return frames_list

def getFramesFiles(transformation_layer_list, stations_info_dict, target_jd, frames_files_paths_list=None, print_activity=False):
    """
    Get the paths to jpg files, in the same order as stations_list using info from stations_info_dict around target_image time.

    Arguments:
        transformation_layer_list: [[list]] list of [stations, time offsets]
        stations_info_dict: [dict] dictionary of station information.
        target_jd: [float] target time for image as a jd float

    Return:
        station_files_list:[[list]] list of [station, path to fits file]
    """


    target_image_time = jdToPyTime(target_jd)
    #target_image_time = datetime.datetime(year=2025, month=8, day=25, hour=3, minute=7, second=9).replace(tzinfo=datetime.timezone.utc)
    stations_files_list = []
    stations_list = stations_info_dict.keys()
    stations_file_list = []

    for s, time_offset_seconds in transformation_layer_list:
        for station, frames_per_station_list in zip(stations_list, frames_files_paths_list):
            if s != station:
                continue
            if print_activity:
                print("Looking for jpg files in station {} time offset {} from {}".format(s, time_offset_seconds,
                                                                                          target_image_time))
            closest_time_index, time_error = getClosestTimeIndex(frames_per_station_list, target_image_time)

            if print_activity:
                print("Index of closest image {} time error {}".format(closest_time_index, time_error))

            if closest_time_index is not None and time_error < 20:
                time_of_closest_file, path_to_closest_file = frames_per_station_list[closest_time_index]
                stations_file_list.append([s, path_to_closest_file])
                if print_activity:
                    print("Added file {}".format(path_to_closest_file))
            else:
                stations_file_list.append([s, None])
    if print_activity:
        print("Stations files list is {}".format(stations_file_list))
    return stations_file_list




def getFitsAsList(stations_files_list, stations_info_dict, print_activity=False, compensation=[50, 80, 80, 99.75]):
    """
    Given a list of lists of stations and paths to fits files, return a list of images from
    the fits compensated to an average intensity of zero.

    Arguments:
        stations_files_list: [[list]] list of [station, path to fits file].
        stations_info_dict: [dict] dictionary of station information keyed by stationID.

    Keyword Arguments:
        print_activity: [bool] Optional, default False.

    Return:
        fits_list: [list] list of compensated fits images as arrays.
    """

    fits_dict, fits_list = {}, []
    for s, f in stations_files_list:
        if print_activity:
            print("Load fits {}".format(f))
        if f is None:
            pp = stations_info_dict[s]['pp']
            fits_list.append(np.array(np.zeros((pp.Y_res, pp.X_res))))
        else:
            ff = readFF(os.path.dirname(f), os.path.basename(f))

            max_pixel = ff.maxpixel.astype(np.float32)
            compensated_image = max_pixel
            min_threshold, max_threshold = np.percentile(compensated_image, compensation[0]), np.percentile(compensated_image, compensation[1])
            if min_threshold == max_threshold:
                compensated_image =  np.full_like(compensated_image, 128)
            else:
                compensated_image = (2 ** 16 * (compensated_image - min_threshold) / (max_threshold - min_threshold)) - 2 ** 15

            fits_list.append(compensated_image)

    return fits_list

def makeTransformation(stations_info_dict, size_x, size_y, minimum_elevation_deg=20, stack_depth=3, time_steps_seconds=256 / 25, print_activity=False):

    """
    Make the transformation from the image coordinates of multiple cameras to the image coordinates of a destination
    polar project image of the sky.

    The calculations for transforming images through time are not robust, and should only be used for short
    offsets, not more than 10 hours. Over these durations the error will be acceptable for producing images.

    Arguments:
        stations_info_dict: [dict] Dictionary of station information.
        size_x: [int] X size of the image in pixels.
        size_y: [int] Y size of the image in pixels.

    Keyword arguments:
        minimum_elevation_deg:[float] Optional, default 20, minimum elevation angle in degrees.
        stack_depth:[int] Optional, default 3, number of fits files to get for stacking.
        time_steps_seconds:[int] Optional, default 256 / 25, number of seconds between images to use for stacking.

    Return:
        stations_info_dict: [dict] Dictionary of station information.
        transformation_layer_list: [[station, time_offset_seconds]] List of stations in same order as source_coordinates_array.
        source_coordinates_array: [array] Array transformation_layer list indices, x_source, y_source.
        intensity_scaling_array: [array] Array of number of source pixels mapped to a destination pixel.
        target_geo: [target_lat, target_lon, elevation] Target latitude, longitude, degrees, elevation meters

    """


    # return stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, [target_lat, target_lon, target_ele]



    # Intialise
    origin_x, origin_y = size_x / 2, size_y / 2
    elevation_range = 2 * (90 - minimum_elevation_deg)
    pixel_to_radius_scale_factor_x = elevation_range / size_x
    pixel_to_radius_scale_factor_y = elevation_range / size_y
    az_vals_deg, el_vals_deg = np.zeros((size_x, size_y)), np.zeros((size_x, size_y))

    # Make transform from target image coordinates to az and el.
    stations_list = sorted(list(stations_info_dict.keys()))

    # Define target image parameters
    pp_target = stations_info_dict[stations_list[0]]['pp']
    target_lat, target_lon, target_ele = pp_target.lat, pp_target.lon, pp_target.elev

    # Define source parameters
    source_coordinates_list, dest_coordinates_list, scaling_list, combined_coordinates_list = [], [], [], []
    transformation_layer_list, transformation_layer = [], 0

    # Form the transformation, working across stations, and then stacked images

    station_stack_count_list = []
    for station in stations_info_dict:
        for stack_count in range(0, stack_depth):
            station_stack_count_list.append([station, stack_count])

    for station, stack_count in station_stack_count_list:
        # Get the source platepar
        pp_source = stations_info_dict[station]['pp']

        # Compute the time offset - +ve is always forward in time
        time_offset_seconds = 0 - time_steps_seconds * stack_count
        offset_date = jd2Date(pp_source.JD, dt_obj=True) + datetime.timedelta(seconds=time_offset_seconds)

        # Convert the source image time to Julian Date relative to platepar reference
        jd_source = date2JD(offset_date.year, offset_date.month, offset_date.day, offset_date.hour, offset_date.minute, offset_date.second, int(offset_date.microsecond / 1000))

        if print_activity:
            print("Making transformation for {:s} with a time offset of {:.1f} seconds - {}".format(station.lower(), 0 - time_offset_seconds, jd2Date(jd_source)))

        # Get the centre of the platepar at creation time in JD - not compensated for time offsets
        _, r_source, d_source, _ = xyToRaDecPP([pp_source.JD], [pp_source.X_res / 2], [pp_source.Y_res / 2], [1], pp_source, jd_time=True, extinction_correction=False, measurement=False)
        r_list, d_list, x_dest_list, y_dest_list = [], [], [], []

        for y_dest in range(1, size_y - 1):
            for x_dest in range(1, size_x - 1):
                # _x, _y, = x_dest - origin_x, y_dest - origin_y

                # Convert the target image (polar projection on cartesian axis) into azimuth and elevation
                # el_deg = 90 - np.hypot(_x * pixel_to_radius_scale_factor_x, _y * pixel_to_radius_scale_factor_y)
                # az_deg = np.degrees(np.arctan2(_x, _y))

                el_deg, az_deg = azimuthalProjectionToAltAz(x_dest, y_dest, 0, size_x, 0, size_y, minimum_elevation_deg)



                # print(x_dest, y_dest )

                # Store
                az_vals_deg[y_dest, x_dest], el_vals_deg[y_dest, x_dest] = az_deg, el_deg

                # Convert to ra and dec at the destination, including any time offset
                r_dest, d_dest = altAz2RADec(az_deg, el_deg, jd_source, target_lat, target_lon)

                # This time delta changes the position of the source pixel
                ang_sep_deg = angularSeparationDeg(r_dest, d_dest, r_source, d_source)

                # Is this still in the FoV
                if ang_sep_deg > np.hypot(pp_source.fov_h, pp_source.fov_v) / 2:
                    continue

                # Compute radec from azimuth and elevation at original platepar time
                r, d = altAz2RADec(az_deg, el_deg, pp_source.JD, pp_source.lat, pp_source.lon)
                r_list.append(r)
                d_list.append(d)
                # x_dest_list.append(size_x - x_dest)
                # y_dest_list.append(size_y - y_dest)
                x_dest_list.append(x_dest)
                y_dest_list.append(y_dest)

        # Compute source image pixels with the time offset
        x_source_array, y_source_array = raDecToXYPP(np.array(r_list), np.array(d_list), jd_source, pp_source)


        for x_source_float, y_source_float, x_dest, y_dest in zip(x_source_array, y_source_array, x_dest_list, y_dest_list):

            x_source, y_source = int(x_source_float), int(y_source_float)
            if not (5 < x_source < (pp_source.X_res - 5) and 5 < y_source < (pp_source.Y_res - 5)):
                continue

            m = stations_info_dict[station]['mask']
            if m is not None:
                if m[y_source, x_source] != 255:
                    continue

            station_index = stations_list.index(station)
            radius = np.hypot((x_source - pp_source.X_res) ** 2, (y_source - pp_source.Y_res) ** 2) ** 0.5
            vignetting_factor = correctVignetting(1,  radius, pp_source.vignetting_coeff)
            source_coordinates_list.append([int(transformation_layer), int(x_source), int(y_source), vignetting_factor])
            dest_coordinates_list.append([x_dest, y_dest])

        transformation_layer_list.append([station, time_offset_seconds])
        transformation_layer += 1

    source_coordinates_array = np.array(source_coordinates_list)
    dest_coordinates_array = np.array(dest_coordinates_list)

    pairs, counts = np.unique(dest_coordinates_array, axis=0, return_counts=True)
    intensity_scaling_array = np.zeros((size_x, size_y))
    for pair, count in zip(pairs, counts):
        intensity_scaling_array[pair[1]][pair[0]] = count

    target_geo = [target_lat, target_lon, target_ele]

    return stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, target_geo

def makeUpload(source_path, upload_to, print_activity=True, color=30):
    """
    Make upload of source_path

    Arguments:
        source_path: [string] source path
        upload_to: [string] upload location i.e. user@host:path/on/remote

    Return:
        Nothing

    """
    if upload_to is None:
        return

    cmd = [
        "rsync",
        "-avz",  # archive mode, verbose, compress
        source_path, upload_to
    ]


    if print_activity:
        print("Uploading to {}".format(upload))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Upload failed with {}".format(e))

    if print_activity:
        print("Uploaded")

    return

def plotConstellations(img, target_image_time_jd, cam_coords, minimum_elevation_deg):

    size_x, size_y = img.shape[0], img.shape[1]


    if plot_constellations:
        constellation_coordinates_list = getConstellationsImageCoordinates(target_image_time_jd, cam_coords, size_x,
                                                                           size_y, minimum_elevation_deg)
        for x, y, x_, y_ in constellation_coordinates_list:
            cv2.line(img, (x, y), (x_, y_), color, 1)

    if print_activity:
        print("Writing output to {:s}".format(output_path))

    return img

def jdToPyTime(jd):

    # Convert jd into python time object

    return jd2Date(jd, dt_obj=True).replace(tzinfo=datetime.timezone.utc)

def pyTimetoJD(python_datetime_object):

    return date2JD(*(python_datetime_object.replace(tzinfo=datetime.timezone.utc).timetuple()[:6]))

def renderAzimuthalProjection(transform_data, annotate=False, target_jd=None, compensation=None, plot_constellations=False, frames_files_paths_list=None):

    # If no compensation values passed use some reasonable values
    if compensation is None:
        compensation = [50, 80, 80, 99.75]

    # Extract individual variables from transform data
    stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, cam_coords = transform_data

    # Get the fits files as a stack of images, one per camera

    if frames_files_paths_list is None:
        # Work with fits
        image_array = np.stack(getFitsAsList(getFitsFiles(transformation_layer_list, stations_info_dict, target_jd, print_activity=True),
                                         stations_info_dict, compensation=compensation), axis=0)
    else:
        # Work with jpg from FramesFiles

        image_array = np.stack(getFramesAsList(getFramesFiles(transformation_layer_list, stations_info_dict, target_jd, frames_files_paths_list=frames_files_paths_list),stations_info_dict))
    # Form the uncompensated and target image arrays
    target_image_array, target_image_array_uncompensated = np.full_like(intensity_scaling_array, 0), np.full_like(
        intensity_scaling_array, 0)

    # Unwrap the source coordinates array into component lists
    camera_no, source_y, source_x, vignetting_factor_array = source_coordinates_array.T

    # And the destination coordinates list
    target_y, target_x = dest_coordinates_array.T

    # Build the uncompensated image by mappings coordinates from each camera
    intensities = image_array[list(map(int, camera_no)), list(map(int, source_x)), list(map(int, source_y))]

    # Stack the images
    np.add.at(target_image_array_uncompensated, (target_x, target_y), intensities * vignetting_factor_array)

    # Replace zero divides with the darkest tone on the image
    div_zero_replacement = np.min(intensities)

    # Average when multiple frames are stacked
    target_image_array = np.divide(target_image_array_uncompensated,
                                   intensity_scaling_array,
                                   out=np.full_like(target_image_array_uncompensated, div_zero_replacement,
                                                    dtype=float),
                                   where=intensity_scaling_array != 0).astype(float)

    # Perform compensation on final image
    min_threshold, max_threshold = np.percentile(intensities, float(compensation[2])), np.percentile(intensities, compensation[3])
    target_image_array = np.clip(255 * (target_image_array - min_threshold) / (max_threshold - min_threshold), 0, 255)

    if plot_constellations:
        target_image_array = plotConstellations(target_image_array, target_jd, cam_coords, minimum_elevation_deg, color=255)

    if annotate:
        stations_as_text = getCommaSeparatedListofStations(stations_info_dict)
        l1 = "{} Stack depth {:.0f}".format( jdToPyTime(target_jd).replace(microsecond=0),
                                                            len(transformation_layer_list) / len(stations_info_dict))
        l2 = "Lat:{:.3f} deg Lon:{:.3f} deg {}".format(cam_coords[0], cam_coords[1], stations_as_text)
        target_image_array = annotateArray(target_image_array, [l1, l2])

    return target_image_array

def annotateArray(target_image_array, lines):


    l1, l2 = lines[0], lines[1]
    size_y = target_image_array.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    position_l1 = (3, size_y - 20)


    cv2.putText(target_image_array, l1, position_l1, font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    position_l2 = (3, size_y - 5)

    cv2.putText(target_image_array, l2, position_l2, font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)

    return target_image_array

def getConstellationsImageCoordinates(jd, cam_coords, size_x, size_y, minimum_elevation_deg, print_activity=False):

    lat, lon = cam_coords[0], cam_coords[1]

    minimum_elevation_deg = max(minimum_elevation_deg, 10)

    if print_activity:
        print("Getting constellation coordinates at jd {} for location lat: {} lon: {}".format(jd, cam_coords[0], cam_coords[1]))
    constellations_path = os.path.join(os.path.expanduser("~/source/RMS/share/constellation_lines.csv"))
    lines = np.loadtxt(constellations_path, delimiter=",")
    array_ra_j2000, array_dec_j2000, array_ra_j2000_ ,array_dec_j2000_ = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    j2000=2451545

    list_ra, list_dec, list_ra_, list_dec_ = [], [], [] ,[]



    for ra_od, dec_od, ra_od_, dec_od_ in zip(array_ra_j2000, array_dec_j2000, array_ra_j2000_, array_dec_j2000_):
        ra_rads, dec_rads = equatorialCoordPrecession(j2000, jd, np.radians(ra_od), np.radians(dec_od))
        ra_rads_, dec_rads_ = equatorialCoordPrecession(j2000, jd, np.radians(ra_od_), np.radians(dec_od_))
        list_ra.append(np.degrees(ra_rads))
        list_dec.append(np.degrees(dec_rads))
        list_ra_.append(np.degrees(ra_rads_))
        list_dec_.append(np.degrees(dec_rads_))


    if False:
        list_ra = [220.35]
        list_dec = [-60.92]
        list_ra_ = [211.4]
        list_dec_ = [-60.50]


    array_ra, array_dec = np.array(list_ra), np.array(list_dec)
    array_ra_, array_dec_ = np.array(list_ra_), np.array(list_dec_)

    array_az, array_alt = raDec2AltAz(array_ra, array_dec, jd, lat, lon)
    array_az_, array_alt_ = raDec2AltAz(array_ra_ ,array_dec_ , jd, lat, lon)
    con = np.stack([array_alt, array_az, array_alt_, array_az_], axis=1)
    constellation_alt_az_above_horizon = con[(con[:, 0] >= minimum_elevation_deg) & (con[:, 2] >= minimum_elevation_deg)]

    image_coordinates = []

    """
    el_deg = 90 - np.hypot(_x * pixel_to_radius_scale_factor_x, _y * pixel_to_radius_scale_factor_y)
    az_deg = np.degrees(np.arctan2(_x, _y))
    """
    if print_activity:
        print("Creating constellation data for an image of size {},{}".format(size_x, size_y))

    origin_x, origin_y = size_x / 2, size_y / 2

    elevation_range = 2 * (90 - minimum_elevation_deg)
    pixel_to_radius_scale_factor_x = elevation_range / size_x
    pixel_to_radius_scale_factor_y = elevation_range / size_y

    for alt, az, alt_, az_ in constellation_alt_az_above_horizon:

        x, y = altAzToAzimuthalProjection(az, alt, 0, size_x, 0, size_y, 20)

        #alt_check, az_check = cartesianToAltAz(x, y, 0, size_x, 0, size_y, 20 )
        #print(alt, alt_check, az, az_check)


        x_, y_ = altAzToAzimuthalProjection(az_, alt_, 0, size_x, 0, size_y, 20)

        #alt_check_, az_check_ = cartesianToAltAz(x_, y_, 0, size_x, 0, size_y, 20 )
        #print(alt_, alt_check_, az_, az_check_)

        image_coordinates.append([int(x), int(y), int(x_), int(y_)])

        pass


    if False:
        img=np.zeros((size_x, size_y), dtype=np.uint8)

        for x, y, x_, y_ in image_coordinates:
            cv2.line(img, (x, y), (x_, y_), 20, 1 )

        imageio.imwrite('cons.png', img)

    return image_coordinates

def getTransformation(path_to_transform, stations_info_dict, size_x, size_y, minimum_elevation_deg, stack_depth, force_recomputation=False, print_activity=False):

    if os.path.exists(path_to_transform) and not force_recomputation:
        with open(path_to_transform, 'rb') as f:
            transform_data = pickle.load(f)

    if not os.path.exists(path_to_transform) or force_recomputation:
        transform_data = makeTransformation(stations_info_dict, size_x, size_y,
                                            minimum_elevation_deg=minimum_elevation_deg,
                                            print_activity=print_activity, stack_depth=stack_depth)

        with open(os.path.expanduser(path_to_transform), 'wb') as f:
            pickle.dump(transform_data, f)

    return transform_data

def singleImage(transform_data, annotate, target_jd, plot_constellations, output_path, upload=None):

    azimuthal_projection = renderAzimuthalProjection(transform_data, annotate=annotate, target_jd=target_jd,
                                                     plot_constellations=plot_constellations)
    if output_path.endswith(".png") or output_path.endswith(".bmp"):
        imageio.imwrite(output_path, azimuthal_projection.astype(np.uint8))
        makeUpload(output_path, upload)

def makeTimelapse(transform_data, annotate, timelapse_start, timelapse_end, plot_constellations, output_path, upload=None):


    frame_count = int(((jd2Date(timelapse_end, dt_obj=True) - jd2Date(timelapse_start,
                                                                      dt_obj=True)).total_seconds()) / seconds_per_frame)
    start_time_obj = datetime.datetime(*jd2Date(timelapse_start, dt_obj=True).timetuple()[:6])
    with imageio.get_writer(output_file_name, fps=25, codec="libx264", quality=8) as writer:
        for frame_no in tqdm.tqdm(range(0, frame_count)):
            frame_time_obj = start_time_obj + datetime.timedelta(seconds=frame_no * seconds_per_frame)
            target_jd = date2JD(*frame_time_obj.timetuple()[:6])

            if print_activity:
                print("Making frame at time {}".format(jd2Date(target_jd, dt_obj=True)))
            azimuthal_projection = renderAzimuthalProjection(transform_data, annotate=annotate, target_jd=target_jd,
                                                             plot_constellations=plot_constellations)
            writer.append_data(azimuthal_projection)
    makeUpload(output_path, upload)

def getTime(file_name):

    file_date = file_name.split("_")[1]
    year = int(file_date[:4])
    month = int(file_date[4:6])
    day = int(file_date[6:8])

    file_time = file_name.split("_")[2]
    hour = int(file_time[:2])
    minute = int(file_time[2:4])
    second = int(file_time[4:6])

    return datetime.datetime(year, month, day, hour, minute, second).replace(tzinfo=datetime.timezone.utc)

def getLatestMP4EndTime(hourly_directory, daily_directory):

    hourly_files = sorted(os.listdir(hourly_directory), reverse=True)
    daily_files = sorted(os.listdir(daily_directory), reverse=True)
    if len(hourly_files):
        latest_hourly_file = getTime(hourly_files[0])
        latest_hour_file_set = True
    else:
        latest_hour_file_set = False
    if len(daily_files):
        latest_daily_file = getTime(daily_files[0])
        latest_daily_file_set = True
    else:
        latest_daily_file_set = False

    if latest_hour_file_set and not latest_daily_file_set:
        latest_complete_file = latest_hourly_file + datetime.timedelta(hours=1)
    elif latest_daily_file_set and not latest_hour_file_set:
        latest_complete_file = latest_daily_file + datetime.timedelta(days=1)
    elif latest_daily_file_set and latest_hour_file_set:
        latest_complete_file = max(latest_daily_file + datetime.timedelta(days=1), latest_hourly_file + datetime.timedelta(hours=1))
    else:
        latest_complete_file = datetime.datetime(2000, 1, 1, 1, 1, 1).replace(tzinfo=datetime.timezone.utc)

    return latest_complete_file

def getEarliestFrame(frames_files_paths_list):

    earliest_frame = frames_files_paths_list[0][0][0]
    for station_frame_list in frames_files_paths_list:
        if len(station_frame_list):
            first_frame_per_station = station_frame_list[0][0]
            if first_frame_per_station < earliest_frame:
                earliest_frame = first_frame_per_station

    return earliest_frame.replace(tzinfo=datetime.timezone.utc)


def waitForNextHour(time_point):

    next_hour = (time_point + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc)
    seconds_until_next_hour = int((next_hour - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
    print("Time now is {}".format(datetime.datetime.now(datetime.timezone.utc)))
    print("Next hour starts in {}".format(seconds_until_next_hour))
    time.sleep(max(0, seconds_until_next_hour + 5))

    # If there is less than a 30 second margin until the end of the previous hour, return True
    # When there is a big buffer of unprocessed frames files, the seconds_until_next hour will be very negative

    return seconds_until_next_hour > (0 - 5)


def getFileTime(file_name):



    file_name = os.path.basename(file_name)
    file_date = file_name.split("_")[1]
    year, month, day = file_date[0:4], file_date[4:6], file_date[6:8]
    file_time = file_name.split("_")[2]
    hour, minute, second = file_time[0:2], file_time[2:4], file_time[4:6]
    time_obj = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
    return time_obj.replace(tzinfo=datetime.timezone.utc)

def pathToFilesToCollate(files_list, hourly_directory_path, daily_directory_path):

    sorted_files_list = sorted(files_list)
    if len(sorted_files_list):
        first_hourly_file_time = getFileTime(os.path.basename(sorted_files_list[0]))
        latest_hourly_file_time = getFileTime(os.path.basename(sorted_files_list[-1]))
        if len(os.listdir(daily_directory_path)):
            latest_daily_file_end = getFileTime(sorted(os.listdir(daily_directory_path))[-1]) + datetime.timedelta(days=1)
        else:
            latest_daily_file_end = datetime.datetime(2000, 1, 1, 1, 1, 1).replace(tzinfo=datetime.timezone.utc)

        start_day_of_latest_hourly_file_time = startOfThisDay(latest_hourly_file_time)
        start_day_of_first_hourly_file_time = startOfThisDay(first_hourly_file_time)
        time_gap = start_day_of_latest_hourly_file_time - start_day_of_first_hourly_file_time


        print("Latest hourly file time is {}".format(latest_hourly_file_time))
        print("Start day of latest hourly file time: {}".format(start_day_of_latest_hourly_file_time))
        print("First hourly file time is {}".format(first_hourly_file_time))
        print("Start day of first hourly file time is {}".format(start_day_of_first_hourly_file_time))
        print("End of latest hourly file time is {}".format(latest_hourly_file_time))
        print("End of latest daily file time is {}".format(latest_daily_file_end))



        if time_gap < datetime.timedelta(hours=24):
            return []
        else:
            start_time_for_collation = max(latest_daily_file_end, first_hourly_file_time)
            return getMP4HourFilePaths(hourly_directory_path, start_time_for_collation, start_day_of_latest_hourly_file_time)



    return []

def getMP4HourFilePaths(hourly_directory_paths, earliest_time_object, latest_time_object):

    hourly_directory_files_list = sorted(os.listdir(hourly_directory_paths))
    hour_files_list = []
    for hourly_directory_file in hourly_directory_files_list:
        if hourly_directory_file.endswith(".mp4"):
            if earliest_time_object <= getFileTime(hourly_directory_file) < latest_time_object:
                hour_files_list.append(os.path.join(hourly_directory_paths, hourly_directory_file))

    return hour_files_list


def runLive(transform_data, annotate=True, plot_constellations=True,  upload=True, frames_files=True, output_file_name=None, print_activity=False):

    # Extract individual variables from transform data
    stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, cam_coords = transform_data

    base_full_path = os.path.expanduser("~/RMS_data/Polar")
    video_full_path = os.path.join(base_full_path, "Video")
    hourly_directory = os.path.join(video_full_path, "Hourly")
    daily_directory = os.path.join(video_full_path, "Daily")
    working_directory = os.path.join(base_full_path, "Working")
    frames_directory = os.path.expanduser("~/RMS_data/FramesFiles")

    mkdirP(os.path.expanduser(hourly_directory))
    mkdirP(os.path.expanduser(daily_directory))
    mkdirP(os.path.expanduser(working_directory))

    # Filename format
    # Hourly files  :   hostname_YYYYMMDD_HHMMSS.mp4
    # Daily files   :   hostname_YYYYMMDD_HHMMSS.mp4

    completed_hour_file_full_path_list = []
    hour_file_names_list = os.listdir(hourly_directory)
    for hour_file in hour_file_names_list:
        completed_hour_file_full_path_list.append(os.path.join(hourly_directory, hour_file))

    while True:

        hour_file_path_list = pathToFilesToCollate(completed_hour_file_full_path_list, hourly_directory, daily_directory)
        print("File paths to be collated {}".format(hour_file_path_list))
        if len(hour_file_path_list):
            hour_file_mp4_list = []
            for hour_file_path in hour_file_path_list:
                    mp4_handle = cv2.VideoCapture(hour_file_path)
                    if mp4_handle.isOpened():
                        hour_file_mp4_list.append(moviepy.VideoFileClip(hour_file_path))
                    else:
                        print("{} could not be read, not adding to full day timelapse video, and removing".format(os.path.basename(hour_file_path)))
                        os.unlink(hour_file_path)


            day_file_mp4 = moviepy.concatenate_videoclips(hour_file_mp4_list, method="compose")
            date_of_first_file = startOfThisDay(getFileTime(sorted(hour_file_path_list)[0]))
            day_file_name = "AP_{}.mp4".format(date_of_first_file.strftime("%Y%m%d_%H%M%S"))
            day_file_full_path = os.path.join(daily_directory, day_file_name)
            day_file_mp4.write_videofile(day_file_full_path, codec="libx264", audio_codec="aac")



        frames_files_paths_list = getFramesFilesPaths(stations_info_dict)
        earliest_frame = getEarliestFrame(frames_files_paths_list)
        latest_mp4_end = getLatestMP4EndTime(hourly_directory, daily_directory)
        start_of_yesterday = startOfPreviousDay()

        timelapse_start = max(earliest_frame, latest_mp4_end, start_of_yesterday)
        # timelapse_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=6)
        seconds_per_frame = 5

        timelapse_end = timelapse_start + datetime.timedelta(hours=1) if timelapse_end = None else timelapse_end
        frame_count = int((timelapse_end - timelapse_start).total_seconds() / seconds_per_frame)
        time_string = timelapse_start.strftime("%Y%m%d_%H%M%S")
        if output_file_name is None:
            output_file_name_with_timestamp = "AP_{}.mp4".format(time_string)
        else:
            output_file_name_with_timestamp = "{}_{}.mp4".format(output_file_name, time_string)
        output_path = os.path.join(hourly_directory, output_file_name_with_timestamp)

        # Wait until there should be a complete hour of frames files available
        if waitForNextHour(timelapse_start):
            # Get frames_files_paths_list again to pick up any new frames
            frames_files_paths_list = getFramesFilesPaths(stations_info_dict, earliest_time=timelapse_start, latest_time=timelapse_end)

        makeVideo(annotate, frame_count, frames_files_paths_list, output_path, plot_constellations, print_activity,
                  seconds_per_frame, timelapse_start, transform_data)



        completed_hour_file_full_path_list.append(output_path)

        makeUpload(output_path, upload)


        pass




    pass


def makeVideo(annotate, frame_count, frames_files_paths_list, output_path, plot_constellations, print_activity,
              seconds_per_frame, timelapse_start, transform_data):
    with imageio.get_writer(output_path, format='mp4', fps=25, codec="libx264", quality=8) as writer:
        print("Making video starting at {}".format(timelapse_start))
        for frame_no in tqdm.tqdm(range(0, frame_count)):
            frame_time_obj = timelapse_start + datetime.timedelta(seconds=frame_no * seconds_per_frame)
            target_jd = pyTimetoJD(frame_time_obj)
            if print_activity:
                print("Making frame at time {}".format(frame_time_obj))
            azimuthal_projection = renderAzimuthalProjection(transform_data, annotate=annotate, target_jd=target_jd,
                                                             plot_constellations=plot_constellations,
                                                             frames_files_paths_list=frames_files_paths_list,
                                                             compensation=[0, 100, 0, 100]).astype(np.uint8)
            writer.append_data(azimuthal_projection)


if __name__ == "__main__":

    # ## PARSE INPUT ARGUMENTS ###
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Produce a projection from multiple cameras""")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, action="append",
                            help="Optional, paths to the config files. If no paths given then will search for config files"
                                 "in a multi-cam linux style file arrangement, or a one camera per usename arrangement.")


    arg_parser.add_argument('-r', '--repeat',  dest='repeat', default=False, action="store_true",
                    help="Run continuously, default false.")

    arg_parser.add_argument('-p', '--period', dest='period', default=[120], type=int, nargs=1,
                            help="Iteration period for continous running, default 120 seconds.")

    arg_parser.add_argument('-s', '--stack', dest='stack', default=[3], type=int, nargs=1,
                            help="Number of images to stack, default 3. This will only take affect if transform is recomputed.")

    arg_parser.add_argument('-o', '--output_file_name', dest='output_file_name', default=None,
                            nargs=1, help="Output filename and path. If only a path to a directory is given, then files will be saved with a timestamp."
                            "YYYYMMSS_HHMMDD. If no output path is given ~/RMS_data will be used.")

    arg_parser.add_argument('-t', '--transform', dest='transform', default=False, action="store_true",
                            help="Force recomputing of transform - needed if platepar has been changed")

    arg_parser.add_argument('-d', '--dimension', dest='dimension', default=[1040], type=int, nargs=1,
                            help="Output image size - only square images are permitted. Default 1040 x 1040.")

    arg_parser.add_argument('-q', '--quiet', dest='quiet', default=False, action="store_true",
                            help="Run quietly")

    arg_parser.add_argument('-u', '--upload', dest='upload', type=str, nargs=1,
                            help="Remote address to upload finished image to.")

    arg_parser.add_argument('-a', '--annotate', dest='annotate', default=False, action="store_true",
                            help="Annotate plot with image time, stations used, and projection origin.")

    arg_parser.add_argument('-n', '--constellations', dest='constellations', default=False, action="store_true",
                            help="Annotate plot with constellations.")

    arg_parser.add_argument('-e', '--elevation', dest='elevation', nargs=1, type=float, default=[20],
                            help="Minimum elevation to use for the plot")

    arg_parser.add_argument('-l', '--timelapse', dest='timelapse', nargs='*', type=float,
                            help="Generate timelapse over the past 24 hours of observations, including spanning"
                            "directories, or if two julian dates are specified, then timelapse between those two dates")

    arg_parser.add_argument('-j', '--julian-date', dest='julian_date', nargs=1, type=float,
                            help="Generate a single projection at the specified julian date")

    arg_parser.add_argument('-m', '--compensation', dest='compensation', nargs=4, type=float,
                            help="Image compensation values 50 80 90 99.85 work well")

    arg_parser.add_argument('-v', '--run-live', dest='run_live', default=False, action="store_true",
                            help="Capture frame files live an make into mp4 videos")

    cml_args = arg_parser.parse_args()



    quiet = cml_args.quiet
    print_activity = not quiet
    path_to_transform = os.path.expanduser("~/RMS_data/camera_combination.transform")
    force_recomputation = cml_args.transform
    repeat = cml_args.repeat
    run_live = cml_args.run_live

    period = cml_args.period[0]

    if cml_args.dimension is not None:
        # round to even number
        size = int(cml_args.dimension[0] / 2) * 2
    else:
        size = 608

    config_paths = []

    if cml_args.config is None:
        path_list = None
    else:
        for path_list in cml_args.config:
            config_paths.append(path_list[0])

    stack_depth = cml_args.stack[0]
    quiet = cml_args.quiet

    if cml_args.upload is None:
        upload = None
    else:
        upload = cml_args.upload[0]

    annotate = cml_args.annotate

    if cml_args.elevation is None:
        minimum_elevation_deg = 0
    else:
        minimum_elevation_deg = cml_args.elevation[0] if cml_args.elevation[0] > 0 else 0

    # Initialise values - these should never be used
    timelapse_start, timelapse_end, seconds_per_frame = None, None, None
    make_timelapse = False


    if cml_args.timelapse is None:
        timelapse_start = None
        timelapse_end = None
        seconds_per_frame = None

    else:
        if len(cml_args.timelapse) == 0:
            timelapse_end = date2JD(*(datetime.datetime.now(datetime.timezone.utc).timetuple()[:6]))
            timelapse_start = timelapse_end - 1
            seconds_per_frame = 256/25
            make_timelapse = True

        elif len(cml_args.timelapse) == 1:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = date2JD(*(datetime.datetime.now(datetime.timezone.utc).timetuple()[:6]))
            seconds_per_frame = 256 / 25
            make_timelapse = True


        elif len(cml_args.timelapse) == 2:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = cml_args.timelapse[1]
            seconds_per_frame = 256 / 25
            make_timelapse = True


        elif len(cml_args.timelapse) == 3:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = cml_args.timelapse[1]
            seconds_per_frame = cml_args.timelapse[2]
            make_timelapse = True

    if cml_args.julian_date is None:
        target_jd = None
        single_image = False
    else:
        target_jd = cml_args.julian_date[0]
        single_image = True


    if cml_args.compensation is None:
        compensation = [80, 95, 50, 99.995]
    else:
        compensation = cml_args.compensation

    plot_constellations = cml_args.constellations

    if cml_args.output_file_name is None:
        output_file_name = None
    else:
        output_file_name = os.path.expanduser(cml_args.output_file_name[0])

    if not run_live:
        if output_file_name is None:
            mkdirP(os.path.expanduser("~/RMS_data/PolarPlot/Projection/"))
            if make_timelapse:
                output_path = os.path.expanduser(
                    "~/RMS_data/PolarPlot/Projection/JD_{}_{}_timelapse.mp4".format(timelapse_start, timelapse_end))

            else:
                output_path = os.path.expanduser(
                    "~/RMS_data/PolarPlot/Projection/{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))

        else:
            if os.path.exists(os.path.expanduser(output_file_name)):
                if os.path.isdir(os.path.expanduser(output_file_name)):
                    if make_timelapse:
                        output_path = os.path.expanduser(
                            "~/RMS_data/PolarPlot/Projection/JD_{}_timelapse.png".format(timelapse_start))
                    else:
                        output_path = os.path.join(os.path.expanduser(output_file_name),
                                               "{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))
                else:
                    output_path = os.path.expanduser(output_file_name)
            elif not os.path.exists(os.path.dirname(os.path.expanduser(output_file_name))):
                mkdirP(os.path.dirname(os.path.expanduser(output_file_name)))
                output_path = os.path.expanduser(output_file_name)
            else:
                output_path = os.path.expanduser(output_file_name)
    else:
        print("Running live")


    stations_info_dict = getStationsInfoDict(config_paths)
    size_x, size_y = size, size
    transform_data = getTransformation(path_to_transform, stations_info_dict, size_x, size_y, minimum_elevation_deg,
                                       stack_depth, force_recomputation, print_activity)


    if single_image:
        singleImage(transform_data, annotate, target_jd, plot_constellations, output_path, upload=upload)

    elif make_timelapse:
        makeTimelapse(transform_data, annotate, target_jd, plot_constellations, output_path, upload=upload)


    elif run_live:
        runLive(transform_data, annotate, plot_constellations, upload=upload, frames_files=True)

        sys.exit()





