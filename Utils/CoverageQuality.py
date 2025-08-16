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

J2000 = 2451545.0
import matplotlib.pyplot as plt
import os
import socket
from curses.ascii import isalnum

from scipy.constants import electron_mass
from scipy.spatial import cKDTree
import pickle

from mpl_toolkits.mplot3d import Axes3D
from RMS.Astrometry.Conversions import ECEF2AltAz, altAz2RADec, raDec2Vector, raDec2AltAz, AER2ECEF
import cv2
from pip._internal import resolution

import RMS.Formats.FFfits as FFfits
import sqlite3
import time
import datetime
import numpy as np
import tempfile
import tarfile
import paramiko
import subprocess
import json
import requests
import tqdm
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, ECEF2AltAz, altAz2RADec, vector2RaDec, J2000_JD, trueRaDec2ApparentAltAz
from RMS.Math import angularSeparationVect, angularSeparationDeg, vectNorm, dimHypot
from RMS.Routines.MaskImage import getMaskFile
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, geoHt2XY, xyHt2Geo, xyToRaDecPP

import RMS.ConfigReader as cr
import gc
import shutil


from RMS.Misc import mkdirP
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
import matplotlib.pyplot as plt

REMOTE_SERVER = 'gmn.uwo.ca'
USER_NAME = "analysis"
STATION_COORDINATES_DICT = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"
STATIONS_DATA_DIR = "StationData"
REMOTE_STATION_PROCESSED_DIR = "/home/$STATION/files/processed"
WORKING_DIRECTORY = os.path.expanduser("~/RMS_data/Coverage")
CHARTS = "charts"



def retrieveBz2File(file_name, server):


    return


def lsRemote(host, username, port, remote_path, sock=None):
    """Return the files in a remote directory.

    Arguments:
        host: [str] remote host.
        username: [str] user account to use.
        port: [int] remote port number.
        remote_pat: [str] path of remote directory to list.

    Return:
        files: [list of strings] Names of remote files.
    """

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
    ssh.connect(hostname=host, port=port, username=username, sock=sock)

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

def extractBz2(input_directory, working_directory, local_target_list=None):

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

def extractBz2Files(bz2_list, input_directory, working_directory, silent=True):
    for bz2 in bz2_list:
        basename_bz2 = os.path.basename(bz2)
        station_directory = os.path.join(working_directory, basename_bz2.split("_")[0]).lower()
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
            print("Redownloading {}".format(basename_bz2))
            path = os.path.join("/home", basename_bz2.split("_")[0].lower(), "files", "processed")
            downloadFile("gmn.uwo.ca", "analysis", 22, path, bz2_directory )
            with tarfile.open(os.path.join(input_directory, basename_bz2), 'r:bz2') as tar:
                tar.extractall(path=bz2_directory)

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
                    if ftp_file_name is None:
                        print("     Could not find a suitable FTP file for {}".format(station))
                        continue
                    ftp_path = os.path.join(working_directory, station, archived_directory)
                    ftp_dict[station] = readFTPdetectinfo(ftp_path, ftp_file_name)

    return ftp_dict

def getFTPFileName(archived_directory, station, working_directory):
    basename_archived_directory = os.path.basename(archived_directory)
    ar_date, ar_time = basename_archived_directory.split("_")[1], basename_archived_directory.split("_")[2]
    ar_milliseconds = basename_archived_directory.split("_")[3]
    ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
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

    else:

        print("     Expected FTP file {} was found".format(ftp_file_name))

    print("Returning {}".format(ftp_file_name))
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

def groupObservationsIntoEvents(observations):
    events = []
    first_observation = True
    observation_list = []
    for observation in observations:
        observation_start_time = observation[0]
        observation_end_time = observation[1]
        if not first_observation:
            time_gap_seconds = (observation_start_time - latest_observation_end_time).total_seconds()
            if time_gap_seconds > 5:
                if len(observation_list) > 1:
                    station_name_list = []
                    for station_name in observation_list:
                        station_name = station_name[2][1]
                        if not station_name in station_name_list:
                            station_name_list.append(station_name)
                            if len(station_name_list) > 1:
                                events.append(sorted(observation_list, key=lambda x: x[0]))
                observation_list = []
                observation_list.append(observation)
                latest_observation_end_time = max(observation_end_time, latest_observation_end_time)
            else:
                observation_list.append(observation)
                latest_observation_end_time = observation_end_time
        else:
            first_observation = False
            observation_list.append(observation)
        _observation_end_time = observation_end_time
        latest_observation_end_time = observation_end_time
    if len(observation_list) > 1:
        events.append(sorted(observation_list))
    return events

def getObservations(ftp_dict, station_list=None, event_time=None, duration=None):
    observations = []
    for station in sorted(ftp_dict):
        if station_list is not None:
            if station not in station_list:
                continue
        for observation in ftp_dict[station]:
            ff_name = observation[0]
            fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
            observation_start_frame = observation[11][0][1]
            observation_end_frame = observation[11][-1][1]
            observation_start_time = fits_date + datetime.timedelta(seconds=observation_start_frame / observation[4])
            observation_end_time = fits_date + datetime.timedelta(seconds=observation_end_frame / observation[4])
            if event_time is None or duration is None:
                observations.append([observation_start_time, observation_end_time, observation])
            else:
                if abs((observation_start_time - event_time).total_seconds()) < 2 and \
                   abs((observation_end_time - (event_time + datetime.timedelta(seconds=duration))).total_seconds()) < 2:
                    observations.append([observation_start_time, observation_end_time, observation])
    observations = sorted(observations, key=lambda x: x[0])
    return observations

def createImagesDict(events, working_area, ftp_dict):

    events_with_fits_dict = {}
    for event in events:
        event_with_fits = addFITSToEvent(event, working_area, ftp_dict)

        if len(event_with_fits) > 1:
            events_with_fits_dict[event[0][0]] = event_with_fits

    return events_with_fits_dict

def addFITSToEvent(event, working_area, ftp_dict):
    event_with_fits = []
    print("Start loading fits files")
    for observation in event:
        fits_file = observation[2][0]
        station_directory = os.path.join(working_area, observation[2][1].lower())
        bz2_directory_list = os.listdir(station_directory)
        for bz2_directory in bz2_directory_list:
            fits_path = os.path.join(station_directory, bz2_directory, fits_file)
            if os.path.exists(fits_path):

                if fits_file.endswith(".bin"):
                    fits_file = fits_file.replace('.bin', '.fits')
                if fits_file.startswith("FR_"):
                    fits_file = fits_file.replace('FR_', 'FF_')
                ff = FFfits.read(os.path.join(station_directory, bz2_directory), fits_file)
                print("     Loading {}".format(fits_file))
                observation_and_fits = [observation, ff]
                event_with_fits.append(observation_and_fits)
    return event_with_fits

def rotateCapture(input_image, angle, rotation_centre, length,run_in=100, run_out=100, y_dim = 100, show_intermediate=False):

    # working area
    size = 4000
    axis_centre = size / 2
    image_centre = (axis_centre, axis_centre)
    working_image_dim = (size, size)
    translated_image = translateToOrigin(axis_centre, input_image, rotation_centre, working_image_dim)
    rotated_image = rotateToHorizontal(angle, image_centre, translated_image, working_image_dim)
    final_image = translateResize(axis_centre, length, rotated_image, run_in, run_out, y_dim)

    return final_image

def translateResize(axis_centre, length, rotated_image, run_in, run_out, y_dim):
    translate_x, translate_y = int(run_in - axis_centre), int(y_dim / 2 - axis_centre)
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    final_img_dim_x, final_img_dim_y = int(length + run_in + run_out), int(y_dim)
    final_img_dim = (final_img_dim_x, final_img_dim_y)
    final_image = cv2.warpAffine(rotated_image, translation_matrix, final_img_dim)
    return final_image

def rotateToHorizontal(angle, image_centre, translated_image, working_image_dim):
    rotation_matrix = cv2.getRotationMatrix2D(image_centre, angle, 1.0)
    rotated_image = cv2.warpAffine(translated_image, rotation_matrix, working_image_dim)
    return rotated_image

def translateToOrigin(axis_centre, input_image, rotation_centre, working_image_dim):
    translate_x, translate_y = 0 - int(rotation_centre[0] - axis_centre), 0 - int(rotation_centre[1] - axis_centre)
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    translated_image = cv2.warpAffine(input_image, translation_matrix, working_image_dim)
    return translated_image

def rotateCoordinateList(points, angle_degrees):
    if not points:
        return []

    angle_radians = np.radians(angle_degrees)
    fx, cx, cy = points[0]  # Pivot point (the first coordinate)

    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    rotated = []
    for frame, x, y in points:
        # Translate point to origin
        tx, ty = x - cx, y - cy

        # Rotate around origin
        rx = tx * cos_theta - ty * sin_theta
        ry = tx * sin_theta + ty * cos_theta

        # Translate back
        rotated.append((frame, rx, ry))

    return rotated

def straightenFlight(img, coordinates, run_in, run_out):



    x = 0

    y_correction = [coordinate[1] for coordinate in coordinates]
    x_correction = [coordinate[2] for coordinate in coordinates]
    working_img = np.zeros_like(img)

    for x in range(img.shape[1]):



        if x - run_in > 0:
            ty = np.interp(x - run_in, x_correction, y_correction)
        else:
            ty = 0


        for y in range(img.shape[0]):
            if 0 < (y + ty) < img.shape[0]:
                working_img[y, x] = img[int(y + ty), x]



    return working_img

def addTimingInformation(image_data, image_array, rotated_coordinates, run_in, run_out, trajectory_summary_report):

    frame_count_list, y_coordinate_list = getFrameCoordinateMapping(rotated_coordinates)

    station = image_data[0][2][1]
    time_correction = getTimeCorrection(station, trajectory_summary_report)

    time_column_list = getObservationTimingData(frame_count_list, image_array, image_data, run_in, run_out, y_coordinate_list, time_correction)
    final_time_stamp, image_start, seconds_per_column = computeTimingsForObservation(run_in, time_column_list, time_correction)
    time_column_list = procesRunIn(image_start, run_in, seconds_per_column, time_column_list)
    time_column_list = processRunOut(final_time_stamp, image_array, run_out, seconds_per_column, time_column_list)
    time_column_list.sort()

    return image_data, image_array, time_column_list


def getTimeCorrection(station, trajectory_summary_report):
    time_correction = 0
    if "TimingOffsets" in trajectory_summary_report:
        value_list = []
        for key in trajectory_summary_report["TimingOffsets"]:
            if key.startswith(station):
                value_list.append(trajectory_summary_report['TimingOffsets'][key])
        time_correction = sum(value_list) / len(value_list) if len(value_list) > 0 else 0
    time_correction = 0
    return datetime.timedelta(seconds=time_correction)


def computeTimingsForObservation(run_in, time_column_list, time_correction):
    first_time_stamp = time_column_list[0][1] + time_correction
    final_time_stamp = time_column_list[-1][1] + time_correction
    observation_duration = (final_time_stamp - first_time_stamp).total_seconds()
    seconds_per_column = observation_duration / len(time_column_list)
    image_start = first_time_stamp - datetime.timedelta(seconds=seconds_per_column * run_in)
    return final_time_stamp, image_start, seconds_per_column

def processRunOut(final_time_stamp, image_array, run_out, seconds_per_column, time_column_list):
    c = 0
    for column in range(len(image_array[0]) - run_out, len(image_array[0])):
        frame_time_stamp = final_time_stamp + datetime.timedelta(seconds=(c * seconds_per_column))
        time_column_list.append([column, frame_time_stamp])
        c += 1
        pass
    return time_column_list

def procesRunIn(image_start, run_in, seconds_per_column, time_column_list):
    c = 0
    for column in range(0, run_in + 1):
        frame_time_stamp = image_start + datetime.timedelta(seconds=c * seconds_per_column)
        time_column_list.append([column, frame_time_stamp])
        c += 1
        pass
    return time_column_list

def getObservationTimingData(frame_count_list, image_array, image_data, run_in, run_out, y_coordinate_list, time_correction):

    fps = image_data[0][2][4]
    c = 0
    time_column_list = []
    first_frame_number = image_data[0][2][11][0][1]
    first_frame = image_data[0][0]

    for column in range(run_in, len(image_array.T) - run_out):
        frame = np.interp(c, y_coordinate_list, frame_count_list)
        frame_time_stamp = first_frame + datetime.timedelta(seconds=(frame - first_frame_number) / fps)
        time_column_list.append([c + run_in, frame_time_stamp + time_correction])
        c += 1


    return time_column_list

def getFrameCoordinateMapping(rotated_coordinates):
    frame_count_list, y_coordinate_list = [], []
    for frame_count, _, y in rotated_coordinates:
        frame_count_list.append(frame_count)
        y_coordinate_list.append(y)
    return frame_count_list, y_coordinate_list

def processEventImages(event_images, event_images_with_timing_dict, run_in, run_out, y_dim, trajectory_summary_report):
    first_image = True
    for image in event_images:
        ff_file = image[0][2][0]
        ftp_entry = image[0][2][11]
        station = image[0][2][1]
        coordinates_list = []
        time_correction = getTimeCorrection(station, trajectory_summary_report)
        observation_end, observation_start = getObservationTimeExtent(coordinates_list, ftp_entry, image)
        observation_end += time_correction
        observation_start += time_correction
        angle_deg, img, length, start_x, start_y = getImageInformation(image)
        rotated_coordinates = rotateCoordinateList(coordinates_list, angle_deg)
        img = rotateCapture(img, angle_deg, (start_x, start_y), length, run_in=run_in, run_out=run_out, y_dim=y_dim)
        img = straightenFlight(img, rotated_coordinates, run_in, run_out)
        image_data, image_array, img_timing = addTimingInformation(image, img, rotated_coordinates, run_in, run_out, trajectory_summary_report)

        if first_image:
            first_image = False
            event_start, event_end = observation_start, observation_end
        else:
            event_start, event_end = min(observation_start, event_start), max(observation_end, event_end)

        event_images_with_timing_dict[ff_file] = [img_timing, image_data, image_array]
    return event_start, event_end, img, event_images_with_timing_dict

def interpolateByTime(target_time, reference_times_list, reference_value_list):

    first_iteration = True

    if target_time < reference_times_list[0]:
        return 0
    if target_time > reference_times_list[-1]:
        return 0

    for reference_time, reference_value in zip(reference_times_list, reference_value_list):
        if first_iteration:
            _reference_time = reference_time
            _reference_value = reference_value
            first_iteration = False
        else:
            if reference_time >= target_time:
                break
            else:
                _reference_time = reference_time
                _reference_value = reference_value

    reference_time_steps = (reference_time - _reference_time).total_seconds()
    reference_value_difference = float(reference_value) - float(_reference_value)
    time_discrepancy = (target_time - _reference_time).total_seconds()

    if reference_time_steps == 0:
        interpolated_value = reference_value
    else:
        interpolated_value = _reference_value + time_discrepancy * (reference_value_difference / reference_time_steps)
    if interpolated_value > 0:
        #print("_reference value, interpolated value, reference value {},{},{}".format(_reference_value, interpolated_value, reference_value))
        pass
    return interpolated_value

def createOutputChartColumnTimings(chart_x_min_time, columns_per_second, output_array):
    column_count, output_column_time_list = 0, []
    for column in output_array:
        column_time = chart_x_min_time + datetime.timedelta(seconds=(column_count / columns_per_second))
        output_column_time_list.append(column_time)
        column_count += 1
    return output_column_time_list

def getImageInformation(image):
    ff = image[1]
    ftp_info = image[0][2]
    img = ff.maxpixel
    start_x, start_y = int(ftp_info[11][0][2]), int(ftp_info[11][0][3])
    end_x, end_y = int(ftp_info[11][-1][2]), int(ftp_info[11][-1][3])
    length = np.hypot(end_x - start_x, end_y - start_y)
    angle_rads = np.arctan2(end_y - start_y, end_x - start_x)
    angle_deg = np.degrees(angle_rads)
    return angle_deg, img, length, start_x, start_y

def getObservationTimeExtent(coordinates_list, ftp_entry, image):
    for coordinate_line in ftp_entry:
        ftp_frame_no, y_coordinate, x_coordinate = coordinate_line[1], coordinate_line[2], coordinate_line[3]
        coordinates_list.append([ftp_frame_no, x_coordinate, y_coordinate])
    observation_start, observation_end = image[0][0], image[0][1]
    return observation_end, observation_start

def annotateChart(chart_x_resolution, event_images, event_images_with_timing_dict, img, output_array,
                  output_column_time_list, y_dim, trajectory_summary_report):
    y_origin = 0
    y_labels, y_label_coords, plot_annotations_dict = [], [], {}
    for image_key in sorted(event_images_with_timing_dict):

        y_labels.append("{}".format(image_key.split('_')[1]))
        y_label_coords.append(y_origin + 0.5 * y_dim)
        station = image_key.split('_')[1]
        time_correction = getTimeCorrection(station, trajectory_summary_report)
        observation_start_time = event_images_with_timing_dict[image_key][1][0][0] + time_correction
        observation_end_time = event_images_with_timing_dict[image_key][1][0][1] + time_correction
        row_timing = event_images_with_timing_dict[image_key][0]
        row_timing = [item[1] for item in row_timing]
        image_array = event_images_with_timing_dict[image_key][2]

        for y in range(0, y_dim):
            row_intensity = image_array[y]
            for x in range(0, chart_x_resolution):
                output_image_time = output_column_time_list[x]

                brightness = interpolateByTime(output_image_time, row_timing, row_intensity)
                output_array[x, y_origin + y] = brightness

        display_array = output_array.T
        for event in event_images:
            pass
            ff_name = event[0][2][0]
            camera_name = event[0][2][1]
            if camera_name == image_key:
                observation_start_time, observation_end_time = event[0][0], event[0][1]
                time_correction = getTimeCorrection(camera_name, trajectory_summary_report)
                break

        annotation = '{}'.format(image_key)
        plot_annotations_dict[(10, y_origin + y_dim - 10)] = annotation
        annotation = "{:.3f}s".format(float(observation_start_time.strftime("%S.%f")))
        c = 0
        for t in output_column_time_list:
            c += 1
            if t > observation_start_time:
                break
        plot_annotations_dict[(c - 20, int(y_origin + y_dim * 0.4))] = annotation

        annotation = "{:.3f}s".format(float(observation_end_time.strftime("%S.%f")))
        c = 0
        for t in output_column_time_list:
            c += 1
            if t > observation_end_time:
                break
        plot_annotations_dict[(c - 20, int(y_origin + y_dim * 0.7))] = annotation

        y_origin += y_dim

    return display_array, plot_annotations_dict, y_label_coords, y_labels

def plotChart(display_array, output_column_time_list, plot_annotations_dict, y_label_coords, y_labels, target_file_name=None, magnitude=None):
    # Plot
    plot_x_range, plot_y_range = display_array.shape[0] / 100, display_array.shape[1] / 100
    fig, ax = plt.subplots(figsize=(plot_y_range, plot_x_range))

    im = ax.imshow(display_array, aspect='auto', cmap='gray')
    # Set x-axis to custom time scale
    earliest_time = output_column_time_list[0]
    latest_time = output_column_time_list[-1]
    duration_seconds = (latest_time - earliest_time).total_seconds()
    if duration_seconds < 2:
        pass
        ticks_per_second = 10
    elif duration_seconds < 5:
        ticks_per_second = 4
    elif duration_seconds < 10:
        ticks_per_second = 2
    elif duration_seconds < 30:
        ticks_per_second = 1
    else:
        ticks_per_second = 0.5
    mark_tick = False
    first_iteration = True
    tick_times = []
    for t in output_column_time_list:
        # Convert to total microseconds
        total_microseconds = (t.second + t.microsecond / 1e6) * ticks_per_second
        rounded_units = np.ceil(total_microseconds)

        # Construct the new time
        rounded_seconds = rounded_units / ticks_per_second
        t_rounded = t.replace(microsecond=0, second=0) + datetime.timedelta(seconds=rounded_seconds)

        if first_iteration:
            _t_rounded = t_rounded
            first_iteration = False
        else:

            if t > _t_rounded:
                tick_times.append(_t_rounded)
                _t_rounded = t_rounded
    tick_positions = []
    for tick in tick_times:
        c = 0
        for col_time in output_column_time_list:

            c += 1
            if col_time >= tick:
                tick_positions.append(c)
                break
    tick_positions.sort()
    tick_times.sort()
    tick_times = tick_times[0:len(tick_positions)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["{:.2f}".format(float(tick.strftime('%S.%f'))) for tick in tick_times], rotation=90, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_yticks(y_label_coords)
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.set_ylabel('Station', fontsize=16)
    # Optional: format with DateFormatter if using mdates
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.colorbar(im, ax=ax, label='Intensity')
    time_without_microseconds = tick_times[int(len(tick_times) / 2)].replace(microsecond=0).isoformat()
    if magnitude is None:
        plt.title('{}'.format(time_without_microseconds), fontsize=20)
    else:
        plt.title('{} Magnitude :{}'.format(time_without_microseconds, magnitude), fontsize=18)

    for (x_coord, y_coord), label in plot_annotations_dict.items():
        plt.annotate(
            label,
            xy=(x_coord, y_coord),
            xytext=(x_coord, y_coord),  # offset text slightly
            fontsize=12,
            color=(0.9, 0.9, 0.9)
        )
    plt.tight_layout()
    if target_file_name is None:
        plt.show()
    else:
        print("Saving {}".format(target_file_name))
        plt.savefig(target_file_name)

def getPathsOfFilesToRetrieve(station_list, event_time):

    files_to_retrieve = []
    for station in station_list:
        remote_path = os.path.join("/home", station.lower(), "files", "processed")
        bz2_files = []
        while bz2_files == []:
            try:
                bz2_files = lsRemote("gmn.uwo.ca", "analysis", 22, remote_path)
            except:
                time.sleep(120)

        bz2_files.sort(reverse=True)
        for file_name in bz2_files:
            file_name_time = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(file_name), "%Y-%m-%d %H:%M:%S.%f")
            if file_name_time < event_time:
                files_to_retrieve.append(os.path.join(remote_path, file_name))
                break

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

def filesNotAvailableLocally(station_list, event_time):

    station_files_to_retrieve = []
    local_dirs_to_use = []
    for station in station_list:
        file_present_locally = False
        local_station_path = os.path.expanduser(os.path.join("~/tmp/collate_working_area/", station.lower()))
        if not os.path.exists(local_station_path):
            station_files_to_retrieve.append(station)
            print("     Must retrieve files for {}".format(station))
            continue
        station_detected_dir_list = os.listdir(local_station_path)
        station_detected_dir_list.sort(reverse=True)

        for detected_dir in station_detected_dir_list:
            file_present_locally = False
            detected_dir_date = detected_dir.split("_")[1]
            detected_dir_time = detected_dir.split("_")[2]
            year, month, day = detected_dir_date[0:4], detected_dir_date[4:6], detected_dir_date[6:8]
            hour, minute, second = detected_dir_time[0:2], detected_dir_time[2:4], detected_dir_time[4:6]
            detected_dir_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            if detected_dir_time < event_time:
                detected_dir_full_path = os.path.join("~/tmp/collate_working_area", station.lower(), detected_dir)
                detected_dir_full_path = os.path.expanduser(detected_dir_full_path)
                detected_dir_list = os.listdir(detected_dir_full_path)
                fits_files_list = []
                for test_file in detected_dir_list:
                    if test_file.startswith("FF_{}".format(station.upper())) and test_file.endswith(".fits"):
                        fits_files_list.append(test_file)
                        found_in = detected_dir


                fits_files_list.sort(reverse=True)

                for ff_name in fits_files_list:
                    fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
                    time_difference_seconds = (fits_date - event_time).total_seconds()

                    if time_difference_seconds < 11:
                        print("Using fits file {}".format(ff_name))
                        file_present_locally = True
                        local_dirs_to_use.append(detected_dir_full_path)
                        break

                if file_present_locally:
                    break

        if not file_present_locally:
            print("No file present locally for station {}, adding to retrieve list".format(station.lower()))
            station_files_to_retrieve.append(station)

    return station_files_to_retrieve, local_dirs_to_use

def produceCollatedChart(input_directory, run_in=100, run_out=100, y_dim=300, x_image_extent=1000, event_run_in=0.05, event_run_out=0.05, target_file_name=None, show_debug_info=False, station_list=None, event_time=None, duration=None, magnitude=None):


    if station_list is not None and duration is not None and event_time is not None:
        station_list_to_get, local_available_directories = filesNotAvailableLocally(station_list, event_time)
        remote_path_list = getPathsOfFilesToRetrieve(station_list_to_get, event_time)
        if len(remote_path_list):
            print("Retrieving from remote:")
            for d in remote_path_list:
                print("     {}".format(d))
        if len(local_available_directories):
            print("These directories already available:")
            for d in local_available_directories:
                print("     {}".format(d))
        local_target_list = []
        for path in remote_path_list:
            basename = os.path.basename(path)
            local_target = os.path.join(os.path.expanduser("~/RMS_data/bz2files/"), basename)
            local_target_list.append(local_target)
            if not os.path.exists(local_target):
                print("Downloading {} to {}".format(basename, local_target))
                downloadFile("gmn.uwo.ca", "analysis", 22, path, local_target )



    working_area = createTemporaryWorkArea("~/tmp/collate_working_area")


    working_area = extractBz2("~/RMS_data/bz2files", working_area, local_target_list)


    ftp_dict = readInFTPDetectInfoFiles(working_area, station_list, event_time=event_time)
    events = clusterByTime(ftp_dict, station_list, event_time, duration)
    # trajectory_summary_report = parseTrajectoryReport("~/RMS_data/bz2files/initial_part/20200109_232639_report.txt")
    trajectory_summary_report = {}
    event_images_dict = createImagesDict(events, working_area, ftp_dict)

    print("Producing chart")
    event_images_with_timing_dict = {}
    for key in event_images_dict:
        event_images = event_images_dict[key]
        event_images_with_timing_dict = {}
        event_start, event_end, img, event_images_with_timing_dict = processEventImages(event_images, event_images_with_timing_dict, run_in,
                                                         run_out, y_dim, trajectory_summary_report)
        if len(event_images_with_timing_dict) < 2:
            continue
        event_duration_seconds = (event_end - event_start).total_seconds()
        columns_per_second = x_image_extent / event_duration_seconds
        chart_x_min_time = event_start - datetime.timedelta(seconds=event_duration_seconds * event_run_in)
        chart_x_max_time = event_end + datetime.timedelta(seconds=event_duration_seconds * event_run_out)
        chart_duration_seconds = (chart_x_max_time - chart_x_min_time).total_seconds()
        chart_x_resolution = round(chart_duration_seconds * columns_per_second)
        number_of_observations = round(len(event_images_with_timing_dict))
        chart_y_resolution = y_dim * number_of_observations

        if show_debug_info:
            print("Chart x min time       :{}".format(chart_x_min_time))
            print("Event start            :{}".format(event_start))
            print("Event end              :{}".format(event_end))
            print("Chart x max time       :{}".format(chart_x_max_time))
            print("Event duration (s)     :{}".format(event_duration_seconds))
            print("Chart duration (s)     :{}".format(chart_duration_seconds))
            print("Columns per second     :{}".format(columns_per_second))
            print("Number of observations: {}".format(number_of_observations))
            print("Output resolution (x,y):({},{})".format(chart_x_resolution, chart_y_resolution))

        output_array = np.zeros((chart_x_resolution, chart_y_resolution))
        output_column_time_list = createOutputChartColumnTimings(chart_x_min_time, columns_per_second, output_array)

        display_array, plot_annotations_dict, y_label_coords, y_labels = annotateChart(chart_x_resolution,
                                                                                     event_images,
                                                                                       event_images_with_timing_dict,
                                                                                       img, output_array,
                                                                                       output_column_time_list,
                                                                                       y_dim, trajectory_summary_report)

        plotChart(display_array, output_column_time_list, plot_annotations_dict, y_label_coords, y_labels, target_file_name=target_file_name, magnitude=magnitude)

    return

def processDatabase(database_path, country_code):
    print("Connecting to {}".format(database_path))
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        'SELECT "Unique trajectory identifier", "Beginning UTC Time", "Duration sec", "Participating Stations", "Peak AbsMag" '
        'FROM Trajectories '
        'WHERE "Peak AbsMag" < -4 '
        'ORDER BY "Peak AbsMag" ASC'.format(country_code))
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

def getStationList(user=USER_NAME, host=REMOTE_SERVER, remote_path=STATION_COORDINATES_DICT):

    station_list = []

    stations_dict = json.loads(requests.get(remote_path).content.decode('utf-8'))

    for station in stations_dict:
        station_list.append(station)

    return sorted(station_list)

def makeConfigPlateParMaskLib(config, station_list, stations_data_dir=STATIONS_DATA_DIR,
                              remote_station_processed_dir=REMOTE_STATION_PROCESSED_DIR, host=REMOTE_SERVER, username=USER_NAME, port=22):

    stations_data_full_path = os.path.join(config.data_dir, stations_data_dir)


    for station in tqdm.tqdm(station_list):
        local_target = os.path.join(stations_data_full_path, station.lower())

        with tempfile.TemporaryDirectory() as t:
            remote_dir = remote_station_processed_dir.replace("$STATION", station.lower())

            extraction_dir = os.path.join(t, "extracted")
            local_target_full_path = os.path.join(local_target)
            local_config_path = os.path.join(local_target_full_path, os.path.basename(config.config_file_name))
            local_platepar_path = os.path.join(local_target_full_path, config.platepar_name)
            local_mask_path = os.path.join(local_target_full_path, config.mask_file)
            if os.path.exists(local_config_path) and os.path.exists(local_platepar_path) and os.path.exists(local_mask_path):
                continue
            remote_files = sorted(lsRemote(host, username, port, remote_dir), reverse=True)
            if len(remote_files):
                latest_remote_file = remote_files[0]
            else:
                continue
            extracted_files_path = os.path.join(extraction_dir, station.lower(), latest_remote_file.split(".")[0])
            extracted_config_path = os.path.join(extracted_files_path, ".config")
            extracted_platepar_path = os.path.join(extracted_files_path, config.platepar_name)
            extracted_mask_path = os.path.join(extracted_files_path, config.mask_file)
            full_remote_path_to_bz2 = os.path.join(remote_dir, latest_remote_file)
            downloadFile(host, username, port, full_remote_path_to_bz2, t)

            mkdirP(extraction_dir)
            extractBz2(t, extraction_dir)

            if os.path.exists(extracted_config_path) and \
               os.path.exists(extracted_platepar_path) and \
               os.path.exists(extracted_mask_path):

                mkdirP(local_target_full_path)
                shutil.move(extracted_config_path, local_config_path)
                shutil.move(extracted_platepar_path, local_platepar_path)
                shutil.move(extracted_mask_path, local_mask_path)


def makeStationsInfoDict(config, stations_data_dir=STATIONS_DATA_DIR):

    stations_info_dict = {}
    stations_data_full_path = os.path.join(config.data_dir, stations_data_dir)
    stations_list = sorted(os.listdir(stations_data_full_path))
    for station in tqdm.tqdm(stations_list):

        station_info_path = os.path.join(stations_data_full_path, station)
        config_path = os.path.join(station_info_path,".config")
        if os.path.exists(config_path):
            config = cr.parse(os.path.join(station_info_path,".config"))
        else:

            continue
        lat_rads, lon_rads, ele_m = np.radians(config.latitude), np.radians(config.longitude), config.elevation
        x, y, z = latLonAlt2ECEF(lat_rads, lon_rads, ele_m)
        mask_struct = getMaskFile(station_info_path, config, silent=True)
        pp = Platepar()
        pp_full_path = os.path.join(station_info_path, config.platepar_name, config.platepar_name)
        if os.path.exists(pp_full_path):
            print(pp_full_path)
        else:
            print("Path does not exist {}".format(pp_full_path))
        pp.read(pp_full_path)
        print(pp.station_code)
        stations_info_dict[station.lower()] =    {'ecef' : (x, y, z),
                                                  'geo': {'lat_rads': lat_rads, 'lon_rads': lon_rads, 'ele_m': ele_m},
                                                  'pp': pp,
                                                  'mask': mask_struct}

    return stations_info_dict

def filterPointsByElevation(points_list, min_ele, max_ele):

    points_filtered_by_elevation = []
    for point in points_list:
        lat_rads, lon_rads, alt = ecef2LatLonAlt(point[0], point[1], point[2])
        if min_ele < alt < max_ele:
            points_filtered_by_elevation.append(point)

    return points_filtered_by_elevation

def roundList(list, resolution_m):

    output_list = []

    for point in list:
        output_coord = []
        for coord in point:
            output_coord.append(round(coord / resolution_m) * resolution_m)
        output_list.append(output_coord)

    return output_list

def makeECEFPointListAroundStations(station_info_dict, max_distance_to_station_m, resolution_m, min_ele_m=20000, max_ele_m=100000):


    if not len(station_info_dict):
        return []

    # Compute a list of offsets to apply to each station - do this only once and reuse
    offsets_list = np.arange(0 - max_distance_to_station_m, 0 + max_distance_to_station_m + resolution_m,
                                       resolution_m)
    # Make the vertices
    x_list, y_list, z_list = np.meshgrid(offsets_list, offsets_list, offsets_list, indexing='ij')

    # Fill the cube
    local_points_cube = np.vstack([x_list.ravel(), y_list.ravel(), z_list.ravel()]).T

    # Trim away anything outside the sphere of max distance - this will leave some points with negative elevations for
    # some stations

    points_template = local_points_cube[np.linalg.norm(local_points_cube, axis=1) <= max_distance_to_station_m]

    # Free up some memory
    del x_list, y_list, z_list, offsets_list, local_points_cube
    gc.collect()

    # Create a list of points within elevation range for all stations, without duplicates

    combined_points_array = np.empty((0,3))

    for station in tqdm.tqdm(station_info_dict):

        # Get the ecef information for this station
        station_ecef = station_info_dict[station]['ecef']

        # create local points by shifting template by station origin
        local_points = points_template + station_ecef

        # Combine with points found so far
        combined_points_array = np.vstack([ np.array(filterPointsByElevation(local_points, min_ele_m, max_ele_m)), combined_points_array])

        # Round to resolution and force to integers for speed (not sure if this applies in python)
        indices = (combined_points_array / resolution_m).round().astype(int)

        # Remove duplicates
        combined_points_array = np.array((list(set([tuple(row) for row in indices])))) * resolution_m

    return combined_points_array


def makeECEFPointList(station_info_dict, min_ele_m=20000, max_ele_m=100000, resolution_m = 200000, max_distance_to_station_m=500000):

    print("Making array of coordinates, radius {:.0f}km at resolution {:.1f}km around {} stations".format(max_distance_to_station_m / 1000, resolution_m / 1000, len(station_info_dict)))
    ecef_point_array_around_stations = makeECEFPointListAroundStations(station_info_dict, max_distance_to_station_m, resolution_m, min_ele_m=min_ele_m, max_ele_m=max_ele_m)

    return ecef_point_array_around_stations

def addStationsToECEFArray(ecef_point_array, station_info_dict, radius=50000):


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



    x_list, y_list = [], []
    mask, i = np.zeros(len(vecs_normalised_array), dtype=bool), 0
    for vec_norm, station in zip(vecs_normalised_array, station_name_list):

        station_info = station_info_dict[station]
        pp = station_info['pp']
        mask_struct = station_info['mask']
        lat_degs = np.degrees(station_info['geo']['lat_rads'])
        lon_degs = np.degrees(station_info['geo']['lon_rads'])
        station_ecef = station_info['ecef']

        check_point_az_deg, check_point_alt_deg = ECEF2AltAz(station_ecef, station_ecef + vec_norm)
        check_point_ra_deg, check_point_dec_deg = altAz2RADec(check_point_az_deg, check_point_alt_deg, pp.JD, lat_degs, lon_degs)
        fov_ra, fov_dec = altAz2RADec(pp.az_centre, pp.alt_centre, pp.JD, lat_degs, lon_degs)
        angular_separation = angularSeparationDeg(fov_ra, fov_dec, check_point_ra_deg, check_point_dec_deg)

        if angular_separation > np.hypot(pp.fov_h, pp.fov_v ) / 2:
            visible = False
        else:
            point = station_ecef + vec_norm * 1000
            point_lat_rads, point_lon_rads, point_alt_m  = ecef2LatLonAlt(point[0], point[1], point[2])
            point_lat_degs, point_lon_degs = np.degrees(point_lat_rads), np.degrees(point_lon_rads)
            #print("Notional point location = {:.6f}, {:.6f}, {:.1f}".format(point_lat_degs, point_lon_degs, point_alt_m))
            #print("Station at                {:.6f}, {:.6f}, {:.1f}".format(pp.lat, pp.lon, pp.elev))

            #print("Angular separation {:.2f} x,y:{},{} ".format(angular_separation, x, y))


            x_float_arr, y_float_arr = geoHt2XY(pp, point_lat_degs, point_lon_degs, point_alt_m)
            x_float, y_float = x_float_arr[0], y_float_arr[0]
            x, y = int(x_float), int(y_float)

            #print("Angular separation {:.2f} x,y:{},{} ".format(angular_separation, x, y))

            # Get mask resolution
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


        mask[i] = visible
        i += 1
        if visible:
            x_list.append(x_float)
            y_list.append(y_float)


    return mask, x_list, y_list


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

            random_ecef = np.array((np.random.random() * 100 - 50, np.random.random() * 100 - 50, np.random.random() * 100 - 50))
            random_angle_x = np.random.random() * 0.5 - 0.25
            random_angle_y = np.random.random() * 0.5 - 0.25
            random_angle_x_in_pixels = pp.X_res * random_angle_x / pp.fov_h
            random_angle_y_in_pixels = pp.Y_res * random_angle_y / pp.fov_v
            random_pixel_x = np.random.random() * 3 - 1.5 + random_angle_x_in_pixels
            random_pixel_y = np.random.random() * 3 - 1.5 + random_angle_y_in_pixels

            distance_to_point_m = RMS.Math.dimHypot(observed_point_array, np.array(station_info['ecef']))

            jd_array, ra_deg_array,  dec_deg_array, _ = xyToRaDecPP(np.array([pp.JD]), np.array([x + random_pixel_x]), np.array([y + random_pixel_y]), np.array([1]), pp, jd_time=True, extinction_correction=False, measurement=False)
            az, alt = raDec2AltAz(ra_deg_array, dec_deg_array, pp.JD, pp.lat, pp.lon)
            ecef_point = AER2ECEF(az, alt, distance_to_point_m, pp.lat, pp.lon, pp.elev)
            ecef_point = (ecef_point[0][0], ecef_point[1][0], ecef_point[2][0])
            ecef_sta_pt = vectNorm(ecef_point - np.array(station_info['ecef'] + random_ecef))
            vector_list.append(ecef_sta_pt)


        calculated_position_list, error_matrix = pairwiseTriangulationError(origin_list, vector_list, observed_point_array)
        error_matrix_list.append(error_matrix)

    # This is ugly, but works more quickly
    error_matrix_over_iterations = np.stack(error_matrix_list)

    # Compute mean error over all iterations per combination
    error_matrix = np.mean(error_matrix_over_iterations, axis=0)


    # Exclude errors of station against same station
    non_reflexive_errors = error_matrix[~np.eye(error_matrix.shape[0], dtype=bool)]

    # Compute stats
    mean_error, std_dev = np.mean(non_reflexive_errors), np.std(non_reflexive_errors)
    best_pair = []



    if len(non_reflexive_errors) > 2:

        min_error = np.min(non_reflexive_errors)
        station_pair_indices = np.where(error_matrix == min_error)


        best_pair.append(station_name_list[station_pair_indices[0][0]])
        best_pair.append(station_name_list[station_pair_indices[1][0]])
    elif len(non_reflexive_errors) == 2:
        min_error = np.min(non_reflexive_errors)
        station_pair_indices = np.where(error_matrix == min_error)

        best_pair.append(station_name_list[station_pair_indices[0][0]])
        best_pair.append(station_name_list[station_pair_indices[1][0]])
    else:
        min_error = None

    return calculated_position_list, error_matrix, min_error, best_pair, mean_error, std_dev

def computeOrigin(points_list):

    x_list, y_list, z_list = np.array(points_list).T

    ele_m_list = []
    min_ele_m_initialised = False
    for x, y, z, in zip(x_list, y_list, z_list):
        lat_rads, lon_rads, ele_m = ecef2LatLonAlt(x, y, z)

        if min_ele_m_initialised:
            if  ele_m < min_ele_m:
                min_ele_m = ele_m
                min_x, min_y, min_z = x, y, z
        else:
            min_ele_m = ele_m
            min_x, min_y, min_z = x, y, z
            min_ele_m_initialised = True


    x = np.mean((np.min(x_list), np.max(x_list)))
    y = np.mean((np.min(y_list), np.max(y_list)))
    z = np.mean((np.min(z_list), np.max(z_list)))

    min_r = (min_x ** 2 + min_y ** 2 + min_z ** 2) ** 0.5
    origin_r = (x ** 2 + y ** 2 + z ** 2) ** 0.5

    correction_factor = min_r / origin_r

    x, y, z = x * correction_factor, y * correction_factor, z * correction_factor

    lat_rads, lon_rads, ele_m = ecef2LatLonAlt(x, y, z)

    lat_deg, lon_deg = np.degrees(lat_rads), np.degrees(lon_rads)

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




def computeAnglesPerPoint(station_info_dict, mapping_list):

    mapping_list_with_angles=[]

    output_mapping_list = []
    for observed_point_array, station_ecef_array, station_name_list in tqdm.tqdm(mapping_list):
        # Vectors from stations to reference_point

        point_lat, point_lon, point_ele = ecef2LatLonAlt(observed_point_array[0], observed_point_array[1], observed_point_array[2])


        vectors_array = observed_point_array - station_ecef_array  # shape (N, 3)
        normalisation_array = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        vecs_normalized_array = vectors_array / normalisation_array  # shape (N, 3)
        visible_mask, x_list, y_list = checkVisible(station_info_dict, vecs_normalized_array, station_name_list)


        station_ecef_array = station_ecef_array[visible_mask]
        station_name_list = station_name_list[visible_mask]
        vecs_normalized_array = vecs_normalized_array[visible_mask]

        if not len(station_name_list):
            continue

        # Dot product matrix
        dot_matrix = np.dot(vecs_normalized_array, vecs_normalized_array.T)  # shape (N, N)
        dot_matrix = np.clip(dot_matrix, -1.0, 1.0)  # numerical safety

        # Angle and Score matrix in sin(degrees)
        angle_matrix = np.degrees(np.arccos(dot_matrix))
        sin_score_matrix = np.sin(np.arccos(dot_matrix))  # shape (N, N)
        calculated_position_list, error_matrix, min_error, best_pair, mean_error, sd = \
            getCalculatedPositionErrorFromImageCoordinates(station_info_dict, station_name_list, x_list, y_list, vecs_normalized_array, observed_point_array, iterations = 50)


        if len(station_name_list) > 6:


            if True:
                # Create scatter plot


                ecef_locations = []
                for station_name in station_name_list:
                    station = station_info_dict[station_name]
                    ecef_locations.append(station['ecef'])
                    s_x, s_y, s_z = station['ecef']
                    _,_,ele_m = ecef2LatLonAlt(s_x, s_y, s_z)

                ecef_locations.append([observed_point_array[0], observed_point_array[1], observed_point_array[2]])

                lat_origin_deg, lon_origin_deg, ele_origin_m, x, y, z = computeOrigin(ecef_locations)
                point_lat_rads, point_lon_rads, point_ht = ecef2LatLonAlt(observed_point_array[0], observed_point_array[1], observed_point_array[2])
                e_list, n_list, u_list, labs_list = [], [], [], []

                lat_deg = np.degrees(point_lat_rads)
                lon_deg = np.degrees(point_lon_rads)
                ele_m  =  point_ht
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
                i_1, i_2 = np.where(station_name_list == best_pair[0]), np.where(station_name_list == best_pair[1])
                convergence_angle = angle_matrix[i_1, i_2][0][0]

                plot_title = "Stations and reference point minimum error {:.0f}m with stations {} {} angle {:.1f} degrees".format(min_error, best_pair[0], best_pair[1], convergence_angle)

                plt.title(plot_title)

                plot_file_name = "{}_{}_angle_{}_deg.png".format(best_pair[0], best_pair[1], convergence_angle)
                plt.grid(True)
                plot_dir = os.path.join(WORKING_DIRECTORY, CHARTS)
                mkdirP(plot_dir)
                plot_full_path =  os.path.join(plot_dir, plot_file_name)
                plt.savefig(plot_full_path)
                pass

        output_mapping_list.append([observed_point_array, station_ecef_array, station_name_list, angle_matrix, sin_score_matrix, calculated_position_list,  error_matrix, mean_error, sd])

    return output_mapping_list

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Compute coverage quality of the GMN \
        """, formatter_class=argparse.RawTextHelpFormatter)


    cwd = os.getcwd()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    mkdirP(WORKING_DIRECTORY)

    station_info_dict_path = os.path.join(WORKING_DIRECTORY, "station_info_dict_path.pkl")
    ecef_array_path = os.path.join(WORKING_DIRECTORY, "ecef_point_array_around_stations.npy")
    ecef_point_to_camera_mapping_path = os.path.join(WORKING_DIRECTORY, "ecef_point_to_camera_mapping.pkl")
    station_point_angle_score_mapping_list_path = os.path.join(WORKING_DIRECTORY, "station_point_angle_score_mapping_list.pkl")

    if False:
        station_list = getStationList()
        makeConfigPlateParMaskLib(config, station_list)

    if not os.path.exists(station_info_dict_path):
        station_info_dict = makeStationsInfoDict(config)
        with open(station_info_dict_path, 'wb') as f:
            pickle.dump(station_info_dict, f)

    station_info_dict = pickle.load(open(station_info_dict_path, 'rb'))

    if not os.path.exists(ecef_array_path):
        ecef_point_array = makeECEFPointList(station_info_dict, min_ele_m=20000, max_ele_m=100000, resolution_m=50000)
        np.save(ecef_array_path, ecef_point_array)

    if not os.path.exists(ecef_point_to_camera_mapping_path):
        ecef_point_array = np.load(ecef_array_path)
        ecef_point_to_camera_mapping = addStationsToECEFArray(ecef_point_array, station_info_dict, radius=500000)


        with open(ecef_point_to_camera_mapping_path, 'wb') as f:
            pickle.dump(ecef_point_to_camera_mapping, f)

    ecef_point_to_camera_mapping = pickle.load(open(ecef_point_to_camera_mapping_path, 'rb'))

    if True:
        station_point_angle_score_mapping_list = computeAnglesPerPoint(station_info_dict, ecef_point_to_camera_mapping)

    with open(station_point_angle_score_mapping_list_path, 'wb') as f:
        pickle.dump(station_point_angle_score_mapping_list, f)

    pass