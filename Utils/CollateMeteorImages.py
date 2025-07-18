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

import os
from curses.ascii import isalnum

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

from RMS.Misc import mkdirP
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from matplotlib import pyplot as plt

def retrieveBz2File(file_name, server):


    return


def lsRemote(host, username, port, remote_path):
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

def extractBz2(input_directory, working_directory):

    bz2_list = []
    input_directory = os.path.expanduser(input_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".bz2"):
            bz2_list.append(filename)

    bz2_list.sort()
    mkdirP(working_directory)
    extractBz2Files(bz2_list, input_directory, working_directory)

    return working_directory

def extractBz2Files(bz2_list, input_directory, working_directory):
    for bz2 in bz2_list:
        station_directory = os.path.join(working_directory, bz2.split("_")[0]).lower()
        mkdirP(station_directory)
        bz2_directory = os.path.join(station_directory, bz2.split(".")[0])
        if os.path.exists(bz2_directory):
            continue
        mkdirP(bz2_directory)
        with tarfile.open(os.path.join(input_directory, bz2), 'r:bz2') as tar:
            tar.extractall(path=bz2_directory)

def readInFTPDetectInfoFiles(working_directory, station_list=None, local_available_directories=None, event_time=None):


    archived_directory_list, station_directories = getArchivedDirectories(working_directory, event_time=event_time, station_list=station_list)
    if local_available_directories is not None:
        archived_directory_list = local_available_directories
    ftp_dict = getFTPFileDictionary(archived_directory_list, station_directories, working_directory, station_list=station_list, event_time=event_time)
    return ftp_dict

def getFTPFileDictionary(archived_directory_list, station_directories, working_directory, station_list=None, event_time=None):

    ftp_dict = {}
    for station, archived_directory in zip(station_directories, archived_directory_list):
        if station_list is not None:
            if not station in station_list:
                continue
            if not event_time is None:
                directory_date = os.path.basename(archived_directory).split("_")[1]
                directory_time = os.path.basename(archived_directory).split("_")[1]
                year, month, day = directory_date[0:4], directory_date[4:6], directory_date[6:8]
                hour, minute, second = directory_time[0:2], directory_time[2:4], directory_time[4:6]
                directory_time_object = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                if abs(event_time - directory_time_object).total_seconds() > 16 * 60 * 60:
                    continue

            ftp_file_name = getFTPFileName(archived_directory, station, working_directory)

            ftp_path = os.path.join(working_directory, station, archived_directory)
            ftp_dict[station] = readFTPdetectinfo(ftp_path, ftp_file_name)

    return ftp_dict

def getFTPFileName(archived_directory, station, working_directory):
    ar_date, ar_time = archived_directory.split("_")[1], archived_directory.split("_")[2]
    ar_milliseconds = archived_directory.split("_")[3]
    ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
    if not os.path.exists(os.path.join(working_directory, station, archived_directory, ftp_file_name)):
        directory_containing_ftp = os.path.join(working_directory, station, archived_directory)
        ftp_file_name = None

        for file_name in os.listdir(directory_containing_ftp):
            if file_name.startswith("FTPdetectinfo") and file_name.endswith(".txt") and "manual" in file_name:
                ftp_file_name = file_name
                return ftp_file_name

        for file_name in os.listdir(directory_containing_ftp):
            if file_name.startswith("FTPdetectinfo") and file_name.endswith(".txt"):
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

def createImagesDict(events, working_area):

    events_with_fits_dict = {}
    for event in events:
        event_with_fits = addFITSToEvent(event, working_area)

        if len(event_with_fits) > 1:
            events_with_fits_dict[event[0][0]] = event_with_fits

    return events_with_fits_dict

def addFITSToEvent(event, working_area):
    event_with_fits = []
    for observation in event:
        fits_file = observation[2][0]
        station_directory = os.path.join(working_area, observation[2][1].lower())
        bz2_directory = os.path.join(station_directory, os.listdir(station_directory)[0])
        if os.path.exists(os.path.join(bz2_directory, fits_file)):

            if fits_file.endswith(".bin"):
                fits_file = fits_file.replace('.bin', '.fits')
            if fits_file.startswith("FR_"):
                fits_file = fits_file.replace('FR_', 'FF_')
            ff = FFfits.read(bz2_directory, fits_file)
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
    bz2_files = []
    for station in station_list:
        remote_path = os.path.join("/home", station.lower(), "files", "processed")
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
        fits_files_list = []
        local_station_path = os.path.expanduser(os.path.join("~/tmp/collate_working_area/", station.lower()))
        if not os.path.exists(local_station_path):
            station_files_to_retrieve.append(station)
            continue
        station_detected_dir_list = os.listdir(local_station_path)
        station_detected_dir_list.sort(reverse=True)
        for detected_dir in station_detected_dir_list:
            detected_dir_date = detected_dir.split("_")[1]
            detected_dir_time = detected_dir.split("_")[2]
            year, month, day = detected_dir_date[0:4], detected_dir_date[4:6], detected_dir_date[6:8]
            hour, minute, second = detected_dir_time[0:2], detected_dir_time[2:4], detected_dir_time[4:6]
            detected_dir_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            if detected_dir_time < event_time:
                detected_dir_full_path = os.path.join("~/tmp/collate_working_area", station.lower(), detected_dir)
                detected_dir_full_path = os.path.expanduser(detected_dir_full_path)
                detected_dir_list = os.listdir(detected_dir_full_path)

                for test_file in detected_dir_list:
                    if test_file.startswith("FF_{}".format(station.upper())) and test_file.endswith(".fits"):
                        fits_files_list.append(test_file)

            fits_files_list.sort(reverse=True)
            file_present_locally = False
            for ff_name in fits_files_list:
                fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
                time_difference_seconds = (fits_date - event_time).total_seconds()

                if time_difference_seconds < 11:
                    file_present_locally = True
                    local_dirs_to_use.append(detected_dir_full_path)
                    print("Not downloading for station {} as {} already downloaded".format(station.lower(), detected_dir))
                    break

            if not file_present_locally:
                station_files_to_retrieve.append(station)

    return station_files_to_retrieve, local_dirs_to_use

def produceCollatedChart(input_directory, run_in=100, run_out=100, y_dim=300, x_image_extent=1000, event_run_in=0.05, event_run_out=0.05, target_file_name=None, show_debug_info=False, station_list=None, event_time=None, duration=None, magnitude=None):


    if station_list is not None and duration is not None and event_time is not None:
        station_list_to_get, local_available_directories =  filesNotAvailableLocally(station_list, event_time)
        remote_path_list = getPathsOfFilesToRetrieve(station_list_to_get, event_time)
        for path in remote_path_list:
            basename = os.path.basename(path)
            local_target = os.path.join(os.path.expanduser("~/RMS_data/bz2files/"), basename)
            if not os.path.exists(local_target):
                print("Downloading {} to {}".format(basename, local_target))
                downloadFile("gmn.uwo.ca", "analysis", 22, path, local_target )


    working_area = createTemporaryWorkArea("~/tmp/collate_working_area")


    working_area = extractBz2("~/RMS_data/bz2files", working_area)


    ftp_dict = readInFTPDetectInfoFiles(working_area, station_list, event_time=event_time)
    events = clusterByTime(ftp_dict, station_list, event_time, duration)
    # trajectory_summary_report = parseTrajectoryReport("~/RMS_data/bz2files/initial_part/20200109_232639_report.txt")
    trajectory_summary_report = {}
    event_images_dict = createImagesDict(events, working_area)

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

def processDatabase(database_path):
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT "Unique trajectory identifier", "Beginning UTC Time", "Duration sec", "Participating Stations", "Peak AbsMag" FROM Trajectories Order by "Beginning UTC Time" ASC')
    for row in cursor:
        uti, utc_time, duration_sec, stations, magnitude = row[0], row[1], row[2], row[3].lower(), row[4]


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

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('input_dir', help='Path to trajectory database')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-o', '--oneshot', dest='one_shot', default=False, action="store_true",
                            help="Run once, and terminate.")

    arg_parser.add_argument('-d', '--deletedb', dest='delete_db', default=False, action="store_true",
                            help="Delete the event_monitor database at initialisation.")

    arg_parser.add_argument('-k', '--keepfiles', dest='keepfiles', default=False, action="store_true",
                            help="Keep working files")

    arg_parser.add_argument('-n', '--noupload', dest='noupload', default=False, action="store_true",
                            help="Do not upload")

    cml_args = arg_parser.parse_args()
    input_directory = os.path.expanduser(cml_args.input_dir)
    mkdirP(os.path.expanduser("~/RMS_data"))
    mkdirP(os.path.expanduser("~/RMS_data/bz2files"))
    mkdirP(os.path.expanduser("~/RMS_data/trajectory_images"))

    processDatabase(os.path.expanduser(input_directory))

