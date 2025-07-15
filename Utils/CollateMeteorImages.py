# RPi Meteor Station
# Copyright (C) 2023
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

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


import os
import sys
import shutil
from types import ClassMethodDescriptorType

import cv2
from matplotlib.testing.decorators import image_comparison

import RMS.Formats.FFfits as FFfits
import datetime
import pickle
import datetime
import time
import dateutil
import glob
import sqlite3
import multiprocessing
import copy
import uuid
import random
import string

from matplotlib import pyplot as plt

from astropy.io.fits.fitstime import fits_to_time

if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os, ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)): 
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request


import numpy as np


import tempfile
import tarfile

from RMS.Astrometry.Conversions import datetime2JD, geo2Cartesian, altAz2RADec, vectNorm, raDec2Vector
from RMS.Astrometry.Conversions import latLonAlt2ECEF, AER2LatLonAlt, AEH2Range, ECEF2AltAz, ecef2LatLonAlt
from RMS.Logger import getLogger
from RMS.Math import angularSeparationVect
from RMS.Formats.FFfile import convertFRNameToFF
from RMS.Formats.Platepar import Platepar
from RMS.UploadManager import uploadSFTP
from Utils.StackFFs import stackFFs
from Utils.FRbinViewer import view
from Utils.BatchFFtoImage import batchFFtoImage
from RMS.CaptureDuration import captureDuration
from RMS.Misc import sanitise, RmsDateTime, mkdirP
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz

log = getLogger("logger")
EM_RAISE = False

def createTemporaryWorkArea():

     #temp_dir = tempfile.TemporaryDirectory()
    temp_dir = os.path.expanduser('~/tmp/collate_working_area')

    return temp_dir


def extractBz2(input_directory, working_directory):

    bz2_list = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".bz2"):
            bz2_list.append(filename)

    bz2_list.sort()
    mkdirP(working_directory)
    for bz2 in bz2_list:
        station_directory = os.path.join(working_directory, bz2.split("_")[0]).lower()
        mkdirP(station_directory)
        bz2_directory = os.path.join(station_directory, bz2.split(".")[0])
        if os.path.exists(bz2_directory):
            continue
        mkdirP(bz2_directory)
        with tarfile.open(os.path.join(input_directory, bz2), 'r:bz2') as tar:
            tar.extractall(path=bz2_directory)

    return working_directory

def readInFTPDetectInfoFiles(working_directory):

    station_directories = sorted(os.listdir((working_directory)))
    archived_directory_list = []
    for station_directory in station_directories:
        extracted_directories_directory_list = os.listdir(os.path.join(working_directory, station_directory))
        if extracted_directories_directory_list is not None:
            archived_directory_list.append(extracted_directories_directory_list[0])



    ftp_dict = {}
    for station, archived_directory in zip(station_directories, archived_directory_list):
        ar_date = archived_directory.split("_")[1]
        ar_time  = archived_directory.split("_")[2]
        ar_milliseconds = archived_directory.split("_")[3]
        #ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
        directory_containing_ftp = os.path.join(working_directory, station, archived_directory)
        for file_name in os.listdir(directory_containing_ftp):
            if file_name.startswith("FTPdetectinfo") and file_name.endswith("manual.txt"):
                ftp_file_name = file_name
                break
        ftp_dict[station] = readFTPdetectinfo(os.path.join(working_directory, station, archived_directory), ftp_file_name)

    return ftp_dict


def clusterByTime(ftp_dict):
    # Rearrange into time
    observations = []
    for station in sorted(ftp_dict):
        for observation in ftp_dict[station]:
            ff_name = observation[0]
            fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
            observation_start_frame = observation[11][0][1]
            observation_end_frame = observation[11][-1][1]
            observation_start_time = fits_date + datetime.timedelta(seconds=observation_start_frame / observation[4])
            observation_end_time = fits_date + datetime.timedelta(seconds=observation_end_frame / observation[4])
            observations.append([observation_start_time, observation_end_time, observation])
    observations = sorted(observations, key=lambda x: x[0])
    events = []
    first_observation = True
    observation_list = []
    for observation in observations:
        observation_start_time = observation[0]
        observation_end_time = observation[1]
        if not first_observation:
            time_gap_seconds = (observation_start_time - _observation_end_time).total_seconds()
            if time_gap_seconds > 1:
                pass

                events.append(sorted(observation_list, key=lambda x:x[0]))
                print("---------------")
                print("camera", observation[2][1])
                print("observation_start_time", observation_start_time)
                print("observation_end_time  ", observation_end_time)
                observation_list = []
                observation_list.append(observation)
            else:
                observation_list.append(observation)
                print("camera", observation[2][1])
                print("observation_start_time", observation_start_time)
                print("observation_end_time  ", observation_end_time)
        else:
            first_observation = False
            observation_list.append(observation)
            print("camera", observation[2][1])
            print("observation_start_time", observation_start_time)
            print("observation_end_time  ", observation_end_time)
            pass
        _observation_end_time = observation_end_time
    events.append(sorted(observation_list))
    pass

    pass
    return events

def createImagesDict(events, working_area):

    events_with_fits_dict = {}
    for event in events:
        event_with_fits = []

        for observation in event:
            fits_file = observation[2][0]
            station_directory = os.path.join(working_area, observation[2][1].lower())
            bz2_directory = os.path.join(station_directory, os.listdir(station_directory)[0])
            if os.path.exists(os.path.join(bz2_directory, fits_file)):
                print("Reading {}".format(fits_file))
                if fits_file.endswith(".bin"):
                    fits_file = fits_file.replace('.bin', '.fits')
                if fits_file.startswith("FR_"):
                    fits_file = fits_file.replace('FR_', 'FF_')
                ff = FFfits.read(bz2_directory, fits_file)
                observation_and_fits = [observation, ff]
                event_with_fits.append(observation_and_fits)


        pass

        if len(event_with_fits) > 2:
            events_with_fits_dict[event[0][0]] = event_with_fits
    return events_with_fits_dict


def rotateCapture(input_image, angle, rotation_centre, length,run_in=100, run_out=100, y_dim = 100):

    # working area
    size = 4000
    axis_centre = size / 2
    image_centre = (axis_centre, axis_centre)
    working_image_dim = (size, size)

    # New image dimensions
    x_range, y_range = length + run_in + run_out, y_dim
    target_start_x, target_start_y  = run_in, y_range / 2

    translate_x = 0 - int(rotation_centre[0] - axis_centre)
    translate_y = 0 - int(rotation_centre[1] - axis_centre)

    translation_matrix = np.float32([[1,0, translate_x], [0,1, translate_y]])

    translated_image = cv2.warpAffine(input_image, translation_matrix, working_image_dim)

    rotation_matrix = cv2.getRotationMatrix2D(image_centre, angle, 1.0)

    rotated_image = cv2.warpAffine(translated_image, rotation_matrix, working_image_dim)

    translate_x = int(run_in - axis_centre)
    translate_y = int(y_dim / 2 - axis_centre)

    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])

    final_img_dim_x = int(length + run_in + run_out)
    final_img_dim_y = int(y_dim)

    final_img_dim = (final_img_dim_x, final_img_dim_y)
    final_image = cv2.warpAffine(rotated_image, translation_matrix, final_img_dim)

    if False:
        plt.imshow(input_image, cmap='gray')
        plt.show()

        plt.imshow(translated_image, cmap='gray')
        plt.show()

        plt.imshow(rotated_image, cmap='gray')
        plt.show()


        plt.imshow(final_image, cmap='gray')
        plt.show()

    # Get rotation matrix

    return final_image

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

def addTimingInformation(image_data, image_array, rotated_coordinates, run_in, run_out):


    first_frame_number = image_data[0][2][11][0][1]
    first_frame = image_data[0][0]

    frame_count_list, y_coordinate_list = [], []
    for frame_count, _, y  in rotated_coordinates:
        frame_count_list.append(frame_count)
        y_coordinate_list.append(y)



    fps = image_data[0][2][4]
    c = 0
    time_column_list = []



    for column in range(run_in, len(image_array.T) - run_out):

        frame = np.interp(c, y_coordinate_list, frame_count_list)

        frame_time_stamp = first_frame + datetime.timedelta(seconds=(frame - first_frame_number) / fps)

        time_column_list.append([c + run_in, frame_time_stamp])
        last_observation_frame_time_stamp = frame_time_stamp
        last_observation_column = column
        c+= 1

    # Infer timing information for run_in and run_out
    first_time_stamp, final_time_stamp = time_column_list[0][1], time_column_list[-1][1]

    observation_duration = (final_time_stamp - first_time_stamp).total_seconds()
    seconds_per_column = observation_duration / c
    print("Observation duration:", observation_duration)
    print("Columns {}".format(c))
    print("Seconds per column:", seconds_per_column)
    image_start = first_time_stamp - datetime.timedelta(seconds = seconds_per_column * run_in)

    print("Length from trace {}".format(len(time_column_list)))
    c = 0
    for column in range(0, run_in + 1):
        frame_time_stamp = image_start + datetime.timedelta(seconds = c * seconds_per_column )
        time_column_list.append([column,frame_time_stamp])
        c += 1
        pass

    print("Length with run_in {}".format(len(time_column_list)))
    c=0
    for column in range(len(image_array[0]) - run_out, len(image_array[0])):
        frame_time_stamp = final_time_stamp + datetime.timedelta(seconds=(c * seconds_per_column))
        time_column_list.append([column, frame_time_stamp])
        c += 1
        pass


    print("Seconds per column {}".format(seconds_per_column))
    print("Length with run in and run out {}".format(len(time_column_list)))
    time_column_list.sort()
    first_col = True
    for col, col_time in time_column_list:
        if first_col:
            _col_time = col_time
            first_col = False
        else:
            col_duration = (col_time - _col_time).total_seconds()
            #print(col, col_time, col_duration)
            _col_time = col_time
    print(len(time_column_list))
    print(run_in, len(image_array[0]), run_out)
    print("Image start {}".format(time_column_list[0]))
    print("Event start {}".format(time_column_list[run_in]))
    print("Event end {}".format(time_column_list[len(image_array[0])]))
    print("Image end {}".format(time_column_list[-1]))


    pass


    return image_data, image_array, time_column_list


def produceCollatedChart(input_directory, run_in=100, run_out=100, y_dim=150, x_image_extent=2000, event_run_in=0.1, event_run_out=0.1):


    if True:
        working_area = createTemporaryWorkArea()

    if False:
        working_area = extractBz2(input_directory, working_area)

    if True:
        ftp_dict = readInFTPDetectInfoFiles(working_area)
        events = clusterByTime(ftp_dict)
        event_images_dict = createImagesDict(events, working_area)

        with open('event_images_dict.pkl', 'wb') as f:
            pickle.dump(event_images_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if True:

        with open('event_images_dict.pkl', 'rb') as f:
            event_images_dict = pickle.load(f)


        with open('event_images.pkl', 'wb') as f:
            pickle.dump(event_images_dict[datetime.datetime(2020,1,9,23,26,39, 186060)], f, protocol=pickle.HIGHEST_PROTOCOL)
            #pickle.dump(event_images_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if True:
        with open('event_images.pkl', 'rb') as f:
            event_images = pickle.load(f)

        event_images_with_timing_dict = {}
        first_image = True

        for image in event_images:
            original_point_list = []
            station = image[0][2][1]
            ff_file = image[0][2][0]
            ftp_entry = image[0][2][11]
            coordinates_list = []
            observation_end, observation_start = getObservationTimeExtent(coordinates_list, ftp_entry, image)


            angle_deg, img, length, start_x, start_y = getImageInformation(image)
            rotated_coordinates = rotateCoordinateList(coordinates_list, angle_deg)



            img = rotateCapture(img, angle_deg, (start_x, start_y), length, run_in=run_in, run_out=run_out, y_dim=y_dim)
            img = straightenFlight(img, rotated_coordinates, run_in, run_out)



            image_data, image_array, img_timing = addTimingInformation(image, img, rotated_coordinates, run_in, run_out)

            if False:
                plt.imshow(image_array, cmap='gray')
                plt.show()

            if first_image:
                first_image = False
                event_start, event_end = observation_start, observation_end
            else:
                event_start, event_end = min(observation_start, event_start), max(observation_end, event_end)



            station = image[0][2][1]
            event_images_with_timing_dict[ff_file] = [img_timing, image_data, image_array]

            h, w = img.shape[:2]
            centre_y = int(h/2)
            event_start_time, event_end_time = observation_start.strftime("%H:%M:%S.%f"), observation_end.strftime("%H:%M:%S.%f")


            #plt.imshow(img, cmap='gray')
            #plt.show()


            pass

    event_duration_seconds = (event_end - event_start).total_seconds()
    columns_per_second = x_image_extent / event_duration_seconds
    chart_x_min_time = event_start - datetime.timedelta(seconds=event_duration_seconds * event_run_in)
    chart_x_max_time = event_end + datetime.timedelta(seconds=event_duration_seconds * event_run_out)
    chart_duration_seconds = (chart_x_max_time - chart_x_min_time).total_seconds()
    chart_x_resolution = round(chart_duration_seconds * columns_per_second)
    number_of_observations = round(len(event_images_with_timing_dict))
    chart_y_resolution = y_dim * number_of_observations
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

    y_origin=0
    y_labels, y_label_coords, plot_annotations_dict = [], [], {}
    for image_key in sorted(event_images_with_timing_dict):

        y_labels.append("{}".format(image_key.split('_')[1]))
        y_label_coords.append(y_origin + 0.5 * y_dim)
        observation_start_time = event_images_with_timing_dict[image_key][1][0][0]
        observation_end_time = event_images_with_timing_dict[image_key][1][0][1]
        row_timing = event_images_with_timing_dict[image_key][0]
        row_timing = [item[1] for item in row_timing]
        print("For image {} minimum {} maximum {}".format(image_key, min(row_timing), max(row_timing)))
        image_data   = event_images_with_timing_dict[image_key][1]
        image_array = event_images_with_timing_dict[image_key][2]

        output_array_transposed = output_array.T
        for y in range(0, y_dim):
            row_intensity = image_array[y]
            for x in range(0, chart_x_resolution):
                output_image_time = output_column_time_list[x]

                brightness = interpolateByTime(output_image_time, row_timing, row_intensity)
                output_array[x, y_origin + y] = brightness

                pass


        pass

        display_array = output_array.T

        h, w = img.shape[:2]
        centre_y = int(h / 2)

        for event in event_images:
            pass


            ff_name = event[0][2][0]
            camera_name = event[0][2][1]
            if camera_name == image_key:

                observation_start_time, observation_end_time = event[0][0], event[0][1]
                print("ff_name                  :{}".format(ff_name))
                print("camera_name              :{}".format(camera_name))
                print("observation_start_time   :{}".format(observation_start_time))
                print("observation_end_time     :{}".format(observation_end_time))
                break




        '''
        plt.imshow(display_array, cmap='gray')
        plt.show()
        '''

        annotation = '{}'.format(image_key)
        print(annotation)
        plot_annotations_dict[(10, y_origin + y_dim - 10)]  = annotation
        annotation = "{}".format(observation_start_time.strftime("%S.%f"))
        c= 0
        for t in output_column_time_list:
            c += 1
            if t > observation_start_time:
                break
        plot_annotations_dict[(c, int(y_origin + y_dim * 0.25 ))] = annotation

        annotation = "{}".format(observation_end_time.strftime("%S.%f"))
        c= 0
        for t in output_column_time_list:
            c += 1
            if t > observation_end_time:
                break
        plot_annotations_dict[(c, int(y_origin + y_dim * 0.75 ))] = annotation


        y_origin += y_dim


    # Plot

    fig, ax = plt.subplots()
    im = ax.imshow(display_array, aspect='auto', cmap='gray')

    # Set x-axis to custom time scale
    earliest_time = output_column_time_list[0]
    latest_time = output_column_time_list[-1]
    duration_seconds = (latest_time - earliest_time).total_seconds()
    if duration_seconds < 2:
        pass
        ticks_per_second = 4
    else:
        pass
        ticks_per_second = 2

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
                print("Tick at {}".format(t_rounded))
        pass
    pass
    tick_positions = []
    for tick in tick_times:
        c = 0
        for col_time in output_column_time_list:

            c += 1
            if col_time >= tick:
                tick_positions.append(c)
                break
    print(tick_positions)
    tick_positions.sort()
    tick_times.sort()
    tick_times = tick_times[0:len(tick_positions)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["{:.2f}".format(float(tick.strftime('%S.%f'))) for tick in tick_times], rotation=90)
    ax.set_xlabel('Time (s)', fontsize=8)

    ax.set_yticks(y_label_coords)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylabel('Station')

    # Optional: format with DateFormatter if using mdates
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    #plt.colorbar(im, ax=ax, label='Intensity')
    plt.title('{}'.format(tick_times[int(len(tick_times)/2)].isoformat()))

    for (x_coord, y_coord), label in plot_annotations_dict.items():
        plt.annotate(
            label,
            xy=(x_coord, y_coord),
            xytext=(x_coord, y_coord),  # offset text slightly
            fontsize=6,
            color=(0.9,0.9,0.9)
        )


    plt.tight_layout()
    plt.show()

    return event_images_with_timing_dict, event_start, event_end


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


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('input_dir', help='Directory containing image frames organized in hour subdirectories')

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
    produceCollatedChart(input_directory)