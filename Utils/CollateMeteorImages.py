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


import os
import sys
import shutil
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

def fitsToJd(ff_name):
    """
    Convert a fits file name to a julian date.

    Arguments:
        ff_name:[str] name of the fits file.

    Return:
        jd [float] JD time of the file

    """
    fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
    return date2JD(fits_date.year, fits_date.month, fits_date.day, fits_date.hour, fits_date.minute,
                      fits_date.second, fits_date.microsecond / 1000)



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
        ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
        ftp_dict[station] = readFTPdetectinfo(os.path.join(working_directory, station, archived_directory), ftp_file_name)

    return ftp_dict

def findTimeRelatedEvents(detectInfoDict):



    return time_list





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
    for column in image_array[0]:
        if run_in < c < len(image_array[0]) - run_out :
            frame = np.interp(c-run_in, y_coordinate_list, frame_count_list)
            frame_time_stamp = first_frame + datetime.timedelta(seconds=(frame - first_frame_number) / fps)
            time_column_list.append([c, frame_time_stamp])
        c+= 1



    # Infer timing information for run_in and run_out
    first_time_stamp, final_time_stamp = time_column_list[0][1], time_column_list[-1][1]

    seconds_per_column = (final_time_stamp - first_time_stamp).total_seconds() / (c - (run_in + run_out))

    image_start = first_time_stamp - datetime.timedelta(seconds = seconds_per_column * run_in)


    for column in range(0, run_in + 1):
        frame_time_stamp = first_frame + datetime.timedelta(seconds = (column - run_in) * seconds_per_column )
        time_column_list.append([column,frame_time_stamp])
        pass

    for column in range(run_in + len(image_array[0]), run_in + len(image_array[0]) + run_out):
        frame_time_stamp = first_frame + datetime.timedelta(seconds=(column - run_in) * seconds_per_column)
        time_column_list.append([column, frame_time_stamp])
        pass

    time_column_list.sort()

    pass


    return image_data, image_array, time_column_list


def produceCollatedChart(input_directory, run_in=200, run_out=200, y_dim=150, x_image_extent=1000, event_run_in=0.1, event_run_out=0.05):


    if False:
        working_area = createTemporaryWorkArea()

        working_area = extractBz2(input_directory, working_area)
        ftp_dict = readInFTPDetectInfoFiles(working_area)
        events = clusterByTime(ftp_dict)
        event_images_dict = createImagesDict(events, working_area)

        with open('event_images_dict.pkl', 'wb') as f:
            pickle.dump(event_images_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if False:

        with open('event_images_dict.pkl', 'rb') as f:
            event_images_dict = pickle.load(f)


        with open('event_images.pkl', 'wb') as f:
            pickle.dump(event_images_dict[datetime.datetime(2025,6,4,17,9,37, 416028)], f, protocol=pickle.HIGHEST_PROTOCOL)

    if True:
        with open('event_images.pkl', 'rb') as f:
            event_images = pickle.load(f)

        event_images_with_timing_dict = {}
        first_image = True
        for image in event_images:
            original_point_list = []
            ftp_entry = image[0][2][11]
            coordinates_list = []
            observation_end, observation_start = getObservationTimeExtent(coordinates_list, ftp_entry, image)

            if first_image:
                first_image = False
                event_start, event_end = observation_start, observation_end
            else:
                event_start, event_end = min(observation_start, event_start), max(observation_end, event_end)

            angle_deg, img, length, start_x, start_y = getImageInformation(image)
            rotated_coordinates = rotateCoordinateList(coordinates_list, angle_deg)



            img = rotateCapture(img, angle_deg, (start_x, start_y), length, run_in=run_in, run_out=run_out, y_dim=y_dim)
            img = straightenFlight(img, rotated_coordinates, run_in, run_out)
            image_data, image_array, img_timing = addTimingInformation(image, img, rotated_coordinates, run_in, run_out)
            station = image[0][2][1]
            event_images_with_timing_dict[station] = [img_timing, image_data, image_array]

            h, w = img.shape[:2]
            centre_y = int(h/2)
            event_start_time, event_end_time = observation_start.strftime("%H:%M:%S.%f"), observation_end.strftime("%H:%M:%S.%f")
            cv2.putText(img, "{}".format(event_start_time), (run_in, centre_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
            cv2.putText(img, "{}".format(event_end_time), (w - run_out, centre_y + 25)  , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                        1)
            cv2.putText(img, "{}".format(image[0][2][0]), (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200),
                        1)



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


    for time_point in output_column_time_list:
        for image_key in event_images_with_timing_dict:
            print(image_key)
            image_timing = event_images_with_timing_dict[image_key][0]
            image_data   = event_images_with_timing_dict[image_key][1]
            image_array = event_images_with_timing_dict[image_key][2]
            pass

    return event_images_with_timing_dict, event_start, event_end


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