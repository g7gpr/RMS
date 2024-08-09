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


"""
Routine to mirror the daily trajectory files from https://globalmeteornetwork.org/data/traj_summary_data/
and from them create a file with all the trajectories, optionally without the duplicates.

"""


import os
import sys
from html.parser import HTMLParser
from dateutil import parser
import time
from collections import Counter
import operator
from datetime import datetime, timedelta
import hdbscan
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from RMS.EventMonitor import gcDistDeg
from RMS.Misc import mkdirP

from sklearn.datasets import make_blobs




traj_summary_all_filename = "traj_summary_all.txt"
data_directory = os.path.expanduser("~/RMS_data/")
trajectory_summary_directory = os.path.join(data_directory, "traj_summary_data")
daily_directory = os.path.join(trajectory_summary_directory, "daily")
monthly_directory = os.path.join(trajectory_summary_directory, "monthly")
trajectory_summary_all_file = os.path.join(trajectory_summary_directory,traj_summary_all_filename)
query_exports = os.path.join(trajectory_summary_directory, "query_results")
association_results_file = os.path.join(trajectory_summary_directory, "association_results")


def clusterMetric(p1, p2):

    """
    Custom metric for HDBSCAN
    :param p1: data point 1
    :param p2: data point 2

    :return: distance
    """

    vector = clusterDistanceVector(p2, degrees=False) - clusterDistanceVector(p1, degrees=False)
    mag = np.sqrt(vector.dot(vector))

    return mag

def clusterDistanceVector(point, degrees=False):


    Sol_lon, BETgeo, LAMhel, Vgeo = point[0], point[1], point[2], point[3]

    """


    :param Sol_lon:
    :param BETgeo:
    :param sun_centered_ecliptic_longitude:
    :param geocentric_velocity:
    :param degrees: [bool] set true if passing degrees
    :return:

    This function implements the Vector in secton 2.1 of
    Meteor Shower Detection with Density-Based Clustering
    Sugar, A. Moorhead, A. Brown, P. Cooke, W



    """


    # Radians
    if degrees:
        Sol_lon, BETgeo, LAMhel = np.radians([Sol_lon, BETgeo, LAMhel])

    # https://arxiv.org/pdf/1702.02656.pdf section 2.1 DBSCAN Vector

    vector = np.array([np.cos(Sol_lon),
                       np.sin(Sol_lon),
                       np.sin(LAMhel) * np.cos(BETgeo),
                       np.cos(LAMhel) * np.cos(BETgeo),
                       np.sin(BETgeo),
                       Vgeo / 72])

    #

    return vector








def angularSeparationDegrees(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """ Calculates the angle between two points on a sphere.

    Arguments:
        ra1: [float] Right ascension 1 (degrees).
        dec1: [float] Declination 1 (degrees).
        ra2: [float] Right ascension 2 (degrees).
        dec2: [float] Declination 2 (degrees).

    Return:
        [float] Angle between two coordinates (degrees).
    """

    ra1, dec1, ra2, dec2 = np.radians(ra1_deg), np.radians(dec1_deg), np.radians(ra2_deg), np.radians(dec2_deg)

    return np.degrees(np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1)))


def readTrajFileCol(trajectory_file, column, length=0, ignore_line_marker ="#"):

    """

    :param traj_summary_all_file:path to the file to hold the daily files combined
    :param column: [bool] column header to read
    :return: list of values
    """


    value_list=[]
    col_no = getHeaders(trajectory_file).index(column)

    with open(trajectory_file) as input_fh:
        previous_line = ""
        for line in input_fh:
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue
            value = line.split(";")[col_no]
            value_list.append(value)

    if length != 0:
        del value_list[length:]

    return value_list

def readTrajFileMultiCol(trajectory_file, column_list, length=0, ignore_line_marker ="#", convert_to_radians = True, solar_lon_range=None):


    """

    Read a GMN format trajectory summary file into a list of lists, one list for each column name provided in
    column list.

    :param traj_summary_all_file:path to the file to hold the daily files combined
    :param column_list: list of column headers to read
    :param length: number of trajectories to remove from the start of the list
    :param ignore_line_marker: lines starting with this string will be ignored
    :param convert_to_radians: Whether the values with "deg" in the field should be converted to radians
    :return: list of values
    """


    value_list, col_no_list=[],[]

    for column in column_list:
        col_no_list.append(getHeaders(trajectory_file).index(column))

    if solar_lon_range is not None:
        print("Seeking between solar longitudes {} and {}".format(solar_lon_range[0],solar_lon_range[1]))
    with open(trajectory_file) as input_fh:

        for line in tqdm.tqdm(input_fh):
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue

            if solar_lon_range is not None:
                if not float(solar_lon_range[0]) < float(line.split(";")[5]) < float(solar_lon_range[1]):
                    continue

            line_value_list = []
            for col_no, field in zip(col_no_list,column_list):
                value = line.split(";")[col_no]

                if "deg" in field and convert_to_radians:
                    line_value_list.append(np.radians(float(value)))
                else:
                    try:
                        line_value_list.append(float(value))
                    except:
                        line_value_list.append(value)



            value_list.append(line_value_list)

    if length != 0:
        del value_list[length:]

    return value_list



def getHeaders(trajectory_file):

    header_list = []
    traj_summary = open(trajectory_file, 'r')

    headerlinecounter = 0
    for line in traj_summary:
        if line != "\n" and line[0] == '#' and ";" in line and not "---" in line:
            headers = line[1:].split(';')  # get rid of the hash at the front
            if headerlinecounter == 0:
                header_list = [""] * len(headers)
            columncount = 0
            for header in headers:
                header_list[columncount] = str(header_list[columncount]).strip() + " " + str(header).strip()
                if header_list[columncount].strip() == "+/- sigma" and columncount > 1:
                    header_list[columncount] = header_list[columncount - 1].strip() + " " + header_list[
                        columncount].strip()
                else:
                    header_list[columncount] = header_list[columncount].strip()
                columncount += 1
            headerlinecounter += 1

    return header_list



class HTMLStripper(HTMLParser):
    """
    HTML to plain text coverter
    """
    def __init__(self):
        self.buffer = ""
        HTMLParser.__init__(self)

    def handle_data(self,newdata):
        self.buffer += newdata

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

def ifNotExistCreate(path):

    """

    :param path: path to the directory to be created if it does not exist
    :return: nothing

    """

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)

def createTrajectoryDataDirectoryStructure():

    """
    Creates a folder structure in line with the format used at https://globalmeteornetwork.org/data/traj_summary_data/
    :return: nothing
    """

    ifNotExistCreate(data_directory)
    ifNotExistCreate(trajectory_summary_directory)
    ifNotExistCreate(daily_directory)
    ifNotExistCreate(monthly_directory)


def downloadStrippedPage(page):

    """

    :param page: url of page to be converted to text
    :return: text from page referred to by url
    """

    if sys.version_info[0] < 3:
        web_page = urllib2.urlopen(page).read()
    else:
        web_page = urllib.request.urlopen(page).read().decode("utf-8")

    stripper = HTMLStripper()
    stripper.feed(web_page)
    return stripper.buffer

def mirror(force_reload,page,max_downloads):

    """

    :param force_reload: ignore any checks, pull in fresh copy of every file
    :param page: url of index page
    :return: nothing
    """

    print("Mirroring {}".format(page))
    print("Force reload is {}".format(force_reload))
    createTrajectoryDataDirectoryStructure()
    line_list = downloadStrippedPage(page).splitlines()
    file_name_list, date_list, size_list = [] , [], []
    files_downloaded = 0
    for this_line in line_list:

        if this_line == '':
            continue
        if this_line[0:8] == 'Index of':
            continue

        column_list = []
        column_list += this_line.split()
        file_name_list.append(column_list[0])
        date_list.append("{} {}".format(column_list[1],column_list[2]))
        file_size_text = column_list[3]
        file_size_final_character = file_size_text[-1]

        if file_size_final_character.isalpha():
            file_size_multiplier = file_size_final_character
            file_size = int(file_size_text[:-1]) - 1
        else:
            file_size_multiplier = ""
            file_size = int(file_size_text)

        file_size = round(file_size) * 1000 if file_size_multiplier == "K" else file_size
        file_size = round(file_size) * 1000 * 1000 if file_size_multiplier == "M" else file_size
        size_list.append(file_size)

    for file_name, remote_date, remote_size in zip(file_name_list,date_list, size_list):

        url = "{}{}".format(page, file_name)
        remote_mod_time = time.mktime(parser.parse(remote_date).timetuple())
        local_target_file = os.path.expanduser(os.path.join(daily_directory, file_name))

        if force_reload:
            print("Downloading {} because force reload selected".format(url))
            urllib.request.urlretrieve(url, local_target_file)

        if os.path.exists(local_target_file):
            local_mod_time, local_size = os.stat(local_target_file).st_mtime, round(os.stat(local_target_file).st_size)

            if local_mod_time < remote_mod_time:
                print("Re-downloading {} because remote file is newer than local file".format(url))
                urllib.request.urlretrieve(url, local_target_file)

            if local_size < remote_size:
                print("Re-downloading {} because remote file is larger than local file".format(url))
                print("Local size  {}".format(local_size))
                print("Remote size {}".format(remote_size))
                urllib.request.urlretrieve(url, local_target_file)

            with open(local_target_file) as input_handle:
                for line in input_handle:
                    if line[0]=="#" or line == "\n":
                        continue
                    value_count = len(line.split(";"))
                    if value_count != 86:
                        print("Re-downloading {} because file is not correct format".format(url))
                        urllib.request.urlretrieve(url, local_target_file)
                        break


        else:
            print("Downloading new file {}".format(url))
            urllib.request.urlretrieve(url, local_target_file)
            files_downloaded += 1
            if files_downloaded >= max_downloads:
                return

def createFileWithHeaders(destination_file_path, reference_directory):

    """

    Creates a file using the headers from another file

    :param destination_file_path: file to be created
    :param header_source_file: file whose headers are to be used
    :return: file handle of the newly created file
    """

    directory = os.listdir(reference_directory)
    directory.sort()
    header_source_file = os.path.join(reference_directory, directory[-1])
    #print("Using file {} to create headers".format(header_source_file))
    if os.path.exists(destination_file_path):
        os.remove(destination_file_path)
    mkdirP(os.path.dirname(destination_file_path))
    output_handle = open(destination_file_path,"w")

    with open(header_source_file) as input_handle:
        line_no = 0
        for line in input_handle:
            if line != "\n":
                line_no += 1
                if line[0] == "#":
                    output_handle.write(line)
    output_handle.close()
    return destination_file_path

def isDuplicate(line_1,line_2):

    """
    Very crude detection of duplicate trajectories

    :param line_1:first line to be checked
    :param line_2:second line to be checked
    :return:

    """

    line_1_split,line_2_split = line_1.split(";"), line_2.split(";")
    if len(line_1_split) < 71 or len(line_2_split) < 71:
        return False
    lat1_s, lon1_s, lat1_e, lon1_e, time_1 = float(line_1_split[63]),float(line_1_split[65]),float(line_1_split[69]), float(line_1_split[71]), line_1_split[2]
    lat2_s, lon2_s, lat2_e, lon2_e, time_2 = float(line_2_split[63]),float(line_2_split[65]),float(line_2_split[69]), float(line_2_split[71]), line_2_split[2]

    if abs(lat1_s - lat2_s) < 0.1 and abs(lon1_s - lon2_s) < 0.1 and \
            abs(lat1_e - lat2_e) < 0.1 and abs(lon1_e - lon2_e) < 0.1 and \
            (abs(parser.parse(time_1)-parser.parse(time_2))).total_seconds() < 2:

        return True
    else:
        return False

def isBetweenSL(line, sl_start, sl_end):

    """
    Test to see if this trajectory is between sl_start and sl_end

    :param line: line to be tested
    :param sl_start: start solar longitude
    :param sl_end: end solar longitude
    :return: [bool]
    """

    between = False
    line_split = line.split(";")
    sl = float(line_split[5])

    if sl_start < sl_end:
        if sl_start < sl < sl_end:
            between = True
    else:
        if sl_start < sl or sl < sl_end:
            between = True

    return between

def isWithinSkyRadius(line, ra, dec, radius):

    """

    :param line: line to be tested
    :param ra: centre of Ra
    :param dec: centre of Dec
    :param radius: sky radius to be checked
    :return: [bool] True if within radius
    """

    split_line = line.split(";")
    time_traj, ra_traj, dec_traj = split_line[2], float(split_line[7]), float(split_line[9])

    angle = angularSeparationDegrees(ra,dec,ra_traj, dec_traj)

    #print("Time: {}, Ra:{} Dec:{} Angle:{} Radius:{}".format(time_traj, ra_traj, dec_traj,angle, radius))


    return angle < radius


def isBetweenVelocity(line, min_v, max_v):

    """
    
    :param min: 
    :param max: 
    :return: [bool] True if velocity is between maximum and minimum
    
    """
    split_line = line.split(";")
    velocity = float(split_line[15])
    #print("Min: {}, Velocity: {}, Max: {}".format(min_v,velocity,max_v))
    if not min_v < max_v:
        return False
    return min_v < velocity < max_v



def fileAppendFilter(output_fh, file_to_append, ignore_line_marker, drop_duplicates, sl_set, sl_start, sl_end,
                     radius_set, radius_ra, radius_dec, radius_radius, velocity_set, velocity_min, velocity_max, included_count):

    """
    sl_set, sl_start, sl_end, radius_set, radius_ra, radius_dec, radius_radius, velocity_set, velocity_min, velocity_max)

    Append a file, to output_fh without lines starting with ignore_line_marker

    :param output_fh: handle of file to be appended to
    :param file_to_append: file to be appended
    :param ignore_line_marker: marker of any line to be ignored
    :param drop_duplicates: allow drop_duplicates function to control append operation
    :return: output_fh, duplicate_count
    """

    duplicate_count = 0
    with open(file_to_append) as input_fh:
        previous_line = ""
        for line in input_fh:
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue
            if sl_set and not isBetweenSL(line, sl_start, sl_end):
                continue
            if radius_set and not isWithinSkyRadius(line, radius_ra, radius_dec, radius_radius):
                continue
            else:
                pass

            if velocity_set and not isBetweenVelocity(line, velocity_min, velocity_max):
                continue
            if previous_line != "":
                if isDuplicate(previous_line, line):
                    duplicate_count += 1
                    if not drop_duplicates:
                        output_fh.write(line)
                        included_count += 1
                else:
                    output_fh.write(line)
                    included_count += 1

            previous_line = line

    return output_fh, duplicate_count, included_count



def fileAppend(output_full_file_path, file_to_append, ignore_line_marker, drop_duplicates):

    """

    Append a file, to output_fh without lines starting with ignore_line_marker

    :param output_fh: handle of file to be appended to
    :param file_to_append: file to be appended
    :param ignore_line_marker: marker of any line to be ignored
    :param drop_duplicates: allow drop_duplicates function to control append operation
    :return: output_fh, duplicate_count
    """

    with open(output_full_file_path, 'a') as output_fh:
        duplicate_count = 0
        with open(file_to_append) as input_fh:
            previous_line = ""
            for line in input_fh:
                if line[0] == "\n" or line[0] == ignore_line_marker:
                    continue
                if previous_line != "":
                    if isDuplicate(previous_line, line):
                        duplicate_count += 1
                        if not drop_duplicates:
                            output_fh.write(line)
                    else:
                        output_fh.write(line)
                previous_line = line
    return duplicate_count

def filter(sol_lon_centre, sol_lon_dev, ra_centre, ra_dev, dec_centre, dec_dev):

    print(getHeaders("~/RMS_data/traj_summary_data/traj_summary_all.txt"))

    pass


def createAllFile(traj_summary_all_file, drop_duplicates):

    """
    :param traj_summary_all_file:path to the file to hold the daily files combined
    :param drop_duplicates: [bool] detect and drop duplicates
    :return: duplicate_count
    """

    print("\n\n")
    print("Creating {}".format(traj_summary_all_file))

    full_file_path = createFileWithHeaders(traj_summary_all_file,daily_directory)

    directory_list = os.listdir(daily_directory)
    directory_list.sort()

    duplicate_count = 0
    for traj_file in directory_list:
        if traj_file[13:20].isnumeric():
            duplicates = fileAppend(full_file_path,os.path.join(daily_directory,traj_file),"#", drop_duplicates)
            duplicate_count += duplicates
        else:
            print("Not adding {} to the {} file".format(traj_file, traj_summary_all_file))

    print("Found {} suspected duplicate trajectories".format(duplicate_count))
    return duplicate_count

def counttrajectories(target_file, lat,lon, range):

    """
    Perform analysis on trajectories

    :param target_file: file to ba analysed
    :return: statistics on the file
    """

    time_between_southern_trajectories = 0
    total, northern, southern = 0,0,0
    shower_list, camera_combination_list, camera_detections = [], [], []
    camera_dates = {}
    first_trajectory,first_northern, first_southern = True, "",""
    within_range = 0
    count_2024 = 0
    within_range_2024 = 0
    with open (target_file, 'r') as fh:
        for line in fh:
            if line == "\n" or line[0] == "#":
                continue

            split_line = line.split(";")

            if len(split_line) >= 63:
                total += 1
                if float(split_line[63].strip()) > 0:
                    northern += 1
                    if first_northern == "":
                        first_northern = split_line[2].strip()

                if float(split_line[63].strip()) < 0:
                    southern += 1
                    if southern%1000 == 1:
                        if southern > 1:
                                this_southern_1000 = parser.parse(split_line[2].strip())
                                time_between_southern_trajectories = (this_southern_1000 - last_southern_1000).total_seconds() /1000
                        last_southern_1000 = parser.parse(split_line[2].strip())
                    if first_southern == "":
                        first_southern = split_line[2].strip()

            else:
                pass

            if split_line[0][0:4] == "2024":
                count_2024 += 1
                lat1, lon1 = float(split_line[63]), float(split_line[65])
                lat2, lon2 = float(split_line[69]), float(split_line[71])
                distance = gcDistDeg(lat1, lon1, lat, lon)
                if distance < range and split_line[85].strip() != ["AU0002,AU0003"]:
                    within_range_2024 += 1



            if len(split_line) >= 4:
                shower_list.append(split_line[4])

            if len(split_line) >= 2:
                last_date = split_line[2].strip()
                if first_trajectory:
                    first_date = last_date
                    first_trajectory = False

            if len(split_line) >= 85:
                camera_list = split_line[85].strip()
                camera_combination_list.append(camera_list)
                cameras = camera_list.split(",")
                for camera in cameras:
                    camera_detections.append(camera.split("_")[0])
                    camera_dates.update({camera.split("_")[0]: split_line[2]})

            lat1, lon1 = float(split_line[63]), float(split_line[65])
            lat2, lon2 = float(split_line[69]), float(split_line[71])

            distance = gcDistDeg(lat1,lon1,lat,lon)

            if distance < range and split_line[85].strip() != "AU0002,AU0003":

                within_range += 1





    shower_data = Counter(shower_list)
    shower_data.most_common()

    camera_detection_data = Counter(camera_detections)
    camera_detection_data.most_common()

    camera_combination_data = Counter(camera_combination_list)
    camera_combination_data.most_common()

    sorted_camera_dates = sorted(camera_dates.items(), key= operator.itemgetter(1))


    return total,  northern, first_northern, southern, first_southern, \
        shower_data, camera_combination_data, camera_detection_data, \
        first_date, last_date, sorted_camera_dates, time_between_southern_trajectories, within_range, count_2024, within_range_2024

def generateStatistics(target_file, duplicate_count):

    """

    :param target_file: path of file to be analysed
    :param duplicate_count: number of duplicates detected
    :return: nothing
    """


    total_trajectories, \
        northern_hemisphere_trajectories, first_northern, \
        southern_hemisphere_trajectories, first_southern, \
        shower_data, camera_combination_data, camera_detection_data, first_date, last_date, camera_last_data, time_between_southern_trajectories, \
        within_range, total_trajectories_2024, within_range_2024 = counttrajectories(target_file, -32.0, 115.6, 1000)


    print("\n")
    print("Statistics")
    print("\n")

    print("Shower data")
    for shower, count in shower_data.most_common():
        print("{}:{}".format(shower,shower_data[shower]), end = "; ")
    print("\n\n")

    print("Camera yield data")
    for cameras, count in camera_detection_data.most_common():
        if count > 500:
            print("{}:{}".format(cameras, count), end="; ")


    print("\n")
    print("Camera combination data")

    for cameras, count in camera_combination_data.most_common():
        if count > 500:
            print("{}:{}".format(cameras, count), end="; ")

    print("\n\n")
    print("Cameras which have reported during the previous two months, but not within the last week")
    for camera, date in camera_last_data:
        dt_last = parser.parse(date)
        dt_now = datetime.now()

        if (dt_now - dt_last).days > 7 and (dt_now - dt_last).days < 60:
            print("{}:{}".format(camera, date), end="; ")


    print("\n\n")
    print("Summary")
    print("\n")
    print("First trajectory     : {}".format(first_date))
    print("Last trajectory      : {}".format(last_date))
    print("Total trajectories   : {}".format(total_trajectories))
    print("First northern       : {}".format(first_northern))
    print("Northern hemisphere  : {}".format(northern_hemisphere_trajectories))
    print("Within range         : {}".format(within_range))
    if total_trajectories > 0:
        print("% within range total : {}".format(100 * within_range / total_trajectories))
    if southern_hemisphere_trajectories > 0:
        print("% within range south : {}".format(100 * within_range / southern_hemisphere_trajectories))
    print("Total 2024           : {}".format(total_trajectories_2024))
    print("Within range 2024    : {}".format(within_range_2024))
    if total_trajectories_2024 > 0:
        print("Within range 2024    : {}%".format(100 * within_range_2024/ total_trajectories_2024))
    print("First southern       : {}".format(first_southern))
    print("Southern hemisphere  : {}".format(southern_hemisphere_trajectories))
    seconds_to_reach_10000 = (100000 - southern_hemisphere_trajectories) * time_between_southern_trajectories
    print("Days to reach 100000 : {:.0f}".format(seconds_to_reach_10000 / 3600 / 24))
    print("Forecasted time      : {}".format(parser.parse(last_date) + timedelta(seconds = seconds_to_reach_10000)))
    print("Suspected duplicates : {}".format(duplicate_count))
    print("Check total          : {}".format(northern_hemisphere_trajectories + southern_hemisphere_trajectories))




def plot(query_filename):
    print("Plot of {} requested".format(query_filename))
    apparent_radiants = readTrajFileMultiCol(query_filename,["Unique trajectory identifier", "RAgeo deg","DECgeo deg","Participating stations", "Sol lon deg"], convert_to_radians=False)
    uti_list, ra_list, dec_list,participating_list, sol_lon_list = [],[],[], [], []
    for apparent_radiant in apparent_radiants:
        print("UTI: {}".format(str(apparent_radiant[0])))
        uti_list.append(str(apparent_radiant[0])[15:20])
        ra_list.append(apparent_radiant[1])
        dec_list.append(apparent_radiant[2])
        participating_list.append(apparent_radiant[3].strip())
        sol_lon_list.append(apparent_radiant[4])
    for ra, dec in zip(ra_list,dec_list):
        print(ra,dec)

    print(participating_list)
    plt.figure(figsize=(14, 8))
    cm = colormaps['viridis']
    plt.scatter(ra_list,dec_list, c=sol_lon_list, cmap='viridis',  marker="+", s=10)
    for i in range(len(uti_list)):
        #lt.annotate(participating_list[i].split(",")[0], (ra_list[i] -0.25, dec_list[i]+0.02), fontsize=8)
        pass
    query_filename = os.path.splitext(query_filename)[0]
    query_filename_list = os.path.basename(query_filename).split("_")
    min_lon, max_lon = query_filename_list[3],query_filename_list[4]
    min_v, max_v = query_filename_list[12], query_filename_list[14]

    plt.title("Sol lon {}-{}° Velocity {}-{}km/s".format(min_lon,max_lon,min_v,max_v), fontsize=10)
    plt.xlabel('Ra °', fontsize=10)
    plt.ylabel('Dec °', fontsize=10)

    plot_filename = "{}.png".format(query_filename)
    plt.colorbar(label="Sol lon °")
    plt.savefig(plot_filename)


def exportRangeFromQuery(query_SL, query_radius, query_velocity, source_file_path, drop_duplicates = False):
    print("Export solar longitude range {}".format(query_SL))
    print("Export radius          range {}".format(query_radius))
    print("Export velocity        range {}".format(query_velocity))

    sl_set, sl_start, sl_end = False,0,0
    radius_set, radius_ra, radius_dec, radius_radius = False,0,0,0
    velocity_set, velocity_min, velocity_max = False,0,0


    if query_SL != "":
        sl_start = float(query_SL.split()[0])
        sl_end = float(query_SL.split()[1])
        sl_set = True


    if query_radius != "":
        radius_ra = float(query_radius.split()[0])
        radius_dec = float(query_radius.split()[1])
        radius_radius = float(query_radius.split()[2])
        radius_set = True

    if query_velocity != "":
        velocity_min = float(query_velocity.split()[0])
        velocity_max = float(query_velocity.split()[1])
        velocity_set = True

    return sl_set, sl_start, sl_end, radius_set, radius_ra, radius_dec, radius_radius, velocity_set, velocity_min, velocity_max

def createFileHeader(traj_summary_all_file, reference_directory):


    full_file_path = createFileWithHeaders(traj_summary_all_file, reference_directory)


    return full_file_path


def exportRange(sl_start = 0, sl_end = 360, ra=180, dec=0,
                radius=90, v_min = 0, v_max = 200):


    sl_set, radius_set, velocity_set, drop_duplicates = True,True,True,True

    full_file_path, query_export_file_path = createQueryOutputFile(dec, ra, radius, sl_end, sl_start, v_max, v_min)



    directory_list = os.listdir(daily_directory)
    directory_list.sort()

    return query_export_file_path


def notSureWhatThisWillDo(dec, directory_list, drop_duplicates, full_file_path, ra, radius, radius_set, sl_end, sl_set,
                          sl_start, source_file_path, v_max, v_min, velocity_set):
    duplicate_count, traj_count = 0, 0
    for traj_file in directory_list:
        if traj_file[13:20].isnumeric():
            output_fh, duplicates, traj_count = fileAppendFilter(full_file_path,
                                                                 os.path.join(daily_directory, traj_file),
                                                                 "#", drop_duplicates, sl_set, sl_start, sl_end,
                                                                 radius_set, ra,
                                                                 dec, radius, velocity_set, v_min, v_max, traj_count)
            duplicate_count += duplicates

        else:
            print("Not adding {} to the {} file".format(traj_file, source_file_path))

            print("File closed with {} trajectories".format(traj_count))
    return traj_count


def createQueryOutputFile(dec, ra, radius, sl_end, sl_start, v_max, v_min, use_dir_structure=True):

    query_export_file_name = "query_results"
    query_export_file_name += "_sl_{:03d}_{:03d}".format(sl_start, sl_end)
    query_export_file_name += "_Ra_{:03d}_Dec_{:03d}_Radius_{:03d}".format(ra, dec, radius)
    query_export_file_name += "_VMin_{:03d}_VMax_{:03d}".format(v_min, v_max)
    query_export_file_name += ".txt"
    # directory_path = os.path.join(query_exports, "Ra_"+ str(ra), "Dec_" + str(dec), "V_" + str((v_min + v_max) /2))
    # ifNotExistCreate(directory_path)

    if use_dir_structure:
        query_export_file_path = os.path.join(query_exports, str(ra).zfill(3), str(dec).zfill(3), str((v_min + v_max) / 2).zfill(3),
                                          query_export_file_name)
    else:
        query_export_file_path = os.path.join(query_exports,query_export_file_name)
    #print("Creating file {}".format(query_export_file_path))
    full_file_path = createFileWithHeaders(query_export_file_path, os.path.join(daily_directory, daily_directory))

    return full_file_path, query_export_file_path

def createDirectoryStructure(sl_dev, radius, v_dev):

    for sl in range(0,360,2):
        for ra in range(0,360,1):
            for dec in range(-90,+90,5):
                for v in range(5,120,5):
                    sl_start, sl_end  = sl - sl_dev, sl + sl_dev,
                    v_min, v_max = v - v_dev, v + v_dev
                    createQueryOutputFile(dec, ra, radius, sl_start, sl_end, v_max, v_min)


                    #export_filename, trajectory_count = exportRange(trajectory_summary_all_file, sl-sl_dev, sl+sl_dev, ra, dec, radius, v-v_dev, v+v_dev)
                    #print(export_filename,trajectory_count)
                    #file_name = os.path.basename(export_filename)
                    #print("Basename {}".format(file_name))
                    #dir_name = os.path.dirname(export_filename)
                    #print("Directory ",format(dir_name))
                    #new_name = "{:0>8}_{}".format(int(trajectory_count),file_name)
                    #new_path = os.path.join(dir_name, new_name)
                    #print("Moving to {}".format(new_path))
                    #os.rename(export_filename, new_path)



if __name__ == "__main__":

    import argparse



    arg_parser = argparse.ArgumentParser(description="""Download new or changed files from a web page of links \
        """, formatter_class=argparse.RawTextHelpFormatter)



    arg_parser.add_argument('-f', '--force_reload', dest='force_reload', default=False, action="store_true", help="Force download of all files")

    arg_parser.add_argument('-a', '--all', dest='all', default=False, action="store_true", help="Create a trajectory_summary_all file")

    arg_parser.add_argument('-p', '--plot', dest='plot', default=False, action="store_true", help="Create a plot of the query")

    arg_parser.add_argument('-s', '--statistics', dest='statistics', default=False, action="store_true", help="Print statistics at end")

    arg_parser.add_argument('-m', '--max_downloads', dest='max_downloads', default = 7, type=int, help="Maximum number of files to download")

    arg_parser.add_argument('-w', '--webpage', dest='page', default="https://globalmeteornetwork.org/data/traj_summary_data/daily/", type=str, help="Webpage to use")

    arg_parser.add_argument('-n', '--no_download', dest='no_download', default=False, action="store_true", help="Do not download anything")

    arg_parser.add_argument('-d', '--drop_dup', dest='drop_duplicates', default=False, action="store_true", help="Detect duplicates and remove from traj_all file")

    arg_parser.add_argument('--sl', dest='solar_longitude', type=str, default="", help="Solar longitude start and end i.e 256 266")

    arg_parser.add_argument('-r', '--radius', dest='radius', type=str, default="", help="Skyradius around ra dec and radius in degrees, i.e. 113.5 32.3 5")

    arg_parser.add_argument('-v', '--velocity', dest='velocity', type=str, default="", help="Minimum and maximum velocity in km/s, i.e. 33.5 34.5")




    cml_args = arg_parser.parse_args()

    if not cml_args.no_download:
        mirror(cml_args.force_reload,cml_args.page, cml_args.max_downloads)
    if cml_args.all:
        duplicate_count = createAllFile(trajectory_summary_all_file,cml_args.drop_duplicates)
    if cml_args.statistics:
        # generateStatistics(trajectory_summary_all_file,duplicate_count)
        pass

    sl_dev = 3
    radius = 10
    v_dev = 5

    createDirectoryStructure(sl_dev, radius, v_dev)

    if cml_args.solar_longitude != "" or cml_args.radius != "" or cml_args.velocity != "":
        export_filename = exportRange(cml_args.solar_longitude, cml_args.radius, cml_args.velocity, trajectory_summary_all_file)
        print("Created {}".format(export_filename))
        if cml_args.plot:
            plot(export_filename)
    header_list,i = getHeaders(trajectory_summary_all_file),0




    data = readTrajFileMultiCol(trajectory_summary_all_file, ["Sol lon deg","BETgeo deg","LAMhel deg","Vgeo km/s"], convert_to_radians=True, solar_lon_range=[191, 197])

    print("Clustering started at {}".format(datetime.utcnow()))
    clusterer = hdbscan.HDBSCAN(min_samples=60, metric=clusterMetric)
    clusterer.fit(data)
    print("Clustering finished at {}".format(datetime.utcnow()))

    print("Number of clusters {}".format(clusterer.labels_.max()))
    unique_trajectory_identifier_list = readTrajFileCol(trajectory_summary_all_file, "Unique trajectory identifier")
    iau_code_list = readTrajFileCol(trajectory_summary_all_file, "IAU code")
    sol_lon_list = readTrajFileCol(trajectory_summary_all_file, "Sol lon deg")
    ra_geo_list = readTrajFileCol(trajectory_summary_all_file, "RAgeo deg")
    lam_geo_list = readTrajFileCol(trajectory_summary_all_file, "LAMgeo deg")
    bet_geo_list = readTrajFileCol(trajectory_summary_all_file, "BETgeo deg")
    RA_list = readTrajFileCol(trajectory_summary_all_file, "RAgeo deg")
    DEC_list = readTrajFileCol(trajectory_summary_all_file, "DECgeo deg")
    vhel_list = readTrajFileCol(trajectory_summary_all_file, "Vhel km/s")

    print("Writing results to {}".format(association_results_file))
    out_fh = open(association_results_file,"w")
    label_code_list = []
    for unique_trajectory_identifier, cluster_label, cluster_probability, iau_code, sol_lon, lam_geo, bet_geo, vhel in zip(unique_trajectory_identifier_list, clusterer.labels_, clusterer.probabilities_, iau_code_list, sol_lon_list, lam_geo_list, bet_geo_list, vhel_list):
        out_fh.write("{},{},{:8.6f},{},{},{},{},{} \n".format(unique_trajectory_identifier, cluster_label, cluster_probability, iau_code, sol_lon, lam_geo, bet_geo, vhel))
        label_code_list.append("{}:{}".format(str(cluster_label).strip(), str(iau_code).strip()))
    out_fh.close()

    cluster_counts = Counter(clusterer.labels_)
    print(cluster_counts)

    label_code_counts = Counter(label_code_list)
    print(label_code_counts)

    cluster_counts_results_file = os.path.join(trajectory_summary_directory, "cluster_code_results")
    out_fh = open(cluster_counts_results_file, "w")
    for cluster,count in cluster_counts.most_common():
        out_fh.write("{}:{} \n".format(count,cluster))
    out_fh.close

    label_code_counts_results_file = os.path.join(trajectory_summary_directory, "label_code_results")
    out_fh = open(label_code_counts_results_file, "w")
    for label_code,count in label_code_counts.most_common():
        out_fh.write("{}:{} \n".format(count,label_code))
    out_fh.close






