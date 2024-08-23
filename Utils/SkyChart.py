from matplotlib import pyplot as plt
import argparse
import os.path
import tqdm as tqdm
import numpy as np
import subprocess
import RMS.ConfigReader as cr
import shutil
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt, JD2HourAngle, datetime2JD, altAz2RADec, raDec2AltAz
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Formats.Platepar import Platepar
from datetime import datetime, timedelta
from RMS.Misc import mkdirP
from RMS.Math import angularSeparation
from RMS.Formats.FFfile import read as readFF
from RMS.Routines.MaskImage import loadMask, MaskStructure
import ephem
import time
import pickle

import sys
import traceback

if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os
        import ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request


captured_dirs_list_of_lists = None
file_list_of_lists_of_lists = None


def angSepDeg(ra1, dec1, ra2, dec2):

    ra1, dec1, ra2, dec2 =  np.radians(ra1), np.radians(dec1) , np.radians(ra2), np.radians(dec2)
    return np.degrees(angularSeparation(ra1,dec1,ra2,dec2))
def rmsTimeExtractor(rms_time, asTuple = False, asJD = False, delimiter = None):
    """
    General purpose function to convert *20240819*010235*123 | 123456 into a datetime object or JD
    Offsets can be given for the positions of date, time, and fractional seconds, however
    the code will try to parse any string that is given.


    Args:
        rms_time (): Any string containing YYYYMMDD and HHMMSS separated by the delimited
        asJD (): optional, default false, if true return julian date, if false return datetime object

    Returns:
        a datetime object or a julian date number

    """

    rms_time = os.path.basename(rms_time)
    # remove any dots, might be filename extension
    rms_time = rms_time.split(".")[0] if "." in rms_time else rms_time

    # find the delimiter, which is probably the first non alpha numberic character
    if delimiter == None:
        for c in rms_time:
            if c.isnumeric() or c.isalpha():
                continue
            else:
                delim = c
                break

    field_list = rms_time.split(delim)
    field_count = len(field_list)
    str_us = "0"

    consecutive_time_date_fields = 0

    # Parse rms filename, datestring into a date time object
    for field, field_no in zip(field_list, range (0, field_count)):
        field = field.split(".")[0] if "." in field else field
        if field.isnumeric():
            consecutive_time_date_fields += 1

        # Handle year month day
        if consecutive_time_date_fields == 1:
            if len(field) == 8 or len(field) == 6:
                # This looks like a date field so process the date field
                str_date = field_list[field_no]
                if len(str_date) == 8:
                    year, month, day = int(str_date[:4]), int(str_date[4:6]), int(str_date[6:8])
                    dt = datetime(year=int(year), month=int(month), day=int(day))
                # Handle 2 digit year format
                if len(str_date) == 6:
                    year, month, day = 2000 + int(str_date[:2]), int(str_date[2:4]), int(str_date[4:6])
                    dt = datetime(year=int(year), month=month, day=day)
            else:
                dt = 0

        # Handle hour minute second
        if consecutive_time_date_fields == 2:
            if len(field) == 6:
                # Found two consecutive numeric fields followed by a non numeric
                # These are date and time
                str_time = field_list[field_no]
                hour, minute, second = int(str_time[:2]), int(str_time[2:4]), int(str_time[4:6])
                dt = datetime(year, month , day, hour, minute, second)
            elif len(field) == 4:
                str_time = field_list[field_no]
                hour, minute, second = int(str_time[:2]), int(str_time[2:4]), 0
                dt = datetime(year, month, day, hour, minute, second)
            else:
                # if the second field is not of length 6 then reset the counter
                consecutive_time_date_fields = 0

        # Handle fractional seconds
        if consecutive_time_date_fields == 3:
            if field.isnumeric():
                # Convert any arbitrary length next field to microseconds
                us = int(field) * (10 ** (6 - len(field)))
                dt = datetime(year, month, day, hour, minute, second, microsecond=int(us))
                # Stop looping in all cases
                break
            else:
                # Stop looping in call cases
                break


    if asTuple:
        return dt, datetime2JD(dt)

    if asJD:
        return datetime2JD(dt)
    else:
        return dt

def testRmsTimeConverter():

    def testLoop(test_cases):

        for input in test_cases:

            dt = rmsTimeExtractor(input)
            jd = rmsTimeExtractor(input, asJD=True)
            print("Input {} gives {} julian date {}".format(input, dt, jd))



    test_cases = []
    test_cases.append("20240820_1600")
    test_cases.append("20240820_160000")

    test_cases.append("log_AU0006_20000101_120000.365999.log")
    test_cases.append("FF_AU0006_20000101_120000_123.fits")
    test_cases.append("FF_AU0006_20000101_120000_123456.fits")
    test_cases.append("FF_AU0006_000101_120000_123456.fits")
    test_cases.append("/home/au0006/RMS_data/CapturedFiles/AU0006_20240820_101710_566632")
    test_cases.append("/home/au0006/RMS_data/CapturedFiles/AU0006_20240820_101710_566632_1345")

    testLoop(test_cases)

    return 0

def progress(current, total, start_time=None, task_name = "Progress", task_name_space=30, show_time_per_it = True, show_total = True, show_eta=True,
            show_remaining = True, show_eta_date=False, bar_len=20, char="-", pointer=">", newline_at_end=False, iteration_estimate = 60):



    progress = int(bar_len * current / total)
    percent = 100 * current / total

    time_elapsed = time.time() - start_time
    if current == 0:
        time_per_it = iteration_estimate
    else:
        time_per_it = time_elapsed / current

    total_time = time_per_it * total
    its_remaining = total - current
    time_remaining = time_per_it * its_remaining

    if show_eta:
        if show_eta_date:
            eta = datetime.fromtimestamp((time.time() + time_remaining)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            eta = datetime.fromtimestamp((time.time() + time_remaining)).strftime('%H:%M:%S')
    else:
        eta = 10 * "  "

    if task_name_space > len(task_name):
        task_name = "{}{}".format(task_name, " " * (task_name_space - len(task_name)))

    p = "\r{:s} : {:3.0f}%|{}{}{}| {}/{}".format(task_name, percent, char * progress, pointer,
                                                        " " * (len(pointer) + bar_len - progress), current, total, eta)

    if show_time_per_it:
        p += " TPI: {:s}".format(str(timedelta(seconds=round(time_per_it))))

    if show_remaining:
        p += " REM: {:s}".format(str(timedelta(seconds=round(time_remaining))))

    if show_total:
        p += " TOT: {:s}".format(str(timedelta(seconds=round(total_time))))

    if show_eta:
        p += " ETA: {:s}".format(eta)

    if current == total:
        print("")

    current += 1
    if current > total:
        p += "\n"
    return current, p

def timeFromDayLight(file_name):

    pp = ppFromFileName(file_name)
    o = ephem.Observer()
    o.lat, o.long, o.elevation,o.date = str(pp.lat), str(pp.lon), pp.elev, rmsTimeExtractor(file_name)

    o.horizon = '-5:26'
    s = ephem.Sun()
    s.compute()

    time_to_rise = abs(o.date.datetime() - o .next_rising(s).datetime()).total_seconds()
    time_to_set = abs(o.date.datetime() - o .previous_setting(s).datetime()).total_seconds()

    return min(time_to_rise, time_to_set)

def tooCloseToDay(file):

    return timeFromDayLight(file) < 3600 * 1

def removeTooCloseToDay(files_list_in):


    files_list_out = []
    for file in files_list_in:
        if not file.endswith(".fits"):
            files_list_out.append(file)
        else:
            if not tooCloseToDay(file):
                files_list_out.append(file)

    return files_list_out

def createTransformationPPToRADEC(pp_dest):

    x_coords_dest, y_coords_dest = np.meshgrid(np.arange(0, pp_dest.X_res), np.arange(0, pp_dest.Y_res))
    x_coords_dest, y_coords_dest = x_coords_dest.ravel(), y_coords_dest.ravel()
    pixels = pp_dest.X_res * pp_dest.Y_res
    time_arr, level_arr = pixels * [pp_dest.JD], pixels * [1]

    jd_arr, ra_coords, dec_coords, mag = xyToRaDecPP(time_arr, x_coords_dest, y_coords_dest,
                                                     level_arr, pp_dest, jd_time=True, precompute_pointing_corr=True)

    return x_coords_dest, y_coords_dest, ra_coords, dec_coords

def createTransformationRADECtoImage(file_name, ra_coords, dec_coords):


    pp_source = ppFromFileName(file_name)
    x_coords_source, y_coords_source = raDecToXYPP(ra_coords, dec_coords, pp_source.JD, pp_source)

    return x_coords_source, y_coords_source

def createTransformationDestinationPPtoImage(pp_dest, file_name):

    x_coords_dest, y_coords_dest, ra_coords, dec_coords = createTransformationPPToRADEC(pp_dest)
    x_coords_source, y_coords_source = createTransformationRADECtoImage(file_name, ra_coords, dec_coords)

    return x_coords_dest, y_coords_dest, x_coords_source, y_coords_source

def setPPTime(pp,file_name):

    pp.time, pp.JD = rmsTimeExtractor(file_name, asTuple=True)
    pp.Ho = JD2HourAngle(pp.JD)

    return pp

def ppFromFileName(file_name):

    if file_name.endswith(".fits"):
        station = file_name.split("_")[1]
        pp = Platepar()
        pp.read(os.path.join(os.path.expanduser("~/tmp/SkyChart/"),station,"platepar_cmn2010.cal"))
        setPPTime(pp, file_name)
        return pp

def configurePlatepar(ppar, config_list, image_time, az=0, el=90, rot =0, angle=90, res=600, print_values=False):


    ppar.resetDistortionParameters()
    ppar.equal_aspect = True
    ppar.time, ppar.JD = rmsTimeExtractor(image_time), rmsTimeExtractor(image_time, asJD=True)
    ppar.Ho = JD2HourAngle(ppar.JD)
    ppar.X_res, ppar.Y_res, ppar.pos_angle_ref = res, res, rot
    average_lat, average_lon, average_elevation = getAveragePosition(config_list)
    ppar.lat, ppar.lon, ppar.ele = average_lat, average_lon, average_elevation
    ppar.fov_h, ppar.fov_v = angle, angle
    ppar.az_centre, ppar.alt_centre = 0, 90
    ppar.F_scale = res / angle

    # calculate hour angle
    ppar.Ho = JD2HourAngle(ppar.JD)
    ppar.RA_d, ppar.dec_d = altAz2RADec(az, el, ppar.JD, ppar.lat, ppar.lon)

    jd_arr = np.array([ppar.JD])
    x_arr = np.array([-0.5 + ppar.X_res / 2])
    y_arr = np.array([-0.5 + ppar.Y_res / 2])
    mag_arr = np.array([1])


    _, centre_ra_2, centre_dec_2, _ = xyToRaDecPP(jd_arr, x_arr, y_arr, mag_arr, ppar,  jd_time=True)

    ang_sep = angSepDeg(ppar.RA_d, ppar.dec_d, centre_ra_2, centre_dec_2)

    if print_values:
        print(ppar)
        print("Field of view {},{} Resolution {},{}".format(ppar.fov_h, ppar.fov_v, ppar.X_res, ppar.Y_res))
        print("Target RADEC {},{} Platepar RADEC {} {} Angle difference {}"
                .format(ppar.RA_d, ppar.dec_d, centre_ra_2[0],  centre_dec_2[0], ang_sep))



    return ppar

def getAveragePosition(config_list):


    position_list = getPositionsFromConfigs(config_list)

    ecef_list = []
    x_list, y_list, z_list = [], [], []

    for position in position_list:
        lat, lon, ele = position
        lat_rads, lon_rads = np.radians(lat), np.radians(lon)
        x,y,z, = latLonAlt2ECEF(lat_rads,lon_rads,ele)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    average_x, average_y, average_z = np.average(x_list), np.average(y_list), np.average(z_list)
    lat_rads, lon_rads,  ele = ecef2LatLonAlt(average_x, average_y, average_z)
    lat, lon = np.degrees(lat_rads), np.degrees(lon_rads)

    return lat, lon, ele

def downloadIfNotExist(source_directory, file_name, target_directory, print_files=False, force=False):

    """
    Use rsync with compression to download a file from remote

    Args:
        source_directory (): source directory to download from, can include domain, such as au000a@123.456.678.90:
        file_name (): the file_name to download
        target_directory (): the target directory
        print_files (): print debugging information

    Returns:
        Nothing
    """

    file_path = os.path.join(source_directory, file_name)
    final_destination_file = os.path.join(target_directory, file_name)
    if os.path.exists(final_destination_file) and not force:
        if print_files:
            print("Skipping {}, already exists".format(final_destination_file))
    else:
        if print_files:
            print("Downloading {} to {}".format(os.path.join(source_directory, file_name),final_destination_file))
        subprocess.run(["rsync", "-z", file_path, final_destination_file])

def getStations(paths):

    """
    From the paths to stations, try to extract the station name. This is only used before the station .config files
    have been retrieved

    Args:
        paths (): From a list of paths to stations, extract the stations names

    Returns:
        A string with a list of stations for priting
    """


    str = ""
    for path in paths:
        str += "{},".format(path.split("@")[0])

    return str[:-1]

def getConfigsMasksPlatepars(config_file_paths_list,temp_dir="~/tmp/SkyChart"):

    """
    Get the .config files, masks and platepars from the remote machines

    Args:
        config_file_paths_list (): list of paths to coni
        temp_dir ():

    Returns:

    """

    config_list = []
    station_list = []
    temp_dir = os.path.expanduser(temp_dir)


    i, start_time = 0, time.time()
    for config_path in config_file_paths_list:
        i, str = progress(i, len(config_file_paths_list), start_time,  task_name="Get configs", pointer="*>")
        print(str, end="")
        config_filename = os.path.basename(config_path.split(':')[1])
        temp_destination_path_and_filename = os.path.join(temp_dir,config_filename)

        while not os.path.exists(temp_destination_path_and_filename):
            subprocess.run(["rsync", "-z", config_path, temp_destination_path_and_filename])
        config = cr.parse(temp_destination_path_and_filename)
        local_user = os.path.basename(os.path.expanduser("~"))
        if local_user in config.data_dir:
            config.data_dir = config.data_dir.replace(local_user, config.stationID.lower())

        config_list.append(config)
        final_destination = os.path.join(temp_dir,config.stationID)
        mkdirP(final_destination)
        final_destination_name = os.path.join(final_destination, config_filename)
        shutil.move(temp_destination_path_and_filename, final_destination_name)
        station_list.append(config.stationID)
        downloadIfNotExist(os.path.dirname(config_path), config.mask_file, final_destination)
        downloadIfNotExist(os.path.dirname(config_path), config.platepar_name, final_destination)


    return station_list, config_list

def getPositionsFromConfigs(config_list):

    """

    Args:
        config_list (): a list of configuration objects

    Returns:

    """

    position_list = []
    for config in config_list:
        position_list.append([config.latitude, config.longitude, config.elevation])
    return position_list

def getFilePaths(config_file_paths_list, image_time):


    files_path_list = []

    start_time, i = time.time(), 0
    for path, config, dirs_list in config_file_paths_list:
        i, str = progress(i, len(config_file_paths_list), start_time, task_name="Get file paths", show_remaining = True, show_eta = True)
        print(str, end= " ")
        directory = getDirectoryByTime(dirs_list, image_time)
        user_domain = path.split(':')[0]
        target = os.path.join(config.data_dir, config.captured_dir, directory)

        o = subprocess.run(["rsync", "-z", "{}:{}/*.fits".format(user_domain, target)],capture_output=True)
        dir_string = o.stdout.decode("utf-8")
        dir_list = dir_string.split("\n")
        prefix, station_id, suffix = "FF", config.stationID, "fits"
        files_path_list.append([getFilePathsbyTime(dir_list, image_time, target, prefix, station_id, suffix), path, config])

    return files_path_list

def getCapturedDirs(path_list, config_list, reverse=False):

    dir_names_list_of_lists = []
    i, start_time = 0, time.time()
    for path, config in zip(path_list, config_list):
        i, str = progress(i, len(path_list), start_time, "Getting captured directories")
        dir_name_list = []
        user_domain = path.split(':')[0]
        target_dir = os.path.join(config.data_dir, config.captured_dir)
        output = subprocess.run(["rsync", "-z", "{}:{}/".format(user_domain,target_dir)], capture_output=True)
        dir_string = output.stdout.decode("utf-8")
        dir_list = dir_string.split("\n")
        for item in dir_list:
            if len(item.split()) == 5:
                dir_name = item.split()[4]
                if dir_name[0:len(config.stationID)] == config.stationID:
                    dir_name_list.append(dir_name)
        if reverse:
            dir_name_list.reverse()
        else:
            dir_name_list.sort()
        dir_names_list_of_lists.append([path, config, dir_name_list])

    return dir_names_list_of_lists

def getDirectoryByTime(dirs_list, image_time):
    # reverse the dirs_list, and find the first dir before the time
    dirs_list.reverse()
    for directory in dirs_list:

        rms_time = "{}_{}".format(directory.split("_")[1], directory.split("_")[2])
        if rmsTimeExtractor(rms_time) < rmsTimeExtractor(image_time):
            break
    return directory

def getPlatePars(station_list, config_list, temp_dir):

    platepar_list = []

    for station, config in zip(station_list, config_list):
        pp = Platepar()
        pp.read(os.path.join(temp_dir,station,config.platepar_name))
        platepar_list.append(pp)

    return platepar_list

def getFilePathsbyTime(dir_list, image_time, target_directory, prefix, stationID, suffix):

    file_name_list, files_path_list = [], []
    for item in dir_list:
        if len(item.split()) == 5:
            file_name = item.split()[4]
            if file_name.startswith("{}_{}".format(prefix, stationID)) and file_name.endswith(suffix):
                file_dt, image_dt = rmsTimeExtractor(file_name), rmsTimeExtractor(image_time)
                if abs((file_dt - image_dt).total_seconds()) < 30:
                    file_path_name = os.path.join(target_directory, file_name)
                    file_name_list.append(file_path_name)
    files_path_list.append(file_name_list)
    return files_path_list

def retrieveFiles(files_path_lists, temp_dir):


    destination_path_list = []
    start_time, i = time.time(), 0
    for file_paths_list, path, config in files_path_lists:
        i, str = progress(i,len(files_path_lists), start_time, task_name="Retrieve initial files", show_eta=True, show_remaining=True)
        print(str, end=" ")
        destination_path = os.path.join(temp_dir, config.stationID)
        user_domain = path.split(':')[0]
        for file_paths in file_paths_list:
                for file_path in file_paths:
                    file_path, file_name = os.path.dirname(file_path), os.path.basename(file_path)
                    downloadIfNotExist("{}:{}".format(user_domain, file_path), file_name, destination_path)
                    destination_path_list.append(os.path.join(destination_path,file_name))

    return destination_path_list

def createLookUpTable(pp):


    pixels = pp.X_res * pp.Y_res
    x_coords, y_coords = np.meshgrid(np.arange(0, pp.X_res), np.arange(0, pp.Y_res))
    x_coords, y_coords = x_coords.ravel(), y_coords.ravel()
    time_arr, level_arr = pixels * [pp.JD], pixels * [1]

    # Map output image pixels to sky coordinates

    jd_arr, ra_coords, dec_coords, _ = xyToRaDecPP(time_arr, x_coords, y_coords, level_arr ,pp, jd_time=True)

    return (x_coords, y_coords, ra_coords, dec_coords)

def getDeviationsPerFits(r, d, temp_dir, station, files_list, dest_pp, s_pp, corrupted_fits, print_values=False):


    file_name_list, time_deviation_list, angle_deviation_list, station_list = [], [], [], []
    for file_name in files_list:
        if file_name.startswith("FF_{}".format(station)) and file_name.endswith(".fits"):
            if file_name in corrupted_fits:
                continue
            if tooCloseToDay(file_name):
                continue
            file_name_list.append(os.path.join(temp_dir, station, file_name))
            fits_time, fits_time_jd = rmsTimeExtractor(file_name, asTuple=True)
            delta = (dest_pp.time - fits_time).total_seconds()
            station_list.append(station)
            time_deviation_list.append(delta)
            ra_fits, dec_fits = altAz2RADec(s_pp.az_centre, s_pp.alt_centre, fits_time_jd, s_pp.lat, s_pp.lon)
            ang_sep = angSepDeg(r,d,ra_fits, dec_fits)
            angle_deviation_list.append(ang_sep)
            if print_values:
                print("File {} Ra Dec required {},{}, Fits centre {},{} sep {}"
                      .format(file_name, r, d, ra_fits, dec_fits, s_pp.az_centre, s_pp.alt_centre, ang_sep))
    return angle_deviation_list, file_name_list, station_list, time_deviation_list

def findFitsLocal(r,d, station_list, temp_dir,dest_pp, corrupted_fits):

    platepar_list, config_list = [], []

    # should look in memory here first

    az, el = raDec2AltAz(r,d,dest_pp.JD, dest_pp.lat, dest_pp.lon)
    #print("Az,el {},{} r,d {},{}".format(az,el,r,d))



    files_list = []
    angle_deviation_list, file_name_list, sta_list, time_deviation_list = [], [], [], []
    for station in station_list:
        if station not in station_list and False:
            station_list.append(station)


        config_path = os.path.join(temp_dir, station, ".config")
        config = cr.parse(config_path)
        config_list.append(config)
        s_pp = Platepar()
        s_pp.read(os.path.join(temp_dir, station, config.platepar_name))
        platepar_list.append(s_pp)

        files_list = os.listdir(os.path.join(temp_dir, station))

        files_list = removeTooCloseToDay(files_list)

        angle_deviation_list_per_s , file_name_list_per_s, station_list_per_s, time_deviation_list_per_s = \
            getDeviationsPerFits(r, d, temp_dir, station, files_list, dest_pp, s_pp, corrupted_fits, print_values=False )

        angle_deviation_list += angle_deviation_list_per_s
        file_name_list += file_name_list_per_s
        sta_list += station_list_per_s
        time_deviation_list += time_deviation_list_per_s

    return sta_list, file_name_list, time_deviation_list, angle_deviation_list

def checkMaskxy(x,y,file_name, temp_dir):

    station_id = file_name.split("_")[1]
    mask_path_file = os.path.join(os.path.expanduser(temp_dir), station_id, "mask.bmp")
    if os.path.exists(mask_path_file):
        m = loadMask(mask_path_file)
    else:
        return True

    if m.img[y,x] == 255:
        return True
    else:
        return False

def plateparContainsRaDec(r, d, source_pp, dest_pp, file_name, temp_dir, check_mask = True):


    # Get the image time from the file_name
    source_JD = rmsTimeExtractor(file_name, asJD=True)

    # Convert r,d to source image coordinates
    r_array = np.array([r])
    d_array = np.array([d])
    source_x, source_y = raDecToXYPP(r_array, d_array, source_JD, source_pp)
    source_x, source_y = round(source_x[0]), round(source_y[0])

    if 0 < source_x < source_pp.X_res and 0 < source_y < source_pp.Y_res:
        if check_mask:
            if checkMaskxy(source_x,source_y,file_name, temp_dir):

                return True
            else:
                print("Mask obstructs")
                return False
        else:
            return True
    else:
        return False

def lsRemote(path, captured_dir, prefix, suffix, reverse=False):



    user_domain = path.split(':')[0]
    output = subprocess.run(["rsync", "-z", "{}:{}/".format(user_domain, captured_dir)],
                            capture_output=True)
    dir_string = output.stdout.decode("utf-8")
    dir_list = dir_string.split("\n")

    file_name_list = []
    for item in dir_list:
        file_name_field_list = item.split()
        if len(file_name_field_list) > 4:
            file_name = file_name_field_list[4]
            if file_name.startswith(prefix) and file_name.endswith(suffix):
                file_name_list.append(file_name)

    if reverse:
        file_name_list.reverse()
    else:
        file_name_list.sort()

    return file_name_list

def getFitsAllStations(remote_path_list, config_list):

    # use a global variable to give persistence, so we don't have do this more than once
    global captured_dirs_list_of_lists
    global file_list_of_lists_of_lists

    if captured_dirs_list_of_lists is None or file_list_of_lists_of_lists is None:

        captured_dirs_list_of_lists = getCapturedDirs(remote_path_list, config_list, reverse=True)
        file_list_of_lists_of_lists = []
        print("Getting remote directory structure {}".format(len(remote_path_list)))
        for path, captured_dirs_list, config in zip(remote_path_list, captured_dirs_list_of_lists, config_list):
            file_list_of_lists = []

            for captured_dir in tqdm.tqdm(captured_dirs_list):
                captured_dir_full_path = os.path.join(config.data_dir, config.captured_dir, captured_dir)
                files = lsRemote(path, captured_dir_full_path, "FF_{}".format(config.stationID), "fits")
                file_list_of_lists.append(files)

            file_list_of_lists_of_lists.append(file_list_of_lists)

        return captured_dirs_list_of_lists, file_list_of_lists_of_lists

    else:
        return captured_dirs_list_of_lists, file_list_of_lists_of_lists

def searchRaDecCoverage(r,d, station_list, remote_path_list, dest_pp, config_list, temp_dir, corrupted_fits):





    ad_amount_list, ad_file_list, station_list_by_fits, captured_dirs_list_by_fits = [], [], [], []
    pp_source_list = getPlatePars(station_list, config_list, temp_dir)
    captured_dirs_list_of_lists, file_list_of_lists_of_lists = getFitsAllStations(remote_path_list, config_list)
    for station, captured_dirs_list, file_list_of_lists, s_pp, config in zip(station_list, captured_dirs_list_of_lists,
                                                                    file_list_of_lists_of_lists, pp_source_list, config_list):
            for captured_dir, files_list in zip(captured_dirs_list, file_list_of_lists):
                ad_list, file_name_list, station_list_by_dir, time_deviation_list = \
                    getDeviationsPerFits(r, d, temp_dir, station, files_list, dest_pp, s_pp, corrupted_fits, print_values=False)
                for file_name in file_name_list:
                    if file_name in corrupted_fits:
                        continue
                    if tooCloseToDay(file_name):
                        continue

                    ad_file_list.append(os.path.join(config.data_dir, config.captured_dir, captured_dir, os.path.basename(file_name)))
                    captured_dirs_list_by_fits.append(captured_dir)
                ad_amount_list += ad_list
                station_list_by_fits += station_list_by_dir

    if len(ad_amount_list):
        min_ad = min(ad_amount_list)

        ad_file_list_index = ad_amount_list.index(min_ad)
        best_fits = ad_file_list[ad_file_list_index]
        best_station = station_list_by_fits[ad_file_list_index]
        best_captured_dir = captured_dirs_list_by_fits[ad_file_list_index]

    print("Remote file from station {:s} provided best fits {:s} with dev {:.2f} from r, d {:.2f}, {:.2f}".format(best_station, best_fits, min_ad,r,d))
    s_pp = pp_source_list[station_list.index(best_station)]

    if plateparContainsRaDec(r,d, s_pp, dest_pp, best_fits, temp_dir):
        return best_station, best_captured_dir, best_fits, min_ad


    return None, None, None, None

def getIntensities(look_up_table, temp_dir, pp_dest, station_list, remote_path_list, config_list):

    corrupted_fits = []
    x_coords, y_coords, ra, dec = look_up_table
    centre_ra, centre_dec = altAz2RADec(0,90, pp_dest.JD, pp_dest.lat, pp_dest.lon)
    print("Ra,Dec {},{}".format(centre_ra, centre_dec))

    jd_arr = np.array([pp_dest.JD])
    x_arr = np.array([pp_dest.X_res / 2])
    y_arr = np.array([pp_dest.Y_res / 2])
    mag_arr = np.array([1])

    _, centre_ra, centre_dec, _ = xyToRaDecPP(jd_arr, x_arr, y_arr, mag_arr, pp_dest,  jd_time=True)
    print("Ra,Dec {},{}".format(centre_ra, centre_dec))

    loaded_fits_names, loaded_fits, loaded_pp = [], [], []
    loaded_ground_masks, loaded_camera_masks = [], []
    max_pixel_arr = np.zeros(shape=(pp_dest.X_res, pp_dest.Y_res), dtype=int)
    ave_pixel_arr = np.zeros(shape=(pp_dest.X_res, pp_dest.Y_res), dtype=int)
    count_arr = np.zeros(shape=(pp_dest.X_res, pp_dest.Y_res), dtype=int)

    hits, misses = 0,0
    first_iteration = True
    best_file = ""
    start_time = time.time()
    for x, y, r, d in zip(x_coords, y_coords, ra, dec):

        print(progress(hits+misses, pp_dest.X_res * pp_dest.Y_res, start_time,eta=True,eta_date=True, bar_len=30, char="-"), end=" ")

        # First look in memory then the local file store
        sta_list, fn_list, td_list, ad_list = findFitsLocal(r, d, station_list, temp_dir, pp_dest, corrupted_fits)
        if len(ad_list):

            # Find the image locally with the closest centre RaDec to the target
            min_ad = min(ad_list)
            index = ad_list.index(min_ad)
            best_file = os.path.basename(fn_list[index])
            station = sta_list[index]
            loaded_fits, loaded_fits_names, loaded_pp, loaded_camera_masks, loaded_ground_masks, corrupted_fits  =  \
                            loadFits(station, best_file, loaded_fits_names, loaded_pp, loaded_fits,
                                            loaded_camera_masks, loaded_ground_masks, temp_dir, corrupted_fits)
            pp_source = loaded_pp[loaded_fits_names.index(best_file)]

            # If this image does not contain the required RaDec then look remotely
            if not plateparContainsRaDec(r, d, pp_source, pp_dest, best_file,temp_dir):

                fits_time_jd = rmsTimeExtractor(best_file, asJD=True)
                az_fits, el_fits = raDec2AltAz(r, d, fits_time_jd, pp_source.lat, pp_source.lon)
                print("Found local non matching file {} for r,d {:.1f},{:.1f}, x, y {:.1f},{:.1f} az el of {:.1f},{:.1f}"
                            .format(best_file, r, d, x,y, az_fits, el_fits))
                print("Looking for any remote files that could contain the RaDec {:.1f},{:.1f}".format(r,d))

                best_station, best_captured_dir, best_unretrieved_file, best_ad = \
                                            searchRaDecCoverage(r,d, station_list,
                                                    remote_path_list, pp_dest, config_list,
                                                                        temp_dir, corrupted_fits)


                if best_station is None:
                    misses += 1
                    continue

                print("Station {:s} has file {:s}, with angular deviation of {:.1f}"
                            .format(best_station, os.path.basename(best_unretrieved_file), best_ad))

                # Download this file

                station_index = station_list.index(best_station)
                remote_path = remote_path_list[station_index].split(':')[0]
                source_directory = "{}:{}".format(remote_path, os.path.dirname(best_unretrieved_file))

                target_directory = os.path.join(temp_dir, best_station)
                best_file = os.path.basename(best_unretrieved_file)

                downloadIfNotExist(source_directory, best_file, target_directory)
                download_path_and_file = os.path.join(target_directory, best_file)
                loaded_fits, loaded_fits_names, loaded_pp, corrupted_fits, loaded_camera_masks, loaded_ground_masks = \
                        loadFits(best_station, download_path_and_file, loaded_fits_names, loaded_pp, loaded_fits, loaded_camera_masks, loaded_ground_masks, temp_dir, corrupted_fits)

        if best_file == "":
            continue

        # Get the index for this file
        if best_file in loaded_fits_names:
            fits_index = loaded_fits_names.index(best_file)


        # Get the az and el for the centre of this image
        fits_time_jd = rmsTimeExtractor(best_file, asJD=True)
        az_fits, el_fits = raDec2AltAz(r,d, fits_time_jd, pp_source.lat, pp_source.lon)
                #print("Found file {} for r,d {:.1f},{:.1f}, az el for that station of {},{}".format(best_file,r,d, az_fits, el_fits))

        # Convert plain numbers to arrays
        r_array, d_array = np.array([r]), np.array([d])

        # Get the platepar for this image
        s_pp = loaded_pp[fits_index]

        # Get the source image coordinates for this RaDec at the source image time
        source_x, source_y = raDecToXYPP(r_array, d_array, fits_time_jd, s_pp)
        source_x, source_y = round(source_x[0]), round(source_y[0])
        max_pixel = loaded_fits[fits_index].maxpixel
        ave_pixel = loaded_fits[fits_index].avepixel


        # If these source coordinates are within the image bounds, plot a point
        if 0 < source_x < s_pp.X_res and 0 < source_y < s_pp.Y_res:
            print("plotting {},{} intensity {} from {} {},{}".format(x,y,max_pixel[source_y][source_x], best_file, source_x, source_y))
            max_pixel_arr[x, y] = max_pixel[source_y][source_x]
            ave_pixel_arr[x, y] = ave_pixel[source_y][source_x]
            count_arr[x, y] = count_arr[x, y] + 1
            hits += 1
        else:
            misses += 1





    return max_pixel_arr, ave_pixel_arr, count_arr

def createTargetImage(pp_dest):

    pass
    return

def loadFits(station, fits_file, loaded_fits_names, loaded_pp, loaded_fits, loaded_camera_masks, loaded_ground_masks, temp_dir, corrupted_fits):

    fits_file = os.path.basename(fits_file)

    if fits_file in loaded_fits_names:
        pass
        #print("Already loaded {} ".format(fits_file))
    else:
        fits_path = os.path.join(temp_dir, station, fits_file)
        print("Reading in new fits")
        print("Directory {}".format(temp_dir))
        print("Station   {}".format(station))
        print("file name {}".format(fits_file))
        new_fits = readFF(os.path.join(temp_dir, station), fits_file)

        if new_fits == None:
            # Fits is probably corrupted try and redownload
            fits_time = rmsTimeExtractor(fits_file)
            print("Found a corrupted file {}".format(fits_file))
            corrupted_fits.append(fits_file)
            return loaded_fits, loaded_fits_names, loaded_pp, loaded_camera_masks, loaded_ground_masks, corrupted_fits

        loaded_fits.append(new_fits)
        pp_source = Platepar()
        pp_source.read(os.path.join(temp_dir, station, "platepar_cmn2010.cal"))
        loaded_pp.append(pp_source)
        loaded_fits_names.append(fits_file)
        #print("{} fits in memory".format(len(loaded_fits_names)))

    return loaded_fits, loaded_fits_names, loaded_pp, loaded_camera_masks, loaded_ground_masks, corrupted_fits

def addDirToAccountNameHostName(config_file_paths_list, target_dir = "source/RMS/.config"):

    config_file_paths_list_with_dir = []
    for config_file in config_file_paths_list:
        if ":" in config_file:
            config_file_paths_list_with_dir.append(config_file)
        else:
            config_file_paths_list_with_dir.append("{}:{}".format(config_file, target_dir))

    return config_file_paths_list_with_dir

def makeTransformations(pp_dest,file_list):

    transformation_list = []
    start_time, i = time.time(), 0
    for file_name in file_list:
        i, str = progress(i,len(file_list), start_time, task_name="Make transforms")
        print(str, end = " ")
        transformation_list.append([file_name, createTransformationDestinationPPtoImage(pp_dest, file_name)])

    return transformation_list

def makeTransformationPickleFromFileList(pp_dest, file_list, pickle_path=None):

    if pickle_path is None:
        pickle_path = "~/tmp/SkyChart/ContemporaryTransformations.pickle"
    mkdirP(os.path.dirname(pickle_path))
    with open(os.path.expanduser(pickle_path),'wb') as f:
        pickle.dump(pp_dest,f)
        pickle.dump(makeTransformations(pp_dest,file_list), f)

    return pickle_path

def generateContemporaryImageTransformations(config_file_paths_list=None, pickle_path=None, daemon_delay=None, image_time=None, resolution = 300):

    if config_file_paths_list is None:

        return

    if daemon_delay is None:
        print("Not running in daemon mode")
    else:
        print("Running in daemon mode")

    temp_dir = "~/tmp/SkyChart"
    temp_dir = os.path.expanduser(temp_dir)
    mkdirP(temp_dir)

    station_list, config_list = getConfigsMasksPlatepars(config_file_paths_list)

    pp_dest = Platepar()
    pp_dest = configurePlatepar(pp_dest, config_list, image_time, res=resolution, angle=120)

    dirs_list_of_lists = getCapturedDirs(config_file_paths_list, config_list)
    files_path_lists = getFilePaths(dirs_list_of_lists, image_time)
    local_file_path_list = retrieveFiles(files_path_lists,temp_dir)
    pickle_path = makeTransformationPickleFromFileList(pp_dest, local_file_path_list,pickle_path=pickle_path)

    return pickle_path

def createArrayList(pp_dest, count):

    array_list = []
    for n in range(0, count):
        array_list.append(np.zeros(shape=(pp_dest.X_res, pp_dest.Y_res), dtype=int))

    return array_list

def inLimits(x,y,pp):

    return 0 < x < pp.X_res and 0 < y < pp.X_res

def transformPixel(array_list,pp_source, ff, x_s, y_s, x_d, y_d):


    x_s, y_s = round(x_s), round(y_s)
    if inLimits(x_s, y_s, pp_source):
        array_list[0][x_d, y_d] = ff.maxpixel[x_s, y_s]
        array_list[1][x_d, y_d] = ff.avepixel[x_s, y_s]
        array_list[2][x_d, y_d] += 1

    return array_list

def transformFF(pp_dest, file_name, transformation):

    print("Transforming image with resolution {}, {}".format(pp_dest.X_res, pp_dest.Y_res))
    pp_source = ppFromFileName(file_name)
    ff_dir, ff_name = os.path.dirname(file_name), os.path.basename(file_name)
    ff = readFF(ff_dir, ff_name)
    array_list = createArrayList(pp_dest, 3)
    x_d_arr, y_d_arr, x_s_arr, y_s_arr = transformation
    i, start_time = 0, time.time()
    for x_d, y_d, x_s, y_s in zip(x_d_arr, y_d_arr, x_s_arr, y_s_arr):
        array_list = transformPixel(array_list, pp_source, ff, x_s, y_s, x_d, y_d)

    return array_list

def applyTransformations(pickle_path):

    with open(os.path.expanduser(pickle_path),'rb') as f:
        pp_dest = pickle.load(f)
        transformation_list = pickle.load(f)

    [max_pixel_arr, ave_pixel_arr, count_arr] = createArrayList(pp_dest,3)
    i, start_time = 0, time.time()
    for data_list in transformation_list:
        file_name, transformation = data_list
        i, p = progress(i, len(transformation_list), start_time, task_name="{}".format(os.path.basename(file_name)), iteration_estimate=60)
        print(p, end=" ")
        [ff_max_pixel_arr, ff_ave_pixel_arr, ff_count_arr] = transformFF(pp_dest, file_name, transformation)
        max_pixel_arr = ff_max_pixel_arr
        ave_pixel_arr = ff_ave_pixel_arr
        count_arr = ff_count_arr
        display(max_pixel_arr)
        display(ave_pixel_arr)
        display(count_arr)



    with open("image_array", 'wb') as fh:
        pickle.dump(max_pixel_arr, fh)
        pickle.dump(ave_pixel_arr, fh)
        pickle.dump(count_arr, fh)


def display(arr):



    plt.imshow(arr, cmap="gray")
    plt.show()
    pass

def resolveNameToIP(account_name_hostname):
    hostname_list = ['coorinja', 'lemongum', 'pemberton', 'pioneer', 'rhodesdale', 'walnut', 'youndegin']
    ip_address_server = "http://rvrgm.asuscomm.com:8243/stations"
    ip = 0

    account_name, hostname = account_name_hostname.split("@")

    if hostname in hostname_list:
        url_for_ip_address = "{}/{}.ip".format(ip_address_server, hostname)
    else:
        return account_name_hostname

    try:
        if sys.version_info[0] < 3:
            web_page = urllib2.urlopen(url_for_ip_address).read().splitlines()
        else:
            web_page = urllib.request.urlopen(url_for_ip_address).read().decode("utf-8").splitlines()

    except Exception:
        # Return an empty list
        traceback.print_exc()

        return None

    address_list = web_page[0].split()
    for address in address_list:
        if address.startswith("10.8.0"):
            return account_name_hostname.replace(hostname,address)


    print("Returned {}".format(web_page))

    return account_name_hostname

def resolveListToIP(accountname_hostname_list):

    output_list = []
    for accountname_hostname in accountname_hostname_list:
        output_list.append(resolveNameToIP(accountname_hostname))
    return output_list

def testIPResolver():

    print(resolveNameToIP("au0001u@coorinja"))

def testPlatePar():

    pp_dest = Platepar()
    pp_dest.JD = rmsTimeExtractor("20240817_143021", asJD = True)
    pp_dest.time = rmsTimeExtractor("20240817_143021")
    print("Time {} jd {}".format(pp_dest.time, pp_dest.JD))
    pp_dest.X_res, pp_dest.Y_res = 4000, 4000
    pp_dest.lat, pp_dest.lon = -32.354, 115.806
    pp_dest.fov_h, pp_dest.fov_v = 180,180
    pp_dest.az_centre, pp_dest.alt_centre = 0, 90

    pp_dest.F_scale = pp_dest.X_res / pp_dest.fov_v
    T = (pp_dest.JD - 2451545.0) / 36525.0
    pp_dest.Ho = (280.46061837
                  + 360.98564736629 * (pp_dest.JD - 2451545.0)
                  + 0.000387933 * T ** 2
                  - T ** 3 / 38710000.0
                  ) % 360
    pp_dest.pos_angle_ref = 0
    pp_dest.resetDistortionParameters(preserve_centre=True)
    pp_dest.asymmetry_corr = False

    """
    RA_data, dec_data = cyXYToRADec(JD_data, np.array(X_data, dtype=np.float64), \
        np.array(Y_data, dtype=np.float64), float(platepar.lat), float(platepar.lon), float(platepar.X_res), \
        float(platepar.Y_res), float(platepar.Ho), float(platepar.JD), float(platepar.RA_d), 
        float(platepar.dec_d), float(platepar.pos_angle_ref), float(platepar.F_scale), platepar.x_poly_fwd, 
        platepar.y_poly_fwd, unicode(platepar.distortion_type), refraction=platepar.refraction, \
        equal_aspect=platepar.equal_aspect, force_distortion_centre=platepar.force_distortion_centre, \
        asymmetry_corr=platepar.asymmetry_corr, precompute_pointing_corr=precompute_pointing_corr)

    """


    centre_ra, centre_dec = altAz2RADec(0,90, pp_dest.JD, pp_dest.lat, pp_dest.lon)
    print("Ra,Dec {},{}".format(centre_ra, centre_dec))

    pp_dest.RA_d = centre_ra
    pp_dest.dec_d = centre_dec
    jd_arr = np.array([pp_dest.JD])
    x_arr = np.array([pp_dest.X_res / 2])
    y_arr = np.array([pp_dest.Y_res / 2])
    mag_arr = np.array([1])




    _, centre_ra_2, centre_dec_2, _ = xyToRaDecPP(jd_arr, x_arr, y_arr, mag_arr, pp_dest,  jd_time=True)

    pass

if __name__ == '__main__':
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Tool for combining multiple cameras.")

    arg_parser.add_argument('-i','--image_time', type=str,
                            help='Time of the image to be created, YYYYMMDD_HHMMSS')


    arg_parser.add_argument('-d', '--daemon', nargs=1, type=int,
                            help="Run as a daemon, with a break in seconds between each run")


    arg_parser.add_argument('-p', '--paths', type=str, nargs='*',   \
                            help="Paths to the .config files for the cameras to be used")

    arg_parser.add_argument('-r', '--resolution', type=str, nargs=1,   \
                            help="Output image resolution, always square")

    arg_parser.add_argument('-a', '--angle', type=int, \
                            help="Angle of the simulated lens")

    arg_parser.add_argument('-t', '--transformations', default=False, action = 'store_true',
                            help="Generate contemporary transformations")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    print("Time of image {}".format(cml_args.image_time))
    print("Daemon delay {}".format(cml_args.daemon))
    print("Paths {}".format(cml_args.paths))

    if cml_args.image_time is None:
        image_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        print(image_time)
    else:
        image_time = cml_args.image_time

    testRmsTimeConverter()

    if cml_args.resolution is None:
        resolution = 300
    else:
        resolution = int(cml_args.resolution[0])

    pickle_path = "~/tmp/SkyChart/ContemporaryTransformations.pickle"

    #testPlatePar()
    if cml_args.transformations:
        account_name_hostname_list = resolveListToIP(cml_args.paths)
        account_name_hostname_list = addDirToAccountNameHostName(account_name_hostname_list)
        generateContemporaryImageTransformations(config_file_paths_list=account_name_hostname_list,
                                                 image_time=image_time, resolution=resolution, pickle_path=pickle_path)

    applyTransformations(pickle_path=pickle_path)

