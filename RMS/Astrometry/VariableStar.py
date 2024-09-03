""" This module contains procedures for collating data from multiple stations in a database
    of magnitudes and can produce a chart of magnitudes close to radec coordinates
"""

# The MIT License

# Copyright (c) 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from RMS.DeleteOldObservations import getNightDirs
import argparse
import copy
import datetime
import os
import shutil
import sys

import numpy as np
# Import Cython functions
import pyximport
import RMS.Formats.Platepar
import scipy.optimize
from RMS.Astrometry.AtmosphericExtinction import \
    atmosphericExtinctionCorrection
from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FTPdetectinfo import (findFTPdetectinfoFile,
                                       readFTPdetectinfo, writeFTPdetectinfo)
from RMS.Math import angularSeparation, cartesianToPolar, polarToCartesian

pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import RMS.ConfigReader as cr
import glob as glob
import sqlite3
import tqdm
import json
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting, photometryFitRobust
from RMS.Misc import rmsTimeExtractor
from RMS.Astrometry.FFTalign import getMiddleTimeFF, alignPlatepar
import matplotlib.pyplot as plt
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent
from RMS.Astrometry.CheckFit import matchStarsResiduals
from RMS.Formats.StarCatalog import readStarCatalog

# Handle Python 2/3 compatibility
if sys.version_info.major == 3:
    unicode = str

EM_RAISE = True

def filterDirectoriesByJD(path, earliest_jd, latest_jd):

    """
    Returns a list of directories inclusive of the earliest and latest jd
    The earliest directory returned will be the first directory dated
    before the earliest jd.
    The latest directory returned will be the last directory dated before
    the latest jd

    Args:
        path (): path to ierate over
        earliest_jd (): directory of the earliest jd to include
        latest_jd (): directory of the latest jd to include

    Returns:
        filtered list of directories
    """

    directory_list = []
    for obj in os.listdir(os.path.expanduser(path)):
        if os.path.isdir(obj):
            directory_list.append(os.path.join(path, obj))

    directory_list.sort(reverse=True)

    filtered_by_jd = []
    for directory in directory_list:

        # If the start time of this directory is less than the latest_target append to the list
        if rmsTimeExtractor(directory, asJD=True) < latest_jd:
            filtered_by_jd.append(directory)

        # As soon as a directory has been added which is before the earliest_jd
        # stop appending break the loop; everything else has already been processed
        if rmsTimeExtractor(directory, asJD=True) < earliest_jd:
            print("Excluding directories before {}, already processed for {}".format(
                                os.path.basename(directory), config.stationID))
            break

    # Sort the list so that the oldest is at the top.
    filtered_by_jd.sort()

    return filtered_by_jd

def readInArchivedCalstars(config, conn):


    """
    Iterates over the ArchivedDirectories for the station to load all
    the calstar files into the database in radec format

    Args:
        config(): config instance
        conn(): database connection instance
    Returns:

    """

    # Load the star catalogue
    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)

    # Deduce the path to the archived directories for this station
    archived_directories_path = os.path.join(config.data_dir, config.archived_dir)
    archived_directories = getNightDirs(archived_directories_path, config.stationID)

    # Reverse this list so that the newest directories are at the front
    archived_directories.reverse()

    # Find the most recent jd in the database for this station
    latest_jd = findMostRecentEntry(config, conn)

    # Initialise the calstar list
    calstar_list, archived_directories_filtered_by_jd = [], []

    # Iterate through the list of archived directories newest first
    # appending to the list of directories to be considered
    for directory in archived_directories:
        archived_directories_filtered_by_jd.append(directory)
        # As soon as a directory has been added which is before the latest_jd
        # stop appending break the loop; everything else has already been processed
        if rmsTimeExtractor(directory, asJD=True) < latest_jd:
            print("Excluding directories before {}, already processed for {}".format(
                                os.path.basename(directory), config.stationID))
            break

    # Reverse the list again, so that the oldest is at the top.
    archived_directories_filtered_by_jd.reverse()

    # Working with each of the remaining archived directories write into the database
    for dir in archived_directories_filtered_by_jd:

        # Get full paths to critical files
        full_path = os.path.join(archived_directories_path, dir)
        full_path_calstars = glob.glob(os.path.join(full_path,"*CALSTARS*" ))
        full_path_platepar = glob.glob(os.path.join(full_path, "platepar_cmn2010.cal"))

        # If no platepar is found or no calstars, then ignore this directory
        if len(full_path_platepar) != 1 or len(full_path_calstars) != 1:
            continue

        full_path_platepar, full_path_calstars = full_path_platepar[0], full_path_calstars[0]
        calstars_path = os.path.dirname(full_path_calstars)
        calstars_name = os.path.basename(full_path_calstars)

        # Read in the CALSTARS file
        calstar = readCALSTARS(calstars_path, calstars_name)

        # Put the CALSTARS list into the database
        calstar_list.append(calstarToDb(calstar, conn, full_path, latest_jd))



def getCatalogueID(r, d, conn, margin=0.3):
    """
    Get the local for the brightest star within margin degrees of passed radec
    Args:
        r ():  right ascension (degreees)
        d (): declination (degrees)
        conn (): database connection
        margin (): optional, default 0.3, degrees margin. This is not a skyarea, simply a box in the
                    interest of computational efficiency

    Returns:
        tuple (id, magnitude, catalogue right ascension, catalogue declination)
    """

    sql_command = ""
    sql_command += "SELECT id, mag, r, d FROM catalogue \n"
    sql_command += "WHERE \n"
    sql_command += "r < {} AND r > {} AND d < {} AND d > {}\n".format(r+margin, r-margin, d+margin, d-margin)
    sql_command += "ORDER BY mag ASC\n"
    id = conn.cursor().execute(sql_command).fetchone()
    if id is not None:
        if len(id):
            return id
        else:
            return 0, 0, 0, 0
    else:
        return 0, 0, 0, 0


def computePhotometry(config, pp_all, calstar, match_radius=2.0, star_margin = 1.2):

    """
    Compute photometric offset and vignetting coefficient from CALSTARS
    Best practice is to to use the vignetting coeffificnt from the platepar
    not a computed number

    Args:
        config (): configuration instance
        pp_all (): a dictionary of all recomputed platepars
        calstar (): calstar data structure
        match_radius (): the pixel radius used by the recalibration routine

    Returns:
        tuple(photometric offset, vignetting coefficient)
    """

    # Extract stars from the catalogue one order of magnitude dimmer than config limit
    lim_mag = config.catalog_mag_limit + 1
    catalog_stars, mag_band_str, config.star_catalog_band_ratios = readStarCatalog(config.star_catalog_path,
                                            config.star_catalog_file, lim_mag=lim_mag,
                                                mag_band_ratios=config.star_catalog_band_ratios)

    # star_dict contains the star data from calstars - indexed by jd
    # ff_name contains the fits file name - indexed by jd
    star_dict, ff_dict = {}, {}
    max_stars = 0
    ff_most_stars = None
    for entry in calstar:
        ff_name, star_data = entry
        d = getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        jd = date2JD(*d)
        star_dict[jd], ff_dict[jd] = star_data, ff_name
        star_count = len(star_dict[jd])
        if star_count > max_stars:
            if ff_name in pp_all:
                max_stars, ff_most_stars, jd_most = star_count, ff_name, jd

    # As the purpose of this code is to get the best magnitude information, discard observation sessions where
    # too few stars were observed, returning none from here will discard the whole observation session
    pp = Platepar()
    if ff_most_stars is None or max_stars < config.min_matched_stars * star_margin:
        print("Too few stars, moving on")
        return None, None

    # Build a list of matched stars for photometry computations
    pp.loadFromDict(pp_all[ff_most_stars])
    n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, pp, catalog_stars,
                                        {jd_most: star_dict[jd_most]}, match_radius, ret_nmatch=True,
                                                                   lim_mag=lim_mag)

    # If jd_most is not in matched stars, then do not use this observation session.
    # This is probably caused by too few stars

    if jd_most not in matched_stars:
        print("Key error, moving on")
        return None, None

    # Split the data return from matched_stars into image stars and catalogue stars
    image_stars, matched_catalog_stars, distances = matched_stars[jd_most]

    # Get the star intensities
    star_intensities = image_stars[:, 2]

    # Transpose the matched_catalog_stars array and extract ra, dec, mag
    cat_ra, cat_dec, cat_mags = matched_catalog_stars.T

    # For every star on the image compute the radius from image centre
    radius_arr = np.hypot(image_stars[:, 0] - pp.Y_res / 2, image_stars[:, 1] - pp.X_res / 2)

    # Correct for extinction
    mag_cat = extinctionCorrectionTrueToApparent(cat_mags, cat_ra, cat_dec, jd, pp)

    # Conduct the photometry fit, probably should do something with the standard deviation
    photom_params, fit_stddev, fit_resid, star_intensities, radius_arr, catalog_mags = \
        photometryFitRobust(star_intensities, radius_arr, mag_cat)

    return photom_params


def getFitsPaths(config, earliest_jd, latest_jd):

    full_path_to_captured = os.path.expanduser(os.path.join(config.data_dir, config.captured_dir))
    directories_to_search = filterDirectoriesByJD(full_path_to_captured, earliest_jd, latest_jd)
    stationID = config.stationID

    fits_paths = []
    for dir in directories_to_search:
        for file_name in os.listdir(dir):
            if file_name.beginswith('FF') and file_name.endswith('.fits') and len(file_name.split('_')) == 5:
                if file_name.split('_')[1] == stationID:
                    fits_paths.append(file_name)

def createThumbnails(config, r, d, earliest_jd, latest_jd):

    paths = getFitsPaths(config, earliest_jd, latest_jd)
    print(paths)
    pass
    return []

def calstarToDb(calstar, conn, archived_directory_path, latest_jd=0):

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

    # Intialise calstar_radec list
    calstar_radec = []

    # Get the path to all the recalibrated platepars for the night and read them in
    platepars_all_recalibrated_path = os.path.join(archived_directory_path, "platepars_all_recalibrated.json")
    with open(platepars_all_recalibrated_path, 'r') as fh:
        pp_recal = json.load(fh)

    # Compute photometry offset and vignetting using the best data from the night
    # vignetting coefficient will be overwritten by platepar value
    offset, vignetting = computePhotometry(config, pp_recal, calstar)

    # If this can't be computed, then probably the night was a poor observation session, so reject all
    if offset is None or vignetting is None:
        print("Nothing found in {}, moving on".format(archived_directory_path))
        return

    # Iterate through the calstar data structure for each image in the whole night
    for fits_file, star_list in tqdm.tqdm(calstar):

        # If too few stars on this specific observation, then ignore
        if len(star_list) < config.min_matched_stars:
            continue

        # Get the data and time of this observation
        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
        # Skip anything which has already been processed
        if jd < latest_jd:
            continue

        # If this fits_file does not have a recalibrated platepar, then skip
        if not fits_file in pp_recal:
            continue

        # If it does, then load the recalibrated platepar for this image
        pp = Platepar()
        pp.loadFromDict(pp_recal[fits_file])

        # Overwrite vignetting coefficient with platepar value
        vignetting = pp.vignetting_coeff
        jd_list, y_list, x_list, bg_list, amp_list, FWHM_list = [], [], [], [], [], []

        # Build up lists of data for this image
        for y, x, bg_intensity, amplitude, FWHM in star_list:
            jd_list.append(jd)
            x_list.append(x)
            y_list.append(y)
            bg_list.append(bg_intensity)
            amp_list.append(amplitude)
            FWHM_list.append(FWHM)

        # Convert to arrays
        jd_arr, x_data, y_data, level_data = np.array(jd_list), np.array(x_list), np.array(y_list), np.array(amp_list)

        # Process data into RaDec and apply magnitude corrections
        jd, ra, dec, mag = xyToRaDecPP(jd_arr, x_data, y_data, level_data, pp,
                                                jd_time=True, extinction_correction=False, measurement=True)

        star_list_radec = []
        for j, x, y, r, d, bg, amp, FWHM, mag in zip(jd, x_list, y_list, ra, dec, bg_list, amp_list, FWHM_list, mag):
            cat_id, cat_mag, cat_r, cat_d = getCatalogueID(r, d, conn)
            az, el = raDec2AltAz(r, d, j, pp.lat, pp.lon)
            radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
            mag = 0 - 2.5 * np.log10(correctVignetting(amp, radius, pp.vignetting_coeff)) + offset
            if mag == np.inf:
                continue
            star_list_radec.append([j, date_time, fits_file, x, y, az, el, r, d, bg, amp,
                                                                    FWHM, mag, cat_id, cat_mag, cat_r, cat_d])
        # Check that we still have enough stars and write to database
        if len(star_list_radec) > config.min_matched_stars:
            insertDB(config, conn, star_list_radec)

        # Add the data to the calstar_radec list
        calstar_radec.append([fits_file, star_list_radec])
    return calstar_radec

def insertDB(config, conn, star_list_radec):
    """
    Write data into the stellar magnitudes database
    Args:
        config (): config instance
        conn (): database connection
        star_list_radec (): star_list in radec format with corrected magnitudes

    Returns:

    """

    for jd, date_time, fits, x, y, az, el, r, d,  bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d in star_list_radec:
        sql_command = ""
        sql_command += "INSERT INTO star_observations \n"
        sql_command += "(jd, date_time, station_id, fits, x, y, az, el, r, d, bg, amp, FWHM, mag, cat_key, cat_mag, cat_r, cat_d )\n"
        sql_command += "VALUES\n"
        sql_command += ("({}, '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})"
                        .format(jd, date_time, config.stationID, fits, x, y, az, el, r, d,
                                                                bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d))
        conn.execute(sql_command)
    conn.commit()

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

    try:
        conn = sqlite3.connect(db_path, timeout=60)
        createTableStarObservations(conn)
        createTableCatalogue(conn)
        return conn

    except:
        return None

def retrieveMagnitudesAroundRaDec(conn, r,d, window=0.5, start_time=None, end_time=None):

    """
    Query the database on conn to find magnitudes around r, d. This might return more than one star
    Args:
        r (): right ascension in degrees
        d (): declination in degrees
        window(): window width in degrees
        start_time (): jd of start
        end_time (): jd of end

    Returns:
        list of tuples (jd, stationID, r, d, amp, mag, cat_mag)
    """
    window = abs(window)
    sql_command = ""
    sql_command += "SELECT jd, station_id, r, d, amp, mag, cat_mag\n"
    sql_command += "FROM star_observations\n"
    sql_command += "WHERE\n"
    sql_command += "r > {} AND r < {} AND\n".format(r - window, r + window, )
    sql_command += "d > {} AND d < {}".format(d - window, d + window)

    values = conn.cursor().execute(sql_command).fetchall()

    return values

def createTableStarObservations(conn):

    """
    If the star_observations table does not exist, then create
    Args:
        conn (): connection to database

    Returns:

    """
    table_name = "star_observations"
    # Returns true if the table exists in the database
    try:
        tables = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' and name = '{}';".format(table_name)).fetchall()

        if len(tables) > 0:
            return conn
    except:
        if EM_RAISE:
            raise
        return None

    sql_command = ""
    sql_command += "CREATE TABLE {} \n".format(table_name)
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    # j, x, y, r, d, bg, amp, FWHM, mag
    sql_command += "jd FLOAT NOT NULL, \n"
    sql_command += "date_time DATETIME NOT NULL, \n"
    sql_command += "station_id TEXT NOT NULL, \n"
    sql_command += "fits TEXT NOT NULL, \n"
    sql_command += "x FLOAT NOT NULL, \n"
    sql_command += "y FLOAT NOT NULL, \n"
    sql_command += "az FLOAT NOT NULL, \n"
    sql_command += "el FLOAT NOT NULL, \n"
    sql_command += "r FLOAT NOT NULL, \n"
    sql_command += "d FLOAT NOT NULL, \n"
    sql_command += "bg FLOAT NOT NULL, \n"
    sql_command += "amp FLOAT NOT NULL, \n"
    sql_command += "FWHM FLOAT NOT NULL, \n"
    sql_command += "mag FLOAT NOT NULL, \n"
    sql_command += "cat_mag FLOAT NOT NULL, \n"
    sql_command += "cat_r FLOAT NOT NULL, \n"
    sql_command += "cat_d FLOAT NOT NULL, \n"
    sql_command += "cat_key INT NOT NULL \n"


    sql_command += ") \n"
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
        if EM_RAISE:
            raise
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

def findMostRecentEntry(config, conn):

    """
    Get the most recent entry for the station id in config object
    in the stellar magnitude database

    Args:
        config (): config instance
        conn (): connection instance

    Returns:
        jd of most recent entry for this station
    """

    sql_command = ""
    sql_command += "SELECT max(jd) FROM star_observations \n"
    sql_command += "WHERE \n"
    sql_command += "station_id = '{}' \n".format(config.stationID)

    jd = conn.cursor().execute(sql_command).fetchone()[0]
    if jd is not None:
        return jd
    else:
        return 0


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

def catalogueToDB(conn):
    """
    Read catalogue into database
    Args:
        conn (): connection instance

    Returns:
        Nothing
    """
    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)
    for star in tqdm.tqdm(catalogue):
        sql_command = "INSERT INTO catalogue (r , d, mag) \n"
        sql_command += "Values ({} , {}, {})".format(star[0], star[1], star[2])
        conn.execute(sql_command)
    conn.commit()

def createPlot(values, r, d, w=0):
    """

    Args:
        values (): list of values to be plotted as (jd, stationID, ra, dec, mag, cat_mag)
        r (): right ascension, used only for title
        d (): declination, used only for title
        w (): window, used only for title

    Returns:

    """

    x_vals, y_vals = [], []
    title = "Plot of magnitudes at RA {} Dec {}, window {}".format(r,d, w)
    for jd, stationID, r, d, amp, mag, cat_mag in values:
        x_vals.append(jd)
        y_vals.append(amp)
    f, ax = plt.subplots()

    plt.title(title)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.xlabel("Julian Date")
    plt.ylim((min(y_vals) * 0.8, max(y_vals) * 1.2))
    ax.scatter(x_vals, y_vals)

    return ax



if __name__ == "__main__":


    # Init the command line arguments parser

    description = "Iterate over archived directories, using the CALSTARS file to generate\n"
    description += "a database of stellar magnitudes against RaDec\n\n"
    description += "For multicamera operation, either start this as a process in each camera\n"
    description += "user account, pointing to the same database location\n"
    description += "Or run multiple proceses in one account, pointing to each cameras config file\n"

    arg_parser = argparse.ArgumentParser(description=description)

    arg_parser.add_argument('-r', '--ra', nargs=1, metavar='RA', type=float,
                            help="Right ascension to plot")

    arg_parser.add_argument('-d', '--dec', nargs=1, metavar='DEC', type=float,
                            help="Declination to plot")

    arg_parser.add_argument('-w', '--window', nargs=1, metavar='WINDOW', type=float,
                            help="Width to plot")

    arg_parser.add_argument("-p", '--dbpath', nargs=1, metavar='DBPATH', type=str,
                            help="Path to Database")

    arg_parser.add_argument("-c", '--config', nargs=1, metavar='CONFIGPATH', type=str,
                            help="Config file to load")

    arg_parser.add_argument("-f", '--format', nargs=1, metavar='FORMAT', type=str,
                            help="Chart output format - default png")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    if cml_args.config is None:
        config_path = "~/source/RMS/.config"
    else:
        config_path = cml_args.config[0]
    config_path = os.path.expanduser(config_path)
    config = cr.parse(config_path)

    if cml_args.format is None:
        format = "png"
    else:
        format = cml_args.format[0]

    if format not in ['png', 'jpg', 'bmp']:
        format = 'png'



    if cml_args.dbpath is None:
        dbpath = "~/RMS_data/magnitudes.db"
    else:
        dbpath = cml_args.dbpath


    dbpath = os.path.expanduser(dbpath)
    conn = getStationStarDBConn(dbpath)
    #createThumbnails(config, 344.4, -29.6)

    if cml_args.ra is None and cml_args.dec is None and cml_args.window is None:
        print("Collecting RaDec Data")

        archived_calstars = readInArchivedCalstars(config, conn)

    else:
        if cml_args.window is None:
            w = 0.1
        else:
            w = cml_args.window[0]
        r, d  = cml_args.ra[0], cml_args.dec[0]
        print("Producing plot around RaDec {}, {} width {}".format(r, d, w))

        values = retrieveMagnitudesAroundRaDec(conn, r, d, window=w)
        ax = createPlot(values, r, d, w)
        ax.plot()
        plt.savefig("magnitudes_at_Ra_{}_Dec_{}_Window_{}.{}".format(r, d, w, format), format=format)

