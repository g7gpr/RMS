""" This module contains procedures for detecting variations in star magnitude.
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
from RMS.Astrometry.CyFunctions import (cyraDecToXY, cyTrueRaDec2ApparentAltAz,
                                        cyXYToRADec,
                                        eqRefractionApparentToTrue,
                                        equatorialCoordPrecession)
from RMS.Misc import RmsDateTime
import RMS.ConfigReader as cr
import glob as glob
import pickle
import sqlite3
import tqdm
import json
import logging
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting, photometryFitRobust
from RMS.Misc import rmsTimeExtractor
from RMS.Astrometry.ApplyRecalibrate import recalibrateFF, recalibratePlateparsForFF
from RMS.Logger import initLogging
from RMS.Misc import getRMSStyleFileName
from RMS.Astrometry.FFTalign import getMiddleTimeFF, alignPlatepar
import matplotlib.pyplot as plt
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent
from RMS.Astrometry.CheckFit import matchStarsResiduals
from RMS.Formats.StarCatalog import readStarCatalog

# Handle Python 2/3 compatibility
if sys.version_info.major == 3:
    unicode = str

EM_RAISE = True

def readInArchivedCalstars(config, conn):


    """
    Iterates over the ArchivedDirectories for the station to load all
    the calstar files into memory

    Args:
        config (): config file

    Returns:

    """

    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)
    archived_directories_path = os.path.join(config.data_dir, config.archived_dir)
    archived_directories = getNightDirs(archived_directories_path, config.stationID)
    archived_directories.reverse()


    print("Archived Directories")
    calstar_list = []
    latest_jd = findMostRecentEntry(config, conn)
    archived_directories_filtered_by_jd = []
    for directory in archived_directories:
        archived_directories_filtered_by_jd.append(directory)
        if rmsTimeExtractor(directory, asJD=True) < latest_jd:
            print("Excluding directories before {}, already processed for {}".format(
                                os.path.basename(directory), config.stationID))
            break
    archived_directories_filtered_by_jd.reverse()
    for dir in archived_directories_filtered_by_jd:
        print("Working on {}".format(dir))
        full_path = os.path.join(archived_directories_path, dir)
        full_path_calstars = glob.glob(os.path.join(full_path,"*CALSTARS*" ))
        full_path_platepar = glob.glob(os.path.join(full_path, "platepar_cmn2010.cal"))
        if len(full_path_platepar) != 1 or len(full_path_calstars) != 1:
            continue
        full_path_platepar, full_path_calstars = full_path_platepar[0], full_path_calstars[0]
        print(full_path_calstars)
        print(full_path_platepar)
        calstars_path = os.path.dirname(full_path_calstars)
        calstars_name = os.path.basename(full_path_calstars)
        calstar = readCALSTARS(calstars_path, calstars_name)



        calstar_list.append(convertRaDec(calstar, conn, catalogue, full_path, latest_jd))



def getCatalogueID(r, d, conn, margin=0.3):

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


def photometry(config, pp_all, calstar, match_radius = 2.0):


    lim_mag = config.catalog_mag_limit + 1

    catalog_stars, mag_band_str, config.star_catalog_band_ratios = readStarCatalog(config.star_catalog_path,
                                            config.star_catalog_file, lim_mag=lim_mag,
                                                mag_band_ratios=config.star_catalog_band_ratios)

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

    pp = Platepar()
    if ff_most_stars is None or max_stars < config.min_matched_stars:
        print("Too few stars, moving on")
        return None, None
    if ff_most_stars is not in pp_all:
        print("Key error, moving on")
        return None, None
    pp.loadFromDict(pp_all[ff_most_stars])
    n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, pp, catalog_stars,
                                        {jd_most: star_dict[jd_most]}, match_radius, ret_nmatch=True,
                                                                   lim_mag=lim_mag)

    print("jd with most stars {} as date {}".format(jd_most, jd2Date(jd, dt_obj=True)))
    image_stars, matched_catalog_stars, distances = matched_stars[jd_most]
    star_intensities = image_stars[:, 2]
    cat_ra, cat_dec, cat_mags = matched_catalog_stars.T
    radius_arr = np.hypot(image_stars[:, 0] - pp.Y_res / 2, image_stars[:, 1] - pp.X_res / 2)
    mag_cat = extinctionCorrectionTrueToApparent(cat_mags, cat_ra, cat_dec, jd, pp)

    photom_params, fit_stddev, fit_resid, star_intensities, radius_arr, catalog_mags = \
        photometryFitRobust(star_intensities, radius_arr, mag_cat)

    return photom_params

def convertRaDec(calstar, conn, catalogue, archived_directories_path, latest_jd=0):

    """
    Parses a calstar data structure, retains all existing data but
    uses the

    Args:
        calstar (): a calstar data structure


    Returns:

    """

    calstar_radec = []

    platepars_all_recalibrated_path = os.path.join(archived_directories_path, "platepars_all_recalibrated.json")
    with open(platepars_all_recalibrated_path, 'r') as fh:
        pp_recal = json.load(fh)

    offset, vignetting = photometry(config, pp_recal, calstar)
    if offset is None or vignetting is None:
        print("Nothing found in {}, moving on".format(archived_directories_path))
        return
    for fits_file, star_list in tqdm.tqdm(calstar):
        if len(star_list) < config.min_matched_stars:
            continue
        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
        # Skip anything which has already been processed
        if jd < latest_jd:
            continue
        if not fits_file in pp_recal:
            continue
        pp = Platepar()
        pp.loadFromDict(pp_recal[fits_file])

        jd_list, y_list, x_list, bg_list, amp_list, FWHM_list = [], [], [], [], [], []
        for y, x, bg_intensity, amplitude, FWHM in star_list:

            jd_list.append(jd)
            x_list.append(x)
            y_list.append(y)
            bg_list.append(bg_intensity)
            amp_list.append(amplitude)
            FWHM_list.append(FWHM)
        jd_arr, x_data, y_data, level_data = np.array(jd_list), np.array(x_list), np.array(y_list), np.array(amp_list)

        jd, ra, dec, mag = xyToRaDecPP(jd_arr, x_data, y_data, level_data, pp, jd_time=True, extinction_correction=False, measurement=True)
        star_list_radec = []
        for j, x, y, r, d, bg, amp, FWHM, mag in zip(jd, x_list, y_list, ra, dec, bg_list, amp_list, FWHM_list, mag):
            cat_id, cat_mag, cat_r, cat_d = getCatalogueID(r, d, conn)
            az, el = raDec2AltAz(r, d, j, pp.lat, pp.lon)
            radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
            mag = 0 - 2.5 * np.log10(correctVignetting(amp, radius, vignetting)) + offset
            if mag == np.inf:
                continue
            star_list_radec.append([j, date_time, fits_file, x, y, az, el, r, d, bg, amp,
                                                                    FWHM, mag, cat_id, cat_mag, cat_r, cat_d])
        if len(star_list_radec) > config.min_matched_stars:
            insertDB(config, conn, star_list_radec)
        calstar_radec.append([fits_file, star_list_radec])
    return calstar_radec

def insertDB(config, conn, star_list_radec):


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
    """ Creates the station star database. Tries only once.

    arguments:
        config: config file
        force_delete: if set then deletes the database before recreating

    returns:
        conn: [connection] connection to database if success else None

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

    Args:
        r (): right ascension in degrees
        d (): declination in degrees
        window(): window width in degrees
        start_time (): jd of start
        end_time (): jd of end

    Returns:

    """
    window = abs(window)
    sql_command = ""
    sql_command += "SELECT jd, station_id, r, d, mag, cat_mag\n"
    sql_command += "FROM star_observations\n"
    sql_command += "WHERE\n"
    sql_command += "r > {} AND r < {} AND\n".format(r - window, r + window, )
    sql_command += "d > {} AND d < {}".format(d - window, d + window)

    values = conn.cursor().execute(sql_command).fetchall()

    return values

def createTableStarObservations(conn):

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

    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)
    for star in tqdm.tqdm(catalogue):
        sql_command = "INSERT INTO catalogue (r , d, mag) \n"
        sql_command += "Values ({} , {}, {})".format(star[0], star[1], star[2])
        conn.execute(sql_command)
    conn.commit()

def createPlot(values, r, d, w):


    x_vals, y_vals = [], []
    title = "Plot of magnitudes at RA {} Dec {}".format(r,d)
    for jd, stationID, r, d, mag, cat_mag in values:
        x_vals.append(jd)
        y_vals.append(mag)
    f, ax = plt.subplots()

    plt.title(title)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.xlabel("Julian Date")
    plt.ylim((12,0))
    ax.invert_yaxis()
    ax.scatter( x_vals,y_vals)
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

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    if cml_args.config is None:
        config_path = "~/source/RMS/.config"
    else:
        config_path = cml_args.config[0]
    config_path = os.path.expanduser(config_path)

    if cml_args.dbpath is None:
        dbpath = "~/RMS_data/magnitudes.db"
    else:
        dbpath = cml_args.dbpath

    dbpath = os.path.expanduser(dbpath)

    conn = getStationStarDBConn(dbpath)


    if cml_args.ra is None and cml_args.dec is None and cml_args.window is None:
        print("Collecting RaDec Data")
        config = cr.parse(config_path)
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
        plt.savefig("magnitudes_at_Ra_{}_Dec_{}".format(r,d), format='png')

