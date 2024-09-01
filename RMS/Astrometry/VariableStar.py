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
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Misc import rmsTimeExtractor
from RMS.Astrometry.ApplyRecalibrate import recalibrateFF, recalibratePlateparsForFF
from RMS.Logger import initLogging
from RMS.Misc import getRMSStyleFileName
from RMS.Astrometry.FFTalign import getMiddleTimeFF, alignPlatepar

# Handle Python 2/3 compatibility
if sys.version_info.major == 3:
    unicode = str

EM_RAISE = True

def readInArchivedCalstars(config, conn, log):


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
    latest_jd = findMostRecentEntry(conn)
    archived_directories_filtered_by_jd = []
    for directory in archived_directories:
        archived_directories_filtered_by_jd.append(directory)
        if rmsTimeExtractor(directory, asJD=True) < latest_jd:
            print("Excluding directories before {}, already processed".format(os.path.basename(directory)))
            break
    archived_directories_filtered_by_jd.reverse()
    for dir in archived_directories_filtered_by_jd:
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
    pass


def getCatalogueID(r, d, conn, margin=0.5):

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

def convertRaDec(calstar, conn, catalogue, archived_directories_path, latest_jd=0):

    """
    Parses a calstar data structure, retains all existing data but
    uses the

    Args:
        calstar (): a calstar data structure


    Returns:

    """

    calstar_radec = []
    calstar_for_recal = dict(calstar)
    #fits_files_in_calstar = dict.keys(calstar_for_recal)
    #full_path_all_recalibrated_json = getRMSStyleFileName(archived_directories_path, "pp_all_recalibrated.json")
    #pp_recal = recalibratePlateparsForFF(pp, fits_files_in_calstar, calstar_for_recal, catalogue, config)

    #with open(full_path_all_recalibrated_json, 'w') as f:
    #    f.write(json.dumps(pp_recal, indent=4, sort_keys=True))


    platepars_all_recalibrated_path = os.path.join(archived_directories_path, "platepars_all_recalibrated.json")
    with open(platepars_all_recalibrated_path, 'r') as fh:
        pp_recal = json.load(fh)


    for fits_file, star_list in calstar:
        if len(star_list) < config.min_matched_stars:
            print("In {} only {} stars".format(fits_file, len(star_list)))
            continue
        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
        # Skip anything which has already been processed
        if jd < latest_jd:
            continue
        jd_list, x_list, y_list, bg_list, amp_list, FWHM_list = [], [], [], [], [], []
        for x, y, bg_intensity, amplitude, FWHM in star_list:

            jd_list.append(jd)
            x_list.append(x)
            y_list.append(y)
            bg_list.append(bg_intensity)
            amp_list.append(amplitude)
            FWHM_list.append(FWHM)
        jd_arr, x_data, y_data, level_data = np.array(jd_list), np.array(x_list), np.array(y_list), np.array(amp_list)

        star_dict_ff = {jd: star_list}
        '''
        calstars_coords = np.array(star_list)
        calstars_coords = np.array(calstars_coords[:, :2])
        calstars_coords[:, [0, 1]] = calstars_coords[:, [1, 0]]
        calstars_time = getMiddleTimeFF(fits_file, config.fps, ret_milliseconds=True)
        pp_recal = alignPlatepar(config, pp, calstars_time, calstars_coords, show_plot=False)
        '''

        if not fits_file in pp_recal:
            continue
        pp = Platepar()
        pp.loadFromDict(pp_recal[fits_file])


        jd, ra, dec, mag = xyToRaDecPP(jd_arr, x_data, y_data, level_data, pp, jd_time=True, extinction_correction=True)
        star_list_radec = []
        for j, x, y, r, d, bg, amp, FWHM, mag in zip(jd, x_list, y_list, ra, dec, bg_list, amp_list, FWHM_list, mag):
            cat_id, cat_mag, cat_r, cat_d = getCatalogueID(r, d, conn)
            star_list_radec.append([j, date_time, x, y, r, d, bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d])
        if len(star_list_radec)  > 30:

            insertDB(conn, star_list_radec)
            print("In {},  {} stars, inserting".format(fits_file, len(star_list_radec)))
        calstar_radec.append([fits_file, star_list_radec])
    return calstar_radec

def insertDB(conn, star_list_radec):


    for jd, date_time, x, y, r, d, bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d in star_list_radec:
        sql_command = ""
        sql_command += "INSERT INTO star_observations \n"
        sql_command += "(jd, date_time, x, y, r, d, bg, amp, FWHM, mag, cat_key, cat_mag, cat_r, cat_d )\n"
        sql_command += "VALUES\n"
        sql_command += "({}, '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(jd, date_time, x, y, r, d, bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d)
        conn.execute(sql_command)
    conn.commit()

def getStationStarDBConn(config, force_delete=False):
    """ Creates the station star database. Tries only once.

    arguments:
        config: config file
        force_delete: if set then deletes the database before recreating

    returns:
        conn: [connection] connection to database if success else None

    """

    # Create the station star database
    db_name = "station_star"

    db_path = os.path.join(config.data_dir,"{}.db".format(db_name))

    if force_delete:
        os.unlink(db_path)

    if not os.path.exists(os.path.dirname(db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(db_path))

    try:
        conn = sqlite3.connect(db_path)
        createTableStarObservations(conn)
        createTableCatalogue(conn)
        return conn

    except:
        return None


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
    sql_command += "x SMALLINT NOT NULL, \n"
    sql_command += "y SMALLINT NOT NULL, \n"
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

def findMostRecentEntry(conn):


    sql_command = ""
    sql_command += "SELECT max(jd) FROM star_observations \n"

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


if __name__ == "__main__":


    config = cr.parse( os.path.expanduser("~/source/RMS/.config"))
    conn = getStationStarDBConn(config)

    # Initialize the logger
    initLogging(config)
    # Get the logger handle
    log = logging.getLogger("logger")

    archived_calstars = readInArchivedCalstars(config, conn, log)

    with open("archived_calstars.pickle", 'wb') as fh:
        pickle.dump(archived_calstars, fh)
