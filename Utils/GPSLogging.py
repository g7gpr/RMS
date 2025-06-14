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

""" Summary text and json files for station and observation session
"""

from __future__ import print_function, division, absolute_import


import sys
import os
import subprocess


from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName, getRmsRootDir, UTCFromTimestamp
import re
import sqlite3
from RMS.ConfigReader import parse
from datetime import datetime
import platform
import git
import shutil
import gpsd
from datetime import timezone
import zoneinfo
import glob
import json
import logging
import numpy as np
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ecef2LatLonAlt
from RMS.GeoidHeightEGM96 import wgs84toMSLHeight, mslToWGS84Height

from RMS.Formats.FFfits import filenameToDatetimeStr
import datetime
from RMS.Formats.Platepar import Platepar

if sys.version_info.major > 2:
    import dvrip as dvr
else:
    # Python2 compatible version
    import Utils.CameraControl27 as dvr

EM_RAISE = True

import socket
import struct
import sys
import time

def addECEFtoLatLonEle(lat_deg, lon_deg, ele_egm96, x ,y, z, config):


    lat_rads, lon_rads = np.radians(lat_deg), np.radians(lon_deg)
    ele_wgs84 = mslToWGS84Height(lat_rads, lon_rads, ele_egm96, config)
    ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(lat_rads, lon_rads, ele_wgs84)
    ecef_x_moved, ecef_y_moved, ecef_z_moved = x + ecef_x, y + ecef_y, z + ecef_z


    lat_rads, lon_rads, alt_wgs84 = ecef2LatLonAlt(ecef_x_moved, ecef_y_moved, ecef_z_moved)
    alt_egm96 = wgs84toMSLHeight(lat_rads, lon_rads, alt_wgs84, config)

    return np.degrees(lat_rads), np.degrees(lon_rads), alt_egm96





def getGPSDBConn(config, force_delete=False):
    """ Creates the GPS Data database. Tries only once.

    arguments:
        config: config file
        force_delete: if set then deletes the database before recreating

    returns:
        conn: [connection] connection to database if success else None

    """

    # Create the Observation Summary database
    gps_records_db_path = os.path.join(config.data_dir,"gps.db")

    if force_delete:
        os.unlink(gps_records_db_path)

    if not os.path.exists(os.path.dirname(gps_records_db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(gps_records_db_path))

    try:
        conn = sqlite3.connect(gps_records_db_path)

    except:
        return None

    # Returns true if the table observation_records exists in the database
    try:
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'records';""").fetchall()

        if len(tables) > 0:
            return conn
    except:
        if EM_RAISE:
            raise
        return None

    sql_command = ""
    sql_command += "CREATE TABLE records \n"
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    sql_command += "TimeStamp_local TEXT NOT NULL, \n"
    sql_command += "TimeStamp_gps TEXT NOT NULL, \n"
    sql_command += "LAT INTEGER NOT NULL, \n"
    sql_command += "LON INTEGER NOT NULL, \n"
    sql_command += "ALT INTEGER NOT NULL, \n"
    sql_command += "ECEF_X_CM INTEGER NOT NULL, \n"
    sql_command += "ECEF_Y_CM INTEGER NOT NULL, \n"
    sql_command += "ECEF_Z_CM INTEGER NOT NULL, \n"
    sql_command += "DELTA_X_MM INTEGER NOT NULL, \n"
    sql_command += "DELTA_Y_MM INTEGER NOT NULL, \n"
    sql_command += "DELTA_Z_MM INTEGER NOT NULL \n"
    sql_command += ") \n"
    conn.execute(sql_command)

    return conn

def getGPSTimeDelta(config):

    print("Getting time delta")
    gpsd.connect()
    time.sleep(1)

    try:
        current_mode = gpsd.get_current().mode
    except:
        return "GPS not available"

    #try:
    start_waiting_for_fix = datetime.datetime.now(tz=timezone.utc)
    while gpsd.get_current().mode < 2:
        time.sleep(10)
        time_now = datetime.datetime.now(tz=timezone.utc)
        elapsed = (time_now - start_waiting_for_fix).total_seconds()
        print("Waited for fix for {:.0f} seconds".format(elapsed))
        if elapsed > 60:
            return "Waited {} for a fix, no time delta available".format(elapsed)


    print("Got fix type {}".format(gpsd.get_current().mode))

    time_stamp_gps_str = gpsd.get_current().time
    time_stamp_gps_str_last = time_stamp_gps_str
    start_waiting_for_second_change = datetime.datetime.now(tz=timezone.utc)
    while time_stamp_gps_str == time_stamp_gps_str_last:
            time_stamp_gps_str_last = time_stamp_gps_str
            p = gpsd.get_current()
            time_stamp_gps_str = p.time
            time_stamp_local = datetime.datetime.now(tz=timezone.utc)
            time_now = datetime.datetime.now(tz=timezone.utc)
            elapsed = (time_now - start_waiting_for_second_change).total_seconds()
            if elapsed > 1:
                return "Waited {} seconds for second change".format(elapsed)
    time_stamp_gps = datetime.datetime.strptime(time_stamp_gps_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

    return (time_stamp_local - time_stamp_gps).total_seconds()


def getAverageDisplacement(config):

    sql_command = "SELECT AVG(DELTA_X_MM), AVG(DELTA_Y_MM), AVG(DELTA_Z_MM) FROM records ;"

    conn = getGPSDBConn(config)
    if conn is None:
        return None, None, None

    returned_query = conn.execute(sql_command)
    dev_x, dev_y, dev_z = returned_query.fetchone()

    return dev_x, dev_y, dev_z

def getStandardDeviation(config):

    sql_command = "SELECT STDDEV(DELTA_X_MM), STDDEV(DELTA_Y_MM), STDDEV(DELTA_Z_MM) FROM records ;"

    conn = getGPSDBConn(config)
    if conn is None:
        return None, None, None
    returned_query = conn.execute(sql_command)

    sd_x, sd_y, sd_z = returned_query.fetchone()

    return sd_x, sd_y, sd_z




def startGPSDCapture(config, duration=3600*4, period=10, force_delete=False):

    """ Enters the parameters known at the start of observation into the database

        arguments:
            config: config file
            duration: the initially calculated duration
            force_delete: forces deletion of the observation summary database, default False

        returns:
            conn: [connection] connection to database

        """
    con_lat_wgs84= config.latitude
    con_lon_wgs84 = config.longitude
    con_lat_wgs84_rads, con_lon_wgs84_rads = np.radians(con_lat_wgs84), np.radians(con_lon_wgs84)
    con_ele_egm96 = config.elevation
    con_alt_wgs84 = mslToWGS84Height(con_lat_wgs84_rads,con_lon_wgs84_rads, con_ele_egm96, config)
    con_ecef_x, con_ecef_y, con_ecef_z = latLonAlt2ECEF(con_lat_wgs84_rads, con_lon_wgs84_rads, con_alt_wgs84)


    conn = getGPSDBConn(config, force_delete=force_delete)


    start_time = datetime.datetime.now(tz=timezone.utc)
    iteration_start_time = start_time
    iteration_end_time = iteration_start_time
    time_elapsed = 0

    gpsd.connect()
    while time_elapsed < duration:
        time.sleep(period - (iteration_end_time - iteration_start_time).total_seconds())
        iteration_start_time = datetime.datetime.now(tz=timezone.utc)
        time_elapsed = (iteration_start_time - start_time).total_seconds()



        packet = gpsd.get_current()
        try:
            time_stamp_gps = datetime.datetime.strptime(packet.time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except:
            print("GPS not providing valid data")
            time.sleep(period)
            continue
        gps_lat_wgs84 = packet.lat
        gps_lon_wgs84 = packet.lon
        gps_alt_egm96 = packet.alt
        gps_lat_wgs84_rads, gps_lon_wgs84_rads = np.radians(gps_lat_wgs84), np.radians(gps_lon_wgs84)
        gps_alt_wgs84 = mslToWGS84Height(gps_lat_wgs84_rads, gps_lon_wgs84_rads, gps_alt_egm96, config)
        ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(np.radians(gps_lat_wgs84),np.radians(gps_lon_wgs84),gps_alt_wgs84)
        d_x, d_y, d_z = ecef_x - con_ecef_x, ecef_y - con_ecef_y, ecef_z - con_ecef_z
        #print("gps   (lat:{},lon:{},alt_egm96:{}, alt_wgs84:{})".format(gps_lat_wgs84, gps_lon_wgs84, gps_alt_egm96, gps_alt_wgs84))
        #print("config(lat:{},lon:{},alt_egm96:{})".format(con_lat_wgs84, con_lon_wgs84, con_ele_egm96))
        #print("delta (x:{}, y:{}, z:{})".format(d_x, d_y, d_z))

        sql_command = ""
        sql_command += "INSERT INTO records ( \n"

        sql_command += "TimeStamp_locaL, \n"
        sql_command += "TimeStamp_gps, \n"
        sql_command += "LAT, \n"
        sql_command += "LON, \n"
        sql_command += "ALT, \n"
        sql_command += "ECEF_X_CM, \n"
        sql_command += "ECEF_Y_CM, \n"
        sql_command += "ECEF_Z_CM, \n"
        sql_command += "DELTA_X_MM, \n"
        sql_command += "DELTA_Y_MM, \n"
        sql_command += "DELTA_Z_MM  \n"
        sql_command += ") \n"
        sql_command += "VALUES \n"
        sql_command += "( \n"
        sql_command += "'{}',".format(datetime.datetime.now(tz=timezone.utc))
        sql_command += "'{}',".format(time_stamp_gps)
        sql_command += "'{}',".format(gps_lat_wgs84)
        sql_command += "'{}',".format(gps_lon_wgs84)
        sql_command += "'{}',".format(gps_alt_egm96)
        sql_command += "'{}',".format(ecef_x * 100)
        sql_command += "'{}',".format(ecef_y * 100)
        sql_command += "'{}',".format(ecef_z * 100)
        sql_command += "'{}',".format(d_x * 1000)
        sql_command += "'{}',".format(d_y * 1000)
        sql_command += "'{}'".format(d_z * 1000)
        sql_command += ") \n"

        conn.execute(sql_command)
        conn.commit()

                #time_stamp_local = datetime.datetime.p(time_stamp_local , tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')

        iteration_end_time = datetime.datetime.now(tz=timezone.utc)

    conn.close()



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Read the config, measure the location for an extended time, average, and propose new location. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-d', '--duration', nargs=1, metavar='DURATION', type=int,
                            help="Period of time to log for in minutes.")

    arg_parser.add_argument('-p', '--period', nargs=1, metavar='PERIOD', type=int,
                            help="Period between logs in seconds.")

    cml_args = arg_parser.parse_args()

    if cml_args.duration is None:
        duration = 120
    else:
        duration = cml_args.duration[0] * 60

    if cml_args.period is None:
        period = 10
    else:
        period = cml_args.period[0]



    print("Logging for {} minutes period {} seconds".format(duration / 60, period))

    logging.getLogger("gpsd").setLevel(logging.ERROR)
    config = parse(os.path.expanduser("~/source/RMS/.config"))

    startGPSDCapture(config, duration=duration,period=period)
    dev_x, dev_y, dev_z  = getAverageDisplacement(config)
    sd_x, sd_y, sd_z = getStandardDeviation(config)
    print("Deviation x,y,z {},{},{} (m)".format(dev_x / 1000, dev_y / 1000, dev_z / 1000))
    print("SD        x,y,z {},{},{} (m)".format(sd_x / 1000, sd_y / 1000, sd_z / 1000))
    new_lat_degs, new_lon_degs, new_ele_egm96 = addECEFtoLatLonEle(config.latitude, config.longitude, config.elevation, dev_x / 1000, dev_y / 1000, dev_z / 1000, config)

    output = "\n"
    output += "; WGS84 +N (degrees) \n"
    output += "latitude: {} \n \n".format(new_lat_degs)
    output += "; WGS84 +E (degrees) \n"
    output += "longitude: {} \n \n".format(new_lon_degs)

    output += "; Mean sea level EGM96 geoidal datum, not WGS84 ellipsoidal (meters) \n"
    output += "; Can be obtained from Google Earth or other apps. Raw GPS altitude should not \n"
    output += "; be used as the software converts from EGM96 to WGS84 internally. \n"
    output += "elevation: {}".format(new_ele_egm96)
    output = "\n"

    print(output)