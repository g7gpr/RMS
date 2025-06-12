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
import glob
import json

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
    sql_command += "TimeStamp_error TEXT NOT NULL, \n"
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


def startGPSDCapture(config, duration, force_delete=False):

    """ Enters the parameters known at the start of observation into the database

        arguments:
            config: config file
            duration: the initially calculated duration
            force_delete: forces deletion of the observation summary database, default False

        returns:
            conn: [connection] connection to database

        """


    conn = getGPSDBConn(config, force_delete=force_delete)

    gpsd.connect()
    try:
        while True:
            time_stamp_local = time.time()
            packet = gpsd.get_current()
            lat = packet.lat
            lon = packet.lon
            alt = packet.altitude
            time_stamp_gps = packet.time

            print("lat {}, lon {}, alt {}, time_gps {}, time_local {}".format(lat, lon, alt, time_stamp_gps, time_stamp_local))
            time.sleep(1)

    except StopIteration:
        print("GPSD has terminated")
    except KeyError:
        pass
    except KeyboardInterrupt:
        print("\nUser interrupted.")

    conn.close()



if __name__ == "__main__":

    config = parse(os.path.expanduser("~/source/RMS/.config"))
    startGPSDCapture(config, 0.1)
