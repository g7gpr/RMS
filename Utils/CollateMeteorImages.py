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
import RMS.Formats.FFfits as FFfits

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

import RMS.ConfigReader as cr
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

    print(archived_directory_list)

    ftp_dict = {}
    for station, archived_directory in zip(station_directories, archived_directory_list):
        print(station, archived_directory)
        ar_date = archived_directory.split("_")[1]
        ar_time  = archived_directory.split("_")[2]
        ar_milliseconds = archived_directory.split("_")[3]
        ftp_file_name = "FTPdetectinfo_{}_{}_{}_{}.txt".format(station.upper(), ar_date, ar_time, ar_milliseconds)
        print(ftp_file_name)
        ftp_dict[station] = readFTPdetectinfo(os.path.join(working_directory, station, archived_directory), ftp_file_name)

    return ftp_dict

def findTimeRelatedEvents(detectInfoDict):



    return time_list


def produceCollatedChart(input_directory):

    working_area = createTemporaryWorkArea()
    print("Working in {}".format(working_area))
    working_area = extractBz2(input_directory, working_area)
    ftp_dict = readInFTPDetectInfoFiles(working_area)


    # Rearrange into time

    events = []
    for station in sorted(ftp_dict):
        ff_name =  ftp_dict[station][1][0]
        fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")

        events.append([fits_date,ftp_dict[station]])
    events = sorted(events)
    print(events)







    pass

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