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
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP

# Handle Python 2/3 compatibility
if sys.version_info.major == 3:
    unicode = str


def readInArchivedCalstars(config):


    """
    Iterates through Archived Directories to load in the Calstars Files
    """
    archived_directories_path = os.path.join(config.data_dir, config.archived_dir)
    archived_directories = getNightDirs(archived_directories_path, config.stationID)
    print("Archived Directories")
    calstar_list = []
    for dir in archived_directories:
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
        pp = Platepar()
        pp.read(full_path_platepar)
        calstar_list.append(convertRaDec(calstar,pp))
    pass

def convertRaDec(calstar,pp):

    print(calstar)
    print(pp)

    xyToRaDecPP()

    return radec_calstar


if __name__ == "__main__":

    config = cr.parse( os.path.expanduser("~/source/RMS/.config"))
    readInArchivedCalstars(config)