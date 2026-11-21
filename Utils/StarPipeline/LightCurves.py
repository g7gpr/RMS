# RPi Meteor Station
# Copyright (C) 2026 David Rollinson Kristen Felker
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
import tempfile
import tarfile

import subprocess
import json
import requests
import RMS.ConfigReader as cr
import shutil
import sys
import datetime
import time
import random
import socket

from RMS.Misc import mkdirP
from pathlib import Path
from CacheBuilder import sortFilesByTime
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar

from Utils.StarPipeline.CacheBuilder import ARCHIVE

# Local directories
ARCHIVE = "/archive"
CALSTARS_DATA_DIR = "CALSTARS"

def getWorkList(archive):


    print(f"Working through {archive}")
    paths_to_files = Path(archive).rglob("*.tar.bz2")
    return sortFilesByTime([str(f) for f in paths_to_files])


def calstarToLightCurves(config, pp, calstar):

    light_curves = {}

    for fits_file in calstar:
        print(fits_file)

    return light_curves

def process(path_to_job):

    print(f"Processing {path_to_job}")
    observation_dir = os.path.basename(path_to_job).split("_min")[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting to {temp_dir}")

        with tarfile.open(path_to_job, "r:bz2") as tar:
            tar.extractall(temp_dir)
            working_dir = os.path.join(temp_dir, observation_dir)
            config_path = os.path.join(working_dir, ".config")
            obs_con = cr.parse(config_path)
            pp = Platepar()
            pp_path = os.path.join(working_dir, obs_con.platepar_name)
            pp.read(pp_path)
            calstars_filename = f"CALSTARS_{observation_dir}.txt"
            calstar = readCALSTARS(working_dir, calstars_filename)

            if len(calstar) == 2:
                light_curves = calstarToLightCurves(config, pp, calstar[0])

        pass

    # Temporary directory is automatically deleted here



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Convert GMN data cache into star light curves \
        """, formatter_class=argparse.RawTextHelpFormatter)

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    os.makedirs(ARCHIVE, exist_ok=True)
    calstars_directory_path = os.path.join(ARCHIVE, CALSTARS_DATA_DIR)
    os.makedirs(calstars_directory_path, exist_ok=True)

    work_list = getWorkList(calstars_directory_path)
    for path_to_job in work_list:
        process(path_to_job)