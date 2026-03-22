# RPi Meteor Station
# Copyright (C) 2025 David Rollinson
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

from datetime import tzinfo


import os
import tempfile
import tarfile
from tkinter.filedialog import Directory

import paramiko
import subprocess
import json
import requests
import RMS.ConfigReader as cr
import shutil
import sys
import datetime
import numpy as np
import time
import random
import socket
import psycopg

from RMS.Formats import FFfile, Platepar
from RMS.Astrometry.CheckFit import starListToDict
from RMS.Astrometry.Conversions import J2000_JD, date2JD
from RMS.Formats.StarCatalog import Catalog
from RMS.Astrometry.Conversions import latLonAlt2ECEF
from RMS.Routines.MaskImage import getMaskFile
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP, raDec2AltAz
from RMS.Formats.CALSTARS import readCALSTARS, maxCalstarsToPNG, calstarsToMP4
from RMS.Misc import mkdirP



DB_SCALE_FACTOR = 1e6
JD_OFFSET = J2000_JD

TYPE_MAP = {
    "stationID": "TEXT",
    "jd": "BIGINT",
    # Add more special cases here as your schema evolves
}


STATION_COORDINATES_JSON = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"
CALSTARS_DATA_DIR = "CALSTARS"
PLATEPARS_ALL_RECALIBRATED_JSON = "platepars_all_recalibrated.json"
DIRECTORY_INGESTED_MARKER = ".ingested"
CALSTAR_FILES_TABLE_NAME = "calstar_files"
STAR_OBSERVATIONS_TABLE_NAME = "star_observations"


CHARTS = "charts"
PORT = 22

from RMS.Logger import LoggingManager, getLogger
from pathlib import Path



import matplotlib.pyplot as plt
from collections import defaultdict



def plotStarLightcurve(conn_params, catalogue_id, jd_start, jd_end):
    """
    Plot observed magnitudes for a given catalogue star across all stations.
    """

    query = """
        SELECT jd, obs_mag, stationID
        FROM star_observations
        WHERE catalogue_id = %s
          AND jd BETWEEN %s AND %s
        ORDER BY jd;
    """

    # psycopg3 uses connect() with keyword args, same as before
    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (catalogue_id, jd_start, jd_end))
            rows = cur.fetchall()

    if not rows:
        print("No observations found for this star in the given JD range.")
        return

    # Group by station so each camera gets its own colour
    by_station = defaultdict(lambda: {"jd": [], "mag": []})

    for jd, mag, station_id in rows:
        by_station[station_id]["jd"].append(jd)
        by_station[station_id]["mag"].append(mag)

    plt.figure(figsize=(12, 6))

    for station_id, data in by_station.items():
        plt.plot(
            data["jd"],
            data["mag"],
            marker='o',
            linestyle='-',
            markersize=3,
            label=station_id
        )

    plt.gca().invert_yaxis()  # astronomy convention
    plt.xlabel("Julian Date")
    plt.ylabel("Observed Magnitude")
    plt.title(f"Light Curve for Catalogue Star {catalogue_id}")
    plt.grid(True)
    plt.legend(title="StationID", fontsize=8)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Ingest CALSTAR data \
        """, formatter_class=argparse.RawTextHelpFormatter)


    arg_parser.add_argument('user_hostname', help="""user@hostname""")

    arg_parser.add_argument('path_template', help="""Template to remote file stores i.e. /home/stationID/files/processed """)

    arg_parser.add_argument('postgresql_host', help="""PostgreSQL server host """)

    arg_parser.add_argument('-l', '--local', dest='run_local', default=False, action="store_true",
                            help="Run using local mirror.")

    arg_parser.add_argument('-d', '--drop', dest='drop', default=False, action="store_true",
                            help="Drop all tables at start - do not use in production")

    arg_parser.add_argument('-r', '--reset_ingestion', dest='reset_ingestion', default=False, action="store_true",
                            help="Reset all ingestion markers")

    arg_parser.add_argument('--country', metavar='COUNTRY', help="""Country code to work on""")

    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))
    country_code = cml_args.country



    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("rmslogger")

    user, _, hostname = cml_args.user_hostname.partition("@")
    path_template = cml_args.path_template
    postgresql_host = cml_args.postgresql_host

    log.info(f"Starting ingestion from {user}@{hostname} with path template {path_template}")
    log.info(f"Postgresql host {postgresql_host}")


    cwd = os.getcwd()

    conn_params = {
        "host": "192.168.1.174",
        "dbname": "meteor_ingest",
        "user": "ingest_user"
    }

    catalogue_id =

    plotStarLightcurve(conn_params, catalogue_id=catalogue_id, jd_start, jd_end)


    pass