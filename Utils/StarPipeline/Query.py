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
import numpy as np

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
import cv2
import ephem
import tqdm

from RMS.Astrometry.Conversions import datetime2JD, jd2Date
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Misc import mkdirP
from pathlib import Path
from Utils.StarPipeline.CacheBuilder import sortFilesByTime
from RMS.Formats.CALSTARS import readCALSTARS, renderStars
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalogQueries import Catalog, renderStarsGaussian
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP, raDec2AltAz
from Utils.Flux import detectMoon

from Utils.StarPipeline.CacheBuilder import ARCHIVE, sortFilesByTime
import matplotlib.pyplot as plt


import uuid

# Local directories
# Local directories
ARCHIVE = "/archive"
CALSTARS_DATA_DIR = "CALSTARS"
QUEUES = "/archive/despatcher/queues"
REGISTRATIONS = "/archive/despatcher/registrations"
PROCESSED_SESSIONS_FILE = "/archive/processed_sessions.txt"
LIGHT_CURVES = Path(Path(ARCHIVE) / "light_curves")

from pathlib import Path

def getStars():

    root = Path(LIGHT_CURVES)

    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
    )

def getSessionCount(star_name):

    star_dir = Path(LIGHT_CURVES) / star_name

    return len(list(star_dir.rglob("*.npy")))

def getStarFiles(star_name):

    star_dir = Path(LIGHT_CURVES) / star_name

    if not star_dir.exists():
        return []

    return sorted(
        star_dir.rglob("*.npy")
    )


def getStarObservations(star_name):

    arrays = []
    stations = set()

    for filename in getStarFiles(star_name):

        arrays.append(loadObservationFile(filename))
        stations.add(filename.parent.name)

    observations = np.concatenate(arrays)

    return star_name, observations, stations

def loadObservationFile(filename):

    dtype = np.dtype([
        ("station", "U6"),
        ("jd", np.float64),
        ("obs_mag", np.float32),
        ("cat_mag", np.float32),
        ("pixel_scale", np.float32),
        ("level", np.int32),
        ("amplitude", np.int32),
        ("fwhm", np.float32),
        ("background", np.int32),
        ("snr", np.float32),
        ("saturated_count", np.int32),
        ("star_count", np.int16),
        ("obs_ra", np.float32),
        ("obs_dec", np.float32)
    ])

    filename = Path(filename)
    stem = filename.stem
    station, date_str, time_str, usec_str, cat_mag_str, scale_str = stem.split("_")
    session_date = datetime.datetime.strptime(f"{date_str}_{time_str}_{usec_str}","%Y%m%d_%H%M%S_%f")
    cat_mag = int(cat_mag_str) / 1e6
    pixel_scale = int(scale_str) / 1000


    data = np.load(filename)

    offset = data[:, 0]
    residual = data[:, 1]


    n = len(offset)
    observations = np.empty(n, dtype=dtype)

    session_jd = datetime2JD(session_date)

    observations["station"] = station
    observations["jd"] = session_jd +  offset.astype(np.float64) / 10000.0

    observations["obs_mag"] = cat_mag + residual / 100.0
    observations["cat_mag"] = cat_mag
    observations["pixel_scale"] = pixel_scale

    observations["level"] = data[:, 2]
    observations["amplitude"] = data[:, 3]
    observations["fwhm"] = data[:, 4] / 100.0
    observations["background"] = data[:, 5]
    observations["snr"] = data[:, 6] / 100.0
    observations["saturated_count"] = data[:, 7]
    observations["star_count"] = data[:, 8]
    observations["obs_ra"] = data[:, 9]  / 90
    observations["obs_dec"] = data[:, 10] / 300

    return observations

def starSummary(star_name):

    files = getStarFiles(star_name)
    message = ""

    message += f"For star {star_name}\n"
    message += f"Files:{len(files)}\t"

    total_points = 0

    for file in files:
        total_points += np.load(file).shape[0]

    message += f"Observations:{total_points} "

    return message

def simpleMagnitudeScatterPlot(plot_data):

    star_name, obs, stations_list = plot_data

    stations_text = ", ".join(sorted(stations_list))

    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()

    #
    # Magnitude scatter coloured by station
    #

    obs_ra, obs_dec = np.mean(obs['obs_ra']), np.mean(obs['obs_dec'])

    for station in stations_list:
        mask = obs["station"] == station

        ax1.scatter(
            obs["jd"][mask],
            obs["obs_mag"][mask],
            s=5,
            label=station
        )


    # Plot station magnitude legends

    ax1.legend(
        title="Stations",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )


    #
    # Catalog magnitude
    #
    ax1.axhline(
        y=obs["cat_mag"][0],
        color="red",
        linestyle=":",
        linewidth=1,
        alpha=0.5
    )

    ax1.invert_yaxis()

    ax1.set_xlabel("Julian Date")
    ax1.set_ylabel("Magnitude")

    plt.ticklabel_format(
        axis="x",
        style="plain",
        useOffset=False
    )

    #
    # Title + station list
    #
    plt.suptitle(f"{star_name}\nRa:{obs_ra:.2f}deg   Dec:{obs_dec:.2f}deg", fontsize=14)

    ax1.set_title(
        stations_text,
        fontsize=8,
        color="gray"
    )

    ax1.grid(True)

    #
    # Save
    #
    safe_name = (
        star_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    filename = f"{safe_name}.png"

    filepath = "/archive/plots"

    os.makedirs(filepath, exist_ok=True)

    plt.savefig(
        os.path.join(filepath, filename),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Despatch start curve processing jobs \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("-r", "--reverse", action="store_true", help="Process newest jobs first")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))






    if True:

        star_list = getStars()
        star_sessions = []
        for star in star_list:
            session_count = getSessionCount(star)
            star_sessions.append([session_count, star])

        star_sessions.sort(key=lambda x: x[0], reverse=True)


        session, most_observed_star = star_sessions[0]

        print(f"Querying the star with the most observation sessions ({session}), {most_observed_star}")
        star_file_list = getStarFiles(most_observed_star)
        star_to_plot =  most_observed_star



    print("Query started")
    query_response = getStarObservations(star_to_plot)
    print("Query completed")
    simpleMagnitudeScatterPlot(query_response)

    pass