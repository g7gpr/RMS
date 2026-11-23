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
from Utils.StarPipeline.LightCurves import decodeOffsetGnomonic
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
        ("xi", np.float32),
        ("eta", np.float32),
        ("ra_cat", np.float32),
        ("dec_cat", np.float32),
        ("ra_obs", np.float32),
        ("dec_obs", np.float32)
    ])

    filename = Path(filename)
    stem = filename.stem
    station, date_str, time_str, usec_str, cat_mag_str, cat_ra_str, cat_dec_str, scale_str = stem.split("_")
    session_date = datetime.datetime.strptime(f"{date_str}_{time_str}_{usec_str}","%Y%m%d_%H%M%S_%f")
    cat_mag = int(cat_mag_str) / 1e6
    cat_ra = int(cat_ra_str) / 1e6
    cat_dec = int(cat_dec_str) / 1e6
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
    observations['ra_cat']  = cat_ra
    observations['dec_cat'] = cat_dec
    observations["xi"] = data[:, 9]  / 10000
    observations["eta"] = data[:, 10] / 10000
    ra_obs, dec_obs = decodeOffsetGnomonic(cat_ra, cat_dec, observations["xi"], observations["eta"])
    observations["ra_obs"] = ra_obs
    observations["dec_obs"] = dec_obs

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
    number_of_observations = len(obs)
    number_of_stations = len(stations_list)
    if number_of_stations > 10:
        stations_text = f"Observed by {len(stations_list)} stations with {number_of_observations} individual observations"

    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()

    #
    # Magnitude scatter coloured by station
    #

    ra_cat, dec_cat = np.mean(obs['ra_cat']), np.mean(obs['dec_cat'])

    for station in stations_list:
        mask = obs["station"] == station

        ax1.scatter(
            obs["jd"][mask],
            obs["obs_mag"][mask],
            s=5,
            label=station
        )


    # Plot station magnitude legends
    #
    #ax1.legend(
    #    title="Stations",
    #    fontsize=8,
    #    title_fontsize=9,
    #    loc="upper left",
    #    bbox_to_anchor=(1.02, 1)
    #)


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
    plt.suptitle(f"{star_name} magnitude plot", fontsize=14)

    ax1.set_title(
        f"Ra:{ra_cat:.2f}deg Dec:{dec_cat:.2f}deg\n{stations_text}",
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

    filename = f"{safe_name}_light_curves.png"

    filepath = "/archive/plots"

    os.makedirs(filepath, exist_ok=True)

    plt.savefig(
        os.path.join(filepath, filename),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")

def plotAstrometryRaDec(plot_data):

    star_name, obs, stations_list = plot_data

    plt.figure(figsize=(8, 8))

    plt.scatter(
        obs["ra_obs"],
        obs["dec_obs"],
        s=2
    )

    plt.scatter(
        obs["ra_cat"][0],
        obs["dec_cat"][0],
        color="red",
        marker="+",
        s=200,
        linewidths=2,
        label="Catalog"
    )

    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")

    plt.title(star_name)

    plt.legend()

    plt.grid(True)

    plt.gca().set_aspect('equal')

    safe_name = (
        star_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    filename = f"{safe_name}_astrometry_radec.png"

    filepath = "/archive/plots"

    os.makedirs(filepath, exist_ok=True)

    plt.savefig(
        os.path.join(filepath, filename),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {filename}")


def plotAstrometryXiEta(plot_data):

    star_name, obs, stations_list = plot_data

    plt.figure(figsize=(8, 8))
    plt.figure(figsize=(8, 8))

    plt.scatter(
        obs["xi"],
        obs["eta"],
        s=2
    )

    plt.axhline(0, color="red", alpha=0.5)
    plt.axvline(0, color="red", alpha=0.5)

    plt.xlabel("Xi (deg)")
    plt.ylabel("Eta (deg)")
    plt.title(star_name)

    plt.grid(True)
    plt.gca().set_aspect("equal")
    plt.title(f"{star_name} astrometry in xi, eta")

    plt.legend()

    plt.grid(True)

    plt.gca().set_aspect('equal')

    safe_name = (
        star_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    filename = f"{safe_name}_astrometry_xi_eta.png"

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

    arg_parser.add_argument("-c", "--continuous", action="store_true", help="Build all the plots for all stars in the background")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    continuous = cml_args.continuous





    while continuous:

        star_list = getStars()
        star_sessions = []
        for star in star_list:
            session_count = getSessionCount(star)
            star_sessions.append([session_count, star])

        star_sessions.sort(key=lambda x: x[0], reverse=True)

        for star in star_sessions:
            observation_count, star_to_plot = star[0], star[1]
            if observation_count < 1:
                continue
            print(f"Querying for {star_to_plot}")
            query_response = getStarObservations(star_to_plot)
            print(f"Plotting {star_to_plot}")
            simpleMagnitudeScatterPlot(query_response)
            plotAstrometryRaDec(query_response)
            plotAstrometryXiEta(query_response)

            time.sleep(60)


    star_to_plot = "HD 23480"
    print("Query started")
    query_response = getStarObservations(star_to_plot)
    print("Query completed")
    simpleMagnitudeScatterPlot(query_response)

    pass