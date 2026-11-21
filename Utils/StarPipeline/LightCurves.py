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

from Utils.StarPipeline.CacheBuilder import ARCHIVE

# Local directories
ARCHIVE = "/archive"
CALSTARS_DATA_DIR = "CALSTARS"

def minSunBelowHorizon(fits_file_list, c, sun_angle=-18, chunk_size=1):

    if not len(fits_file_list):
        return [], np.array([]), 0, 0


    # Initialize observer
    o = ephem.Observer()
    o.lat, o.lon, o.elevation  = str(c.latitude), str(c.longitude), float(c.elevation)
    sun = ephem.Sun()
    sun.compute(o)
    sun_alt_deg = np.degrees(float(sun.alt))
    last_sun_alt_deg = sun_alt_deg
    angle_list, astronomical_night_list = [], []
    setting_count, rising_count = 0, 0
    for i, fits_file in enumerate(fits_file_list):

        # Recompute Sun altitude every chunk_size frames
        if i % chunk_size == 0:
            o.date = getMiddleTimeFF(fits_file, c.fps, dt_obj=True)
            sun.compute(o)
            sun_alt_deg = np.degrees(float(sun.alt))

            last_sun_alt = sun_alt_deg
        angle_list.append(sun_alt_deg)

        if sun_alt_deg < sun_angle:
            astronomical_night_list.append(fits_file)
        else:
            if last_sun_alt_deg > sun_alt_deg:
                setting_count += 1
            elif last_sun_alt_deg < sun_alt_deg:
                rising_count += 1
            else:
                pass
        last_sun_alt_deg = sun_alt_deg

    return astronomical_night_list, np.array(angle_list), setting_count, rising_count


def getWorkList(archive):


    print(f"Working through {archive}")
    paths_to_files = Path(archive).rglob("*.tar.bz2")



    return sortFilesByTime([str(f) for f in paths_to_files])


def calstarToLightCurves(cat, obs_con, pp, calstar, platepars_all_recalibrated_path, video_path=None):

    light_curves = {}
    produce_debug_timelapse = False

    if os.path.exists(platepars_all_recalibrated_path):
        try:
            with open(platepars_all_recalibrated_path, 'r') as fh:
                pp_recal_json = json.load(fh)
                #log.info(f"Read {os.path.basename(local_recal_path)}")
        except:
            pp_recal_json = None
            print(f"{platepars_all_recalibrated_path} was corrupted")
    else:
        print(f"No file {os.path.basename(platepars_all_recalibrated_path)}`")
        pp_recal_json = None


    if video_path is not None:
        produce_debug_timelapse = True
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                30, (pp.X_res * 2, pp.Y_res), isColor=False)

    fits_files_from_calstar_list = [calstar_entry[0] for calstar_entry in calstar]
    fits_files_without_moon_list = detectMoon(fits_files_from_calstar_list, pp, obs_con)
    astronomical_night_list, sun_below_horizon_angle_list, setting_count, rising_count = minSunBelowHorizon(fits_files_from_calstar_list, obs_con, sun_angle=-18)

    counter = 0
    for fits_file, star_list in tqdm.tqdm(calstar):

        if counter > 10:
            break

        if len(star_list) < 40:
            #print(f"Skipping {fits_file} only {len(star_list)} stars")
            continue

        if fits_file not in fits_files_without_moon_list:
            #print(f"Skipping {fits_file} - moon in FoV")
            continue

        if fits_file not in astronomical_night_list:
            #print(f"Skipping {fits_file} - not astronomical night")
            continue

        if pp_recal_json is not None:
            if fits_file in pp_recal_json:
                pp.loadFromDict(pp_recal_json[fits_file])

        jd = datetime2JD(getMiddleTimeFF(fits_file, obs_con.fps, dt_obj=True))
        p, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd, n_brightest=10*len(star_list))

        y, x, level, amplitude, fwhm, background, snr, saturated_count = zip(*star_list)
        jd_list = [jd] * len(star_list)
        _arr_jd, arr_obs_ra, arr_obs_dec, arr_obs_mag = xyToRaDecPP(jd_list, list(x), list(y), list(amplitude), pp,
                                                                    jd_time=True, measurement=True,
                                                                    precompute_pointing_corr=True,
                                                                    extinction_correction=True)
        if produce_debug_timelapse:
            query_img, agc_level = renderStarsGaussian(x_array, y_array, mag_array, (pp.Y_res, pp.X_res))
            calstar_img, agc_level = renderStarsGaussian(list(x), list(y), arr_obs_mag, (pp.Y_res, pp.X_res))
            combined_img = np.hstack((query_img, calstar_img))
            video.write(combined_img)

        for observation_index, star in enumerate(star_list):
            if len(star) == 8:
                obs_x, obs_y = star[1], star[0]
                img_dist = (x_array - obs_x)**2 + (y_array - obs_y)**2
                nearest_idx = np.argmin(img_dist)
                if img_dist[nearest_idx] <= 2 ** 2:
                    # Match found within 2 pixels
                    cat_x = x_array[nearest_idx]
                    cat_y = y_array[nearest_idx]
                    name = names_array[nearest_idx]
                    #print(f"{fits_file} {name} {img_dist[nearest_idx]} {obs_x},{obs_y} {cat_x},{cat_y}")
                    if name not in light_curves:
                        light_curves[name] = {}
                    if jd not in light_curves[name]:
                        light_curves[name][jd] = {}
                        light_curves[name][jd]['obs'] = arr_obs_mag[observation_index]
                        light_curves[name][jd]['cat'] = mag_array[nearest_idx]

                    counter += 1


                    pass
        pass
    if produce_debug_timelapse:
        video.release()

    return light_curves

def process(path_to_job, cat):


    print(f"Processing {path_to_job}")
    observation_dir = os.path.basename(path_to_job).split("_min")[0]


    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting to {temp_dir}")

        with tarfile.open(path_to_job, "r:bz2") as tar:

            try:
                tar.extractall(temp_dir, filter="fully_trusted")
            except:
                tar.extractall(temp_dir)

            working_dir = os.path.join(temp_dir, observation_dir)
            config_path = os.path.join(working_dir, ".config")
            obs_con = cr.parse(config_path)
            pp = Platepar()
            pp_path = os.path.join(working_dir, obs_con.platepar_name)
            pp.read(pp_path)
            calstars_filename = f"CALSTARS_{observation_dir}.txt"
            calstar = readCALSTARS(working_dir, calstars_filename)
            video_file = f"{observation_dir}_timelapse.mp4"
            video_path = os.path.join(os.path.expanduser(config.data_dir), video_file)
            platepars_all_recalibrated_path = os.path.join(working_dir, "platepars_all_recalibrated.json")

            if len(calstar) == 2:

                light_curves = calstarToLightCurves(cat, config, pp, calstar[0], platepars_all_recalibrated_path, video_path=None)

                station, date_str, time_str, us_str = observation_dir.split('_')
                session_date = datetime.datetime.strptime(f"{date_str}_{time_str}_{us_str}", "%Y%m%d_%H%M%S_%f")
                for star_name, observations in light_curves.items():

                    daily_data = {}

                    for jd_str, data in observations.items():

                        jd = float(jd_str)


                        if observation_dir not in daily_data:
                            daily_data[observation_dir] = []
                        observation_date = jd2Date(jd, dt_obj=True)

                        offset = int((observation_date - session_date).total_seconds() / 10)

                        daily_data[observation_dir].append([offset, int(round(100 * (data['obs'] - data['cat'])))])

                    for session, records in daily_data.items():
                        dirname = Path("/archive/light_curves/") / star_name / obs_con.stationID
                        dirname.mkdir(parents=True, exist_ok=True)

                        filename = dirname / f"{session}_{int(data['cat'] * 1e6)}.npy"

                        np.save(filename, np.array(records, dtype=np.int16))
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


    open("/archive/processed_sessions.txt", "a").close()

    with open("/archive/processed_sessions.txt") as f:
        processed_sessions = {
            line.strip()
            for line in f
            if line.strip()
        }

    work_list = getWorkList(calstars_directory_path)
    cat = Catalog(config)


    for path_to_job in work_list:
        if os.path.basename(path_to_job) not in processed_sessions:
            process(path_to_job, cat)
            processed_sessions.add(os.path.basename(path_to_job))
            with open("/archive/processed_sessions.txt", "a") as f:
                f.write(path_to_job + "\n")
        else:
            print(f"{os.path.basename(path_to_job)} already processed")