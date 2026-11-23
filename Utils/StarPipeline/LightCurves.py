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
from RMS.Math import angularSeparationDeg
from RMS.Misc import mkdirP
from pathlib import Path
from Utils.StarPipeline.CacheBuilder import sortFilesByTime
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalogQueries import Catalog, renderStarsGaussian
from RMS.Astrometry.ApplyAstrometry import geoHt2XY, xyToRaDecPP, raDec2AltAz
from Utils.Flux import detectMoon
from Utils.StarPipeline.Despatcher import getWorkerID, registerWorker, waitForQueue, getJob, jobToPath, getWorkList, jobCompleted

# Local directories
ARCHIVE = "/archive"
CALSTARS_DATA_DIR = "CALSTARS"
QUEUES = "/archive/despatcher/queues"
REGISTRATIONS = "/archive/despatcher/registrations"


def encodeOffSetTangential(ra1, dec1, ra2, dec2):

    # _ denotes delta
    _ra = ((ra1 - ra2 + 180) % 360) - 180
    _x = _ra * np.cos(np.radians(dec2))
    _y = dec1 - dec2

    return _x, _y

def decodeOffsetTangential(ra1, dec1, _x, _y):

    dec2 = dec1 + _y
    ra2 = ra1  + _x / np.cos(np.radians(dec1))
    ra2 %= 360

    return ra2, dec2

import numpy as np


def encodeOffsetGnomonic(ra1, dec1, ra2, dec2):

    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)

    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    dra = ra1 - ra2

    cos_c = np.sin(dec2) * np.sin(dec1) + np.cos(dec2) * np.cos(dec1) * np.cos(dra)

    xi = (np.cos(dec1) * np.sin(dra)) / cos_c

    eta = (np.cos(dec2) * np.sin(dec1) - np.sin(dec2) * np.cos(dec1) * np.cos(dra)) / cos_c

    return np.degrees(xi), np.degrees(eta)



def decodeOffsetGnomonic(ra2, dec2, xi, eta):

    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    xi = np.radians(xi)
    eta = np.radians(eta)

    ra1 = ra2 + np.arctan2(
        xi,
        np.cos(dec2) - eta * np.sin(dec2)
    )

    dec1 = np.arctan2(
        np.sin(dec2) + eta * np.cos(dec2),
        np.sqrt(
            (np.cos(dec2) - eta * np.sin(dec2))**2
            + xi**2
        )
    )

    ra1 = np.degrees(ra1) % 360.0
    dec1 = np.degrees(dec1)

    return ra1, dec1



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




def calstarToLightCurves(cat, obs_con, pp, calstar, platepars_all_recalibrated_path, video_path=None, debug_mode=False):

    light_curves = {}
    produce_debug_timelapse = False


    # Test to see if recal file is available
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

    # If not then reject this observation session
    if pp_recal_json is None:
        print("No recalibration file found, discarding session")
        return

    if video_path is not None:
        produce_debug_timelapse = True
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                30, (pp.X_res * 2, pp.Y_res), isColor=False)

    print("Building fits file list")
    fits_files_from_calstar_list = [calstar_entry[0] for calstar_entry in calstar]
    print(f"Filter {len(fits_files_from_calstar_list)} fits files for astronomical night list")
    astronomical_night_list, sun_below_horizon_angle_list, setting_count, rising_count = minSunBelowHorizon(fits_files_from_calstar_list, obs_con, sun_angle=-18)
    print(f"Filter {len(astronomical_night_list)} fits files to find files without the moon")
    fits_files_to_process_list = detectMoon(astronomical_night_list, pp, obs_con)
    fits_files_to_process = len(fits_files_to_process_list)
    print(f"Starting to process {fits_files_to_process} fits files - astronomical night and no moon")

    counter = 0
    recalibrated_platepar_loaded = False
    print("Starting to iterate through fits_files")
    for fits_file, star_list in calstar:
        counter += 1
        if counter > 10 and debug_mode:
            break
        star_count = len(star_list)
        if star_count < 80:
            continue

        if fits_file not in fits_files_to_process_list:
            continue

        # We could improve this by picking the nearest platepar
        if pp_recal_json is not None:
            if fits_file in pp_recal_json:
                if recalibrated_platepar_loaded:
                    pass
                    #print(f"Loaded recalibrated platepar for : {fits_file} {counter}/{fits_files_to_process}")
                else:
                    print(f"Loaded first recalibrated platepar for : {fits_file}")

                recalibrated_platepar_loaded = True
                pp.loadFromDict(pp_recal_json[fits_file])


        # Throw away observations until we have our first recalibrated platepar
        if not recalibrated_platepar_loaded:
            continue

        jd = datetime2JD(getMiddleTimeFF(fits_file, obs_con.fps, dt_obj=True))


        p, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd, n_brightest=5*len(star_list))



        limiting_mag = np.max(mag_array) + 0.5

        y, x, level, amplitude, fwhm, background, snr, saturated_count = zip(*star_list)
        jd_list = [jd] * len(star_list)
        _arr_jd, arr_obs_ra, arr_obs_dec, arr_obs_mag = xyToRaDecPP(jd_list, list(x), list(y), list(level), pp,
                                                                    jd_time=True, measurement=True,
                                                                    precompute_pointing_corr=True,
                                                                    extinction_correction=True)

        img_stars = np.column_stack([x, y, arr_obs_mag])
        catalog_stars = np.column_stack([ra_array, dec_array, mag_array])
        res_success, res_fun, inlier_fraction, inlier_rmsd = pp.fitPointingNN(jd, img_stars, catalog_stars)
        if not res_success:
            continue

        # Filter catalog return so we do not try to match against faint stars

        limiting_mag = np.max(arr_obs_mag) + 0.5
        good = mag_array <= limiting_mag
        x_array, y_array = x_array[good], y_array[good]
        ra_array, dec_array = ra_array[good], dec_array[good]
        mag_array = mag_array[good]
        names_array = names_array[good]

        if produce_debug_timelapse:
            query_img, agc_level = renderStarsGaussian(x_array, y_array, mag_array, (pp.Y_res, pp.X_res))
            calstar_img, agc_level = renderStarsGaussian(list(x), list(y), arr_obs_mag, (pp.Y_res, pp.X_res))
            combined_img = np.hstack((query_img, calstar_img))
            video.write(combined_img)

        star_count = len(star_list)
        for observation_index, star in enumerate(star_list):

            if len(star) == 8:
                obs_x, obs_y = star[1], star[0]
                ra_obs, dec_obs, mag_obs = arr_obs_ra[observation_index], arr_obs_dec[observation_index], arr_obs_mag[observation_index]
                img_dist = (x_array - obs_x) ** 2 + (y_array - obs_y) ** 2

                candidates = np.where(img_dist <= 2 ** 2)[0]

                if len(candidates):

                    brightest_idx = candidates[
                        np.argmin(mag_array[candidates])
                    ]

                    nearest_idx = brightest_idx
                    name = names_array[nearest_idx]
                    ra_cat, dec_cat = ra_array[nearest_idx], dec_array[nearest_idx]
                    xi, eta = encodeOffsetGnomonic(ra_obs, dec_obs, ra_cat, dec_cat)
                    if False:
                        xi_test, eta_test = int(xi * 10000) / 10000, int(eta * 10000) / 10000
                        ra_decoded, dec_decoded = decodeOffsetGnomonic(ra_cat, dec_cat, xi_test, eta_test)
                        error = angularSeparationDeg(ra_decoded, dec_decoded, ra_obs, dec_obs)
                        print(f"Small angle encoding error {error} degrees")

                    if name not in light_curves:
                        light_curves[name] = {}


                    light_curves[name][jd] = {}
                    light_curves[name][jd]['obs'] = mag_obs
                    light_curves[name][jd]['cat'] = mag_array[nearest_idx]
                    light_curves[name][jd]['ra_cat'] = ra_cat
                    light_curves[name][jd]['dec_cat'] = dec_cat
                    light_curves[name][jd]['level'] = star[2]
                    light_curves[name][jd]['amplitude'] = star[3]
                    light_curves[name][jd]['fwhm'] = star[4]
                    light_curves[name][jd]['background'] = star[5]
                    light_curves[name][jd]['snr'] = star[6]
                    light_curves[name][jd]['saturated_count'] = star[7]
                    light_curves[name][jd]['star_count'] = star_count
                    light_curves[name][jd]['xi'] = xi
                    light_curves[name][jd]['eta'] = eta


        if produce_debug_timelapse:
            video.release()

    return light_curves

def process(path_to_job, cat):


    print(f"Processing {path_to_job}")

    observation_dir = os.path.basename(path_to_job).split("_min")[0]


    with tempfile.TemporaryDirectory() as temp_dir:


        with tarfile.open(path_to_job, "r:bz2") as tar:

            try:
                tar.extractall(temp_dir, filter="fully_trusted")
            except:
                tar.extractall(temp_dir)

            working_dir = os.path.join(temp_dir, observation_dir)
            config_path = os.path.join(working_dir, ".config")

            if os.path.isfile(config_path):
                obs_con = cr.parse(config_path)
            else:
                print(f"No config file for {observation_dir}")
                return

            pp = Platepar()
            pp_path = os.path.join(working_dir, obs_con.platepar_name)
            pp.read(pp_path)
            calstars_filename = f"CALSTARS_{observation_dir}.txt"
            print(f"Reading {calstars_filename}")
            calstar = readCALSTARS(working_dir, calstars_filename)
            video_file = f"{observation_dir}_timelapse.mp4"
            video_path = os.path.join(os.path.expanduser(config.data_dir), video_file)
            platepars_all_recalibrated_path = os.path.join(working_dir, "platepars_all_recalibrated.json")

            if len(calstar) == 2:

                print(f"Starting light curve work")
                light_curves = calstarToLightCurves(cat, config, pp, calstar[0], platepars_all_recalibrated_path, video_path=None)
                if light_curves is None:
                    return

                station, date_str, time_str, us_str = observation_dir.split('_')
                session_date = datetime.datetime.strptime(f"{date_str}_{time_str}_{us_str}", "%Y%m%d_%H%M%S_%f")
                session_jd = datetime2JD(session_date)
                print(f"Starting filing work for {len(light_curves)} stars")
                for star_name, observations in light_curves.items():

                    daily_data = {}

                    for jd_str, data in observations.items():

                        observation_jd = float(jd_str)


                        if observation_dir not in daily_data:
                            daily_data[observation_dir] = []

                        offset = int((observation_jd - session_jd) * 10000)

                        if not np.isfinite(data['obs']):
                            print(f"Skipping observation of {star_name}")

                            continue

                        daily_data[observation_dir].append([offset,
                                                            int(round(100 * (data['obs'] - data['cat']))),
                                                            data['level'],
                                                            data['amplitude'],
                                                            data['fwhm'] * 100,
                                                            data['background'],
                                                            data['snr'] * 100,
                                                            data['saturated_count'],
                                                            data['star_count'],
                                                            data['xi'] * 10000,
                                                            data['eta'] * 10000,
                                                            ])

                    for session, records in daily_data.items():
                        cat_ra, cat_dec = data['ra_cat'], data['dec_cat']
                        dirname = Path("/archive/light_curves/") / star_name / obs_con.stationID
                        dirname.mkdir(parents=True, exist_ok=True)

                        filename = dirname / f"{session}_{int(data['cat'] * 1e6)}_{int(cat_ra * 1e6)}_{int(cat_dec* 1e6)}_{int(round(pp.F_scale * 1000))}.npy"
                        array_to_save = np.array(records, dtype=np.int16)
                        np.save(filename, array_to_save)
    print("Completed")


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Convert GMN data cache into star light curves \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("-r", "--reverse", action="store_true", help="Process newest jobs first")

    arg_parser.add_argument("-s", "--single", action="store_true", help="Do not use despatcher")



    if False:

        print("Test encode offset")

        print("Test tangential")
        # Test Alcyone at ra,dec 56.87740, 24.1062
        # Against Meope at ra,dec, 56.58783, 23.9492

        alcyone_ra, alcyone_dec = 56.87740, 24.1062
        meope_ra, meope_dec = 56.58783, 23.9492
        _x, _y = encodeOffSetTangential(alcyone_ra, alcyone_dec, meope_ra, meope_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_decoded, dec_decoded = decodeOffsetTangential(meope_ra, meope_dec, _x, _y)
        ang_sep = angularSeparationDeg(alcyone_ra, alcyone_dec, ra_decoded, dec_decoded)
        print(f"regenerated from offset: {ra_decoded}, {dec_decoded} - angular separtion deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        _x, _y = encodeOffSetTangential(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_orig, dec_orig = decodeOffsetTangential(s2_ra, s2_dec, _x, _y)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        _x, _y = encodeOffSetTangential(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_orig, dec_orig = decodeOffsetTangential(s2_ra, s2_dec, _x, _y)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Ra Wrp
        s1_ra, s1_dec = 359.40511, 73.8707
        s2_ra, s2_dec = 0.42559, 73.6172
        _x, _y = encodeOffSetTangential(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_orig, dec_orig = decodeOffsetTangential(s2_ra, s2_dec, _x, _y)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        _x, _y = encodeOffSetTangential(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_orig, dec_orig = decodeOffsetTangential(s2_ra, s2_dec, _x, _y)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Ra Wrp
        s1_ra, s1_dec = 359.40511, 73.8707
        s2_ra, s2_dec = 0.42559, 73.6172
        _x, _y = encodeOffSetTangential(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {_x}, {_y}")
        ra_orig, dec_orig = decodeOffsetTangential(s2_ra, s2_dec, _x, _y)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        print("Test gnomonic")

        # Test Alcyone at ra,dec 56.87740, 24.1062
        # Against Meope at ra,dec, 56.58783, 23.9492

        alcyone_ra, alcyone_dec = 56.87740, 24.1062
        meope_ra, meope_dec = 56.58783, 23.9492
        xi, eta = encodeOffsetGnomonic(alcyone_ra, alcyone_dec, meope_ra, meope_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_decoded, dec_decoded = decodeOffsetGnomonic(meope_ra, meope_dec, xi, eta)
        ang_sep = angularSeparationDeg(alcyone_ra, alcyone_dec, ra_decoded, dec_decoded)
        print(f"regenerated from offset: {ra_decoded}, {dec_decoded} - angular separtion deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        xi, eta = encodeOffsetGnomonic(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_orig, dec_orig = decodeOffsetGnomonic(s2_ra, s2_dec, xi, eta)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        xi, eta = encodeOffsetGnomonic(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_orig, dec_orig = decodeOffsetGnomonic(s2_ra, s2_dec, xi, eta)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Ra Wrp
        s1_ra, s1_dec = 359.40511, 73.8707
        s2_ra, s2_dec = 0.42559, 73.6172
        xi, eta = encodeOffsetGnomonic(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_orig, dec_orig = decodeOffsetGnomonic(s2_ra, s2_dec, xi, eta)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Going over the declination, so around ra
        s1_ra, s1_dec = 29.49906, 89.7957
        s2_ra, s2_dec = 320.10718, 89.8860
        xi, eta = encodeOffsetGnomonic(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_orig, dec_orig = decodeOffsetGnomonic(s2_ra, s2_dec, xi, eta)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

        # Ra Wrp
        s1_ra, s1_dec = 359.40511, 73.8707
        s2_ra, s2_dec = 0.42559, 73.6172
        xi, eta = encodeOffsetGnomonic(s1_ra, s1_dec, s2_ra, s2_dec)
        print(f"Encoded offset: {xi}, {eta}")
        ra_orig, dec_orig = decodeOffsetGnomonic(s2_ra, s2_dec, xi, eta)
        ang_sep = angularSeparationDeg(ra_orig, dec_orig, s1_ra, s1_dec)
        print(f"regenerated from offset: {ra_orig}, {dec_orig} - angular separation deg: {ang_sep}")

    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    os.makedirs(ARCHIVE, exist_ok=True)
    calstars_directory_path = os.path.join(ARCHIVE, CALSTARS_DATA_DIR)
    os.makedirs(calstars_directory_path, exist_ok=True)



    print("Loading catalog")
    cat = Catalog(config)
    print("Catalog loaded")

    if cml_args.single:

        open("/archive/processed_sessions.txt", "a").close()

        with open("/archive/processed_sessions.txt") as f:
            processed_sessions = {
                line.strip()
                for line in f
                if line.strip()
            }

        work_list = getWorkList(calstars_directory_path)
        if cml_args.reverse:
            work_list.reverse()



        for path_to_job in work_list:
            job = os.path.basename(path_to_job)
            if job not in processed_sessions:
                process(path_to_job, cat)
                processed_sessions.add(job)
                with open("/archive/processed_sessions.txt", "a") as f:
                    f.write(job + "\n")
            else:
                print(f"{job} already processed")

    else:

        print("Running using despatcher")

        worker_id = getWorkerID()
        registerWorker(worker_id)

        while True:
            waitForQueue(worker_id)
            job = getJob(worker_id)
            if job is None:
                print("Queue empty")
                time.sleep(10)
            else:
                # Re-register with the same id to show despatch we are still alive
                registerWorker(worker_id)
                path_to_job = jobToPath(job)
                process(path_to_job, cat)
                jobCompleted(worker_id, job)