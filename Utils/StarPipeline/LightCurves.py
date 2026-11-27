# RPi Meteor Station
# Copyright (C) 2026 David Rollinson
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
from typing import Any

import numpy as np
import json
import RMS.ConfigReader as cr
import datetime
import time
import cv2
import ephem
import tqdm

from RMS.Astrometry.Conversions import datetime2JD, jd2Date
from RMS.Formats.FFfile import getMiddleTimeFF
from pathlib import Path
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalogQueries import Catalog, renderStarsGaussian
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from Utils.Flux import detectMoon
from Utils.StarPipeline.Despatcher import getWorkerID, registerWorker, waitForQueue, getJob , getArchiveList, jobCompleted
from Utils.StarPipeline.Cache import filenameToLocalPath

# CONSTANTS

LIGHT_CURVES_PATH = "/srv/lightcurves/stars/byobservationsession"
from Utils.StarPipeline.Cache import LOCAL_PATH_TEMPLATE, MINIMAL_ARCHIVE_SUFFIX



def encodeOffSetTangential(ra1, dec1, ra2, dec2):
    """Encode a small angular offset using a tangential approximation.

    Arguments:
        ra1: [float]
            Observation right ascension in degrees.

        dec1: [float]
            Observation declination in degrees.

        ra2: [float]
            Reference right ascension in degrees.

        dec2: [float]
            Reference declination in degrees.

    Return:
        _x: [float]
            Tangential right ascension offset in arcseconds.

        _y: [float]
            Tangential declination offset in arcseconds.
    """

    # _ denotes delta
    _ra = ((ra1 - ra2 + 180) % 360) - 180
    _x = _ra * np.cos(np.radians(dec2))
    _y = dec1 - dec2

    return _x / 3600, _y / 3600

def decodeOffsetTangential(ra1, dec1, _x, _y):
    """Reconstruct celestial coordinates from tangential offsets.

    Arguments:
        ra1: [float]
            Reference right ascension in degrees.

        dec1: [float]
            Reference declination in degrees.

        _x: [float]
            Tangential right ascension offset in arcseconds.

        _y: [float]
            Tangential declination offset in arcseconds.

    Return:
        ra2: [float]
            Reconstructed right ascension in degrees.

        dec2: [float]
            Reconstructed declination in degrees.
    """


    dec2 = dec1 + _y * 3600
    ra2 = ra1  + (_x * 3600) / np.cos(np.radians(dec1))
    ra2 %= 360

    return ra2, dec2


def encodeOffsetGnomonic(ra1, dec1, ra2, dec2):
    """Encode a celestial offset using a gnomonic projection.

    Arguments:
        ra1: [float]
            Observation right ascension in degrees.

        dec1: [float]
            Observation declination in degrees.

        ra2: [float]
            Reference right ascension in degrees.

        dec2: [float]
            Reference declination in degrees.

    Return:
        xi: [float]
            Gnomonic projection offset in arcseconds.

        eta: [float]
            Gnomonic projection offset in arcseconds.
    """
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)

    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    dra = ra1 - ra2

    cos_c = np.sin(dec2) * np.sin(dec1) + np.cos(dec2) * np.cos(dec1) * np.cos(dra)

    xi = (np.cos(dec1) * np.sin(dra)) / cos_c

    eta = (np.cos(dec2) * np.sin(dec1) - np.sin(dec2) * np.cos(dec1) * np.cos(dra)) / cos_c

    return np.degrees(xi) / 3600, np.degrees(eta) / 3600



def decodeOffsetGnomonic(ra2, dec2, xi, eta):
    """Reconstruct celestial coordinates from gnomonic offsets.

    Arguments:
        ra2: [float]
            Reference right ascension in degrees.

        dec2: [float]
            Reference declination in degrees.

        xi: [float]
            Gnomonic projection offset in arcseconds.

        eta: [float]
            Gnomonic projection offset in arcseconds.

    Return:
        ra1: [float]
            Reconstructed right ascension in degrees.

        dec1: [float]
            Reconstructed declination in degrees.
    """
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    xi = np.radians(xi * 3600)
    eta = np.radians(eta * 3600)

    ra1 = ra2 + np.arctan2(xi, np.cos(dec2) - eta * np.sin(dec2))

    dec1 = np.arctan2(np.sin(dec2) + eta * np.cos(dec2), np.sqrt((np.cos(dec2) - eta * np.sin(dec2))**2 + xi**2))

    ra1 = np.degrees(ra1) % 360.0
    dec1 = np.degrees(dec1)

    return ra1, dec1



def minSunBelowHorizon(fits_file_list, c, sun_angle=-18, chunk_size=10):
    """Filter FITS files based on Solar altitude.

    Arguments:
        fits_file_list: [list]
            List of FITS filenames.

        c: [Config]
            RMS station configuration.

    Keyword arguments:
        sun_angle: [float]
            Solar altitude threshold in degrees.

        chunk_size: [int]
            Number of frames between Solar altitude calculations.

    Return:
        astronomical_night_list: [list]
            FITS files recorded with the Sun below the threshold.

        angle_list: [ndarray]
            Solar altitude in degrees for each FITS file.

        setting_count: [int]
            Number of frames for which the Sun was setting.

        rising_count: [int]
            Number of frames for which the Sun was rising.
    """
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

            last_sun_alt_deg = sun_alt_deg
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


def getFitsFilesToProcess(calstar, obs_con, pp):



    fits_files_from_calstar_list = [calstar_entry[0] for calstar_entry in calstar]
    astronomical_night_list, sun_below_horizon_angle_list, setting_count, rising_count = minSunBelowHorizon(
        fits_files_from_calstar_list, obs_con, sun_angle=-18)
    fits_files_to_process_list = detectMoon(astronomical_night_list, pp, obs_con)
    return fits_files_to_process_list

def getRecalibratedPlatepars(platepars_all_recalibrated_path):


    # Test to see if recal file is available
    if os.path.exists(platepars_all_recalibrated_path):
        try:
            with open(platepars_all_recalibrated_path, 'r') as fh:
                pp_recal_json = json.load(fh)
                # log.info(f"Read {os.path.basename(local_recal_path)}")
        except:
            pp_recal_json = None
            print(f"{platepars_all_recalibrated_path} was corrupted")
    else:
        print(f"No file {os.path.basename(platepars_all_recalibrated_path)}`")
        pp_recal_json = None
    return pp_recal_json

def filterCatalogByMagnitude(limiting_mag, ra_array, dec_array, mag_array, x_array, y_array, names_array):
    """Filter catalog stars by limiting magnitude.

    Remove catalog stars fainter than the specified limiting
    magnitude while preserving alignment between all catalog arrays.

    Arguments:
        limiting_mag: [float]
            Faintest catalog magnitude to include.

        ra_array: [ndarray]
            Catalog right ascensions in degrees.

        dec_array: [ndarray]
            Catalog declinations in degrees.

        mag_array: [ndarray]
            Catalog magnitudes.

        x_array: [ndarray]
            Catalog X image coordinates.

        y_array: [ndarray]
            Catalog Y image coordinates.

        names_array: [ndarray]
            Catalog star identifiers.

    Return:
        ra_array: [ndarray]
            Filtered catalog right ascensions.

        dec_array: [ndarray]
            Filtered catalog declinations.

        mag_array: [ndarray]
            Filtered catalog magnitudes.

        x_array: [ndarray]
            Filtered catalog X coordinates.

        y_array: [ndarray]
            Filtered catalog Y coordinates.

        names_array: [ndarray]
            Filtered catalog star identifiers.
    """


    include = mag_array <= limiting_mag
    x_array, y_array = x_array[include], y_array[include]
    ra_array, dec_array = ra_array[include], dec_array[include]
    mag_array = mag_array[include]
    names_array = names_array[include]

    return ra_array, dec_array, mag_array, x_array, y_array, names_array

def alignPlatePar(arr_obs_mag, dec_array, jd, mag_array, pp, ra_array, x, y):
    """Align a platepar using observed and catalog stars.

    Build star lists in image and celestial coordinates and
    attempt to refine the platepar pointing solution.

    Arguments:
        arr_obs_mag: [ndarray]
            Observed stellar magnitudes.

        dec_array: [ndarray]
            Catalog declinations in degrees.

        jd: [float]
            Observation Julian date.

        mag_array: [ndarray]
            Catalog magnitudes.

        pp: [Platepar]
            Platepar object to align.

        ra_array: [ndarray]
            Catalog right ascensions in degrees.

        x: [ndarray]
            Observed star X coordinates.

        y: [ndarray]
            Observed star Y coordinates.

    Return:
        res_success: [bool]
            True if the platepar alignment was successful.
    """

    # Image stars in image coordinates
    img_stars = np.column_stack([x, y, arr_obs_mag])

    # Catalog stars in sky coordinates
    catalog_stars = np.column_stack([ra_array, dec_array, mag_array])

    # Align pointing
    res_success, res_fun, inlier_fraction, inlier_rmsd = pp.fitPointingNN(jd, img_stars, catalog_stars)
    return res_success

def populateLightCurves(jd, star, light_curves, arr_obs_ra, arr_obs_dec, arr_obs_mag, ra_array, dec_array, mag_array,
                        names_array, x_array, y_array, observation_index, star_count):
    """Associate an observation with a catalog star and store it.

    Find catalog candidates near an observed star, select the
    brightest candidate within the matching radius, calculate
    astrometric offsets, and store the resulting light-curve
    measurement.

    Arguments:
        jd: [float]
            Observation Julian date.

        star: [tuple]
            CALSTARS observation entry.

        light_curves: [dict]
            Light curve dictionary to update.

        arr_obs_ra: [ndarray]
            Observed right ascensions in degrees.

        arr_obs_dec: [ndarray]
            Observed declinations in degrees.

        arr_obs_mag: [ndarray]
            Observed magnitudes.

        ra_array: [ndarray]
            Catalog right ascensions in degrees.

        dec_array: [ndarray]
            Catalog declinations in degrees.

        mag_array: [ndarray]
            Catalog magnitudes.

        names_array: [ndarray]
            Catalog star identifiers.

        x_array: [ndarray]
            Catalog X image coordinates.

        y_array: [ndarray]
            Catalog Y image coordinates.

        observation_index: [int]
            Index of the observed star within the frame.

        star_count: [int]
            Number of stars detected in the frame.

    Return:
        None

    Notes:
        Catalog stars are matched to observations using a
        two-pixel search radius. If multiple catalog stars are
        found within this radius, the brightest candidate is
        selected.
    """

    if len(star) == 8:
        obs_x, obs_y = star[1], star[0]
        ra_obs, dec_obs, mag_obs = arr_obs_ra[observation_index], arr_obs_dec[observation_index], arr_obs_mag[
            observation_index]
        img_dist = (x_array - obs_x) ** 2 + (y_array - obs_y) ** 2

        # Find the stars within 2 pixels
        candidates = np.where(img_dist <= 2 ** 2)[0]

        if len(candidates):

            # Find the brightest star within the limit
            brightest_idx = candidates[np.argmin(mag_array[candidates])]

            name = names_array[brightest_idx]
            ra_cat, dec_cat = ra_array[brightest_idx], dec_array[brightest_idx]
            xi, eta = encodeOffsetGnomonic(ra_obs, dec_obs, ra_cat, dec_cat)

            if name not in light_curves:
                light_curves[name] = {}

            light_curves[name][jd] = {}
            light_curves[name][jd]['obs'] = mag_obs
            light_curves[name][jd]['cat'] = mag_array[brightest_idx]
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

def calstarToLightCurves(cat, obs_con, pp, calstar, platepars_all_recalibrated_path, min_star_count=80):
    """Convert CALSTARS observations into stellar light curves.

    Load recalibrated platepars, reject observations affected by
    daylight or moonlight, match observed stars against a catalog,
    and generate per-star photometric measurements.

    Arguments:
        cat: [Catalog]
            Stellar catalog.

        obs_con: [Config]
            Observation configuration.

        pp: [Platepar]
            Initial platepar.

        calstar: [list]
            CALSTARS observations.

        platepars_all_recalibrated_path: [str]
            Path to the recalibrated platepar JSON file.

    Keyword arguments:
        min_star_count: Optional, default: 80. Minimum number of stars for a frame to be accepted

    Return:
        light_curves: [dict]
            Light curve measurements indexed by star name and
            Julian date.

        None:
            Returned if the observation session cannot be
            recalibrated.
    """

    light_curves = {}

    pp_recal_json = getRecalibratedPlatepars(platepars_all_recalibrated_path)

    # If not then reject this observation session
    if pp_recal_json is None:
        print("No recalibration file found, discarding session")
        return

    fits_files_to_process_list = getFitsFilesToProcess(calstar, obs_con, pp)

    recalibrated_platepar_loaded = False

    # Iterate over the fits files
    for fits_file, star_list in calstar:

        star_count = len(star_list)
        if star_count < min_star_count:
            continue

        if fits_file not in fits_files_to_process_list:
            continue


        if pp_recal_json is not None:
            if fits_file in pp_recal_json:

                recalibrated_platepar_loaded = True
                pp.loadFromDict(pp_recal_json[fits_file])

        # Throw away observations until we have our first recalibrated platepar
        if not recalibrated_platepar_loaded:
            continue

        jd = datetime2JD(getMiddleTimeFF(fits_file, obs_con.fps, dt_obj=True))

        # Get the stars in the platepar
        p, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd, n_brightest=5*len(star_list))

        # Get individual variables from the star list extracted from the calstar file.
        y, x, level, amplitude, fwhm, background, snr, saturated_count = zip(*star_list)

        # Create a list of julian dates the same shape as the other parameters
        jd_list = [jd] * len(star_list)
        _arr_jd, arr_obs_ra, arr_obs_dec, arr_obs_mag = xyToRaDecPP(jd_list, list(x), list(y), list(level), pp,
                                                                    jd_time=True, measurement=True,
                                                                    precompute_pointing_corr=True,
                                                                    extinction_correction=True)

        # If we could get a successful alignment, then discard only this frame
        if not alignPlatePar(arr_obs_mag, dec_array, jd, mag_array, pp, ra_array, x, y):
            continue

        # Get a magnitude slightly fainter than the faintest observation
        limiting_mag = np.max(arr_obs_mag) + 0.5

        ra_array, dec_array, mag_array, x_array, y_array, names_array = filterCatalogByMagnitude(limiting_mag,
                                                                                                 ra_array, dec_array, mag_array,
                                                                                                 x_array, y_array, names_array)

        for observation_index, star in enumerate(star_list):
            populateLightCurves(jd, star, light_curves, arr_obs_ra, arr_obs_dec, arr_obs_mag, ra_array, dec_array, mag_array,
                                names_array, x_array, y_array, observation_index, star_count)

    return light_curves



def process(sys_con, path_to_job, cat, cache_path):
    """Process a cached observation archive.

    Extract a minimal archive, load calibration products, generate
    light curves, and save per-star observation files.

    Arguments:
        sys_con: [Config]
            Processing system configuration.

        path_to_job: [str]
            Path to the cached observation archive.

        cat: [Catalog]
            Stellar catalog.

    Return:
        None
    """

    print(f"Processing {path_to_job}")

    observation_dir = os.path.basename(path_to_job).split("_min")[0]


    with tempfile.TemporaryDirectory() as temp_dir:


        with tarfile.open(path_to_job, "r:bz2") as tar:

            try:
                tar.extractall(temp_dir, filter="fully_trusted")
            except:
                tar.extractall(temp_dir)

            working_dir = os.path.join(temp_dir, observation_dir)
            obs_con_path = os.path.join(working_dir, ".config")

            if os.path.isfile(obs_con_path):
                obs_con = cr.parse(obs_con_path)
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
            video_path = os.path.join(os.path.expanduser(sys_con.data_dir), video_file)
            platepars_all_recalibrated_path = os.path.join(working_dir, "platepars_all_recalibrated.json")

            if len(calstar) == 2:

                print(f"Starting light curve work")
                light_curves = calstarToLightCurves(cat, obs_con, pp, calstar[0], platepars_all_recalibrated_path, video_path=None)
                if light_curves is None:
                    return

                station, date_str, time_str, us_str = observation_dir.split('_')
                session_date = datetime.datetime.strptime(f"{date_str}_{time_str}_{us_str}", "%Y%m%d_%H%M%S_%f")
                session_jd = datetime2JD(session_date)
                print(f"Starting filing work for {len(light_curves)} stars")
                for star_name, observations in light_curves.items():

                    observation_data = {}

                    if not observations:
                        continue

                    value = next(iter(observations.values()))
                    cat_ra, cat_dec = value['ra_cat'], value['dec_cat']
                    cat_mag = value['cat']

                    for jd_str, data in observations.items():

                        observation_jd = float(jd_str)


                        if observation_dir not in observation_data:
                            observation_data[observation_dir] = []

                        offset = int((observation_jd - session_jd) * 10000)

                        if not np.isfinite(data['obs']):
                            print(f"Skipping observation of {star_name}")

                            continue

                        observation_data[observation_dir].append([offset,
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



                    for session, records in observation_data.items():

                        dirname = Path(cache_path) / star_name / obs_con.stationID
                        dirname.mkdir(parents=True, exist_ok=True)

                        filename = dirname / f"{session}_{int(cat_mag * 1e6)}_{int(cat_ra * 1e6)}_{int(cat_dec* 1e6)}_{int(round(pp.F_scale * 1000))}.npy"
                        array_to_save = np.array(records, dtype=np.int16)
                        print(f"Saving to {filename}")
                        np.save(filename, array_to_save)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Convert GMN data cache into star light curves \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("-r", "--reverse", action="store_true", help="Process newest jobs first")

    arg_parser.add_argument("-s", "--single", action="store_true", help="Do not use despatcher")



    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    sys_con = cr.parse(os.path.join(os.getcwd(),".config"))


    cache_path = os.path.dirname(LOCAL_PATH_TEMPLATE)
    os.makedirs(cache_path, exist_ok=True)



    cat = Catalog(sys_con)
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
            path_to_job = filenameToLocalPath(job, local_path_template=LOCAL_PATH_TEMPLATE, minimal_archive_suffix=MINIMAL_ARCHIVE_SUFFIX)
            process(sys_con, path_to_job, cat, LIGHT_CURVES_PATH)
            jobCompleted(worker_id, job)