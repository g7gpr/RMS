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
import numpy as np
import datetime
import time
import json

import matplotlib.pyplot as plt

import RMS.ConfigReader as cr

from pathlib import Path
from RMS.Astrometry.Conversions import datetime2JD
from Utils.StarPipeline.LightCurves import decodeOffsetGnomonic, LIGHT_CURVE_TEMPLATE_PATH
from RMS.Formats.StarCatalogQueries import Catalog




def getStars():

    root = Path(LIGHT_CURVE_TEMPLATE_PATH)

    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
    )

def getSessionCount(star_name):

    star_dir = Path(LIGHT_CURVE_TEMPLATE_PATH) / star_name

    return len(list(star_dir.rglob("*.npy")))

def getStarFiles(star_name, stations_list=None):

    star_dir = Path(LIGHT_CURVE_TEMPLATE_PATH) / star_name

    if not star_dir.exists():
        return []

    if stations_list is None:
        return sorted(
            star_dir.rglob("*.npy")
        )

    files = []

    for station_id in stations_list:

        station_dir = star_dir / station_id

        if not station_dir.exists():
            continue

        files.extend(
            station_dir.rglob("*.npy")
        )

    return sorted(files)

def getStarObservations(star_name, stations_list=None):

    arrays = []
    stations = set()

    archived_stars = getStarFiles(star_name, stations_list)
    for filename in archived_stars:

        arrays.append(loadObservationFile(filename))
        stations.add(filename.parent.name)

    if not len(arrays):
        return None, arrays, []

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


    # Force everything to be NxM - to avoid single row returns causing vectors
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2:
        raise ValueError(
            f"Unexpected array shape {data.shape} in {filename}"
        )

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

    observations["xi"] = data[:, 9]
    observations["eta"] = data[:, 10]


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

def stationsText(obs, stations_list):


    ra_cat = np.mean(obs['ra_cat'])
    dec_cat = np.mean(obs['dec_cat'])

    stations_text = ", ".join(sorted(stations_list))

    number_of_observations = len(obs)
    number_of_stations = len(stations_list)

    if number_of_stations > 10:
        stations_text = (
            f"Observed by {number_of_stations} stations "
            f"with {number_of_observations} individual observations"
        )

    return stations_text

def simpleMagnitudeScatterPlot(plot_data):

    star_name, obs, stations_list = plot_data
    stations_text = stationsText(obs, stations_list)
    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    plt.suptitle(f"{star_name} magnitude plot", fontsize=14)
    ra_cat, dec_cat = np.mean(obs['ra_cat']), np.mean(obs['dec_cat'])
    ax1.set_title(f"Ra:{ra_cat:.2f}deg Dec:{dec_cat:.2f}deg\n{stations_text}", fontsize=8, color="gray")

    for station in stations_list:
        mask = obs["station"] == station
        ax1.scatter(obs["jd"][mask], obs["obs_mag"][mask], s=5, label=station)

    ax1.axhline(y=obs["cat_mag"][0], color="red", linestyle=":", linewidth=1, alpha=0.5)

    ax1.invert_yaxis()

    ax1.set_xlabel("Julian Date")
    ax1.set_ylabel("Magnitude")

    plt.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax1.grid(True)

    safe_name = star_name.replace("/", "_").replace(" ", "_")

    filename = f"{safe_name}_light_curves.png"

    filepath = "/archive/lightcurves/plots"

    os.makedirs(filepath, exist_ok=True)

    plt.savefig(os.path.join(filepath, filename), dpi=150, bbox_inches="tight")

    plt.close()

    print(f"Saved {filename}")

def plotAstrometryRaDec(plot_data):

    star_name, obs, stations_list = plot_data
    stations_text = stationsText(obs, stations_list)
    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    ra_cat, dec_cat = np.mean(obs['ra_cat']), np.mean(obs['dec_cat'])
    residual = obs["obs_mag"] - obs["cat_mag"]

    ax1.set_title(f"{star_name} in Ra Dec\nRa:{ra_cat:.2f}deg Dec:{dec_cat:.2f}deg\n{stations_text}",fontsize=10,color="Black")
    plt.scatter(obs["ra_obs"], obs["dec_obs"], c=residual, s=2, cmap="coolwarm", vmin=np.percentile(residual, 5), vmax=np.percentile(residual, 95))
    plt.colorbar(label="Obs - Cat Magnitude")
    plt.scatter(obs["ra_cat"][0], obs["dec_cat"][0], color="red", marker="+", s=80, linewidths=1, alpha=0.5, label="Catalog")
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_box_aspect(1)

    ra_centre = np.mean(obs["ra_cat"])
    dec_centre = np.mean(obs["dec_cat"])

    r = max(
        np.max(np.abs(obs["ra_obs"] - ra_centre)),
        np.max(np.abs(obs["dec_obs"] - dec_centre))
    )

    r *= 1.1  # 10% margin

    x_low, x_high = ra_centre - r, ra_centre + r
    y_low, y_high = dec_centre - r, dec_centre + r

    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)

    plt.tight_layout()
    safe_name = star_name.replace("/", "_").replace(" ", "_")
    filename = f"{safe_name}_astrometry_radec.png"
    filepath = "/archive/lightcurves/plots"
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(os.path.join(filepath, filename), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {filename}")

def plotAstrometryXiEta(plot_data):


    star_name, obs, stations_list = plot_data

    stations_text = stationsText(obs, stations_list)

    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()


    ra_cat, dec_cat = np.mean(obs['ra_cat']), np.mean(obs['dec_cat'])

    ax1.set_title(f"{star_name} in Xi Eta\nRa:{ra_cat:.2f}deg Dec:{dec_cat:.2f}deg\n{stations_text}", fontsize=10, color="Black")



    residual = obs["obs_mag"] - obs["cat_mag"]



    plt.scatter(
        obs["xi"],
        obs["eta"],
        c=residual,
        s=2,
        cmap="coolwarm",
        vmin=np.percentile(residual, 5),
        vmax=np.percentile(residual, 95)
    )

    plt.colorbar(label="Obs - Cat Magnitude")

    plt.axhline(0, color="red",alpha=0.25, linewidth=1)

    plt.axvline(0,color="red",alpha=0.25,linewidth=1)

    plt.scatter(0,0, color="red",marker="+",s=80, linewidths=1, alpha=0.5,label="Catalog")

    plt.xlabel("Xi (arcseoncds)")
    plt.ylabel("Eta (arcseconds)")


    plt.grid(True)


    r = max(np.max(np.abs(obs["xi"])), np.max(np.abs(obs["eta"])))

    # Add 10% margin
    r *= 1.1

    plt.xlim(-r, r)
    plt.ylim(-r, r)

    plt.gca().set_box_aspect(1)

    plt.gca().set_box_aspect(1)

    safe_name = (
        star_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.tight_layout()

    filename = f"{safe_name}_astrometry_xi_eta.png"

    filepath = "/archive/lightcurves/plots"

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

    arg_parser.add_argument("-s", "--star",  type=str, help="Plot the light curve for this star")


    cwd = os.getcwd()
    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))

    cat = Catalog(config)


    target_name = "HD 30836"
    target_ra, target_dec = 72.80, 5.62

    names, ra, dec, mags, thetas, b, v, r = zip(*cat.queryRaDec(target_ra, target_dec, 10, 50)[0])


    names = np.array(names)
    ra = np.array(ra)
    dec = np.array(dec)
    mags = np.array(mags)

    target_mag = mags[names == target_name][0]

    mask = (names != target_name)
    names = names[mask]
    ra = ra[mask]
    dec = dec[mask]
    mags = mags[mask]

    target_star_name_returned, target_star_observations, target_star_stations = getStarObservations(target_name)
    print(f"Target star {target_star_name_returned} catalog mag:{target_mag} has {len(target_star_observations)} observations from {len(target_star_stations)} stations")

    target_star_dict_by_station = {}

    # Create the dict of target star observations
    for station in np.unique(target_star_stations):

        target_star_dict_by_jd = {}
        mask = target_star_observations['station'] == station
        for obs in target_star_observations[mask]:
            target_star_dict_by_jd[obs['jd']] = obs

        target_star_dict_by_station[station] = target_star_dict_by_jd


    # Create the dict of reference star observations
    reference_stars_dict_per_star = {}
    for star_name, mag in zip(names, mags):
        if mag > target_mag +1:
            continue
        query_response = getStarObservations(star_name, stations_list=target_star_stations)
        star_name_returned, observations, stations_returned_list = query_response
        print(f"{star_name} mag:{mag} has {len(observations)} observations from {len(stations_returned_list)} stations")
        pass

        reference_stars_dict_per_station = {}

        for station in stations_returned_list:
            reference_stars_dict_per_jd = {}
            mask = (observations['station'] == station)
            observations_masked_by_station = observations[mask]

            for obs in observations_masked_by_station:
                reference_stars_dict_per_jd[obs['jd']] = obs

            reference_stars_dict_per_station[station] = reference_stars_dict_per_jd

    with open("target_star.json", "w") as f:
        json.dump(
            target_star_dict_by_station,
            f,
            indent=2
        )

    with open("reference_stars.json", "w") as f:
        json.dump(
            reference_stars_dict_per_star,
            f,
            indent=2
        )

    if cml_args.star is None:
        continuous = cml_args.continuous
    else:
        continuous = False
        star_to_plot = cml_args.star






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

            time.sleep(10)



    print(f"Query started for {star_to_plot}")
    query_response = getStarObservations(star_to_plot)
    print("Query completed")
    simpleMagnitudeScatterPlot(query_response)
    plotAstrometryRaDec(query_response)
    plotAstrometryXiEta(query_response)

    pass