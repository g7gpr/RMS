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
import numpy as np
import datetime
import time
import json
import tarfile

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




def getDifferentialData(cat, target_ra, target_dec, frame_json_path=None):

    names, ra, dec, mags, thetas, b, v, r = zip(
        *cat.queryRaDec(
            target_ra,
            target_dec,
            radius_deg=0.1,
            n_brightest=1,
        )[0]
    )

    target_name = names[0]

    names, ra, dec, mags, thetas, b, v, r = zip(
        *cat.queryRaDec(
            target_ra,
            target_dec,
            radius_deg=10,
            n_brightest=30,
        )[0]
    )

    names = np.array(names)
    mags = np.array(mags)

    target_mag = mags[names == target_name][0]

    frame_dict = {}

    #
    # Build list of stars we want.
    #
    # Catalog is sorted by increasing magnitude.
    #

    selected_stars = [(target_name, target_mag)]

    for star_name, mag in zip(names, mags):

        if star_name == target_name:
            continue

        if mag < target_mag - 1:
            continue

        if mag > target_mag + 1:
            break

        selected_stars.append((star_name, mag))

    #
    # Load all stars into the same frame structure.
    #

    target_star_stations = None

    for star_name, mag in selected_stars:

        if target_star_stations is None:

            (
                _,
                observations,
                target_star_stations,
            ) = getStarObservations(star_name)

        else:

            (
                _,
                observations,
                _,
            ) = getStarObservations(
                star_name,
                stations_list=target_star_stations,
            )

        print(
            f"{star_name} "
            f"mag:{mag} "
            f"has {len(observations)} observations"
        )

        if len(observations) == 0:
            continue

        for obs in observations:

            jd = float(obs["jd"])
            station = str(obs["station"])

            if jd not in frame_dict:
                frame_dict[jd] = {}

            if station not in frame_dict[jd]:
                frame_dict[jd][station] = {}

            frame_dict[jd][station][star_name] = {

                "cat_mag": float(obs["cat_mag"]),
                "obs_mag": float(obs["obs_mag"]),
                "residual": float(
                    obs["obs_mag"] - obs["cat_mag"]
                ),

                "snr": float(obs["snr"]),
                "fwhm": float(obs["fwhm"]),
                "pixel_scale": float(obs["pixel_scale"]),

                "level": int(obs["level"]),
                "amplitude": int(obs["amplitude"]),
                "background": int(obs["background"]),

                "saturated_count": int(
                    obs["saturated_count"]
                ),

                "star_count": int(obs["star_count"]),

                "xi": float(obs["xi"]),
                "eta": float(obs["eta"]),

                "ra_cat": float(obs["ra_cat"]),
                "dec_cat": float(obs["dec_cat"]),

                "ra_obs": float(obs["ra_obs"]),
                "dec_obs": float(obs["dec_obs"]),
            }


    if frame_json_path is not None:
        json_path = os.path.dirname(frame_json_path)
    else:
        json_path = "/archive/lightcurves/json"

    os.makedirs(json_path, exist_ok=True)
    json_filename = os.path.join(json_path, f"{target_name}.json")

    with open(json_filename, "w") as f:
        json.dump(frame_dict, f)

    archive_filename = os.path.join(json_path, f"{target_name}.tar.bz2")

    with tarfile.open(archive_filename, "w:bz2") as tar:
        tar.add(json_filename)

    os.remove(json_filename)

    return frame_dict


def buildBinnedLightCurves(frame_dict, bin_seconds=60):

    bin_days = bin_seconds / 86400.0

    stars = {}

    for jd, stations in frame_dict.items():

        jd = float(jd)

        for station_data in stations.values():

            for star_name, star_data in station_data.items():

                stars.setdefault(star_name, [])

                stars[star_name].append(
                    (
                        jd,
                        star_data["obs_mag"],
                    )
                )

    binned_stars = {}

    for star_name, points in stars.items():

        points.sort()

        jd_array = np.array([p[0] for p in points])
        mag_array = np.array([p[1] for p in points])

        jd0 = jd_array.min()

        bin_index = np.floor(
            (jd_array - jd0) / bin_days
        ).astype(int)

        unique_bins = np.unique(bin_index)

        binned_jd = []
        binned_mag = []

        for b in unique_bins:

            mask = (bin_index == b)

            binned_jd.append(
                np.mean(jd_array[mask])
            )

            binned_mag.append(
                np.mean(mag_array[mask])
            )

        binned_stars[star_name] = (
            np.array(binned_jd),
            np.array(binned_mag)
        )

    return binned_stars


def plotAllStars(frame_dict):

    stars = {}

    for jd, stations in frame_dict.items():

        jd = float(jd)

        for station, station_data in stations.items():

            for star_name, star_data in station_data.items():

                stars.setdefault(star_name, {
                    "jd": [],
                    "mag": [],
                })

                stars[star_name]["jd"].append(jd)
                stars[star_name]["mag"].append(
                    star_data["obs_mag"]
                )

    plt.figure(figsize=(16, 10))

    for star_name, data in stars.items():

        plt.scatter(
            data["jd"],
            data["mag"],
            s=1,
            alpha=0.5,
            label=star_name,
        )

    plt.gca().invert_yaxis()

    plt.xlabel("Julian Date")
    plt.ylabel("Observed Magnitude")
    plt.title("All Stars")

    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.show()


def saveBinnedPlots(binned_stars):

    for star_name, (jd, mag) in binned_stars.items():

        plt.figure(figsize=(12, 6))

        plt.plot(
            jd,
            mag,
            ".",
            markersize=2,
        )

        plt.gca().invert_yaxis()

        plt.xlabel("Julian Date")
        plt.ylabel("Observed Magnitude")
        plt.title(star_name)

        filename = (
            star_name
            .replace(" ", "_")
            .replace("/", "_")
            + ".png"
        )

        file_path = os.path.expanduser("~/RMS_data/plots")
        os.makedirs(file_path, exist_ok=True)
        filename = os.path.join(file_path, filename)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()





def extractStar(frame_dict, target_star):

    jd_list = []
    mag_list = []

    for jd, stations in frame_dict.items():

        jd = float(jd)

        for station_data in stations.values():

            if target_star not in station_data:
                continue

            star = station_data[target_star]

            jd_list.append(jd)

            mag_list.append(
                star["obs_mag"]
            )

    jd = np.array(jd_list)
    mag = np.array(mag_list)

    order = np.argsort(jd)

    return jd[order], mag[order]

def phaseFold(jd, mag, period, epoch=None, bin_seconds=60):

    if epoch is None:
        epoch = np.min(jd)

    phase = ((jd - epoch) / period) % 1.0

    phase[phase > 0.5] -= 1.0

    phase = np.concatenate([
        phase - 1.0,
        phase,
        phase + 1.0,
    ])

    mag = np.concatenate([
        mag,
        mag,
        mag,
    ])

    return phase, mag, epoch

def binLightCurve(jd, mag, bin_seconds=60):

    bin_days = bin_seconds / 86400.0

    jd0 = np.min(jd)

    bin_index = np.floor(
        (jd - jd0) / bin_days
    ).astype(int)

    unique_bins = np.unique(bin_index)

    binned_jd = []
    binned_mag = []

    for b in unique_bins:

        mask = bin_index == b

        binned_jd.append(
            np.mean(jd[mask])
        )

        binned_mag.append(
            np.mean(mag[mask])
        )

    return (
        np.array(binned_jd),
        np.array(binned_mag),
    )


import numpy as np


def differentialEnsemble(
    frame_dict,
    target_name,
    min_references=3,
):
    """
    Simple ensemble differential photometry.

    For each frame:

        target_differential =
            target_residual
            -
            mean(reference_residuals)

    Returns:

        jd
        differential_mag
        reference_mean
        reference_std
        reference_count
    """

    jd_list = []
    differential_list = []
    reference_mean_list = []
    reference_std_list = []
    reference_count_list = []

    for jd, stations in frame_dict.items():

        jd = float(jd)

        for station, frame in stations.items():

            #
            # Target must exist in this frame
            #

            if target_name not in frame:
                continue

            target = frame[target_name]

            reference_residuals = []

            for star_name, star_data in frame.items():

                if star_name == target_name:
                    continue

                reference_residuals.append(
                    star_data["residual"]
                )

            if len(reference_residuals) < min_references:
                continue

            reference_residuals = np.array(
                reference_residuals
            )

            reference_mean = np.mean(
                reference_residuals
            )

            reference_std = np.std(
                reference_residuals
            )

            differential = (
                target["residual"]
                - reference_mean
            )

            jd_list.append(jd)

            differential_list.append(
                differential
            )

            reference_mean_list.append(
                reference_mean
            )

            reference_std_list.append(
                reference_std
            )

            reference_count_list.append(
                len(reference_residuals)
            )

    order = np.argsort(jd_list)

    return (
        np.array(jd_list)[order],
        np.array(differential_list)[order],
        np.array(reference_mean_list)[order],
        np.array(reference_std_list)[order],
        np.array(reference_count_list)[order],
    )


def plotPhaseFoldedStar(
    frame_dict,
    star_name,
    period_days,
    epoch=None,
    phase_bins=100,
    output_dir=".",
):

    #
    # Extract star
    #

    jd_list = []
    mag_list = []

    for jd, stations in frame_dict.items():

        jd = float(jd)

        for station_data in stations.values():

            if star_name not in station_data:
                continue

            jd_list.append(jd)

            mag_list.append(
                station_data[star_name]["obs_mag"]
            )

    if len(jd_list) == 0:
        raise RuntimeError(
            f"No observations found for {star_name}"
        )

    jd = np.array(jd_list)
    mag = np.array(mag_list)

    #
    # Sort by JD
    #

    order = np.argsort(jd)

    jd = jd[order]
    mag = mag[order]

    #
    # Fold
    #

    if epoch is None:
        epoch = np.min(jd)

    phase = ((jd - epoch) / period_days) % 1.0

    phase[phase > 0.5] -= 1.0

    #
    # Bin the central cycle only
    #

    edges = np.linspace(
        -0.5,
         0.5,
         phase_bins + 1
    )

    phase_bin = []
    mag_bin = []

    for i in range(phase_bins):

        mask = (
            (phase >= edges[i]) &
            (phase < edges[i + 1])
        )

        if not np.any(mask):
            continue

        phase_bin.append(
            0.5 * (
                edges[i]
                + edges[i + 1]
            )
        )

        mag_bin.append(
            np.mean(mag[mask])
        )

    phase_bin = np.array(phase_bin)
    mag_bin = np.array(mag_bin)

    #
    # Duplicate left and right
    #

    phase_plot = np.concatenate([
        phase_bin - 1.0,
        phase_bin,
        phase_bin + 1.0,
    ])

    mag_plot = np.concatenate([
        mag_bin,
        mag_bin,
        mag_bin,
    ])

    #
    # Plot
    #

    plt.figure(figsize=(12, 6))

    plt.plot(
        phase_plot,
        mag_plot,
        "o-",
        markersize=3,
        linewidth=1,
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Phase")
    plt.ylabel("Magnitude")

    plt.title(
        f"{star_name}\n"
        f"Period = {period_days:.6f} d\n"
        f"Bins = {phase_bins}"
    )

    plt.xlim(-1, 1)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = os.path.join(
        os.path.expanduser(output_dir),
        star_name.replace(" ", "_") + "_phase.png"
    )

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.close()

    print(f"Saved {filename}")

def binLightCurve(jd, mag, bin_seconds=60):

    bin_days = bin_seconds / 86400.0

    jd0 = np.min(jd)

    bin_index = np.floor(
        (jd - jd0) / bin_days
    ).astype(int)

    unique_bins = np.unique(bin_index)

    binned_jd = []
    binned_mag = []

    for b in unique_bins:

        mask = bin_index == b

        binned_jd.append(
            np.mean(jd[mask])
        )

        binned_mag.append(
            np.mean(mag[mask])
        )

    return (
        np.array(binned_jd),
        np.array(binned_mag),
    )


def loadFrameDict(filename):

    with tarfile.open(filename, "r:bz2") as tar:
        member = tar.getmembers()[0]
        with tar.extractfile(member) as f:
            return json.load(f)

def getQueryMetadata(frame_dict, target_name):

    cat_mag = None
    ra_cat = None
    dec_cat = None

    observation_count = 0

    reference_stars = set()

    for stations in frame_dict.values():

        for frame in stations.values():

            if target_name not in frame:
                continue

            observation_count += 1

            target = frame[target_name]

            if cat_mag is None:

                cat_mag = target["cat_mag"]
                ra_cat = target["ra_cat"]
                dec_cat = target["dec_cat"]

            for star_name in frame:

                if star_name != target_name:
                    reference_stars.add(star_name)

    if cat_mag is None:
        raise RuntimeError(
            f"Could not find {target_name}"
        )

    return {
        "target_name": target_name,
        "target_ra": ra_cat,
        "target_dec": dec_cat,
        "target_mag": cat_mag,
        "observation_count": observation_count,
        "reference_count": len(reference_stars),
        "reference_stars": sorted(reference_stars),
    }


def binPhaseCurve(phase, mag, n_bins=100):

    edges = np.linspace(
        -0.5,
         0.5,
         n_bins + 1,
    )

    centres = []
    mean_mag = []
    counts = []

    for i in range(n_bins):

        mask = (
            (phase >= edges[i]) &
            (phase <  edges[i + 1])
        )

        if not np.any(mask):
            continue

        centres.append(
            0.5 * (
                edges[i] +
                edges[i + 1]
            )
        )

        mean_mag.append(
            np.mean(mag[mask])
        )

        counts.append(
            np.sum(mask)
        )

    return (
        np.array(centres),
        np.array(mean_mag),
        np.array(counts),
    )


import os
import numpy as np
import matplotlib.pyplot as plt


def plotDifferentialEnsemble(
    frame_dict,
    target_name,
    period_days,
    epoch=None,
    phase_bins=100,
    output_dir=".",
    min_references=3,
):

    # Get catalog data

    query_metadata = getQueryMetadata(frame_dict, target_name)

    #
    # Build differential light curve
    #

    jd, diff_mag, _, _, _ = differentialEnsemble(
        frame_dict,
        target_name,
        min_references=min_references,
    )

    if len(jd) == 0:
        raise RuntimeError(f"No differential data for {target_name}")

    #
    # Fold
    #

    phase, diff_mag, epoch = phaseFold(
        jd,
        diff_mag,
        period_days,
        epoch,
    )

    #
    # Use only central cycle for binning
    #

    mask = (
        (phase >= -0.5)
        &
        (phase <= 0.5)
    )

    phase = phase[mask]
    diff_mag = diff_mag[mask]

    #
    # Phase bins
    #

    edges = np.linspace(
        -0.5,
         0.5,
         phase_bins + 1,
    )

    phase_bin = []
    mag_bin = []
    count_bin = []

    for i in range(phase_bins):

        mask = (
            (phase >= edges[i])
            &
            (phase < edges[i + 1])
        )

        if not np.any(mask):
            continue

        phase_bin.append(
            0.5 * (
                edges[i]
                + edges[i + 1]
            )
        )

        mag_bin.append(
            np.mean(
                diff_mag[mask]
            )
        )

        count_bin.append(
            np.sum(mask)
        )

    phase_bin = np.array(phase_bin)
    mag_bin = np.array(mag_bin)
    count_bin = np.array(count_bin)

    #
    # Create -1..+1 display
    #

    phase_plot = np.concatenate([
        phase_bin - 1.0,
        phase_bin,
        phase_bin + 1.0,
    ])

    mag_plot = np.concatenate([
        mag_bin,
        mag_bin,
        mag_bin,
    ])

    #
    # Plot
    #

    plt.figure(figsize=(12, 6))
    plt.gca().invert_yaxis()
    plt.plot(
        phase_plot,
        mag_plot,
        "o-",
        markersize=3,
        linewidth=1,
    )

    plt.axvline(
        0.0,
        color="grey",
        linestyle="--",
        alpha=0.5,
    )

    plt.xlabel(f"Phase\n(given period of {period_days} days and {phase_bins} bins)")
    plt.ylabel(f"Differential Residual against Mag={query_metadata['target_mag']:.3f}")

    plt.title(
        f"{target_name}\n"
        f"Total of ={query_metadata['observation_count']} "
        f"RA={query_metadata['target_ra']:.4f} "
        f"Dec={query_metadata['target_dec']:.4f}\n {query_metadata['reference_count']} reference stars"

    )

    plt.xlim(-1.0, 1.0)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = os.path.join(
        os.path.expanduser(output_dir),
        target_name.replace(" ", "_")
        + "_differential_phase.png"
    )

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {filename}"
    )


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

    #star_data = getDifferentialData(cat, 78.858, 32.6880, "/archive/lightcurves/json/kw_aur.json")

    # frame_dict = loadFrameDict("HD 33959A.tar.bz2")
    # plotAllStars(frame_dict)
    # binned_light_curves = buildBinnedLightCurves(frame_dict, 240)
    # saveBinnedPlots(binned_light_curves)
    # plotPhaseFoldedStar(frame_dict, "HD 33959A", 0.88088, output_dir="~/RMS_data/plots/")

    #frame_dict = loadFrameDict("HD 33959A.tar.bz2")

    frame_dict = getDifferentialData(cat, 119.56549, -49.2409)

    plotDifferentialEnsemble(frame_dict, "HD 65818", 1.454485, output_dir="/archive/lightcurves/plots")

    frame_dict = getDifferentialData(cat, 78.858, 32.6880)
    plotDifferentialEnsemble(frame_dict, "HD 33959A", 0.88088, output_dir="/archive/lightcurves/plots")


    if False:

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