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
import RMS.ConfigReader as cr
import numpy as np
import psycopg

from RMS.Astrometry.Conversions import J2000_JD, date2JD

DB_SCALE_FACTOR = 1e6
JD_OFFSET = J2000_JD

TYPE_MAP = {
    "stationID": "TEXT",
    "jd": "BIGINT",
    # Add more cases as schema evolves
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

def plotStarLightcurve(conn, catalogue_id, jd_start, jd_end):
    """
    Plot observed magnitudes for a given catalogue star,
    binned into 10-minute intervals across all stations.
    """

    jd_start = int(jd_start * DB_SCALE_FACTOR)
    jd_end   = int(jd_end   * DB_SCALE_FACTOR)

    query = """
        SELECT jd, obs_mag, stationID
        FROM star_observations
        WHERE catalogue_id = %s
          AND jd BETWEEN %s AND %s AND stationID != 'AU0007'
        ORDER BY jd;
    """

    with conn.cursor() as cur:
        cur.execute(query, (catalogue_id, jd_start, jd_end))
        rows = cur.fetchall()

    summary_query = """
                    SELECT COUNT(*)                  AS n_obs, \
                           COUNT(DISTINCT stationID) AS n_instr
                    FROM star_observations
                    WHERE catalogue_id = %s
                      AND jd BETWEEN %s AND %s; \
                    """


    with conn.cursor() as cur:
        cur.execute(summary_query, (catalogue_id, jd_start, jd_end))
        n_obs, n_instr = cur.fetchone()

        # Fetch catalogue magnitude
        catmag_query = """
                       SELECT cat_mag
                       FROM star_observations
                       WHERE catalogue_id = %s
                         AND cat_mag IS NOT NULL LIMIT 1; \
                       """
        cur.execute(catmag_query, (catalogue_id,))
        row = cur.fetchone()
        cat_mag = row[0] / DB_SCALE_FACTOR if row else None

    if not rows:
        print("No observations found for this star in the given JD range.")
        return

    BIN_SIZE_DAYS = 10.0 / 1440.0
    bins = defaultdict(list)

    # Bin all observations together
    for jd, mag, station_id in rows:
        jd_days = jd / DB_SCALE_FACTOR
        bin_index = int(jd_days / BIN_SIZE_DAYS)
        bins[bin_index].append(mag  / DB_SCALE_FACTOR)

    # Compute bin centers + median magnitudes
    bin_centers = []
    bin_medians = []

    for bin_index, mags in bins.items():
        jd_center = (bin_index + 0.5) * BIN_SIZE_DAYS
        bin_centers.append(jd_center)
        bin_medians.append(np.median(mags))

    # Sort by time
    bin_centers, bin_medians = zip(*sorted(zip(bin_centers, bin_medians)))

    # Plot
    plt.figure(figsize=(12, 6))




    plt.gca().invert_yaxis()
    plt.ylim(12,0)
    plt.xlabel("Julian Date")
    plt.ylabel("Median Magnitude (10-minute bins)")
    plt.title(
        f"Light Curve for {catalogue_id}\n"
        f"{n_obs} observations from {n_instr} instruments (10‑min bins)"
    )

    plt.grid(True)


    # Scatter plot of the binned light curve
    plt.scatter(
        bin_centers,
        bin_medians,
        s=12,  # marker size
        color='black',
        label='10‑min median')

    # Add catalogue magnitude line
    if cat_mag is not None:
        plt.axhline(
            y=cat_mag,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f"Catalogue mag {cat_mag:.3f}"
        )

    # Add legend
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plotObservedRadec(rows):
    """
    rows: iterable of (ra_deg, dec_deg)
    """

    # Extract arrays
    ra_rad = np.array([r[0] for r in rows]) * np.pi / 180.0
    dec_deg = np.array([r[1] for r in rows])

    # Masks
    north_mask = dec_deg >= 0
    south_mask = dec_deg < 0

    # Convert to polar radii
    # North: r = 90 - dec
    # South: r = 90 + dec
    r_north = np.deg2rad(90 - dec_deg[north_mask])
    r_south = np.deg2rad(90 + dec_deg[south_mask])

    # --- Plotting ---
    fig, (ax_north, ax_south) = plt.subplots(
        1, 2,
        subplot_kw={'projection': 'polar'},
        figsize=(12, 6)
    )

    # Northern hemisphere
    ax_north.scatter(ra_rad[north_mask], r_north, s=2, color='white')
    ax_north.set_title("Northern Hemisphere")
    ax_north.set_facecolor("black")
    ax_north.set_ylim(0, np.deg2rad(90))

    # Southern hemisphere
    ax_south.scatter(ra_rad[south_mask], r_south, s=2, color='white')
    ax_south.set_title("Southern Hemisphere")
    ax_south.set_facecolor("black")
    ax_south.set_ylim(0, np.deg2rad(90))

    plt.show()


def fetchObservedRadec(conn):
    """
    Fetch observed RA/Dec pairs from PostgreSQL.

    Arguments:
        conn: psycopg.Connection object

    Returns:
        list of (ra_deg, dec_deg)
    """

    query = """
                SELECT DISTINCT ON (catalogue_id)
                       catalogue_id,
                       obs_ra,
                       obs_dec
                FROM observation
                WHERE obs_ra IS NOT NULL
                  AND obs_dec IS NOT NULL
                ORDER BY catalogue_id;
            """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return rows


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Plot data \
        """, formatter_class=argparse.RawTextHelpFormatter)






    cml_args = arg_parser.parse_args()
    config = cr.parse(os.path.join(os.getcwd(),".config"))



    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("rmslogger")



    cwd = os.getcwd()

    conn_params = {
        "host": "192.168.1.174",
        "dbname": "star_data",
        "user": "ingest_user"
    }

    with psycopg.connect(**conn_params) as conn:

        rows = fetchObservedRadec(conn)
        plotObservedRadec(rows)

        catalogue_id = 'HD 92305'
        jd_start = 2461114.0
        jd_end = 2461118.0

        plotStarLightcurve(conn, catalogue_id, jd_start, jd_end)


        catalogue_id = 'HD 74956'
        jd_start = 2461114.0
        jd_end = 2461118.0

        plotStarLightcurve(conn, catalogue_id, jd_start, jd_end)



    pass