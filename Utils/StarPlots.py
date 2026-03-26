import os
import RMS.ConfigReader as cr
import numpy as np
import psycopg
import matplotlib.pyplot as plt

from RMS.Logger import LoggingManager, getLogger


def radecToPolar(ra_deg, dec_deg):
    """
    Convert RA/Dec to polar coordinates for sky plots.
    Returns (theta_rad, r_rad, hemisphere)
    """

    ra_rad = np.deg2rad(ra_deg)

    if dec_deg >= 0:
        r_rad = np.deg2rad(90 - dec_deg)
        hemisphere = "north"
    else:
        r_rad = np.deg2rad(90 + dec_deg)
        hemisphere = "south"

    return ra_rad, r_rad, hemisphere


def plotHemisphereDensity(rows, hemisphere, gridsize=200):
    """
    Plot density for either the northern or southern celestial hemisphere.

    rows: iterable of (ra_deg, dec_deg)
    hemisphere: 'north' or 'south'
    """

    theta_vals = []
    r_vals = []

    for ra_deg, dec_deg in rows:
        theta_rad, r_rad, hemi = radecToPolar(ra_deg, dec_deg)
        if hemi == hemisphere:
            theta_vals.append(theta_rad)
            r_vals.append(r_rad)

    theta_vals = np.array(theta_vals)
    r_vals = np.array(r_vals)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")
    ax.set_title(
        f"{hemisphere.capitalize()} Celestial Hemisphere\nObservation Density",
        color="blue"
    )

    hb = ax.hexbin(
        theta_vals,
        r_vals,
        gridsize=gridsize,
        cmap="viridis",
        mincnt=1,
        linewidths=0
    )

    plt.colorbar(hb, ax=ax, label="Observation density")
    ax.set_ylim(0, np.deg2rad(90))

    ax.tick_params(colors="blue")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("blue")

    plt.show()


def plotGlobalDensity(rows):
    plotHemisphereDensity(rows, "north")
    plotHemisphereDensity(rows, "south")


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