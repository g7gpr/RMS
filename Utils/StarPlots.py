import numpy as np
import matplotlib.pyplot as plt
import psycopg
import os
import csv

DB_SCALE_FACTOR = 1e6


def loadConstellationLines(csv_path):
    lines = []
    ra1_col, dec1_col, ra2_col, dec2_col = 0,1,2,3

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 4:
                continue
            ra1 = float(row[ra1_col])
            dec1 = float(row[dec1_col])
            ra2 = float(row[ra2_col])
            dec2 = float(row[dec2_col])
            lines.append((ra1, dec1, ra2, dec2))
    return lines

def fetchHemisphereRadec(conn, hemisphere="south", limit_rows=500000, mag_limit=3.5):
    """
    Efficiently fetch RA/Dec for a single hemisphere.
    Returns arrays of ra_deg, dec_deg.
    """

    mag_scaled_limit = int(mag_limit * DB_SCALE_FACTOR)

    if hemisphere == "south":
        dec_filter = "dec < 0"
    else:
        dec_filter = "dec >= 0"

    query = f"""
        SELECT ra, dec
        FROM observation
        WHERE ra IS NOT NULL
          AND dec IS NOT NULL
          AND mag < {mag_scaled_limit}
          AND {dec_filter}
          LIMIT {limit_rows};
    """

    ra_list = []
    dec_list = []

    with conn.cursor() as cur:
        cur.execute(query)
        for ra_scaled, dec_scaled in cur:
            ra_list.append(ra_scaled / DB_SCALE_FACTOR)
            dec_list.append(dec_scaled / DB_SCALE_FACTOR)

    return np.array(ra_list), np.array(dec_list)



def radecToPolarVectorised(ra_deg_array, dec_deg_array):
    """
    Convert arrays of RA/Dec to polar coordinates.
    Vectorised for speed with millions of rows.
    Returns (theta_rad_array, r_rad_array)
    """

    # Convert RA to radians
    theta_rad = np.deg2rad(ra_deg_array)

    # Clamp Dec to [-90, +90] to avoid invalid radii
    dec_clamped = np.clip(dec_deg_array, -90.0, 90.0)

    # Southern hemisphere radius:
    # r = deg2rad(90 - |dec|)
    r_rad = np.deg2rad(90.0 - np.abs(dec_clamped))

    return theta_rad, r_rad



def plotHemisphereDensity(rows_ra, rows_dec, constellation_list, gridsize=200):
    """
    Plot a hemisphere density map using hexbin.
    rows_ra, rows_dec: numpy arrays of degrees.
    """

    # Convert to polar coordinates (vectorised)
    theta_vals, r_vals = radecToPolarVectorised(rows_ra, rows_dec)

    # Diagnostic print (optional)
    print("r min:", np.min(r_vals))
    print("r max:", np.max(r_vals))

    fig = plt.figure(figsize=(16, 16), dpi=400)
    ax = fig.add_subplot(111, projection="polar")


    # White background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Scatter stars in pale blue
    ax.scatter(
        theta_vals,
        r_vals,
        s=4,
        c="#66aaff",   # pale blue
        alpha=0.6
    )

    # Constellation lines in darker blue
    plotConstellationLines(
        ax,
        constellation_list,
        color="#0044aa",
        alpha=0.22,
        lw=0.35
    )

    ax.grid(color="#0044aa", alpha=0.15)

    # RA ticks every 30°
    ra_ticks = np.deg2rad(np.arange(0, 360, 30))
    ra_labels = [f"{deg}°" for deg in np.arange(0, 360, 30)]
    ax.set_xticks(ra_ticks)
    ax.set_xticklabels(ra_labels, color="#0044aa")

    # Dec ticks from centre outward
    dec_degs = np.array([-90, -60, -30, 0])
    dec_r = np.deg2rad(90 - np.abs(dec_degs))
    ax.set_yticks(dec_r)
    ax.set_yticklabels([f"{d}°" for d in dec_degs], color="#0044aa")


    fig.suptitle(
        "The Global Meteor Network survey of the Southern Hemisphere",
        color="#0044aa",
        y=0.05  # near the bottom
    )


    #plt.colorbar(hb, ax=ax, label="Observation density")
    ax.set_ylim(0, np.deg2rad(90))

    # Clean atlas-style ticks
    ax.tick_params(
        axis="both",
        which="both",
        length=6,
        width=0.8,
        color="#0044aa",  # tick marks
        labelcolor="#6699cc",  # softer, less prominent labels
        pad=8
    )

    # Faint grid for declination
    ax.grid(color="lightgray", alpha=0.3)

    plt.savefig(
        "southern_sky.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.2
    )

    plt.close()

    pass

def radecToPolarPoint(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_clamped = max(min(dec_deg, 90.0), -90.0)
    r_rad = np.deg2rad(90.0 - abs(dec_clamped))
    return ra_rad, r_rad


def plotConstellationLines(ax, lines, color="gray", alpha=0.22, lw=0.35):
    for ra1, dec1, ra2, dec2 in lines:
        t1, r1 = radecToPolarPoint(ra1, dec1)
        t2, r2 = radecToPolarPoint(ra2, dec2)
        ax.plot([t1, t2], [r1, r2], color=color, alpha=alpha, lw=lw)


def filterSouthernConstellations(lines):
    south = []
    for ra1, dec1, ra2, dec2 in lines:
        if dec1 < 0 and dec2 < 0:
            south.append((ra1, dec1, ra2, dec2))
    return south


constellations_list = loadConstellationLines(os.path.expanduser("~/source/RMS/share/constellation_lines.csv"))
constellations_list = filterSouthernConstellations(constellations_list)



with psycopg.connect(host="192.168.1.190", dbname="star_data", user="ingest_user") as conn:


    ra_deg, dec_deg = fetchHemisphereRadec(conn, "south")
    plotHemisphereDensity(ra_deg, dec_deg, constellations_list)