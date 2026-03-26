import numpy as np
import matplotlib.pyplot as plt
import psycopg

DB_SCALE_FACTOR = 1e6


def fetchHemisphereRadec(conn, hemisphere="south", limit_rows=5000000, mag_limit=4.0):
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



def plotHemisphereDensity(rows_ra, rows_dec, gridsize=200):
    """
    Plot a hemisphere density map using hexbin.
    rows_ra, rows_dec: numpy arrays of degrees.
    """

    # Convert to polar coordinates (vectorised)
    theta_vals, r_vals = radecToPolarVectorised(rows_ra, rows_dec)

    # Diagnostic print (optional)
    print("r min:", np.min(r_vals))
    print("r max:", np.max(r_vals))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    # Black background for heatmap
    ax.set_facecolor("black")

    ax.set_title(
        "Southern Celestial Hemisphere\nObservation Density",
        color="white"
    )

    print("theta_vals shape:", theta_vals.shape)
    print("r_vals shape:", r_vals.shape)

    print("theta min/max:", np.min(theta_vals), np.max(theta_vals))
    print("r min/max:", np.min(r_vals), np.max(r_vals))

    # Check for NaNs or infs
    print("theta NaNs:", np.isnan(theta_vals).sum())
    print("r NaNs:", np.isnan(r_vals).sum())
    print("theta infs:", np.isinf(theta_vals).sum())
    print("r infs:", np.isinf(r_vals).sum())

    # Check a few samples
    print("Sample theta:", theta_vals[:10])
    print("Sample r:", r_vals[:10])

    ax.scatter(
        theta_vals,
        r_vals,
        s=1,
        c=r_vals,  # or a constant color
        cmap="inferno",
        alpha=0.3
    )

    #plt.colorbar(hb, ax=ax, label="Observation density")
    ax.set_ylim(0, np.deg2rad(90))

    # White tick labels for contrast
    ax.tick_params(colors="white")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")

    plt.show()
    pass


with psycopg.connect(host="192.168.1.190", dbname="star_data", user="ingest_user") as conn:

    ra_deg, dec_deg = fetchHemisphereRadec(conn, "south")
    plotHemisphereDensity(ra_deg, dec_deg)