import numpy as np
import matplotlib.pyplot as plt
import psycopg

# Your integer scaling factor
SCALE_OUT = 1.0 / 1_000_000.0


# ------------------------------------------------------------
#  FETCH FUNCTIONS
# ------------------------------------------------------------

def fetchObservedBrightStars(conn, mag_limit=4.0):
    """
    Return (name, ra_deg, dec_deg, mag) for stars brighter than mag_limit
    that appear in the observation table.
    """
    mag_limit_scaled = int(mag_limit * 1_000_000)

    sql = """
        SELECT s.star_name, s.ra, s.dec, s.mag
        FROM star AS s
        JOIN (
            SELECT DISTINCT star_name
            FROM observation
        ) AS o ON o.star_name = s.star_name
        WHERE s.mag <= %s
        ORDER BY s.mag;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (mag_limit_scaled,))
        rows = cur.fetchall()

    # Convert scaled integers → floats
    return [
        (name, ra * SCALE_OUT, dec * SCALE_OUT, mag * SCALE_OUT)
        for (name, ra, dec, mag) in rows
    ]


# ------------------------------------------------------------
#  PLOTTING FUNCTIONS
# ------------------------------------------------------------

def plotBrightStars(rows):
    """
    Plot bright stars on two polar hemisphere charts.
    rows: list of (name, ra_deg, dec_deg, mag)
    """

    names = [r[0] for r in rows]
    ra_deg = np.array([r[1] for r in rows])
    dec_deg = np.array([r[2] for r in rows])
    mags = np.array([r[3] for r in rows])

    # Convert RA to radians for polar plotting
    ra_rad = np.deg2rad(ra_deg)

    # Hemisphere masks
    north_mask = dec_deg >= 0
    south_mask = dec_deg < 0

    # Polar radius: zenith = 0, horizon = 90°
    r_north = np.deg2rad(90 - dec_deg[north_mask])
    r_south = np.deg2rad(90 + dec_deg[south_mask])

    # Exaggerated brightness scaling
    size = 40 / (mags + 0.3)

    # High-resolution figure
    fig, (ax_north, ax_south) = plt.subplots(
        1, 2,
        subplot_kw={'projection': 'polar'},
        figsize=(32, 16),
        dpi=300
    )

    # -------------------------
    # Northern Hemisphere
    # -------------------------
    ax_north.scatter(
        ra_rad[north_mask],
        r_north,
        s=size[north_mask],
        color='white'
    )
    ax_north.set_title("Bright Stars — Northern Hemisphere", fontsize=18)
    ax_north.set_facecolor("black")
    ax_north.set_ylim(0, np.deg2rad(90))
    ax_north.set_theta_zero_location("S")
    ax_north.set_theta_direction(-1)

    # Label very bright stars
    for name, ra, dec, mag in rows:
        if dec >= 0 and mag < 2.0:
            ax_north.text(
                np.deg2rad(ra),
                np.deg2rad(90 - dec),
                name,
                color='yellow',
                fontsize=10
            )

    # -------------------------
    # Southern Hemisphere
    # -------------------------
    ax_south.scatter(
        ra_rad[south_mask],
        r_south,
        s=size[south_mask],
        color='white'
    )
    ax_south.set_title("Bright Stars — Southern Hemisphere", fontsize=18)
    ax_south.set_facecolor("black")
    ax_south.set_ylim(0, np.deg2rad(90))
    ax_south.set_theta_zero_location("S")
    ax_south.set_theta_direction(-1)

    # Label very bright stars
    for name, ra, dec, mag in rows:
        if dec < 0 and mag < 2.0:
            ax_south.text(
                np.deg2rad(ra),
                np.deg2rad(90 + dec),
                name,
                color='yellow',
                fontsize=10
            )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
#  MAIN ENTRY POINT (example)
# ------------------------------------------------------------

def runBrightStarDiagnostic(conn, mag_limit=4.0):
    """
    Fetch bright stars and plot them.
    """
    rows = fetchObservedBrightStars(conn, mag_limit=mag_limit)
    print(f"Loaded {len(rows)} bright stars (mag < {mag_limit})")
    plotBrightStars(rows)



if __name__ == "__main__":

    conn_str = (
        "host=192.168.1.174 "
        "dbname=star_data "
        "user=ingest_user "

    )
    with psycopg.connect(conn_str) as conn:
        runBrightStarDiagnostic(conn, mag_limit=4.0)



