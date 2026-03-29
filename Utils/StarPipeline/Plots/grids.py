import numpy as np

from transforms import radecToPolar, computeEclipticEquatorial, radecToPolarNoClip


def addRaTicksOld(ax):
    """Add major and minor RA ticks and labels."""
    major_rad = np.deg2rad(np.arange(0, 360, 30))
    minor_rad = np.deg2rad(np.arange(0, 360, 10))

    ax.set_xticks(major_rad)
    ax.set_xticklabels([f"{deg}°" for deg in range(0, 360, 30)], color="#6699cc")

    ax.set_xticks(minor_rad, minor=True)

    ax.tick_params(axis="x", which="major", length=8, width=1.0, color="#0044aa")
    ax.tick_params(axis="x", which="minor", length=4, width=0.6, color="#88aadd")


def addRaTicks(ax, hemisphere):

    ra_ticks_deg = np.arange(0, 360, 30)

    if hemisphere == "south":
        # 0° at bottom, increasing anticlockwise
        theta_ticks = np.deg2rad(ra_ticks_deg - 90.0)

    elif hemisphere == "north":
        # 0° at bottom, increasing clockwise, plus the 180° correction
        theta_ticks = np.deg2rad(90.0 - ra_ticks_deg) + np.pi

    theta_ticks = np.mod(theta_ticks, 2*np.pi)

    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([f"{int(ra)}°" for ra in ra_ticks_deg],color="#6699cc")

    ax.tick_params(axis="y", which="major", length=8, width=1.0, color="#0044aa")
    ax.tick_params(axis="y", which="minor", length=4, width=0.6, color="#88aadd")

    # Convert degree labels → hours
    ra_ticks_hours = (ra_ticks_deg / 15.0) % 24.0
    ax.set_xticklabels([f"{int(h)}h" for h in ra_ticks_hours])


def addDecTicks(ax):
    """Add major and minor Dec ticks and labels."""
    dec_major = np.array([-90, -60, -30, 0])
    dec_minor = np.array([-80, -70, -50, -40, -20, -10])

    r_major = np.deg2rad(90.0 - np.abs(dec_major))
    r_minor = np.deg2rad(90.0 - np.abs(dec_minor))

    ax.set_yticks(r_major)

    ax.set_yticks(r_minor, minor=True)
    # Declination ticks you want to show (omit +90°)
    dec_labels = ["60", "30", "", "30", "60"]

    # Corresponding declinations in degrees (north → south)
    dec_degs = np.array([+60, +30, 0, -30, -60])

    # Convert to polar radii
    dec_r = np.deg2rad(90 - np.abs(dec_degs))

    ax.set_yticks(dec_r)
    ax.set_yticklabels(dec_labels, color="#6699cc")
    ax.tick_params(axis="y", which="minor", length=4, width=0.6, color="#0044aa")


def addRaGrid(ax):
    """Add faint RA gridlines."""
    ax.grid(which="major", axis="x", color="#0044aa", alpha=0.15)


def addDecGrid(ax):
    """Add faint Dec gridlines."""
    ax.grid(which="major", axis="y", color="#0044aa", alpha=0.15)


def addEquator(ax):
    """Draw the celestial equator (Dec = 0°)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    r = np.deg2rad(90.0)
    r_vals = np.full_like(theta, r)
    ax.plot(theta, r_vals, color="#0044aa", lw=1.2, alpha=0.6)


def addEcliptic(ax, hemisphere="south"):
    """Draw the true ecliptic in soft red."""
    ra_deg, dec_deg = computeEclipticEquatorial()
    theta, r = radecToPolarNoClip(ra_deg, dec_deg, hemisphere=hemisphere)
    ax.plot(theta, r, color="#cc4444", lw=0.6, alpha=0.3)
