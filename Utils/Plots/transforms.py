import numpy as np

obliquity_rad = np.deg2rad(23.4392911)


def radecToPolar(ra_deg, dec_deg, hemisphere="south"):
    """Convert RA/Dec (deg) to polar (theta, r) for a given hemisphere."""
    theta = np.deg2rad(ra_deg)

    if hemisphere == "north":
        # 0° at bottom, increasing clockwise
        theta = np.deg2rad(90.0) - theta + np.pi

    elif hemisphere == "south":
        # 0° at bottom, increasing anticlockwise
        theta = theta - np.deg2rad(90.0)

    #theta = np.mod(theta, 2 * np.pi)

    dec_clamped = np.clip(dec_deg, -90.0, 90.0)

    if hemisphere == "south":
        r = np.deg2rad(90.0 - np.abs(dec_clamped))
    else:
        r = np.deg2rad(90.0 - dec_clamped)

    return theta, r

def radecToPolarNoClip(ra_deg, dec_deg, hemisphere="south"):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    theta = ra_rad
    r = np.deg2rad(90 - np.abs(dec_deg))

    return theta, r



def computeEclipticEquatorial():
    """Return RA, Dec (deg) for the true astronomical ecliptic."""
    lam_rad = np.deg2rad(np.linspace(0.0, 360.0, 720))

    sin_lam = np.sin(lam_rad)
    cos_lam = np.cos(lam_rad)

    ra_rad = np.arctan2(sin_lam * np.cos(obliquity_rad), cos_lam)
    dec_rad = np.arcsin(sin_lam * np.sin(obliquity_rad))

    ra_rad = np.mod(ra_rad, 2.0 * np.pi)

    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)

    return ra_deg, dec_deg
