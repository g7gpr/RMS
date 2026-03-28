import numpy as np
from sklearn.cluster import DBSCAN
import psycopg

def clusterDetections(ra_deg, dec_deg, jd,
                      t_window_min=1.0, ang_tol_deg=0.05):
    """
    Spatiotemporal clustering of detections using RA/Dec and JD timestamps.

    Parameters
    ----------
    ra_deg, dec_deg : arrays
        RA/Dec of detections (degrees).
    jd : array
        Julian Date timestamps (days).
    t_window_min : float
        Coincidence window in minutes.
    ang_tol_deg : float
        Angular tolerance for clustering (degrees).

    Returns
    -------
    labels : ndarray
        Cluster labels for each detection.
    """

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # --- 1. Convert RA/Dec to unit vectors ---
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    # Scale JD so that 10.24 seconds ≈ 1 arcsec chord distance
    time_scale = 0.042  # empirically correct for your cadence
    t_scaled = (jd - jd.min()) / time_scale

    # --- 3. Build 4-D coordinate array ---
    coords = np.column_stack((x, y, z, t_scaled))

    # --- 4. Angular tolerance → chord distance ---
    theta = np.deg2rad(ang_tol_deg)
    eps_spatial = 2 * np.sin(theta / 2)

    # Combined 4-D radius: spatial + temporal
    eps_4d = eps_spatial

    # --- 5. Run DBSCAN ---
    clustering = DBSCAN(eps=eps_4d, min_samples=1).fit(coords)
    labels = clustering.labels_

    return labels


def buildPhotometricPoint(idx, ra_deg, dec_deg, jd,
                          mag, snr, camera_id):
    """
    Combine detections in a cluster into a single photometric point.
    """

    ra_c  = ra_deg[idx]
    dec_c = dec_deg[idx]
    jd_c  = jd[idx]
    mag_c = mag[idx]
    snr_c = snr[idx]
    cams  = camera_id[idx]

    # --- centroid position ---
    ra_rad  = np.deg2rad(ra_c)
    dec_rad = np.deg2rad(dec_c)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    x_m = x.mean()
    y_m = y.mean()
    z_m = z.mean()

    r = np.sqrt(x_m*x_m + y_m*y_m + z_m*z_m)
    x_m, y_m, z_m = x_m/r, y_m/r, z_m/r

    dec_mean = np.rad2deg(np.arcsin(z_m))
    ra_mean  = np.rad2deg(np.arctan2(y_m, x_m)) % 360.0

    # --- weighted photometry ---
    flux = 10**(-0.4 * mag_c)
    w = snr_c**2
    w_sum = w.sum()

    flux_mean = np.sum(w * flux) / w_sum
    mag_mean = -2.5 * np.log10(flux_mean)

    sigma_flux = np.sqrt(1.0 / w_sum)
    mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

    # --- time: median JD ---
    jd_mean = np.median(jd_c)

    return {
        'ra_deg': ra_mean,
        'dec_deg': dec_mean,
        'jd': jd_mean,
        'mag': mag_mean,
        'mag_err': mag_err,
        'n_det': len(idx),
        'n_cam': len(set(cams))
    }


def buildLightCurve(ra_deg, dec_deg, jd,
                    mag, snr, camera_id,
                    t_window_min=1.0, ang_tol_deg=0.05,
                    min_cameras=2):

    labels = clusterDetections(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        jd=jd,
        t_window_min=t_window_min,
        ang_tol_deg=ang_tol_deg
    )

    points = []
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        point = buildPhotometricPoint(
            idx=idx,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            jd=jd,
            mag=mag,
            snr=snr,
            camera_id=camera_id
        )

        if point['n_cam'] >= min_cameras:
            points.append(point)

    return points




def loadDetections(conn, jd_start=None, jd_end=None):
    """
    Load detections from the database, optionally filtered by JD range.

    Parameters
    ----------
    conn : psycopg2 connection
        Database connection.
    jd_start : float or None
        Start JD (days). If None, no lower bound.
    jd_end : float or None
        End JD (days). If None, no upper bound.

    Returns
    -------
    detections : dict of numpy arrays
        {
            'ra_deg': ...,
            'dec_deg': ...,
            'jd': ...,
            'mag': ...,
            'snr': ...,
            'camera_id': ...
        }
    """

    # Build WHERE clause
    where = []
    params = []

    if jd_start is not None:
        where.append("frame.jd_mid >= %s")
        params.append(int(jd_start * 1e6))

    if jd_end is not None:
        where.append("frame.jd_mid <= %s")
        params.append(int(jd_end * 1e6))

    where_clause = ""
    if where:
        where_clause = "WHERE " + " AND ".join(where)

    sql = f"""
        SELECT
            obs.ra,
            obs.dec,
            frame.jd_mid,
            obs.mag,
            obs.snr,
            obs.station_name
        FROM observation AS obs
        JOIN frame ON obs.frame_name = frame.frame_name
        {where_clause}
    """



    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return None

    # Convert to numpy arrays and apply scaling
    ra_deg     = np.array([r[0] for r in rows], dtype=float) / 1e6
    dec_deg    = np.array([r[1] for r in rows], dtype=float) / 1e6
    jd         = np.array([r[2] for r in rows], dtype=float) / 1e6
    mag        = np.array([r[3] for r in rows], dtype=float) / 1e6
    snr        = np.array([r[4] for r in rows], dtype=float)
    camera_id  = np.array([r[5] for r in rows], dtype=str)

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'jd': jd,
        'mag': mag,
        'snr': snr,
        'camera_id': camera_id
    }

def spatialFilter(ra, dec, ra0, dec0, radius_deg):
    ra_rad  = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)

    cosang = (np.sin(dec_rad)*np.sin(dec0_rad) +
              np.cos(dec_rad)*np.cos(dec0_rad)*np.cos(ra_rad - ra0_rad))

    ang = np.rad2deg(np.arccos(cosang))
    return ang <= radius_deg

def convertPointsToArrays(points):
    return {
        'jd':      np.array([p['jd'] for p in points]),
        'mag':     np.array([p['mag'] for p in points]),
        'mag_err': np.array([p['mag_err'] for p in points]),
        'ra_deg':  np.array([p['ra_deg'] for p in points]),
        'dec_deg': np.array([p['dec_deg'] for p in points]),
        'n_det':   np.array([p['n_det'] for p in points]),
        'n_cam':   np.array([p['n_cam'] for p in points])
    }


def binByCadence(lc, cadence_sec=10.24):
    """
    Bin light curve points using the natural image cadence.
    """

    # Convert cadence to days
    dt_days = cadence_sec / 86400.0

    jd = lc['jd']
    jd0 = jd.min()

    # Integer cadence index
    bins = np.floor((jd - jd0) / dt_days).astype(int)

    out = {k: [] for k in lc}

    for b in np.unique(bins):
        idx = np.where(bins == b)[0]

        # Weighted mean magnitude
        w = 1.0 / lc['mag_err'][idx]**2
        mag_mean = np.sum(w * lc['mag'][idx]) / np.sum(w)
        mag_err  = np.sqrt(1.0 / np.sum(w))

        out['jd'].append(np.mean(lc['jd'][idx]))
        out['mag'].append(mag_mean)
        out['mag_err'].append(mag_err)
        out['ra_deg'].append(np.mean(lc['ra_deg'][idx]))
        out['dec_deg'].append(np.mean(lc['dec_deg'][idx]))
        out['n_det'].append(np.sum(lc['n_det'][idx]))
        out['n_cam'].append(np.max(lc['n_cam'][idx]))

    for k in out:
        out[k] = np.array(out[k])

    return out





def generateLightCurve(conn,
                       ra_center=None,
                       dec_center=None,
                       radius_deg=None,
                       jd_start=None,
                       jd_end=None,
                       t_window_min=1.0,
                       ang_tol_deg=0.05,
                       min_cameras=2,
                       bin_minutes=None):

    # --- 1. Load detections ---
    det = loadDetections(
        conn=conn,
        jd_start=jd_start,
        jd_end=jd_end
    )

    if det is None:
        return []

    ra_deg = det['ra_deg']
    dec_deg = det['dec_deg']
    jd     = det['jd']
    mag    = det['mag']
    snr    = det['snr']
    cam    = det['camera_id']

    # --- 2. Optional spatial filtering ---
    if ra_center is not None and dec_center is not None and radius_deg is not None:
        mask = spatialFilter(ra_deg, dec_deg, ra_center, dec_center, radius_deg)
        ra_deg = ra_deg[mask]
        dec_deg = dec_deg[mask]
        jd = jd[mask]
        mag = mag[mask]
        snr = snr[mask]
        cam = cam[mask]

    if len(jd) == 0:
        return []

    # --- 3. Cluster detections ---
    points = buildLightCurve(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        jd=jd,
        mag=mag,
        snr=snr,
        camera_id=cam,
        t_window_min=t_window_min,
        ang_tol_deg=ang_tol_deg,
        min_cameras=min_cameras
    )

    if not points:
        return []

    # --- 4. Convert list of dicts → arrays ---
    lc = convertPointsToArrays(points)

    # --- 5. Optional binning ---
    if bin_minutes is not None:
        lc = binByCadence(lc, cadence_sec=bin_minutes * 60.0)

    # --- 6. Sort by JD ---
    order = np.argsort(lc['jd'])
    for k in lc:
        lc[k] = lc[k][order]

    return lc

def generateStarLightCurve(conn,
                           ra_star_deg,
                           dec_star_deg,
                           search_radius_deg=0.05,
                           jd_start=None,
                           jd_end=None,
                           t_window_min=1.0,
                           ang_tol_deg=0.05,
                           min_cameras=2,
                           cadence_sec=10.24):
    """
    Generate a light curve for a star near (ra_star_deg, dec_star_deg).
    """

    # 1. Load detections in time window
    det = loadDetections(conn=conn,
                         jd_start=jd_start,
                         jd_end=jd_end)
    if det is None:
        return None

    ra_deg = det['ra_deg']
    dec_deg = det['dec_deg']
    jd      = det['jd']
    mag     = det['mag']
    snr     = det['snr']
    cam     = det['camera_id']

    # 2. Spatial filter around target star
    mask = spatialFilter(ra_deg, dec_deg,
                         ra_star_deg, dec_star_deg,
                         search_radius_deg)

    if not np.any(mask):
        return None

    ra_deg = ra_deg[mask]
    dec_deg = dec_deg[mask]
    jd      = jd[mask]
    mag     = mag[mask]
    snr     = snr[mask]
    cam     = cam[mask]

    # 3. Cluster + build photometric points
    points = buildLightCurve(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        jd=jd,
        mag=mag,
        snr=snr,
        camera_id=cam,
        t_window_min=t_window_min,
        ang_tol_deg=ang_tol_deg,
        min_cameras=min_cameras
    )

    if not points:
        return None

    lc = convertPointsToArrays(points)

    # 4. Cadence-based binning
    lc_binned = binByCadence(lc, cadence_sec=cadence_sec)

    # 5. Sort by time
    order = np.argsort(lc_binned['jd'])
    for k in lc_binned:
        lc_binned[k] = lc_binned[k][order]

    return lc_binned

with psycopg.connect(host="192.168.1.190", dbname="star_data", user="ingest_user") as conn:


    lc = generateStarLightCurve(
        conn,
        ra_star_deg=131.175,
        dec_star_deg=-54.708,
        search_radius_deg=0.03,  # you can nudge this if needed
        jd_start=2460927,
        jd_end=2460928,
        t_window_min=1.0,
        ang_tol_deg=0.05,
        min_cameras=2,
        cadence_sec=10.24)




    np.savez("alsephina_lightcurve.npz", **lc)
