import numpy as np
from sklearn.cluster import DBSCAN
import psycopg
import json
import matplotlib.pyplot as plt

BIN_BY_CADENCE = True




# =========================
# Geometry helpers
# =========================

def radecToUnit(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z


def chordEpsFromDeg(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.sqrt(2 - 2 * np.cos(angle_rad))


# =========================
# Clustering
# =========================

def clusterDetectionsStar(ra_deg, dec_deg, ang_tol_deg=0.05):
    """
    Spatial-only clustering for static stars.
    """
    x, y, z = radecToUnit(ra_deg, dec_deg)
    coords = np.column_stack((x, y, z))

    eps = chordEpsFromDeg(ang_tol_deg)
    db = DBSCAN(eps=eps, min_samples=1, metric="euclidean")
    return db.fit_predict(coords)


def clusterDetectionsEvent(ra_deg, dec_deg, jd,
                           t_window_min=1.0, ang_tol_deg=0.05):

    # --- spatial coords ---
    x, y, z = radecToUnit(ra_deg, dec_deg)

    # --- convert time window (minutes) to days ---
    dt_days = t_window_min / (24.0 * 60.0)
    jd0 = jd.min()

    # 1 cadence window → 1 unit
    t_scaled = (jd - jd0) / dt_days

    # --- scale time so that 1 cadence window = eps_spatial ---
    eps_spatial = chordEpsFromDeg(ang_tol_deg)
    t_scaled *= eps_spatial

    # --- build 4D coords ---
    coords = np.column_stack((x, y, z, t_scaled))

    # --- DBSCAN with spatial epsilon ---
    eps_4d = eps_spatial

    db = DBSCAN(eps=eps_4d, min_samples=1, metric="euclidean")
    return db.fit_predict(coords)


# =========================
# Photometric point building
# =========================

def buildPhotometricPoint(idx, ra_deg, dec_deg, jd,
                          mag, snr, camera_id):
    ra_c = ra_deg[idx]
    dec_c = dec_deg[idx]
    jd_c = jd[idx]
    mag_c = mag[idx]
    snr_c = snr[idx]
    cams = camera_id[idx]

    # --- centroid position ---
    ra_rad = np.deg2rad(ra_c)
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
    ra_mean = np.rad2deg(np.arctan2(y_m, x_m)) % 360.0

    # --- weighted photometry ---
    flux = 10**(-0.4 * mag_c)
    w = snr_c**2
    w_sum = w.sum()

    flux_mean = np.sum(w * flux) / w_sum
    mag_mean = -2.5 * np.log10(flux_mean)

    sigma_flux = np.sqrt(1.0 / w_sum)
    mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

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


def binByCadence(lc, cadence_sec=10.24):
    dt_days = cadence_sec / 86400.0

    jd = lc['jd']
    jd0 = jd.min()

    bins = np.floor((jd - jd0) / dt_days).astype(int)

    out = {k: [] for k in lc}

    for b in np.unique(bins):
        idx = np.where(bins == b)[0]

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



# =========================
# High-level cluster wrapper
# =========================

def buildLightCurvePoints(ra_deg, dec_deg, jd,
                          mag, snr, camera_id,
                          ang_tol_deg=0.05,
                          min_cameras=2,
                          mode="star",
                          t_window_min=1.0):
    """
    Build photometric points by clustering detections in space and time.

    For stars, we use spatiotemporal clustering so that:
      - detections close in time (multi-camera) are merged
      - detections far apart in time become separate light-curve points

    For events, the same clustering logic applies but with a larger time window.
    """

    # --- Always use spatiotemporal clustering for light curves ---

    if True:

        (jd_bin, ra_bin, dec_bin, mag_bin, mag_err_bin,
         n_det_bin, n_cam_bin, cam_sets) = binNetworkDetections(
            jd, ra_deg, dec_deg, mag, snr, camera_id
        )

        plotTimeBinnedLightCurve(jd_bin, mag_bin, mag_err_bin, n_cam_bin)

        labels = clusterDetectionsStar5D(
            ra_bin, dec_bin, jd_bin, mag_bin,
            t_window_min=10.0,
            ang_tol_deg=0.05,
            mag_tol_mag=0.1
        )


    else:

        labels = clusterDetectionsEvent(
            ra_deg, dec_deg, jd,
            t_window_min=t_window_min,
            ang_tol_deg=ang_tol_deg
        )

    points = []
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        if lab == -1:
            continue  # DBSCAN noise

        idx = np.where(labels == lab)[0]

        # 🔍 DEBUG PRINT HERE
        print(
            "Cluster", lab,
            "JD range:", float(np.min(jd[idx])), float(np.max(jd[idx])),
            "Cameras:", set(camera_id[idx])
        )

        point = buildPhotometricPoint(
            idx, ra_deg, dec_deg, jd, mag, snr, camera_id
        )

        if point['n_cam'] >= min_cameras:
            points.append(point)

    return points



# =========================
# DB access
# =========================

def loadDetections(conn, jd_start=None, jd_end=None,
                   ra_center=None, dec_center=None, radius_deg=None):

    where = []
    params = []

    # Time filtering
    if jd_start is not None:
        where.append("frame.jd_mid >= %s")
        params.append(int(jd_start * 1e6))

    if jd_end is not None:
        where.append("frame.jd_mid <= %s")
        params.append(int(jd_end * 1e6))

    # Spatial filtering (cone search)
    if ra_center is not None and dec_center is not None and radius_deg is not None:
        where.append("""
            ACOS(
                LEAST(
                    GREATEST(
                        SIN(RADIANS(%s)) * SIN(RADIANS(obs.dec/1e6)) +
                        COS(RADIANS(%s)) * COS(RADIANS(obs.dec/1e6)) *
                        COS(RADIANS(obs.ra/1e6 - %s)),
                    -1.0),
                1.0)
            ) * 180.0 / PI() <= %s
        """)
        params.extend([dec_center, dec_center, ra_center, radius_deg])

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

    ra_deg = np.array([r[0] for r in rows], dtype=float) / 1e6
    dec_deg = np.array([r[1] for r in rows], dtype=float) / 1e6
    jd = np.array([r[2] for r in rows], dtype=float) / 1e6
    mag = np.array([r[3] for r in rows], dtype=float) / 1e6
    snr = np.array([r[4] for r in rows], dtype=float) / 1e6
    camera_id = np.array([r[5] for r in rows], dtype=str)

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'jd': jd,
        'mag': mag,
        'snr': snr,
        'camera_id': camera_id
    }


# =========================
# Cadence binning
# =========================

def binNetworkDetectionsOld(jd, ra_deg, dec_deg, mag, snr, camera_id,
                         bin_minutes=1.0):
    """
    Network-wide time binning with SNR-weighted flux averaging.
    """

    jd = np.asarray(jd)
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)
    mag = np.asarray(mag)
    snr = np.asarray(snr)
    camera_id = np.asarray(camera_id)

    dt_days = bin_minutes / (24.0 * 60.0)
    jd0 = jd.min()
    bin_index = np.floor((jd - jd0) / dt_days).astype(int)

    jd_bins = []
    ra_bins = []
    dec_bins = []
    mag_bins = []
    mag_err_bins = []
    n_det_bins = []
    n_cam_bins = []
    cam_sets = []

    for b in np.unique(bin_index):
        idx = np.where(bin_index == b)[0]

        # time & position
        jd_bins.append(np.mean(jd[idx]))
        ra_bins.append(np.mean(ra_deg[idx]))
        dec_bins.append(np.mean(dec_deg[idx]))

        # SNR-weighted flux averaging
        flux = 10**(-0.4 * mag[idx])
        w = snr[idx]**2
        w_sum = np.sum(w)

        flux_mean = np.sum(w * flux) / w_sum
        mag_mean = -2.5 * np.log10(flux_mean)

        sigma_flux = np.sqrt(1.0 / w_sum)
        mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err)

        # metadata
        cams = set(camera_id[idx])
        cam_sets.append(cams)

        n_det_bins.append(len(idx))
        n_cam_bins.append(len(cams))

    return (
        np.array(jd_bins),
        np.array(ra_bins),
        np.array(dec_bins),
        np.array(mag_bins),
        np.array(mag_err_bins),
        np.array(n_det_bins),
        np.array(n_cam_bins),
        cam_sets
    )



from sklearn.cluster import DBSCAN
import numpy as np

def binNetworkDetections(jd, ra_deg, dec_deg, mag, snr, camera_id,
                         bin_minutes=1.0):
    """
    Network-wide time binning with SNR-weighted flux averaging.

    Returns per-bin:
      jd_bin, ra_bin, dec_bin, mag_bin, mag_err_bin,
      n_det_bin, n_cam_bin, cam_sets
    """

    jd = np.asarray(jd)
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)
    mag = np.asarray(mag)
    snr = np.asarray(snr)
    camera_id = np.asarray(camera_id)

    dt_days = bin_minutes / (24.0 * 60.0)
    jd0 = jd.min()
    bin_index = np.floor((jd - jd0) / dt_days).astype(int)

    # --- declare all output lists ---
    jd_bins = []
    ra_bins = []
    dec_bins = []
    mag_bins = []
    mag_err_bins = []
    n_det_bins = []
    n_cam_bins = []
    cam_sets = []

    # --- fill bins ---
    for b in np.unique(bin_index):
        idx = np.where(bin_index == b)[0]

        # time & position
        jd_bins.append(np.mean(jd[idx]))
        ra_bins.append(np.mean(ra_deg[idx]))
        dec_bins.append(np.mean(dec_deg[idx]))

        # SNR-weighted flux averaging
        flux = 10**(-0.4 * mag[idx])
        w = snr[idx]**2
        w_sum = np.sum(w)

        flux_mean = np.sum(w * flux) / w_sum
        mag_mean = -2.5 * np.log10(flux_mean)

        # SNR-weighted uncertainty
        sigma_flux = np.sqrt(1.0 / w_sum)
        mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err)

        # metadata
        cams = set(camera_id[idx])
        cam_sets.append(cams)

        n_det_bins.append(len(idx))
        n_cam_bins.append(len(cams))

    # --- convert to arrays ---
    return (
        np.array(jd_bins),
        np.array(ra_bins),
        np.array(dec_bins),
        np.array(mag_bins),
        np.array(mag_err_bins),
        np.array(n_det_bins),
        np.array(n_cam_bins),
        cam_sets
    )



def clusterDetectionsStar5D(ra_deg, dec_deg, jd, mag,
                            t_window_min=10.0,
                            ang_tol_deg=0.05,
                            mag_tol_mag=0.1):
    """
    5D clustering for stars: (x, y, z, t, mag).

    - ang_tol_deg: max angular separation on sky
    - t_window_min: characteristic time scale for linking detections
    - mag_tol_mag: characteristic magnitude difference to still be "same state"
    """

    # --- spatial: unit vector on the sphere ---
    x, y, z = radecToUnit(ra_deg, dec_deg)

    # --- time: scale so that 1 t_window_min -> 1 unit ---
    dt_days = t_window_min / (24.0 * 60.0)
    jd0 = jd.min()
    t_scaled = (jd - jd0) / dt_days  # 1 unit = t_window_min

    # --- spatial epsilon in chord distance ---
    eps_spatial = chordEpsFromDeg(ang_tol_deg)

    # scale time so that 1 time window ~ eps_spatial
    t_scaled *= eps_spatial

    # --- magnitude: center and scale ---
    mag_center = np.median(mag)
    mag_scaled = mag - mag_center

    # mag_tol_mag difference -> ~eps_spatial
    if mag_tol_mag <= 0:
        raise ValueError("mag_tol_mag must be > 0")
    mag_scaled *= (eps_spatial / mag_tol_mag)

    # --- build 5D coords ---
    coords = np.column_stack((x, y, z, t_scaled, mag_scaled))

    # --- DBSCAN in 5D ---
    eps_5d = eps_spatial
    db = DBSCAN(eps=eps_5d, min_samples=1, metric="euclidean")
    return db.fit_predict(coords)



# =========================
# High-level API
# =========================

def generateStarLightCurve(conn,
                           ra_star_deg,
                           dec_star_deg,
                           search_radius_deg=0.05,
                           jd_start=None,
                           jd_end=None,
                           ang_tol_deg=0.05,
                           min_cameras=2,
                           cadence_sec=10.24):

    det = loadDetections(
        conn,
        jd_start=jd_start,
        jd_end=jd_end,
        ra_center=ra_star_deg,
        dec_center=dec_star_deg,
        radius_deg=search_radius_deg
    )

    print(f"Unique cameras: {set(det['camera_id'])}")

    if det is None:
        return None

    points = buildLightCurvePoints(
        det['ra_deg'], det['dec_deg'], det['jd'],
        det['mag'], det['snr'], det['camera_id'],
        ang_tol_deg=ang_tol_deg,
        min_cameras=min_cameras,
        mode="star",
        t_window_min=cadence_sec / 60.0  # e.g. ~0.85 min for 51.2 s cadence
    )

    if not points:
        return None

    lc = {
        'jd':      np.array([p['jd'] for p in points]),
        'mag':     np.array([p['mag'] for p in points]),
        'mag_err': np.array([p['mag_err'] for p in points]),
        'ra_deg':  np.array([p['ra_deg'] for p in points]),
        'dec_deg': np.array([p['dec_deg'] for p in points]),
        'n_det':   np.array([p['n_det'] for p in points]),
        'n_cam':   np.array([p['n_cam'] for p in points])
    }

    if BIN_BY_CADENCE:
        lc = binByCadence(lc, cadence_sec=cadence_sec)

    order = np.argsort(lc['jd'])
    for k in lc:
        lc[k] = lc[k][order]

    return lc


# =========================
# JSON export
# =========================

def saveLightCurveAsJson(lc, filename):
    if lc is None:
        raise ValueError("Light curve is None; nothing to save.")

    serializable = {k: v.tolist() for k, v in lc.items()}

    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)


import matplotlib.pyplot as plt
import numpy as np

def plotTimeBinnedLightCurve(jd_bin, mag_bin, mag_err_bin, n_cam_bin):
    fig, (ax_mag, ax_cam) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ============================================================
    #  TOP PANEL — Magnitude with discrete uncertainty patches
    # ============================================================

    for x, y, dy in zip(jd_bin, mag_bin, mag_err_bin):
        ax_mag.fill_between(
            [x - 0.0001, x + 0.0001],
            y - dy,
            y + dy,
            color='lightblue',
            alpha=0.35,
            linewidth=0
        )

    ax_mag.scatter(
        jd_bin,
        mag_bin,
        s=14,
        color='tab:blue',
        alpha=0.9
    )

    ax_mag.set_ylabel("Magnitude", color='tab:blue')
    ax_mag.invert_yaxis()
    ax_mag.tick_params(axis='y', labelcolor='tab:blue')

    # Clamp y-axis to magnitude range only
    y_min = np.max(mag_bin)
    y_max = np.min(mag_bin)
    ax_mag.set_ylim(8, -1)   # faintest at top, brightest at bottom


    ax_mag.set_title("Time-Binned Light Curve (SNR-weighted)")

    # ============================================================
    #  LOWER PANEL — Camera count histogram
    # ============================================================

    # Use bar plot for discrete camera counts
    ax_cam.bar(
        jd_bin,
        n_cam_bin,
        width=(jd_bin[1] - jd_bin[0]) * 0.8 if len(jd_bin) > 1 else 0.001,
        color='tab:red',
        alpha=0.7
    )

    ax_cam.set_ylabel("Cameras")
    ax_cam.set_xlabel("JD")

    fig.tight_layout()
    plt.show()

















if __name__ == "__main__":
    with psycopg.connect(
        host="192.168.1.212",
        dbname="star_data",
        user="ingest_user"
    ) as conn:

        lc = generateStarLightCurve(
            conn,
            ra_star_deg=186.66,
            dec_star_deg=-63.10,
            search_radius_deg=0.5,
            jd_start=2460913,
            jd_end=2460922.4,
            ang_tol_deg=0.2,
            min_cameras=1,
            cadence_sec=10.24 * 5
        )

        if lc is None:
            print("No light curve generated.")
        else:
            saveLightCurveAsJson(lc, "debug_lightcurve.json")
            print("Saved debug_lightcurve.json")
