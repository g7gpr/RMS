import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import RegularGridInterpolator
import psycopg
import json
import matplotlib.pyplot as plt

BIN_BY_CADENCE = True

# =========================
# Frame-level + spatial corrections
# =========================

def loadFramePhotometry(conn, frame_name):
    """
    Load all observations for a given frame, including:
      - observed magnitude (obs_mag)
      - catalogue magnitude (cat_mag)
      - detector coordinates (x, y)
      - star_name (for diagnostics)
    """
    sql = """
        SELECT
            obs.mag      AS obs_mag,
            star.mag     AS cat_mag,
            obs.x,
            obs.y,
            obs.star_name
        FROM observation AS obs
        JOIN star
          ON star.station_name = obs.station_name
         AND star.star_name    = obs.star_name
        WHERE obs.frame_name = %s
          AND obs.mag IS NOT NULL
          AND star.mag IS NOT NULL;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (frame_name,))
        rows = cur.fetchall()

    if not rows:
        return None

    obs_mag = np.array([r[0] for r in rows], dtype=float) / 1e6
    cat_mag = np.array([r[1] for r in rows], dtype=float) / 1e6
    x       = np.array([r[2] for r in rows], dtype=float)
    y       = np.array([r[3] for r in rows], dtype=float)
    names   = np.array([r[4] for r in rows], dtype=str)

    return {
        'obs_mag': obs_mag,
        'cat_mag': cat_mag,
        'x': x,
        'y': y,
        'star_name': names
    }


def computeFrameOffset(frame_data):
    """
    Compute robust frame-level zero-point offset using median residual.
    """
    residuals = frame_data['obs_mag'] - frame_data['cat_mag']
    return np.median(residuals)


def buildSpatialCorrectionMap(x, y, residuals, bins=40):
    """
    Build a 2D correction map Δmag(x,y) using mean residuals in bins.
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=residuals)
    N, _, _ = np.histogram2d(x, y, bins=bins)

    grid = np.where(N > 0, H / N, 0.0)

    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    ymid = 0.5 * (yedges[:-1] + yedges[1:])

    interp = RegularGridInterpolator(
        (xmid, ymid),
        grid,
        bounds_error=False,
        fill_value=0.0
    )

    return interp


def applyDetectionCorrections(conn, det):
    """
    Apply frame-level and spatial (x,y) corrections to each detection.

    det: dict from loadDetections()
    Returns a new dict with corrected 'mag'.
    """
    frame_names = det['frame_name']
    x_det = det['x']
    y_det = det['y']
    mag_det = det['mag']

    unique_frames = np.unique(frame_names)
    frame_cache = {}

    for fname in unique_frames:
        frame_data = loadFramePhotometry(conn, fname)
        if frame_data is None:
            frame_cache[fname] = (0.0, None)
            continue

        frame_offset = computeFrameOffset(frame_data)
        residuals = frame_data['obs_mag'] - frame_data['cat_mag'] - frame_offset
        spatial_map = buildSpatialCorrectionMap(
            frame_data['x'], frame_data['y'], residuals
        )
        frame_cache[fname] = (frame_offset, spatial_map)

    mag_corr = np.empty_like(mag_det)

    for i in range(len(mag_det)):
        fname = frame_names[i]
        frame_offset, spatial_map = frame_cache.get(fname, (0.0, None))

        m = mag_det[i] - frame_offset

        if spatial_map is not None:
            spatial_offset = spatial_map((x_det[i], y_det[i]))
            m -= spatial_offset

        mag_corr[i] = m

    det_corr = dict(det)
    det_corr['mag'] = mag_corr
    return det_corr


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
    x, y, z = radecToUnit(ra_deg, dec_deg)
    coords = np.column_stack((x, y, z))

    eps = chordEpsFromDeg(ang_tol_deg)
    db = DBSCAN(eps=eps, min_samples=1, metric="euclidean")
    return db.fit_predict(coords)


def clusterDetectionsEvent(ra_deg, dec_deg, jd,
                           t_window_min=1.0, ang_tol_deg=0.05):

    x, y, z = radecToUnit(ra_deg, dec_deg)

    dt_days = t_window_min / (24.0 * 60.0)
    jd0 = jd.min()

    t_scaled = (jd - jd0) / dt_days

    eps_spatial = chordEpsFromDeg(ang_tol_deg)
    t_scaled *= eps_spatial

    coords = np.column_stack((x, y, z, t_scaled))

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

    ra_rad = np.deg2rad(ra_c)
    dec_rad = np.deg2rad(dec_c)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    x_m = x.mean()
    y_m = y.mean()
    z_m = z.mean()

    r = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    x_m, y_m, z_m = x_m / r, y_m / r, z_m / r

    dec_mean = np.rad2deg(np.arcsin(z_m))
    ra_mean = np.rad2deg(np.arctan2(y_m, x_m)) % 360.0

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

        jd_bin, ra_bin, dec_bin, mag_bin, mag_err_bin = jd, ra_deg, dec_deg, mag, np.zeros_like(mag)
        n_det_bin = np.ones_like(mag, dtype=int)
        n_cam_bin = np.ones_like(mag, dtype=int)

    points = []
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        print(
            "Cluster", lab,
            "JD range:", float(np.min(jd_bin[idx])), float(np.max(jd_bin[idx])),
            "Cameras:", "N/A"
        )

        point = buildPhotometricPoint(
            idx, ra_bin, dec_bin, jd_bin, mag_bin, mag_err_bin, n_cam_bin
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

    if jd_start is not None:
        where.append("frame.jd_mid >= %s")
        params.append(int(jd_start * 1e6))

    if jd_end is not None:
        where.append("frame.jd_mid <= %s")
        params.append(int(jd_end * 1e6))

    if ra_center is not None and dec_center is not None and radius_deg is not None:
        where.append("""
            ACOS(
                LEAST(
                    GREATEST(
                        SIN(RADIANS(%s)) * SIN(RADIANS(obs.dec/1e6)) +
                        COS(RADIANS(%s)) * COS(RADIANS(obs.dec/1e6)) *
                        COS(RADIANS(obs.ra/1e6) - RADIANS(%s)),
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
            obs.station_name,
            obs.frame_name,
            obs.x,
            obs.y
        FROM observation AS obs
        JOIN frame ON obs.frame_name = frame.frame_name
        {where_clause}
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return None

    ra_deg    = np.array([r[0] for r in rows], dtype=float) / 1e6
    dec_deg   = np.array([r[1] for r in rows], dtype=float) / 1e6
    jd        = np.array([r[2] for r in rows], dtype=float) / 1e6
    mag       = np.array([r[3] for r in rows], dtype=float) / 1e6
    snr       = np.array([r[4] for r in rows], dtype=float) / 1e6
    camera_id = np.array([r[5] for r in rows], dtype=str)
    frame_name = np.array([r[6] for r in rows], dtype=str)
    x         = np.array([r[7] for r in rows], dtype=float)
    y         = np.array([r[8] for r in rows], dtype=float)

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'jd': jd,
        'mag': mag,
        'snr': snr,
        'camera_id': camera_id,
        'frame_name': frame_name,
        'x': x,
        'y': y
    }


# =========================
# Cadence binning
# =========================

def binNetworkDetectionsOld(jd, ra_deg, dec_deg, mag, snr, camera_id,
                            bin_minutes=1.0):

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

        jd_bins.append(np.mean(jd[idx]))
        ra_bins.append(np.mean(ra_deg[idx]))
        dec_bins.append(np.mean(dec_deg[idx]))

        flux = 10**(-0.4 * mag[idx])
        w = snr[idx]**2
        w_sum = np.sum(w)

        flux_mean = np.sum(w * flux) / w_sum
        mag_mean = -2.5 * np.log10(flux_mean)

        sigma_flux = np.sqrt(1.0 / w_sum)
        mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err)

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


def binNetworkDetections(jd, ra_deg, dec_deg, mag, snr, camera_id,
                         bin_minutes=1.0):

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

        jd_bins.append(np.mean(jd[idx]))
        ra_bins.append(np.mean(ra_deg[idx]))
        dec_bins.append(np.mean(dec_deg[idx]))

        flux = 10**(-0.4 * mag[idx])
        w = snr[idx]**2
        w_sum = np.sum(w)

        flux_mean = np.sum(w * flux) / w_sum
        mag_mean = -2.5 * np.log10(flux_mean)

        sigma_flux = np.sqrt(1.0 / w_sum)
        mag_err = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err)

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


def clusterDetectionsStar5D(ra_deg, dec_deg, jd, mag,
                            t_window_min=10.0,
                            ang_tol_deg=0.05,
                            mag_tol_mag=0.1):

    x, y, z = radecToUnit(ra_deg, dec_deg)

    dt_days = t_window_min / (24.0 * 60.0)
    jd0 = jd.min()
    t_scaled = (jd - jd0) / dt_days

    eps_spatial = chordEpsFromDeg(ang_tol_deg)

    t_scaled *= eps_spatial

    mag_center = np.median(mag)
    mag_scaled = mag - mag_center

    if mag_tol_mag <= 0:
        raise ValueError("mag_tol_mag must be > 0")
    mag_scaled *= (eps_spatial / mag_tol_mag)

    coords = np.column_stack((x, y, z, t_scaled, mag_scaled))

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
                           min_cameras=1,
                           cadence_sec=10.24):

    det = loadDetections(
        conn,
        jd_start=jd_start,
        jd_end=jd_end,
        ra_center=ra_star_deg,
        dec_center=dec_star_deg,
        radius_deg=search_radius_deg
    )

    if det is None:
        return None

    print(f"Unique cameras: {set(det['camera_id'])}")

    det = applyDetectionCorrections(conn, det)

    points = buildLightCurvePoints(
        det['ra_deg'], det['dec_deg'], det['jd'],
        det['mag'], det['snr'], det['camera_id'],
        ang_tol_deg=ang_tol_deg,
        min_cameras=min_cameras,
        mode="star",
        t_window_min=cadence_sec / 60.0
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


# =========================
# Plotting
# =========================

def plotTimeBinnedLightCurve(jd_bin, mag_bin, mag_err_bin, n_cam_bin):
    fig, (ax_mag, ax_cam) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

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

    ax_mag.set_ylim(8, -1)

    ax_mag.set_title("Time-Binned Light Curve (SNR-weighted)")

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


# =========================
# Main
# =========================

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
            jd_start=2460938.9,
            jd_end=2460939.05,
            ang_tol_deg=0.2,
            min_cameras=1,
            cadence_sec=10.24 * 5
        )

        if lc is None:
            print("No light curve generated.")
        else:
            saveLightCurveAsJson(lc, "debug_lightcurve.json")
            print("Saved debug_lightcurve.json")
