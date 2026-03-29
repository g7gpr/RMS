import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import RegularGridInterpolator
import psycopg
import json
import matplotlib.pyplot as plt
import tqdm
import os

BIN_BY_CADENCE = True
CHARTS_DIR = os.path.expanduser("~/RMS_data/Plots")
os.makedirs(CHARTS_DIR, exist_ok=True)

SPATIAL_METHOD = "gaussian"

# =========================
# Frame-level + spatial corrections
# =========================


import numpy as np
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Astrometry.Conversions import datetime2JD


class GaussianSpatialModel:
    """
    2D anisotropic Gaussian spatial correction model.

    delta_mag(x, y) = offset + A * exp(-0.5 * Q)

    where Q is the Mahalanobis distance squared:

        Q = (1 / (1 - rho^2)) * [
                (dx/sx)^2 + (dy/sy)^2 - 2*rho*(dx/sx)*(dy/sy)
            ]

        dx = x - x0
        dy = y - y0
    """

    def __init__(self, x0, y0, amp, sigma_x, sigma_y, rho=0.0, offset=0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.amp = float(amp)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.rho = float(rho)
        self.offset = float(offset)

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=float)

        # Accept [x, y] or [[x, y], ...]
        if xy.ndim == 1:
            xy = xy[None, :]

        x = xy[:, 0]
        y = xy[:, 1]

        dx = x - self.x0
        dy = y - self.y0

        sx = self.sigma_x
        sy = self.sigma_y
        rho = self.rho

        # Avoid degenerate sigmas
        if sx <= 0 or sy <= 0:
            return np.full_like(x, self.offset)

        denom = 1.0 - rho * rho
        if denom <= 0:
            rho = 0.0
            denom = 1.0

        q = (
            (dx / sx)**2 +
            (dy / sy)**2 -
            2.0 * rho * (dx / sx) * (dy / sy)
        ) / denom

        return self.offset + self.amp * np.exp(-0.5 * q)

    def to_params(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "amp": self.amp,
            "sigma_x": self.sigma_x,
            "sigma_y": self.sigma_y,
            "rho": self.rho,
            "offset": self.offset,
        }

    @classmethod
    def from_params(cls, params):
        if params is None:
            return None
        return cls(
            x0=params["x0"],
            y0=params["y0"],
            amp=params["amp"],
            sigma_x=params["sigma_x"],
            sigma_y=params["sigma_y"],
            rho=params.get("rho", 0.0),
            offset=params.get("offset", 0.0),
        )

def fitGaussianSpatialModel(x, y, residuals):
    """
    Fit a simple Gaussian model to spatial residuals.
    This is a heuristic fit, not a full optimizer.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(residuals)

    # Center at weighted centroid
    w = np.abs(r) + 1e-6
    x0 = np.average(x, weights=w)
    y0 = np.average(y, weights=w)

    # Amplitude = peak residual
    amp = np.median(r)

    # Widths = fraction of detector range
    sigma_x = 0.25 * (x.max() - x.min())
    sigma_y = 0.25 * (y.max() - y.min())

    # No correlation initially
    rho = 0.0

    # Offset = median residual
    offset = np.median(r)

    return GaussianSpatialModel(x0, y0, amp, sigma_x, sigma_y, rho, offset)





def loadCachedSpatialMap(conn, frame_name, method='binned', version=1):
    sql = """
        SELECT grid_mag, xmid, ymid, params
        FROM spatial_correction
        WHERE frame_name = %s AND method = %s AND version = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (frame_name, method, version))
        row = cur.fetchone()

    if not row:
        return None

    grid_mag, xmid, ymid, params = row

    # Binned method
    if method == 'binned':
        grid_mag = np.array(grid_mag)
        xmid = np.array(xmid)
        ymid = np.array(ymid)

        interp = RegularGridInterpolator(
            (xmid, ymid),
            grid_mag,
            bounds_error=False,
            fill_value=0.0
        )
        return interp

    # Gaussian method
    if method == 'gaussian':
        if params is None:
            return None
        return GaussianSpatialModel.from_params(params)

    raise ValueError(f"Unknown spatial correction method: {method}")


def saveCachedSpatialMap(conn, frame_name, method, version,
                         grid_mag=None, xmid=None, ymid=None,
                         params=None):

    sql = """
        INSERT INTO spatial_correction
            (frame_name, method, version, grid_mag, xmid, ymid, params)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (frame_name, method, version) DO UPDATE
        SET grid_mag = EXCLUDED.grid_mag,
            xmid     = EXCLUDED.xmid,
            ymid     = EXCLUDED.ymid,
            params   = EXCLUDED.params,
            created_at = NOW();
    """

    with conn.cursor() as cur:
        cur.execute(sql, (
            frame_name,
            method,
            version,
            json.dumps(grid_mag.tolist()) if grid_mag is not None else None,
            json.dumps(xmid.tolist()) if xmid is not None else None,
            json.dumps(ymid.tolist()) if ymid is not None else None,
            json.dumps(params) if params is not None else None
        ))
    conn.commit()


def lookupBrightestStar(conn, ra_deg, dec_deg, radius_deg=0.05):
    sql = """
        SELECT star_name, mag
        FROM star
        WHERE ACOS(
            LEAST(
                GREATEST(
                    SIN(RADIANS(%s)) * SIN(RADIANS(dec/1e6)) +
                    COS(RADIANS(%s)) * COS(RADIANS(dec/1e6)) *
                    COS(RADIANS(ra/1e6) - RADIANS(%s)),
                -1.0),
            1.0)
        ) * 180.0 / PI() <= %s
        ORDER BY mag ASC
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (dec_deg, dec_deg, ra_deg, radius_deg))
        row = cur.fetchone()
    return row if row else None



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
            obs.y
        FROM observation AS obs
        JOIN star
          ON star.station_name = obs.station_name
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

    return {
        'obs_mag': obs_mag,
        'cat_mag': cat_mag,
        'x': x,
        'y': y
    }


def computeFrameOffset(frame_data):
    """
    Compute robust frame-level zero-point offset using median residual.
    """
    residuals = frame_data['obs_mag'] - frame_data['cat_mag']

    if len(frame_data['obs_mag']) < 3:
        return np.mean(residuals)

    return np.median(residuals)

def buildSpatialCorrectionMap(x, y, residuals_mag, bins=40):
    """
    Build a 2D correction map delta mag(x,y) using mean flux ratios in bins.

    Spatial corrections are derived from the mean flux ratio in each detector bin, converted back to magnitudes.

    """

    # Convert mag residuals to flux ratios
    flux_ratio = 10**(-0.4 * residuals_mag)

    # Bin the flux ratios
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=flux_ratio)
    N, _, _ = np.histogram2d(x, y, bins=bins)

    # Mean flux ratio per bin (1.0 = no correction)
    grid_ratio = np.where(N > 0, H / N, 1.0)

    # Convert back to magnitude correction
    grid_mag = -2.5 * np.log10(grid_ratio)

    # Build interpolator
    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    ymid = 0.5 * (yedges[:-1] + yedges[1:])

    interp = RegularGridInterpolator(
        (xmid, ymid),
        grid_mag,
        bounds_error=False,
        fill_value=0.0
    )

    return interp, grid_mag, xmid, ymid



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

    for fname in tqdm.tqdm(unique_frames):

        method = SPATIAL_METHOD
        version = 1


        cached_map = loadCachedSpatialMap(conn, fname, method, version)

        # 1. Try cache first
        if cached_map is not None:
            print(f"[cache hit]  spatial map for frame {fname}")
            frame_data = loadFramePhotometry(conn, fname)
            frame_offset = computeFrameOffset(frame_data) if frame_data else 0.0
            frame_cache[fname] = (frame_offset, cached_map)
            continue

        print(f"[cache miss] building spatial map for frame {fname}")

        # 2. Otherwise compute
        frame_data = loadFramePhotometry(conn, fname)
        if frame_data is None:
            frame_cache[fname] = (0.0, None)
            continue

        frame_offset = computeFrameOffset(frame_data)
        residuals = frame_data['obs_mag'] - frame_data['cat_mag'] - frame_offset

        if SPATIAL_METHOD == 'binned':

            spatial_map, grid_mag, xmid, ymid = buildSpatialCorrectionMap(
                frame_data['x'], frame_data['y'], residuals)

            # Save binned map
            saveCachedSpatialMap(
                conn,
                frame_name=fname,
                method="binned",
                version=1,
                grid_mag=grid_mag,
                xmid=xmid,
                ymid=ymid
            )

            frame_cache[fname] = (frame_offset, spatial_map)

        else:

            gauss_model = fitGaussianSpatialModel(
                frame_data['x'], frame_data['y'], residuals)

            saveCachedSpatialMap(
                conn,
                frame_name=fname,
                method="gaussian",
                version=1,
                params=gauss_model.to_params()
            )

            frame_cache[fname] = (frame_offset, gauss_model)


    mag_corr = np.empty_like(mag_det)

    for i in range(len(mag_det)):
        fname = frame_names[i]
        frame_offset, spatial_map = frame_cache.get(fname, (0.0, None))

        m = mag_det[i] - frame_offset

        if spatial_map is not None:
            if np.isnan(x_det[i]) or np.isnan(y_det[i]):
                spatial_offset = 0.0
            else:
                spatial_offset = spatial_map([[x_det[i], y_det[i]]])[0]


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
                          mag, snr, camera_id, cat_mag=None,
                          ang_tol_deg=0.05,
                          min_cameras=2,
                          mode="star",
                          star_name=None,
                          t_window_min=1.0):

    if True:
        (jd_bin, ra_bin, dec_bin, mag_bin, mag_err_bin,
         n_det_bin, n_cam_bin, cam_sets) = binNetworkDetections(
            jd, ra_deg, dec_deg, mag, snr, camera_id
        )

        plot_filename = plotTimeBinnedLightCurve(jd_bin, mag_bin, mag_err_bin, n_cam_bin, ra_bin, dec_bin,
                                 n_stations=len(set(camera_id)), n_observations=len(jd), cat_mag=cat_mag,
                                                 star_name=star_name)

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

    return points, plot_filename


# =========================
# DB access
# =========================

def loadDetections(conn, jd_start=None, jd_end=None,
                   ra_center=None, dec_center=None, radius_deg=None,
                   prove=True, prove_n=100):

    where = []
    params = []

    # Optional spatial cone filter (SQL-side)
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
            obs.mag,
            obs.snr,
            obs.station_name,
            obs.frame_name,
            obs.x,
            obs.y,
            obs.mag_err
        FROM observation AS obs
        {where_clause}
    """

    with conn.cursor() as cur:
        #print(f"Executing {sql}")
        cur.execute(sql, params)
        #print("Completed")
        rows = cur.fetchall()

    if not rows:
        return None

    # Extract columns
    ra_deg     = np.array([r[0] for r in rows], dtype=float) / 1e6
    dec_deg    = np.array([r[1] for r in rows], dtype=float) / 1e6
    mag        = np.array([r[2] for r in rows], dtype=float) / 1e6
    snr        = np.array([r[3] for r in rows], dtype=float) / 1e6
    camera_id  = np.array([r[4] for r in rows], dtype=str)
    frame_name = np.array([r[5] for r in rows], dtype=str)
    x          = np.array([r[6] for r in rows], dtype=float)
    y          = np.array([r[7] for r in rows], dtype=float)

    # mag_err may be NULL → convert to nan
    mag_err = np.array([
        (r[8] / 1e6) if r[8] is not None else np.nan
        for r in rows
    ], dtype=float)

    # Reconstruct catalogue magnitude
    cat_mag = mag - mag_err

    # Compute JD mid from frame name using your existing function
    jd = np.array([
        datetime2JD(getMiddleTimeFF(f"FF_{fn}_000000",fps=25, dt_obj=True))  # returns JD in natural units
        for fn in frame_name
    ], dtype=float)

    # Optional time filtering AFTER JD reconstruction
    if jd_start is not None:
        mask = jd >= jd_start
        jd = jd[mask]
        ra_deg = ra_deg[mask]
        dec_deg = dec_deg[mask]
        mag = mag[mask]
        snr = snr[mask]
        camera_id = camera_id[mask]
        frame_name = frame_name[mask]
        x = x[mask]
        y = y[mask]
        cat_mag = cat_mag[mask]
        mag_err = mag_err[mask]

    if jd_end is not None:
        mask = jd <= jd_end
        jd = jd[mask]
        ra_deg = ra_deg[mask]
        dec_deg = dec_deg[mask]
        mag = mag[mask]
        snr = snr[mask]
        camera_id = camera_id[mask]
        frame_name = frame_name[mask]
        x = x[mask]
        y = y[mask]
        cat_mag = cat_mag[mask]
        mag_err = mag_err[mask]

    # Optional proving mode
    if prove:
        print("\n=== PROVING lookupBrightestStar() ===")
        for i in range(min(prove_n, len(ra_deg))):
            ra = ra_deg[i]
            dec = dec_deg[i]

            row = lookupBrightestStar(conn, ra, dec, radius_deg=0.2)

            print(f"[{i}] RA={ra:.6f}, Dec={dec:.6f}")
            print(f"     observed mag = {mag[i]:.3f}")
            print(f"     mag_err = {mag_err[i]:.3f}")
            print(f"     reconstructed cat_mag = {cat_mag[i]:.3f}")

            if row is None:
                print("     lookupBrightestStar returned None\n")
                continue

            star_name, cat_mag_catalogue = row
            cat_mag_catalogue /= 1e6

            print(f"     lookupBrightestStar returned {star_name}, mag={cat_mag_catalogue:.3f}\n")
            pass

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'jd': jd,
        'mag': mag,
        'snr': snr,
        'camera_id': camera_id,
        'frame_name': frame_name,
        'x': x,
        'y': y,
        'cat_mag': cat_mag,
        'mag_err': mag_err}


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

    star_name_from_db, mag_from_db = lookupBrightestStar(conn, ra_star_deg, dec_star_deg, radius_deg=0.05)

    print(f"Working on star {star_name_from_db}, RA: {ra_star_deg}, DEC: {dec_star_deg} MAG:{mag_from_db/1e6}")

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


    cat_mag = float(np.nanmedian(det['cat_mag']))

    points, plot_filename = buildLightCurvePoints(
        det['ra_deg'], det['dec_deg'], det['jd'],
        det['mag'], det['snr'], det['camera_id'], cat_mag=cat_mag,
        ang_tol_deg=ang_tol_deg,
        min_cameras=min_cameras,
        mode="star",
        star_name=star_name_from_db,
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
        'n_cam':   np.array([p['n_cam'] for p in points]),

    }


    if BIN_BY_CADENCE:
        lc = binByCadence(lc, cadence_sec=cadence_sec)

    order = np.argsort(lc['jd'])
    for k in lc:
        lc[k] = lc[k][order]

    # Metadata
    stations = set(det['camera_id'])
    n_observations = len(det['jd'])


    # Filename
    ra_str  = f"{ra_star_deg:.3f}"
    dec_str = f"{dec_star_deg:.3f}"
    jd0_str = f"{jd_start:.5f}"
    jd1_str = f"{jd_end:.5f}"

    filename = f"lc_jd{jd0_str}-{jd1_str}_ra{ra_str}_dec{dec_str}.json"

    # Save JSON with metadata
    saveLightCurveAsJson(
        lc,
        filename,
        stations=stations,
        n_observations=n_observations,
        ra_star_deg=ra_star_deg,
        dec_star_deg=dec_star_deg,
        jd_start=jd_start,
        jd_end=jd_end,
        cat_mag=cat_mag
    )

    json_filename = filename
    png_filename = plot_filename  # from plotTimeBinnedLightCurve

    saveLightCurveSidecarTxt(
        star_name_from_db,
        ra_star_deg,
        dec_star_deg,
        jd_start,
        jd_end,
        cat_mag,
        n_observations,
        len(stations),
        lc,
        json_filename,
        plot_filename
    )

    return lc


def saveLightCurveSidecarTxt(
    star_name,
    ra_star_deg,
    dec_star_deg,
    jd_start,
    jd_end,
    cat_mag,
    n_observations,
    n_stations,
    lc,
    json_filename,
    png_filename
):
    txt_filename = json_filename.replace(".json", ".txt")
    txt_path = os.path.join(CHARTS_DIR, txt_filename)



    ra_mean = float(np.mean(lc['ra_deg']))
    dec_mean = float(np.mean(lc['dec_deg']))

    mag_med = float(np.nanmedian(lc['mag']))
    mag_min = float(np.nanmin(lc['mag']))
    mag_max = float(np.nanmax(lc['mag']))
    mag_err_med = float(np.nanmedian(lc['mag_err']))

    with open(txt_path, "w") as f:
        f.write(f"Star: {star_name}\n")
        f.write(f"Catalogue magnitude: {cat_mag:.3f}\n\n")

        f.write(f"RA (target):  {ra_star_deg:.6f} deg\n")
        f.write(f"Dec (target): {dec_star_deg:.6f} deg\n")
        f.write(f"RA (mean):    {ra_mean:.6f} deg\n")
        f.write(f"Dec (mean):   {dec_mean:.6f} deg\n\n")

        f.write(f"JD start: {jd_start}\n")
        f.write(f"JD end:   {jd_end}\n")
        f.write(f"Detections: {n_observations}\n")
        f.write(f"Stations:   {n_stations}\n")
        f.write(f"Binned points: {len(lc['jd'])}\n\n")

        f.write(f"Mag median: {mag_med:.3f}\n")
        f.write(f"Mag min:    {mag_min:.3f}\n")
        f.write(f"Mag max:    {mag_max:.3f}\n")
        f.write(f"Mag_err median: {mag_err_med:.3f}\n\n")

        f.write(f"JSON: {json_filename}\n")
        f.write(f"Plot: {png_filename}\n")


# =========================
# JSON export
# =========================

def saveLightCurveAsJson(lc, filename,
                         stations=None,
                         n_observations=None,
                         ra_star_deg=None,
                         dec_star_deg=None,
                         jd_start=None,
                         jd_end=None,
                         cat_mag=None):


    serializable = {k: v.tolist() for k, v in lc.items()}
    file_path = os.path.join(CHARTS_DIR, filename)

    if stations is not None:
        serializable["stations"] = list(stations)

    if n_observations is not None:
        serializable["n_observations"] = int(n_observations)

    if ra_star_deg is not None:
        serializable["ra_star_deg"] = float(ra_star_deg)

    if dec_star_deg is not None:
        serializable["dec_star_deg"] = float(dec_star_deg)

    if jd_start is not None:
        serializable["jd_start"] = float(jd_start)

    if jd_end is not None:
        serializable["jd_end"] = float(jd_end)

    if cat_mag is not None:
        serializable["cat_mag"] = float(cat_mag)


    with open(file_path, "w") as f:
        json.dump(serializable, f, indent=2)


def plotTimeBinnedLightCurve(jd_bin, mag_bin, mag_err_bin, n_cam_bin,
                             ra_bin, dec_bin,
                             n_stations, n_observations, cat_mag, star_name=None):

    jd0 = jd_bin[0]
    jd_rel = jd_bin - jd0

    ra_mean  = np.mean(ra_bin)
    dec_mean = np.mean(dec_bin)

    fig, (ax_mag, ax_cam) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax_mag.set_title(
        f"{star_name} — Time-Binned Light Curve\n"
        f"RA={ra_mean:.3f}°, Dec={dec_mean:.3f}°\n"
        f"{n_stations} stations, {n_observations} detections"
    )


    for x, y, dy in zip(jd_rel, mag_bin, mag_err_bin):
        ax_mag.fill_between(
            [x - 0.0001, x + 0.0001],
            y - dy,
            y + dy,
            color='lightblue',
            alpha=0.35,
            linewidth=0
        )

    ax_mag.scatter(jd_rel, mag_bin, s=14, color='tab:blue', alpha=0.9)
    ax_mag.set_ylabel("Magnitude")
    ax_mag.invert_yaxis()
    ax_mag.set_ylim(10, -2)

    if cat_mag is not None:
        ax_mag.axhline(
            y=cat_mag,
            color='grey',
            linestyle=':',
            linewidth=1.0,
            alpha=0.6,
            label=f"Catalogue mag {cat_mag:.3f}"
        )

    ax_cam.bar(
        jd_rel,
        n_cam_bin,
        width=(jd_rel[1] - jd_rel[0]) * 0.8 if len(jd_rel) > 1 else 0.001,
        color='tab:red',
        alpha=0.7
    )



    ax_cam.set_ylabel("Cameras")
    ax_cam.set_xlabel(f"Time since JD {jd0:.5f} (days)")

    fig.tight_layout()
    #plt.show()

    plot_filename = (
        f"{star_name.replace(' ', '_')}"
        f"_jd{jd_bin[0]:.5f}-{jd_bin[-1]:.5f}"
        f"_ra{ra_mean:.3f}_dec{dec_mean:.3f}.png"
    ).replace(" ", "_").replace("/", "_")


    plot_filepath = os.path.join(CHARTS_DIR, plot_filename)
    fig.savefig(plot_filepath, dpi=150)

    return plot_filename

if __name__ == "__main__":
    with psycopg.connect(
        host="192.168.1.212",
        dbname="star_data",
        user="ingest_user"
    ) as conn:

        lc = generateStarLightCurve(
            conn,
            ra_star_deg=84.06,
            dec_star_deg=-1.2024,
            search_radius_deg=0.5,
            jd_start=2460310,
            jd_end=2460311,
            ang_tol_deg=0.2,
            min_cameras=1,
            cadence_sec=10.24 * 5
        )

        if lc is None:
            print("No light curve generated.")
        else:

            print("Saved debug_lightcurve.json")
