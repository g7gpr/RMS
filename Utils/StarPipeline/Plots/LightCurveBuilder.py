import datetime
from tkinter.constants import NONE

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import RegularGridInterpolator, Rbf
from scipy.spatial import ConvexHull
import psycopg
import json
import matplotlib.pyplot as plt
import math
import os
import argparse

BIN_BY_CADENCE = True
CHARTS_DIR = os.path.expanduser("~/RMS_data/Plots")
os.makedirs(CHARTS_DIR, exist_ok=True)

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

def makeLightcurveFilename(ra_deg, dec_deg, jd_start, jd_end,
                           star_name=None, spatial_method=None):

    ra_str  = f"{ra_deg:.3f}"
    dec_str = f"{dec_deg:.3f}"
    jd0_str = f"{jd_start:.5f}"
    jd1_str = f"{jd_end:.5f}"

    if star_name:
        safe_name = star_name.replace(" ", "_").replace("/", "_")
        base = f"{safe_name}_jd{jd0_str}-{jd1_str}_ra{ra_str}_dec{dec_str}"
    else:
        base = f"lc_jd{jd0_str}-{jd1_str}_ra{ra_str}_dec{dec_str}"

    if spatial_method:
        base += f"_model_{spatial_method}"

    return base


def loadCachedSpatialMap(conn, frame_name, method='binned', version=1):
    sql = """
        SELECT grid_mag, xmid, ymid, params
        FROM spatial_model
        WHERE frame_name = %s AND model_type = %s AND version = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (frame_name, method, version))
        row = cur.fetchone()

    if not row:
        return None

    grid_mag, xmid, ymid, params = row

    # -------------------------
    # Binned method
    # -------------------------
    if method == 'binned':
        if grid_mag is None or xmid is None or ymid is None:
            return None

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

    # -------------------------
    # Gaussian method
    # -------------------------
    if method == 'gaussian':
        if params is None:
            return None
        return GaussianSpatialModel.from_params(params)

    # -------------------------
    # TPS method
    # -------------------------
    if method == 'tps':
        if params is None:
            return None

        # Expecting params = { "x": [...], "y": [...], "residuals": [...], "smooth": value }
        x = np.array(params.get("x", []))
        y = np.array(params.get("y", []))
        residuals = np.array(params.get("residuals", []))
        smooth = params.get("smooth", 0.1)

        if len(x) == 0 or len(y) == 0 or len(residuals) == 0:
            return None

        # Rebuild the TPS model
        try:
            tps_model = Rbf(x, y, residuals, function='thin_plate', smooth=smooth)
            return tps_model
        except Exception:
            return None

    # -------------------------
    # Unknown method
    # -------------------------
    raise ValueError(f"Unknown spatial correction method: {method}")


def loadSpatialModel(conn, frame_name, spatial_model=None, version=1):
    """
    Load a spatial correction model from the spatial_model table.

    Returns
    -------
    - RegularGridInterpolator instance (for binned)
    - GaussianSpatialModel instance (for gaussian)
    - None (if no model found or model_type == 'none')
    """

    sql = """
        SELECT grid_mag, xmid, ymid, params
        FROM spatial_model
        WHERE frame_name = %s AND model_type = %s AND version = %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (frame_name, spatial_model, version))
        row = cur.fetchone()

    if not row:
        return None

    grid_mag_json, xmid_json, ymid_json, params_json = row

    # -------------------------
    # Model type: NONE
    # -------------------------
    if spatial_model == "none":
        return None

    # -------------------------
    # Model type: TPS
    # -------------------------
    if spatial_model == "tps":
        if params_json is None:
            return None

        params = json.loads(params_json)
        x = np.array(params["x"], float)
        y = np.array(params["y"], float)
        r = np.array(params["residuals"], float)
        smooth = params.get("smooth", 0.1)

        return Rbf(x, y, r, function='thin_plate', smooth=smooth)

    # -------------------------
    # Model type: BINNED
    # -------------------------
    if spatial_model == "binned":
        if grid_mag_json is None or xmid_json is None or ymid_json is None:
            return None

        grid_mag = np.array(json.loads(grid_mag_json))
        xmid     = np.array(json.loads(xmid_json))
        ymid     = np.array(json.loads(ymid_json))

        return RegularGridInterpolator(
            (xmid, ymid),
            grid_mag,
            bounds_error=False,
            fill_value=0.0
        )

    # -------------------------
    # Model type: GAUSSIAN
    # -------------------------
    if spatial_model == "gaussian":
        if params_json is None:
            return None

        params = json.loads(params_json)
        return GaussianSpatialModel.from_params(params)

    # -------------------------
    # Future model types
    # -------------------------
    raise ValueError(f"Unknown spatial model type: {spatial_model}")


def saveSpatialModel(
    conn,
    frame_name,
    spatial_model=None,
    version=1,
    grid_mag=None,
    xmid=None,
    ymid=None,
    params=None,
    n_points=None,
    rms_mag=None,
    median_resid=None):
    """
    Persist a spatial correction model into the spatial_model table.

    All model payload fields are optional. Only the fields relevant to the
    model_type need to be provided.

    Parameters
    ----------
    frame_name : str
    model_type : str      # 'binned', 'gaussian', 'none', etc.
    version    : int
    grid_mag   : np.ndarray or None
    xmid       : np.ndarray or None
    ymid       : np.ndarray or None
    params     : dict or None
    n_points   : int or None
    rms_mag    : float or None
    median_resid : float or None
    """

    # Convert numpy arrays to JSON-serializable lists
    grid_mag_json = json.dumps(grid_mag.tolist()) if grid_mag is not None else None
    xmid_json     = json.dumps(xmid.tolist())     if xmid is not None else None
    ymid_json     = json.dumps(ymid.tolist())     if ymid is not None else None
    params_json   = json.dumps(params)            if params is not None else None

    sql = """
        INSERT INTO spatial_model
            (frame_name, model_type, version,
             grid_mag, xmid, ymid, params,
             n_points, rms_mag, median_resid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (frame_name, model_type, version) DO UPDATE
        SET grid_mag     = EXCLUDED.grid_mag,
            xmid         = EXCLUDED.xmid,
            ymid         = EXCLUDED.ymid,
            params       = EXCLUDED.params,
            n_points     = EXCLUDED.n_points,
            rms_mag      = EXCLUDED.rms_mag,
            median_resid = EXCLUDED.median_resid,
            created_at   = NOW();
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                frame_name,
                spatial_model,
                version,
                grid_mag_json,
                xmid_json,
                ymid_json,
                params_json,
                n_points,
                rms_mag,
                median_resid
            )
        )

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
            obs.cat_mag  AS cat_mag,
            obs.x,
            obs.y
        FROM observation AS obs
        WHERE obs.frame_name = %s
          AND obs.mag IS NOT NULL
          AND obs.cat_mag IS NOT NULL
          AND obs.mag_err < 0.5e6
          and obs.mad > 0.1e6
          AND flags = 0;

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

def buildTPSCorrectionMap(x, y, residuals_mag, smooth=0.1):
    """
    Build a thin-plate spline correction model using SciPy Rbf.
    smooth: smoothing parameter (lambda)
    Returns an Rbf instance.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = np.asarray(residuals_mag, float)

    rbf = Rbf(x, y, r, function='thin_plate', smooth=smooth)
    return rbf


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

def tryBuildTPS(x, y, residuals, smooth=0.1):

    # this is very inefficient, but the best I can do for now.

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(residuals)
    x = x[mask]
    y = y[mask]
    residuals = residuals[mask]

    if len(x) < 40:
        return None

    coords = np.column_stack((x, y))
    if len(np.unique(coords, axis=0)) < len(coords):
        return None

    # 2. Collinearity check
    if len(x) >= 3:
        hull = ConvexHull(coords)
        if hull.area < 1e-3:
            return None

    # 3. Residual structure check
    if np.nanstd(residuals) < 1e-4:
        return None

    # 4. Try TPS fit
    try:
        return Rbf(x, y, residuals, function='thin_plate', smooth=smooth)
    except np.linalg.LinAlgError:
        return None



def applyDetectionCorrections(conn, det, spatial_method):
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

    start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    total = len(unique_frames)
    cache_hit = 0

    for i, fname in enumerate(unique_frames, start=1):

        if i % 500 == 0 or i == 10:
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            elapsed_s = (now - start_time).total_seconds()

            # Forecast total duration
            forecast_total_s = elapsed_s * total / i
            forecast_completion_time = start_time + datetime.timedelta(seconds=forecast_total_s)
            cache_hit_pc = 100 * cache_hit / i
            cache_hit_txt = f"Cache hit {cache_hit_pc:.1f}%" if spatial_method != 'none' else ""
            print(f"[{i}/{total}] Forecast completion: {forecast_completion_time.isoformat().split('.')[0]} {cache_hit_txt}")

        version = 1
        cached_map = loadCachedSpatialMap(conn, fname, spatial_method, version)

        # 1. Try cache first
        if cached_map is not None:
            #print(f"[cache hit] spatial map type {spatial_method} for frame {fname}")
            cache_hit += 1
            frame_data = loadFramePhotometry(conn, fname)
            frame_offset = computeFrameOffset(frame_data) if frame_data else 0.0
            frame_cache[fname] = (frame_offset, cached_map)
            continue

        if spatial_method != "none":
            pass
            #print(f"[cache miss] building spatial map type {spatial_method} for frame {fname}")

        # 2. Otherwise compute
        frame_data = loadFramePhotometry(conn, fname)
        if frame_data is None:
            frame_cache[fname] = (0.0, None)
            continue

        frame_offset = computeFrameOffset(frame_data)
        residuals = frame_data['obs_mag'] - frame_data['cat_mag'] - frame_offset

        if spatial_method == 'binned':

            spatial_map, grid_mag, xmid, ymid = buildSpatialCorrectionMap(
                frame_data['x'], frame_data['y'], residuals)

            # Save binned map
            saveSpatialModel(conn, frame_name=fname, spatial_model=spatial_method, version=1, grid_mag=grid_mag, xmid=xmid, ymid=ymid)
            frame_cache[fname] = (frame_offset, spatial_map)

        elif spatial_method == 'gaussian':

            gauss_model = fitGaussianSpatialModel(
                frame_data['x'], frame_data['y'], residuals)

            saveSpatialModel(conn, spatial_model="gaussian", frame_name=fname, version=1, params=gauss_model.to_params())
            frame_cache[fname] = (frame_offset, gauss_model)


        elif spatial_method == 'tps':

            # Build TPS model
            smooth = 0.1  # or configurable

            tps_model = tryBuildTPS(frame_data['x'], frame_data['y'], residuals)

            if tps_model is None:
                frame_cache[fname] = (frame_offset, None, 0, 0)
                continue


            try:
                resid_after = residuals - tps_model(frame_data['x'], frame_data['y'])
                rms_frame = float(np.sqrt(np.mean(resid_after ** 2)))
            except Exception:
                rms_frame = 1.0  # fallback

            n_points = len(residuals)

            # Weight: many stars + low RMS = high weight

            if rms_frame > 0:
                w_frame = n_points / (rms_frame * rms_frame)
            else:
                w_frame = 0.0

                # Clip extreme weights
                w_frame = float(min(w_frame, 1e6))

            # Save TPS model
            params = {
                "x": frame_data['x'].tolist(),
                "y": frame_data['y'].tolist(),
                "residuals": residuals.tolist(),
                "smooth": smooth
            }

            saveSpatialModel(conn, spatial_model="tps", frame_name=fname, version=1, params=params, n_points=n_points,
                             rms_mag=rms_frame, median_resid=float(np.median(residuals)))
            frame_cache[fname] = (frame_offset, tps_model, w_frame)


        else:

            # This is the "do not apply a compensation branch - do nothing for now"
            pass

    mag_corr = np.empty_like(mag_det)

    for i in range(len(mag_det)):
        fname = frame_names[i]
        frame_offset, spatial_map, w_frame = frame_cache.get(fname, (0.0, None, 0.0))


        m = mag_det[i] - frame_offset

        if spatial_map is not None:
            if np.isnan(x_det[i]) or np.isnan(y_det[i]):
                spatial_offset = 0.0
            else:
                if spatial_method == 'tps':
                    # SciPy Rbf: call as rbf(x, y)
                    spatial_offset = spatial_map(x_det[i], y_det[i])
                else:
                    # Binned or Gaussian: call as interpolator([[x, y]])[0]
                    spatial_offset = spatial_map([[x_det[i], y_det[i]]])[0]

            m -= spatial_offset

        mag_corr[i] = m

    det_corr = dict(det)
    det_corr['mag'] = mag_corr
    det_corr['frame_weight'] = np.array([frame_cache[f][2] for f in frame_names])

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

def buildLightCurvePoints(det, cat_mag=None, ang_tol_deg=0.05, min_cameras=2, star_name=None, t_window_min=1.0, base_name=None, cadence_sec=10.24):

    ra_bin, dec_bin, jd_bin = det['ra_deg'], det['dec_deg'], det['jd']
    mag_bin, mag_err_bin, snr, camera_id = det['mag'], det['mag_err'], det['snr'], det['camera_id']

    labels = clusterDetectionsStar5D(
        ra_bin, dec_bin, jd_bin, mag_bin,
        t_window_min=10.0,
        ang_tol_deg=0.05,
        mag_tol_mag=0.1)

    n_cam_bin = np.ones_like(mag_bin, dtype=int)

    points = []
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        print(
            "Cluster", lab,
            "JD range:", float(np.min(jd_bin[idx])), float(np.max(jd_bin[idx])),
            "Cameras:", "N/A")

        point = buildPhotometricPoint(
            idx, ra_bin, dec_bin, jd_bin, mag_bin, mag_err_bin, n_cam_bin
        )

        if point['n_cam'] >= min_cameras:
            points.append(point)

    return points


def getJSONFilepath(base_name):

    return os.path.join(CHARTS_DIR, f"{base_name}.json")


def getTXTFilepath(base_name):

    return os.path.join(CHARTS_DIR, f"{base_name}.txt")

def getPNGFilepath(base_name):

    return os.path.join(CHARTS_DIR, f"{base_name}.png")

# =========================
# DB access
# =========================

def loadDetections(conn, jd_start=None, jd_end=None,
                   prove=False, prove_n=100, star_name=None):
    # Star-name lookup (fast path)
    where = []
    params = []

    if star_name is not None:
        where.append("obs.star_name = %s")
        params.append(star_name)

    # Quality filters
    where.append("abs(obs.mag_err) < 1e6")
    where.append("obs.flags = 0")

    # Time filters
    where.append("jd_mid > %s")
    params.append(1e6 * jd_start)
    where.append("jd_mid < %s")
    params.append(1e6 * jd_end)
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
        print("Starting bulk SQL query")
        cur.execute(sql, params)
        print("SQL query completed, fetching rows")
        rows = cur.fetchall()
        print("Row fetch complete")

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

    # Compute JD mid from frame name
    jd = np.array([
        datetime2JD(getMiddleTimeFF(f"FF_{fn}_000000",fps=25, dt_obj=True))  # returns JD in natural units
        for fn in frame_name
    ], dtype=float)

    # Optional time filtering this now gets done in SQL - but leaving here for safety
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

            print(f"[{i}] RA={ra:.6f}, DEC={dec:.6f}")
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



def binNetworkDetections(det, bin_seconds=10.24, period_jd=None, period_repeats=1):
    """
    Bin network detections either in time (default) or in phase if period_jd is given.
    period_jd: period in Julian days (float)
    """

    jd = det['jd']
    ra_deg = det['ra_deg']
    dec_deg = det['dec_deg']
    mag = det['mag']
    snr = det['snr']
    camera_id = det['camera_id']

    if jd.size == 0:
        return

    # Convert to arrays
    jd = np.asarray(jd)
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)
    mag = np.asarray(mag)
    snr = np.asarray(snr)
    camera_id = np.asarray(camera_id)

    # Reference epoch
    jd0 = jd.min()

    # --- TIME OR PHASE COORDINATE ---
    if period_jd is not None:
        # Phase in [0,1)
        phase = ((jd - jd0) / period_jd) % 1.0

        phase_extended = phase + np.floor((jd - jd0) / period_jd)

        phase_extended = phase_extended % period_repeats

        time_coord = phase_extended

        # Bin width in phase units
        bin_width_phase = bin_seconds / (period_jd * 86400.0)
        bin_index = np.floor(phase_extended / bin_width_phase).astype(int)

    else:
        # Normal time binning
        dt_days = bin_seconds / 86400.0
        time_coord = jd
        bin_index = np.floor((jd - jd0) / dt_days).astype(int)

    # --- OUTPUT ARRAYS ---
    jd_bins = []
    ra_bins = []
    dec_bins = []
    mag_bins = []
    mag_err_bins = []
    n_det_bins = []
    n_cam_bins = []
    cam_sets = []

    # --- BINNING LOOP ---
    for b in np.unique(bin_index):
        idx = np.where(bin_index == b)[0]

        jd_bins.append(np.mean(time_coord[idx]))
        ra_bins.append(np.mean(ra_deg[idx]))
        dec_bins.append(np.mean(dec_deg[idx]))

        # Convert magnitudes to flux
        flux = 10**(-0.4 * mag[idx])

        # Photometric weight (inverse variance)
        w_phot = snr[idx] ** 2

        # Frame-level TPS weight
        w_frame = det['frame_weight'][idx]

        # Combined weight
        w = w_phot * w_frame

        w_sum = np.sum(w)

        if np.any(snr[idx] == -1):
            # Old-style detections without SNR
            flux_mean = np.mean(flux)
            sigma_flux = np.std(flux, ddof=1)
        else:
            # Weighted mean flux
            flux_mean = np.sum(w * flux) / w_sum
            sigma_flux = np.sqrt(1.0 / w_sum)

        # Convert back to magnitude
        mag_mean = -2.5 * np.log10(flux_mean)
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
        cam_sets,
        jd0
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
                           cadence_sec=10.24,
                           spatial_method=None, period_jd=None, period_repeats=1):

    star_name_from_db, mag_from_db = lookupBrightestStar(conn, ra_star_deg, dec_star_deg, radius_deg=0.05)

    print(f"Working on star {star_name_from_db}, RA: {ra_star_deg:.2f}, DEC: {dec_star_deg:.2f} MAG:{mag_from_db/1e6:.2f}")
    base_name = makeLightcurveFilename(ra_star_deg, dec_star_deg, jd_start, jd_end, star_name_from_db, spatial_method=spatial_method)


    det = loadDetections(conn, jd_start=jd_start, jd_end=jd_end, star_name=star_name_from_db)

    if det is None:
        return None

    if len(det['camera_id']) == 0:
        print("No station made an observation")
        return None

    contributing_stations = sorted(set(det['camera_id']))
    print(f"Unique stations: {contributing_stations}")

    det = applyDetectionCorrections(conn, det, spatial_method=spatial_method)

    if det is None:
        return []

    cat_mag = float(np.nanmedian(det['cat_mag']))
    print(f"Detections {len(det['jd'])}")
    binned_detections = binNetworkDetections(det, bin_seconds=cadence_sec, period_jd=period_jd, period_repeats=period_repeats)


    arr_jd_bins, arr_ra_bins, arr_dec_bins, arr_mag_bins, arr_mag_err_bins, arr_n_det_bins, arr_n_cam_bins, cam_sets, jd0 = binned_detections
    print(f"Bins {len(arr_jd_bins)}")

    plotTimeBinnedLightCurve(binned_detections, n_stations=len(set(det['camera_id'])), n_observations=len(det['jd']), cat_mag=mag_from_db/1e6, bin_length=cadence_sec, star_name=star_name_from_db, base_name=base_name, sub_title=contributing_stations, period_jd=period_jd)

    points = buildLightCurvePoints(det, cat_mag=cat_mag, ang_tol_deg=ang_tol_deg, min_cameras=min_cameras,
                                   star_name=star_name_from_db, t_window_min=cadence_sec / 60.0, base_name=base_name, cadence_sec=cadence_sec)
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



    lc = binByCadence(lc, cadence_sec=cadence_sec)

    order = np.argsort(lc['jd'])
    for k in lc:
        lc[k] = lc[k][order]

    # Metadata
    stations = set(det['camera_id'])
    n_observations = len(det['jd'])

    # Save JSON with metadata
    saveLightCurveAsJson(
        lc,
        base_name,
        stations=stations,
        n_observations=n_observations,
        ra_star_deg=ra_star_deg,
        dec_star_deg=dec_star_deg,
        jd_start=jd_start,
        jd_end=jd_end,
        cat_mag=cat_mag
    )

    saveLightCurveSidecarTxt(
        star_name_from_db,
        ra_star_deg,
        dec_star_deg,
        jd_start,
        jd_end,
        cat_mag,
        n_observations,
        stations,
        lc,
        base_name,
        spatial_method=spatial_method)

    return lc


def saveLightCurveSidecarTxt(
    star_name,
    ra_star_deg,
    dec_star_deg,
    jd_start,
    jd_end,
    cat_mag,
    n_observations,
    stations,
    lc,
    base_name=None,
    spatial_method=None):


    n_stations = len(stations)
    ra_mean = float(np.mean(lc['ra_deg']))
    dec_mean = float(np.mean(lc['dec_deg']))

    mag_med = float(np.nanmedian(lc['mag']))
    mag_min = float(np.nanmin(lc['mag']))
    mag_max = float(np.nanmax(lc['mag']))
    mag_err_med = float(np.nanmedian(lc['mag_err']))

    with open(getTXTFilepath(base_name), "w") as f:
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
        f.write(f"Spatial method: {spatial_method}\n\n")

        f.write(f"Mag median: {mag_med:.3f}\n")
        f.write(f"Mag min:    {mag_min:.3f}\n")
        f.write(f"Mag max:    {mag_max:.3f}\n")
        f.write(f"Mag_err median: {mag_err_med:.3f}\n\n")

        f.write(f"JSON: {os.path.basename(getJSONFilepath(base_name))}\n")
        f.write(f"Plot: {os.path.basename(getPNGFilepath(base_name))}\n")

        f.write("\n\nContributing stations:\n")
        for s in sorted(stations):
            f.write(f"  - {s}\n")
        f.write("\n")

    return getTXTFilepath(base_name)

# =========================
# JSON export
# =========================

def saveLightCurveAsJson(lc, base_name,
                         stations=None,
                         n_observations=None,
                         ra_star_deg=None,
                         dec_star_deg=None,
                         jd_start=None,
                         jd_end=None,
                         cat_mag=None):


    serializable = {k: v.tolist() for k, v in lc.items()}

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


    with open(getJSONFilepath(base_name), 'w') as f:
        json.dump(serializable, f, indent=2)


def plotTimeBinnedLightCurve(binned_detections, n_stations, n_observations, cat_mag, bin_length=None, star_name=None, base_name=None, sub_title=None, period_jd=None):

    arr_jd_bins, arr_ra_bins, arr_dec_bins, arr_mag_bins, arr_mag_err_bins, arr_n_det_bins, arr_n_cam_bins, cam_sets, jd_period_start = binned_detections


    jd0 = arr_jd_bins[0]
    jd_rel = arr_jd_bins - jd0

    ra_mean  = np.mean(arr_ra_bins)
    dec_mean = np.mean(arr_dec_bins)

    fig, (ax_mag, ax_cam) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    n = 30
    if sub_title is not None:
        sub_title_as_list = [",".join(sub_title[i:i + n]) for i in range(0, len(sub_title), n)]
        ax_cam.set_title("\n".join(sub_title_as_list), fontsize=6, alpha=0.7)



    if period_jd is not None:
        ax_mag.set_title(
            f"{star_name} — Time-Binned Light Curve\n"
            f"RA={ra_mean:.2f} deg, DEC={dec_mean:.2f} deg MAG={cat_mag:.2f} PERIOD={period_jd:.3f} days\n"
            f"{n_stations} stations, {n_observations} detections, {len(binned_detections[0])} bins of length {bin_length:.1f} seconds")
    else:
        ax_mag.set_title(
            f"{star_name} — Time-Binned Light Curve\n"
            f"RA={ra_mean:.2f} deg, DEC={dec_mean:.2f} deg MAG={cat_mag:.2f}\n"
            f"{n_stations} stations, {n_observations} detections, {len(binned_detections[0])} bins of length {bin_length:.1f} seconds")







    for x, y, dy in zip(jd_rel, arr_mag_bins, arr_mag_err_bins):
        ax_mag.fill_between(
            [x - 0.0001, x + 0.0001],
            y - dy,
            y + dy,
            color='lightblue',
            alpha=0.35,
            linewidth=0
        )

    ax_mag.scatter(jd_rel, arr_mag_bins, s=14, color='tab:blue', alpha=0.9)
    ax_mag.set_ylabel("Magnitude")
    ax_mag.invert_yaxis()
    # --- Dynamic scaling: 75% of points in middle 3/5 of axis ---
    q25, q75 = np.percentile(arr_mag_bins, [25, 75])
    middle_span = q75 - q25

    # Full axis span so that middle 75% occupies middle 3/5
    full_span = (5 / 3) * middle_span


    # Axis limits (remember: magnitude axis is inverted)

    y_top, y_bottom = min(arr_mag_bins), max(arr_mag_bins)

    # Produce a positive margin
    margin = (y_bottom - y_top) * 0.01

    # subtract from top, add to bottom
    y_top -= margin
    y_bottom += margin

    ax_mag.set_ylim(y_bottom, y_top)

    if cat_mag is not None:
        ax_mag.axhline(
            y=cat_mag,
            color='grey',
            linestyle=':',
            linewidth=1.0,
            alpha=0.6,
            label=f"Catalogue mag {cat_mag:.3f}"
        )

    # --- Robust bar width calculation ---
    if len(jd_rel) > 1:
        # Compute all spacings
        spacings = np.diff(jd_rel)

        # Use median spacing for robustness
        median_spacing = np.median(spacings)

        # Scale to leave a small gap between bars
        bar_width = median_spacing * 0.8

        # Enforce reasonable bounds (in days)
        bar_width = np.clip(bar_width, 0.0001, 0.1)
    else:
        bar_width = 0.001  # fallback for single-bin case

    ax_cam.bar(jd_rel, arr_n_cam_bins, width=bar_width, color='tab:red', alpha=0.7)





    ax_cam.set_ylabel("Stations")
    if jd0 is None:
        ax_cam.set_xlabel(f"Time since JD {jd0:.5f} (days)")
    else:
        ax_cam.set_xlabel(f"Period starting from jd {jd_period_start}")

    fig.tight_layout()
    #plt.show()
    fig.savefig(getPNGFilepath(base_name), dpi=150)

    return

from urllib.parse import urlparse

def parseDBArg(db_arg):
    """
    Convert a PostgreSQL URI into a dict suitable for psycopg.connect().
    Password is intentionally omitted (use .pgpass).
    """
    parsed = urlparse(db_arg)

    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "user": parsed.username,
        "dbname": parsed.path.lstrip("/"),
        # password intentionally not included — .pgpass handles it
    }


def parseRaDec(value):
    """
    Parse RA/Dec in either:
      - decimal degrees: "123.45,-22.3"
      - sexagesimal:     "12:34:56 -22:33:44"
    Returns (raDeg, decDeg).
    """
    value = value.strip()

    # Decimal degrees
    if "," in value:
        raStr, decStr = value.split(",", 1)
        return float(raStr), float(decStr)

    # Sexagesimal "HH:MM:SS ±DD:MM:SS"
    if ":" in value:
        raStr, decStr = value.split()

        # RA: HH:MM:SS
        h, m, s = map(float, raStr.split(":"))
        raDeg = 15 * (h + m/60 + s/3600)

        # Dec: ±DD:MM:SS
        d, dm, ds = map(float, decStr.split(":"))
        sign = -1 if d < 0 else 1
        decDeg = sign * (abs(d) + dm/60 + ds/3600)

        return raDeg, decDeg

    raise ValueError(f"Unrecognized RA/Dec format: {value!r}")

def parseJdRange(s):
    if s is None:
        return None, None

    # Accept "a,b" or "a:b"
    if ',' in s:
        a, b = s.split(',', 1)
    elif ':' in s:
        a, b = s.split(':', 1)
    else:
        raise ValueError("jd_range must be in form start,end or start:end")

    return float(a), float(b)

def getDbJdRange(conn):
    sql = """
        SELECT MIN(jd_mid), MAX(jd_mid)
        FROM observation
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()

    if row is None or row[0] is None or row[1] is None:
        return None, None

    # Convert BIGINT micro-JD → float JD
    jd_min = row[0] / 1e6
    jd_max = row[1] / 1e6
    return jd_min, jd_max



if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepar and mask files.
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('db_connection_string', metavar='db_connection_string', type=str,
                    help="DB Connection string in the format postgresql://user@host:port/database ")

    arg_parser.add_argument('--radec', metavar='RADEC', type=str,
                    help="Star coordinates in decimal or sexagesimal degrees, if none are passed then defaults to Betelgeux (88.79 7.41)")

    arg_parser.add_argument('--jd_range', metavar='JD_RANGE', type=str,
                            help="Range of jd to use in curve plotting")

    arg_parser.add_argument('-b', '--bins_per_jd', metavar='BINS_PER_JD', type=int,
                            help="Number of bins of intensity in each julian day, default 100")

    arg_parser.add_argument('-p', '--period_jd', metavar='PERIOD_JD', type=float,
                            help="Period length wrapping")

    arg_parser.add_argument('-r', '--period_repeats', metavar='PERIOD_REPEATS', type=int,
                            help="Period repeats to show")

    group = arg_parser.add_mutually_exclusive_group()

    group.add_argument("--tps", action="store_true", help="Use thin-plate spline spatial model")

    group.add_argument("--binned", action="store_true", help="Use binned spatial model")

    group.add_argument("--gaussian", action="store_true", help="Use Gaussian spatial model")



    ###

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    period_jd = cml_args.period_jd

    if cml_args.tps:
        spatial_method = "tps"
    elif cml_args.binned:
        spatial_method = "binned"
    elif cml_args.gaussian:
        spatial_method = "gaussian"
    else:
        spatial_method = "none"

    if cml_args.period_repeats is None:
        period_repeats = 1
    else:
        period_repeats = int(cml_args.period_repeats)

    db_dict = parseDBArg(cml_args.db_connection_string)
    radec = cml_args.radec

    if radec is None:
        target_ra = 88.79187
        target_dec = 7.4056
    else:
        target_ra, target_dec = parseRaDec(radec)

    bins_per_day = 100

    if cml_args.bins_per_jd is None and cml_args.period_jd is None:
        bins_per_day = 100
    elif cml_args.bins_per_jd is not None and cml_args.period_jd is None:
        bins_per_day = cml_args.bins_per_jd
    elif cml_args.bins_per_jd is None and cml_args.period_jd is not None:
        bins_per_day = 100  / cml_args.period_jd
    elif cml_args.bins_per_jd is not None and cml_args.period_jd is not None:
        bins_per_day = cml_args.bins_per_jd  / cml_args.period_jd

    cadence_sec = (24 * 60 * 60) / bins_per_day
    print(f"Running with a bin length of {cadence_sec:.1f} seconds, or {bins_per_day:.1f} bins/day")

    jd_start, jd_end = parseJdRange(cml_args.jd_range)

    # Current time in JD
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    jd_now = datetime2JD(now_dt)
    print(f"JD now is {jd_now}")

    # Clamp jd_end to now
    if jd_end is None:
        jd_end = jd_now
    else:
        jd_end = min(jd_end, jd_now)


    with psycopg.connect(
            host=db_dict['host'],
            dbname=db_dict['dbname'],
            user=db_dict['user'],
            port=db_dict['port']
    ) as conn:

        # If jd_start missing, fill from DB
        if jd_start is None:
            db_jd_min, db_jd_max = getDbJdRange(conn)
            db_jd_max = min(db_jd_max, jd_now)
            jd_start = db_jd_min

        # Safety: ensure ordering
        if jd_start > jd_end:
            jd_start, jd_end = jd_end, jd_start

        print(f"Using JD range: {jd_start:.5f} to {jd_end:.5f}")

        lc = generateStarLightCurve(
            conn,
            ra_star_deg=target_ra,
            dec_star_deg=target_dec,
            search_radius_deg=0.1,
            jd_start=jd_start,
            jd_end=jd_end,
            ang_tol_deg=0.2,
            min_cameras=3,
            cadence_sec=cadence_sec,
            spatial_method = spatial_method, period_jd=period_jd, period_repeats = period_repeats)

        if lc is None:
            print("No light curve generated.")
        else:

            print("Saved debug_lightcurve.json")
