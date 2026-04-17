import datetime
import argparse
import psycopg
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import contextlib

from scipy.interpolate import SmoothBivariateSpline
from pathlib import Path
from scipy.spatial import ConvexHull
from collections import Counter
from urllib.parse import urlparse
from matplotlib.ticker import FormatStrFormatter


def storeCompensatedFrame(conn, frame_name, frame_offset, tps_spline, frame_data):

    if tps_spline is None:
        tps_x = tps_y = tps_coeff = None
        smoothing = None
        quality = "BAD"
    else:
        knots = tps_spline.get_knots()
        tps_x = knots[0].tolist()
        tps_y = knots[1].tolist()
        tps_coeff = tps_spline.get_coeffs().tolist()
        smoothing = None      # SciPy does NOT expose smoothing
        quality = "GOOD"

    residuals = frame_data["obs_mag"] - frame_data["cat_mag"] - frame_offset
    rms = float(np.sqrt(np.mean(residuals**2)))

    sql = """
        INSERT INTO compensated_frame
        (frame_name, frame_offset, tps_x, tps_y, tps_coeff, tps_smoothing,
         num_cal_stars, rms_residual, quality)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (frame_name) DO UPDATE SET
            frame_offset = EXCLUDED.frame_offset,
            tps_x = EXCLUDED.tps_x,
            tps_y = EXCLUDED.tps_y,
            tps_coeff = EXCLUDED.tps_coeff,
            tps_smoothing = EXCLUDED.tps_smoothing,
            num_cal_stars = EXCLUDED.num_cal_stars,
            rms_residual = EXCLUDED.rms_residual,
            quality = EXCLUDED.quality;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (
            frame_name,
            frame_offset,
            tps_x, tps_y, tps_coeff,
            smoothing,
            len(frame_data["obs_mag"]),
            rms,
            quality
        ))


def lookupBrightestStar(conn, ra_deg, dec_deg, radius_deg=0.02):
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

def lookupCatalogueStar(conn, star_name):
    """
    Given a star name, return its catalogue RA, Dec (in degrees) and magnitude.
    Returns None if the star is not found.
    """

    sql = """
        SELECT ra, dec, mag
        FROM star
        WHERE star_name = %s
        LIMIT 1;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (star_name,))
        row = cur.fetchone()

    if row is None:
        return None, None, None

    r = row[0] / 1e6
    d = row[1] / 1e6
    m = row[2] / 1e6 if row[2] is not None else None

    return r, d, m


# ============================================================
# Filename helpers
# ============================================================



def binDetections(det, bin_seconds=10.24*10):
    jd = det["jd"]
    mag = det["mag"]
    snr = det["snr"]
    station = det["station"]

    dt_days = bin_seconds / 86400.0
    jd0 = jd.min()
    bin_index = np.floor((jd - jd0) / dt_days).astype(int)

    jd_bins = []
    mag_bins = []
    mag_err_bins = []
    station_bins = []
    n_det_bins = []          # <-- NEW

    for b in np.unique(bin_index):
        idx = np.where(bin_index == b)[0]
        if len(idx) == 0:
            continue

        # Record number of detections in this time bin
        n_det_bins.append(len(idx))     # <-- NEW

        unique_stations = np.unique(station[idx])
        station_bins.append(np.asarray(unique_stations))

        flux = 10 ** (-0.4 * mag[idx])
        sigma_flux_i = flux / snr[idx]
        w = 1.0 / (sigma_flux_i ** 2)
        w_sum = np.sum(w)

        flux_mean = np.sum(w * flux) / w_sum
        sigma_flux = np.sqrt(1.0 / w_sum)

        mag_mean = -2.5 * np.log10(flux_mean)
        mag_err_mean = (2.5 / np.log(10)) * (sigma_flux / flux_mean)

        jd_bins.append(np.mean(jd[idx]))
        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err_mean)

    return {
        "jd": np.array(jd_bins),
        "mag": np.array(mag_bins),
        "mag_err": np.array(mag_err_bins),
        "station": station_bins,
        "n_det": np.array(n_det_bins)   # <-- NEW
    }



# ============================================================
# DB loading
# ============================================================

def loadFramePhotometry(conn, frame_name):
    """
    Load all observations for a given frame:
      - obs_mag
      - cat_mag
      - x, y
      - star_name
    """
    sql = """
        SELECT
            obs.mag,
            obs.cat_mag,
            obs.x,
            obs.y,
            obs.star_name
        FROM observation AS obs
        WHERE obs.frame_name = %s
          AND obs.mag IS NOT NULL
          AND obs.cat_mag IS NOT NULL
          AND obs.star_name <> ''
          AND obs.mag_err < 3e6
          AND obs.mad > 0.1e6
          AND obs.flags = 0
          AND obs.intens_sum != 0;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (frame_name,))
        rows = cur.fetchall()

    if not rows:
        return None

    star_name = np.array([r[4] for r in rows], dtype=str)
    counts = Counter(star_name)
    dupes = {name for name, c in counts.items() if c > 1}

    if dupes:
        rows = [r for r in rows if r[4] not in dupes]

    obs_mag = np.array([r[0] for r in rows], float) / 1e6
    cat_mag = np.array([r[1] for r in rows], float) / 1e6
    x       = np.array([r[2] for r in rows], float)
    y       = np.array([r[3] for r in rows], float)

    return {
        "obs_mag": obs_mag,
        "cat_mag": cat_mag,
        "x": x,
        "y": y,
    }


def loadDetections(conn, jd_start, jd_end, star_name=None):
    """
    Load raw detections for a star or region.
    """

    where = []
    params = []

    if star_name is not None:
        where.append("obs.star_name = %s")
        params.append(star_name)

    #where.append("abs(obs.mag_err) < 3e6")
    #where.append("obs.flags = 0")

    where.append("obs.jd_mid > %s")
    params.append(1e6 * jd_start)
    where.append("obs.jd_mid < %s")
    params.append(1e6 * jd_end)
    where.append("obs.flags = 0")
    where.append("obs.intens_sum != 0")
    where.append("abs(obs.mag_err) < 3e6")

    where.append("""
        NOT EXISTS (
            SELECT 1 FROM rejected_frame rf
            WHERE rf.frame_name = obs.frame_name
        )
    """)

    where_clause = "WHERE " + " AND ".join(where)

    sql = f"""
        SELECT
            obs.ra,
            obs.dec,
            obs.mag,
            obs.snr,
            obs.frame_name,
            obs.x,
            obs.y,
            obs.mag_err,
            obs.jd_mid,
            obs.cat_mag,
            obs.intens_sum
        FROM observation AS obs
        {where_clause}
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return None

    ra_deg      = np.array([r[0] for r in rows], float) / 1e6
    dec_deg     = np.array([r[1] for r in rows], float) / 1e6
    mag         = np.array([r[2] for r in rows], float) / 1e6
    snr         = np.array([r[3] for r in rows], float) / 1e6
    frame_name  = np.array([r[4] for r in rows], str)
    x           = np.array([r[5] for r in rows], float)
    y           = np.array([r[6] for r in rows], float)
    mag_err     = np.array([(r[7] / 1e6) if r[7] is not None else np.nan for r in rows])
    jd          = np.array([r[8] for r in rows], float) / 1e6
    cat_mag     = np.array([(r[9] / 1e6) if r[9] is not None else np.nan for r in rows])
    station     = np.array([fn.split("_")[0] for fn in frame_name], dtype=str)

    return {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "jd": jd,
        "mag": mag,
        "snr": snr,
        "frame_name": frame_name,
        "x": x,
        "y": y,
        "cat_mag": cat_mag,
        "mag_err": mag_err,
        "station": station
    }


# ============================================================
# Frame offset + TPS smoothing
# ============================================================

def computeFrameOffset(frame_data):
    residuals = frame_data["obs_mag"] - frame_data["cat_mag"]
    if len(residuals) < 3:
        return np.mean(residuals)
    return np.median(residuals)


def buildSplineWithDiagnostics(x, y, r, smooth):
    stderr_buffer = io.StringIO()

    with warnings.catch_warnings(record=True) as wlist, \
         contextlib.redirect_stderr(stderr_buffer):

        warnings.simplefilter("error", UserWarning)

        try:
            spline = SmoothBivariateSpline(x, y, r, s=smooth)
        except UserWarning as uw:
            # FITPACK message is in stderr
            fitpack_text = stderr_buffer.getvalue().strip()
            python_warning = str(uw)

            full_msg = f"{fitpack_text}\n{python_warning}".strip()
            return None, full_msg

        except Exception as e:
            return None, f"Exception: {e}"

    # No warnings → good fit
    return spline, None

def tryBuildTPS(x, y, residuals, smooth=5.0):
    """
    Build a TPS spline with strict quality gating.
    Returns (model_function, spline_object) or None.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(residuals)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x, y, r = x[mask], y[mask], r[mask]

    if len(x) < 40:
        return None

    coords = np.column_stack((x, y))
    if len(np.unique(coords, axis=0)) < len(coords):
        return None

    if len(x) >= 3:
        hull = ConvexHull(coords)
        if hull.area < 1e-3:
            return None

    if np.nanstd(r) < 1e-4:
        return None

    try:
        spline = SmoothBivariateSpline(x, y, r, s=smooth)
    except Exception:
        return None

    def model(xq, yq, s=spline):
        return s.ev(xq, yq)

    return model, spline





def tryBuildTPSDummy():

    def model(xq, yq):
        return 0.0

    return model

def loadCompensatedFrame(conn, frame_name):
    """
    Load a previously stored calibrated frame.
    Reconstructs the SmoothBivariateSpline if knots/coeffs exist.
    """

    sql = """
        SELECT frame_offset, tps_x, tps_y, tps_coeff, tps_smoothing
        FROM compensated_frame
        WHERE frame_name = %s
        LIMIT 1;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (frame_name,))
        row = cur.fetchone()

    if row is None:
        return None

    frame_offset, tps_x, tps_y, tps_coeff, smoothing = row

    if tps_x is None or tps_y is None or tps_coeff is None:
        return {
            "frame_offset": frame_offset,
            "tps_model": None,
        }

    knots_x = np.array(tps_x)
    knots_y = np.array(tps_y)
    coeffs  = np.array(tps_coeff)

    # Create dummy spline
    dummy_x = np.array([0, 1, 2, 3])
    dummy_y = np.array([0, 1, 2, 3])
    dummy_z = np.zeros_like(dummy_x)

    spline = SmoothBivariateSpline(dummy_x, dummy_y, dummy_z)

    # Overwrite internal representation
    spline.tck = (knots_x, knots_y, coeffs, 3, 3)

    # Recreate wrapper
    def model(xq, yq, s=spline):
        return s.ev(xq, yq)

    return {
        "frame_offset": frame_offset,
        "tps_model": model,
    }

def applyDetectionCorrections(conn, det, dummy_run=False):
    """
    Apply frame offset + TPS spatial correction.
    Uses cached calibration when available.
    """

    if dummy_run:
        det_corr = dict(det)
        det_corr["mag"] = det["mag"].copy()
        return det_corr

    frame_names = det["frame_name"]
    x_det = det["x"]
    y_det = det["y"]
    mag_det = det["mag"]

    unique_frames = np.unique(frame_names)

    frame_offset_map = {}
    spatial_model_map = {}

    build_tps_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    reject_tps, attempted_tps, splines_created, splines_loaded, rejected_frames = 0, 0, 0, 0, []
    frames_to_process = len(unique_frames)

    for i, fname in enumerate(unique_frames):

        if i % 200 == 0 and i > 0:
            if len(rejected_frames):
                insertRejectedFrames(conn, rejected_frames)
                rejected_frames.clear()
            time_elapsed = datetime.datetime.now(tz=datetime.timezone.utc) - build_tps_start_time
            time_per_iteration_seconds = time_elapsed.total_seconds() / i
            iterations_remaining = len(unique_frames) - i
            completion_time = datetime.datetime.now(
                tz=datetime.timezone.utc
            ) + datetime.timedelta(seconds=time_per_iteration_seconds * iterations_remaining)
            print(
                f"Forecast end time={completion_time.replace(microsecond=0).isoformat()} "
                f"{100*i/frames_to_process:.1f}% frame rejection="
                f"{100 * reject_tps / max(attempted_tps, 1):.1f}% "
                f"Splines created/loaded {splines_created}/{splines_loaded}"
            )


        # 1) Try cached calibration
        cached = loadCompensatedFrame(conn, fname)
        if cached is not None:
            frame_offset_map[fname] = cached["frame_offset"]
            spatial_model_map[fname] = cached["tps_model"]
            splines_loaded += 1
            continue

        # 2) Compute calibration fresh
        frame_data = loadFramePhotometry(conn, fname)

        if frame_data is None or len(frame_data["cat_mag"]) < 40:
            frame_offset_map[fname] = 0.0
            spatial_model_map[fname] = None
            rejected_frames.append(fname)
            continue

        frame_offset = computeFrameOffset(frame_data)
        frame_offset_map[fname] = frame_offset

        residuals = frame_data["obs_mag"] - frame_data["cat_mag"] - frame_offset
        attempted_tps += 1

        result = tryBuildTPS(
            frame_data["x"],
            frame_data["y"],
            residuals,
            smooth=0.1 * len(frame_data["obs_mag"])
        )

        if result is None:
            spatial_model_map[fname] = None
            rejected_frames.append(fname)
        else:
            model, spline = result
            spatial_model_map[fname] = model

        # Store calibration
        storeCompensatedFrame(conn, fname, frame_offset, spline if result is not None else None, frame_data)
        splines_created += 1

    insertRejectedFrames(conn, rejected_frames)
    conn.commit()

    # Apply corrections
    mag_corr = np.empty_like(mag_det)

    for i in range(len(mag_det)):
        fname = frame_names[i]
        m = mag_det[i] - frame_offset_map.get(fname, 0.0)

        model = spatial_model_map.get(fname)
        if model is not None and not (np.isnan(x_det[i]) or np.isnan(y_det[i])):
            m -= model(x_det[i], y_det[i])

        mag_corr[i] = m

    det_corr = dict(det)
    det_corr["mag"] = mag_corr

    return det_corr



def insertRejectedFrames(conn, rejected_frame_list):
    sql = """
        INSERT INTO rejected_frame (frame_name) VALUES (%s)
        ON CONFLICT (frame_name) DO NOTHING;
    """

    with conn.cursor() as cur:
        for rejected_frame in rejected_frame_list:
            cur.execute(sql, (rejected_frame, ))
    conn.commit()
# ============================================================
# Plotting
# ============================================================
def phaseBinFolded(folded, n_phase_bins=50):
    phase = folded["phase"]
    mag = folded["mag"]
    mag_err = folded["mag_err"]
    station = folded["station"]
    n_det = folded["n_det"]

    # Define bin edges
    bins = np.linspace(-1.0, 1.0, n_phase_bins + 1)
    bin_index = np.digitize(phase, bins) - 1

    phase_bins = []
    mag_bins = []
    mag_err_bins = []
    station_bins = []
    n_det_phase = []



    for b in range(n_phase_bins):
        idx = np.where(bin_index == b)[0]
        if len(idx) == 0:
            continue

        # Sum n_det from all time bins that fall into this phase bin
        n_det_phase.append(np.sum(n_det[idx]))

        # Stations contributing to this phase bin
        stations_here = np.unique(np.concatenate([station[i] for i in idx]))
        station_bins.append(stations_here)

        # Convert to flux, average, convert back
        flux = 10 ** (-0.4 * mag[idx])
        flux_mean = np.mean(flux)
        mag_mean = -2.5 * np.log10(flux_mean)

        # --- RMS ERROR PROPAGATION ---
        if len(idx) > 1:
            # RMS scatter of magnitudes in this phase bin
            rms = np.sqrt(np.mean((mag[idx] - mag_mean)**2))
            # Error on the mean
            mag_err_mean = rms / np.sqrt(len(idx))
        else:
            # Only one point → use its own error
            mag_err_mean = mag_err[idx][0]

        # Representative phase = bin centre
        phase_center = 0.5 * (bins[b] + bins[b+1])

        phase_bins.append(phase_center)
        mag_bins.append(mag_mean)
        mag_err_bins.append(mag_err_mean)

    return {
        "phase": np.array(phase_bins),
        "mag": np.array(mag_bins),
        "mag_err": np.array(mag_err_bins),
        "station": station_bins,
        "n_det": np.array(n_det_phase)
    }



def foldLightCurve(binned, raw_det, period_days):
    # -------------------------
    # Corrected (binned) data
    # -------------------------
    jd       = binned["jd"]
    mag      = binned["mag"]
    mag_err  = binned["mag_err"]
    station  = binned["station"]

    # -------------------------
    # Raw (uncompensated) data
    # -------------------------
    raw_jd   = raw_det["jd"]
    raw_mag  = raw_det["mag"]
    raw_station = raw_det.get("station", None)

    # -------------------------
    # Compute phase for binned data
    # -------------------------
    phase = (((jd - jd.min()) / (period_days * 2)) % 1.0)
    phase_sym = phase - 0.5
    phase_2p = 2.0 * phase_sym

    # -------------------------
    # Compute phase for raw data
    # -------------------------
    raw_phase = (((raw_jd - jd.min()) / (period_days * 2)) % 1.0)
    raw_phase_sym = raw_phase - 0.5
    raw_phase_2p = 2.0 * raw_phase_sym

    # -------------------------
    # Return both folded datasets
    # -------------------------
    return {
        # Corrected folded data
        "phase": phase_2p,
        "mag": mag,
        "mag_err": mag_err,
        "station": station,
        "n_det": binned['n_det'],

        # Raw folded data (for top panel)
        "raw_phase": raw_phase_2p,
        "raw_mag": raw_mag,
        "raw_station": raw_station,
    }



def plotFoldedWithStations(det_phase_folded_binned, folded, cat_mag=None, base_name=None, nbins=100, titles=None, output_dir=None):


    # Extract corrected folded data
    phase   = det_phase_folded_binned["phase"]
    mag     = det_phase_folded_binned["mag"]
    station = det_phase_folded_binned["station"]

    # Extract raw folded data (must exist in folded dict)
    raw_phase = folded["raw_phase"]
    raw_mag   = folded["raw_mag"]
    mag_err = folded["mag_err"]

    # Titles dict with safe defaults
    if titles is None:
        titles = {}

    raw_title    = titles.get("raw",    "Raw Folded Light Curve (Uncorrected)")
    top_title    = titles.get("top",    "")
    bottom_title = titles.get("bottom", "Station Participation")
    super_title  = titles.get("super",  None)
    footer_text  = titles.get("footer", None)

    # ---------------------------------------------------------
    # Create figure with THREE vertically stacked axes
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 3, 0.75, 0.75])
    ax0 = fig.add_subplot(gs[0])  # Raw folded
    ax1 = fig.add_subplot(gs[1], sharex=ax0)  # Corrected folded
    ax2 = fig.add_subplot(gs[2], sharex=ax0)  # Station participation
    ax3 = fig.add_subplot(gs[3], sharex=ax0)  # NEW: detections per folded bin

    # ---------------------------------------------------------
    # SUPERTITLE (optional)
    # ---------------------------------------------------------
    if super_title:
        fig.suptitle(
            super_title,
            fontsize=20,
            fontweight="normal",
            y=0.965,
            color="black"
        )

    plt.tight_layout(rect=[0.04, 0.1, 0.975, 0.95])

    # =========================================================
    #  RAW TOP PANEL (ax0)
    # =========================================================
    ax0.scatter(raw_phase, raw_mag, s=2, alpha=0.0025, color="blue")
    ax0.invert_yaxis()
    ax0.set_ylabel("Magnitude")

    ax0.set_title(
        raw_title,
        fontsize=10,
        color="#666666",
        fontweight="normal"
    )

    # Clamp y-axis
    ax0.set_ylim(np.mean(raw_mag) + 2, np.mean(raw_mag) - 2)

    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax0.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # =========================================================
    #  CORRECTED FOLDED PANEL (ax1)
    # =========================================================
    ax1.scatter(phase, mag, s=12, alpha=0.7, color="black")
    ax1.invert_yaxis()
    ax1.set_ylabel("Compensated magnitude")

    ax1.set_title(
        top_title,
        fontsize=10,
        color="#666666",
        fontweight="normal"
    )

    # Catalogue magnitude line
    if cat_mag is not None and np.isfinite(cat_mag):
        ax1.axhline(
            y=cat_mag,
            color="grey",
            linestyle=":",
            linewidth=1.0,
            alpha=0.5,
            label="Catalogue magnitude"
        )
        ax1.legend(fontsize=8)

    # Clamp y-axis
    ax1.set_ylim(max(max(mag), cat_mag) + 0.05, min(min(mag), cat_mag) - 0.05)

    # Faint red connecting line
    order = np.argsort(phase)
    ax1.plot(
        phase[order],
        mag[order],
        color="red",
        alpha=0.25,
        linewidth=0.8,
        zorder=0
    )

    ax1.errorbar(phase, mag, yerr=mag_err[order], fmt='none', ecolor='#88c0ff', alpha=0.1)

    # =========================================================
    #  STATION PARTICIPATION PANEL (ax2)
    # =========================================================
    bins = np.linspace(-1, 1, nbins + 1)
    station_counts = []

    for i in range(nbins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        idx = np.where(mask)[0]

        if len(idx) == 0:
            station_counts.append(0)
            continue

        stations_here = []
        for j in idx:
            stations_here.extend(station[j])

        unique_stations = np.unique(stations_here)
        station_counts.append(len(unique_stations))

    ax2.bar(
        (bins[:-1] + bins[1:]) / 2,
        station_counts,
        width=(2 / nbins),
        color="steelblue",
        alpha=0.7,
        edgecolor="none"
    )

    ax2.set_ylabel("Stations")
    ax1.set_xlabel("Phase")
    ax2.set_xlim(-1, 1)

    # =========================================================
    #  DETECTION COUNT PER FOLDED BIN (ax3)
    # =========================================================
    det_counts = []

    # =========================================================
    #  DETECTIONS PER TIME BIN (ax3)
    # =========================================================
    n_det = det_phase_folded_binned["n_det"]

    ax3.bar(
        phase,  # phase of each binned point
        n_det,  # number of detections in that bin
        width=0.02,  # small width so bars don't overlap
        color="lightgrey",
        alpha=0.8,
        edgecolor="none"
    )

    ax3.set_ylabel("Detections")
    ax3.set_xlim(-1, 1)
    ax3.set_xlabel("Phase")

    ax3.set_ylabel("Detections")
    ax3.set_xlabel("Phase")
    ax3.set_xlim(-1, 1)

    # ---------------------------------------------------------
    # FOOTER (optional)
    # ---------------------------------------------------------
    if footer_text:
        fig.text(
            0.95, 0.01,
            footer_text,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666"
        )

    fig.text(0.057, 0.033,f"Contributing stations\n\n{bottom_title}", ha="left", va="bottom", fontsize=6, color="#666666")

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    if base_name and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        full_path = os.path.expanduser(os.path.join(output_dir, base_name))
        plt.savefig(f"{full_path}.png", dpi=600)
        print(f"Saved as {full_path}.png")

    #plt.show()




# ============================================================
# Command-line interface
# ============================================================

def summariseStationCounts(det, max_chars=180):
    """
    Given the dict returned by loadDetections(), produce:
      - a comma-separated summary of station counts sorted by decreasing detections
      - with a newline inserted after max_chars characters
      - the number of distinct stations

    Returns:
        summary_str, n_stations
    """

    stations = det["station"]          # array of strings, one per detection
    unique, counts = np.unique(stations, return_counts=True)

    # Sort by count descending
    order = np.argsort(-counts)

    # Build "AU0004-500" style entries
    parts = [f"{unique[i]}-{counts[i]}" for i in order]

    # Character-based wrapping
    lines = []
    current = ""

    for part in parts:
        # +2 accounts for ", " that will be added
        addition = (", " if current else "") + part

        if len(current) + len(addition) > max_chars:
            # Start a new line
            lines.append(current)
            current = part
        else:
            current += addition

    # Append final line
    if current:
        lines.append(current)

    summary_str = "\n".join(lines)
    n_stations = len(unique)

    return summary_str, n_stations

def generateTitles(conn, star_name=None, jd_start=None, jd_end=None, det=None, period_days=None, file_path_name=None, preferred_name=None):


    n_det = len(det["station"])
    contributing_station_list, n_station = summariseStationCounts(det)
    r, d, m = lookupCatalogueStar(conn, star_name)

    base_name = generateBaseName(star_name, r,d, jd_start, jd_end, period_days)

    if preferred_name is not None:
        star_name = f"{preferred_name} ({star_name})"

    titles = {
        "super": f"{star_name} with period {period_days:.6f} days",
        "raw": f"RA={r:.2f} DEC={d:.2f} MAG={m:.2f} Time range {jd_start:.2f} to {jd_end:.2f}JD {n_det} observations from {n_station} stations",
        "bottom": f"{contributing_station_list}",
        "footer": f"Saved as {base_name}"
    }


    return titles, r, d, m

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


def generateBaseName(star_name, r, d, jd_start, jd_end, period_days):


    base_name = f"{star_name}_RA_{r*1e6:.0f}_DEC_{d*1e6:.0f}_jds_{jd_start*1e6:.0f}_jde_{jd_end*1e6:.0f}_period_{period_days*1e6:.0f}"


    return base_name

def resolveJdRange(conn, star_name, jd_start=None, jd_end=None):
    """
    Resolve JD start/end for a given star.
    If jd_start or jd_end is None, query the observation table
    to find the earliest and latest jd_mid for that star.

    Returns:
        (jd_start, jd_end)
    """

    # If both are already provided, nothing to do
    if jd_start is not None and jd_end is not None:
        return jd_start, jd_end

    sql = """
        SELECT MIN(jd_mid), MAX(jd_mid)
        FROM observation
        WHERE star_name = %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (star_name,))
        row = cur.fetchone()

    if row is None or row[0] is None:
        raise ValueError(f"No observations found for star '{star_name}'")

    db_min, db_max = row

    # Fill in missing values

    jd_start = db_min / 1e6 if jd_start is None else jd_start
    jd_end = db_max / 1e6 if jd_end is None else jd_end

    return jd_start, jd_end


def main():
    arg_parser = argparse.ArgumentParser(
        description="Apply TPS photometric corrections and plot corrected magnitudes vs JD."
    )

    arg_parser.add_argument("--jd-start", type=float, default=None,
                        help="Start of JD range (inclusive).")
    arg_parser.add_argument("--jd-end", type=float, default=None,
                        help="End of JD range (exclusive).")

    group_star = arg_parser.add_mutually_exclusive_group()

    group_star.add_argument('--radec', metavar='RADEC', type=str,
                    help="Star coordinates in decimal or sexagesimal degrees, if none are passed then defaults to Betelgeux (88.79 7.41)")

    group_star.add_argument("--star-name", type=str, default=None,
                        help="Optional star name filter (quote if it contains spaces).")


    arg_parser.add_argument("--smooth", type=float, default=1.0,
                        help="TPS smoothing factor (default 1.0).")

    arg_parser.add_argument("--period_days", type=float, default=None,
                        help="Period for folding, in days.")

    arg_parser.add_argument("--db-uri", type=str, required=True,
                        help="PostgreSQL connection URI, e.g. postgresql://user@host:5432/dbname")

    arg_parser.add_argument("-o", "--output-dir", type=str, default=os.path.expanduser("~/RMS_data/Plots/LightCurves"),
                        help="Output directory defaults to ~/RMS_data/Plots/LightCurves ")

    arg_parser.add_argument("-f", "--file-name", type=str, default=None,
                            help="Output filename, defaults to ")

    arg_parser.add_argument("-p", "--preferred-name", type=str, default=None,
                            help="Preferred star name for the plot")

    cml_args = arg_parser.parse_args()

    # ------------------------------------------------------------
    # Parse PostgreSQL URI
    # ------------------------------------------------------------


    parsed = urlparse(cml_args.db_uri)

    conninfo = (
        f"host={parsed.hostname} "
        f"port={parsed.port or 5432} "
        f"user={parsed.username} "
        f"dbname={parsed.path.lstrip('/')}"
    )


    conn = psycopg.connect(conninfo)

    if cml_args.radec:
        r, d = parseRaDec(cml_args.radec)
        star_name = lookupBrightestStar(conn, r, d, radius_deg=0.1)[0]
        print(f"Searched at {r} {d} and found {star_name}")
    else:
        star_name = cml_args.star_name

    jd_start = cml_args.jd_start
    jd_end = cml_args.jd_end
    period_days = cml_args.period_days
    output_dir = cml_args.output_dir
    base_name = cml_args.file_name
    preferred_name = cml_args.preferred_name


    # ------------------------------------------------------------
    # Load detections
    # ------------------------------------------------------------

    jd_start, jd_end = resolveJdRange(conn, star_name, jd_start, jd_end)

    det = loadDetections(
        conn,
        jd_start=jd_start,
        jd_end=jd_end,
        star_name=star_name
    )




    titles, ra_cat, dec_cat, mag_cat = generateTitles(conn, star_name=star_name, jd_start=np.min(det['jd']), jd_end=np.max(det['jd']),
                            period_days=period_days, det=det, preferred_name=preferred_name)

    if base_name is None:
        base_name = generateBaseName(star_name, ra_cat, dec_cat, jd_start, jd_end, period_days)



    if det is None:
        print("No detections found in the given JD range.")
        return

    print("Plotting corrected light curve...")
    cat_mag = det['cat_mag'][0]

    # ------------------------------------------------------------
    # Apply TPS corrections
    # ------------------------------------------------------------
    print("Applying TPS corrections...")
    det_corr = applyDetectionCorrections(conn, det, dummy_run=False)

    # ------------------------------------------------------------
    # Plot corrected light curve
    # ------------------------------------------------------------

    #plotCorrectedLightCurve(det, base_name=args.output_name, cat_mag=cat_mag)
    det_binned = binDetections(det_corr, bin_seconds=5)
    #plotCorrectedLightCurve(det_binned, base_name=args.output_name, cat_mag = cat_mag)

    if cml_args.period_days is not None:
        det_folded = foldLightCurve(det_binned, det, cml_args.period_days)
        det_phase_folded_binned = phaseBinFolded(det_folded, n_phase_bins=200)

        plotFoldedWithStations(det_phase_folded_binned, det_folded, cat_mag=cat_mag, titles=titles, base_name=base_name, output_dir=output_dir)



    print("Done.")



if __name__ == "__main__":
    main()
