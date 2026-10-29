from copy import deepcopy

import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.AstrometryNet import astrometryNetSolveLocal
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalog import readStarCatalog
from RMS.Astrometry.Conversions import datetime2JD, JD2HourAngle,  raDec2AltAz, altAz2RADec
from datetime import datetime, UTC, timezone

from RMS.Math import angularSeparationDeg
from scipy.spatial import cKDTree
import scipy

import os


J2000 = datetime(2000, 1, 1, 12, 0, 0).replace(tzinfo=timezone.utc)





def inferColumnPositions(line):
    """
    From a line of text infer column positions

    Arguments:
        line: [str] line of text

    Return:
        names: [list] list of column names
        positions: [list] list of column positions
    """

    positions,names = [], []
    field_count = 0
    in_field = False
    for i, char in enumerate(line):
        if not in_field and not char.isspace():
            field_count += 1
            if field_count > 1:
                end = i
                field_name = line[start:end].strip()
                if field_name.replace("#","") != "":
                    field_name = field_name.replace("#","")
                names.append(field_name)
                positions.append((start, end))
            if i > 0:
                start = i - 1
            else:
                start = i
            in_field = True
        elif in_field and char.isspace():
            in_field = False
    end = i
    names.append(line[start:end].strip())
    positions.append((start, len(line)))

    if in_field:
        positions.append((start, len(line)))
    return names, positions


class Catalog:
    """
    Load a star catalogue, build a spherical KD-tree, and expose query methods.
    """

    def __init__(self, config, catalogue_time=None, ra_col=0, dec_col=1, mag_col=2, name_col=0, lim_mag=None):

        """Initialise a catalog in a spherical tree object/

        Arguments:
            config[config]: RMS config instance.

        Keyword Arguments:
            catalogue_time: Time point for the catalogue generation if none build for now.
            ra_col: Optional, default 0 - array column with ra data in degrees.
            dec_col: Optional, default 1 - array column with dec data in degrees.
            mag_col: Optional, default 2 - array column of magnitude data.
            name_col: Optional, default 3 - array column of star names.

        """

        self.ra_col, self.dec_col, self.mag_col = ra_col, dec_col, mag_col
        self.name_col = name_col

        if catalogue_time is None:
            catalogue_time = datetime.now(timezone.utc)

        # Compute the number of years from J2000
        years_from_J2000 = (catalogue_time - J2000).total_seconds() / (365.25 * 24 * 3600)






        star_catalog_status = readStarCatalog(
            config.star_catalog_path,
            config.star_catalog_file,
            lim_mag=lim_mag,
            years_from_J2000=years_from_J2000,
            mag_band_ratios=config.star_catalog_band_ratios,
            additional_fields=['preferred_name', 'common_name', 'bayer_name', 'B', 'V', 'R'])

        pass

        if not star_catalog_status:
            print("Could not load star catalogue")

        catalog_stars, _, config.star_catalog_band_ratios, extras = star_catalog_status

        combined_extras = {}
        #combined_catalog_stars = np.vstack((iau_csn_coordinates, catalog_stars))
        #combined_extras['preferred_name'] = np.hstack((iau_extras['Name/ASCII'], extras['preferred_name']))

        if False:
            catalog_stars, extras = combined_catalog_stars, combined_extras

        # Do some cleaning on the data - this is not required at present
        maskFinite = np.isfinite(catalog_stars[:, 0:3]).all(axis=1)
        maskRange = (
                (catalog_stars[:, 0] >= 0.0) & (catalog_stars[:, 0] < 360.0) &  # RA
                (catalog_stars[:, 1] >= -90.0) & (catalog_stars[:, 1] <= 90.0)  # Dec
        )

        mask = maskFinite & maskRange

        self.cat = catalog_stars[mask]
        self.names = extras['preferred_name'][mask]
        self.b = extras['B']
        self.v = extras['V']
        self.r = extras['R']

        # Convert to arrays of radians
        ra, dec = np.radians(self.cat[:, ra_col]), np.radians(self.cat[:, dec_col])

        # Build tree of spherical unit vectors
        self.tree = cKDTree(np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))))
        self.entry_count = len(self.names)

    def queryRaDec(self, ra_deg, dec_deg, radius_deg=0.1, n_brightest=1):
        """
        KD-tree search for RA/Dec coordinates.

        Always returns a list of length N (one per detection):
            None  -> no match
            [ [name, ra, dec, mag, theta], ... ]  -> matches
        """

        # Normalize inputs to arrays of radians
        ra_deg, dec_deg = np.atleast_1d(ra_deg), np.atleast_1d(dec_deg)
        ra, dec = np.radians(ra_deg), np.radians(dec_deg)

        # Build query vectors
        query_vectors = np.column_stack((
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ))

        # Euclidean chord distance for spherical radius
        ecd = 2 * np.sin(np.radians(radius_deg) / 2)

        results = []

        for qvec, ra0, dec0 in zip(query_vectors, ra_deg, dec_deg):

            idx = np.array(self.tree.query_ball_point(qvec, ecd), dtype=int)

            if len(idx) == 0:
                results.append(None)
                continue

            mags = self.cat[idx, self.mag_col]
            chosen = idx[np.argsort(mags)[:n_brightest]]

            names = [x.decode("utf-8", errors="replace").strip() for x in self.names[chosen]]
            ras = [float(x) for x in self.cat[chosen, self.ra_col]]
            decs = [float(x) for x in self.cat[chosen, self.dec_col]]
            mags = [float(x) for x in self.cat[chosen, self.mag_col]]
            b, v, r = [float(x) for x in self.b[chosen]], [float(x) for x in self.v[chosen]], [float(x) for x in self.r[chosen]]


            thetas = angularSeparationDeg(ra0, dec0, ras, decs)
            thetas = [float(t) for t in thetas]

            matched = [
                [names[i], ras[i], decs[i], mags[i], thetas[i], b[i], v[i], r[i]]
                for i in range(len(names))
            ]

            results.append(matched)

        return results




    def queryPlatepar(self, pp, jd, n_brightest=1000, az=None, el=None):

        p = deepcopy(pp)

        if az is None or el is None:
            pass
        else:
            # Repoint the platepar
            p = repointPlatepar(p, jd, az, el)

        # Find the corners
        cx_array, cy_array = np.array([0, p.X_res, p.X_res, 0]), np.array([0, 0, p.Y_res, p.Y_res])
        jd_array, mag_array = np.full_like(cy_array, jd, dtype=float), np.full_like(cy_array, 1, dtype=float)
        _, ra_array, dec_array, _ = xyToRaDecPP(jd_array, cx_array, cy_array, mag_array, p, jd_time=True)

        # Get the optical centre in sky coordinates
        x_c, y_c = p.getDistortionCentre()
        _, ra_c, dec_c, _ = xyToRaDecPP([jd], [x_c], [y_c], [1], p, jd_time=True)

        # Compute the maximum angular radius from the centre to the furthest corner.
        max_separation = max(angularSeparationDeg(ra_c, dec_c, ra_array, dec_array))

        # Query for stars and map to arrays
        stars_list = cat.queryRaDec(ra_c, dec_c, max_separation, n_brightest=n_brightest * 4)[0]
        names_array, ra_array, dec_array, mag_array, theta_array, b_array, v_array, r_array = map(np.array, zip(*stars_list))

        # Compute image coordinates.
        x_array, y_array = raDecToXYPP(ra_array, dec_array, jd, p)

        # Cut any stars which are outside the image
        mask = ((x_array >= 0) & (x_array < p.X_res) & (y_array >= 0) & (y_array < p.Y_res))
        ra_array, dec_array = ra_array[mask][:n_brightest], dec_array[mask][:n_brightest]
        x_array, y_array = x_array[mask][:n_brightest], y_array[mask][:n_brightest]
        mag_array, names_array = mag_array[mask][:n_brightest], names_array[mask][:n_brightest]

        return p, ra_array, dec_array, x_array, y_array, mag_array, names_array



def renderStarsSimple(x_array, y_array, shape):
    """
    Render stars as single bright pixels.

    x_array, y_array : pixel coordinates
    shape            : (height, width)
    """

    H, W = shape

    img = np.zeros((H, W), dtype=np.uint8)

    x = np.round(x_array).astype(int)
    y = np.round(y_array).astype(int)

    # Keep only stars inside image bounds
    mask = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )

    img[y[mask], x[mask]] = 255

    return img





def renderStarsGaussian(
        x_array,
        y_array,
        mag_array,
        names_array,
        shape,
        sigma=1.0,
        min_brightness=20,
        max_brightness=250,
        sensor_saturation=240,
        add_noise=False,
        noise_sigma=1,
        sky_background=5, label=False):
    """
    Render a synthetic star field.

    Parameters
    ----------
    x_array, y_array : ndarray
        Star pixel coordinates.

    mag_array : ndarray
        Stellar magnitudes (smaller = brighter).

    shape : tuple
        (height, width)

    sigma : float
        Gaussian PSF width in pixels.

    min_brightness : float
        Brightness assigned to faintest star.

    max_brightness : float
        Brightness assigned to brightest star.

    add_noise : bool
        Add Gaussian image noise.

    noise_sigma : float
        Noise standard deviation.
    """

    H, W = shape

    image = np.zeros(
        (H, W),
        dtype=np.float32
    )

    #
    # Pixel coordinates
    #
    x = np.round(x_array).astype(int)
    y = np.round(y_array).astype(int)



    #
    # Clip to image bounds
    #
    mask = (
        (x >= 10) & (x < W - 10) &
        (y >= 10) & (y < H - 10)
    )

    #
    # Convert magnitude -> brightness
    #
    # Bright stars have SMALL magnitudes,
    # so invert the scale.
    #
    flux = np.interp(
        mag_array,
        (mag_array.min(), mag_array.max()),
        (max_brightness, min_brightness)
    )

    #
    # Render delta-function stars
    #
    image[y[mask], x[mask]] += flux[mask]

    #x0 = np.floor(x[mask]).astype(int)
    #y0 = np.floor(y[mask]).astype(int)
    #flux = flux[mask]

    #dx = x[mask] - x0
    #dy = y[mask] - y0

    #image[y0, x0] += flux * (1 - dx) * (1 - dy)
    #image[y0, x0 + 1] += flux * dx * (1 - dy)
    #image[y0 + 1, x0] += flux * (1 - dx) * dy
    #image[y0 + 1, x0 + 1] += flux * dx * dy


    #
    # Apply common stellar PSF
    #
    image = scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma
    )

    image -= image.min()
    if image.max() > 0:
        image *= 255 / image.max()

    image += sky_background

    #
    # Optional sensor noise
    #
    if add_noise:
        image += np.random.normal(
            0,
            noise_sigma,
            image.shape
        )



    #
    # Clip to display range
    #
    image = np.clip(
        image,
        0,
        sensor_saturation
    )

    #
    # Burn labels into image
    #
    if label:
        pil_img = Image.fromarray(image.astype(np.uint8))

        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype(
                "DejaVuSans.ttf",
                12
            )
        except:
            font = ImageFont.load_default()

        for i in range(min(10, len(names_array))):

            if not mask[i]:
                continue

            draw.text(
                (x[i] + 5, y[i] - 5),
                str(names_array[i]),
                fill=255,
                font=font
            )

        image = np.array(pil_img)




    return image.astype(np.uint8)



def writePNG(image, filename):
    """
    Save a numpy image array to a PNG.

    Parameters
    ----------
    image : ndarray
        Image data. Can be uint8, uint16, float32, etc.

    filename : str or Path
        Output filename.
    """

    image = np.asarray(image)

    if image.dtype.kind == "f":
        if image.max() > image.min():
            image = (
                255 * (image - image.min())
                / (image.max() - image.min())
            ).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    Image.fromarray(image).save(filename)


def repointPlatepar(pp, jd, az, alt):


    pp.az_centre, pp.alt_centre = az, alt
    ra_centre, dec_centre = altAz2RADec(pp.az_centre, pp.alt_centre, jd, pp.lat, pp.lon)
    pp.RA_d, pp.dec_d = ra_centre, dec_centre

    pp.updateRefRADec(skip_rot_update=False, preserve_rotation=True)
    pp.updateRefAltAz()

    return pp



if __name__ == "__main__":

    config = loadConfigFromDirectory(".config", ".")
    jd_now = datetime2JD(datetime.now(UTC))

    cat = Catalog(config)

    pp = Platepar()
    pp.read(os.path.join(config.rms_root_dir, config.platepar_name))



    pp, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd_now, az=175, el=-0.5)

    img = renderStarsSimple(x_array, y_array, (pp.Y_res, pp.X_res))
    writePNG(img, os.path.join(os.path.expanduser("~/simple_output.png")))

    img = renderStarsGaussian(x_array, y_array, mag_array, names_array,(pp.Y_res, pp.X_res))
    writePNG(img, os.path.join(os.path.expanduser("~/gaussian_output.png")))

    solve_result = astrometryNetSolveLocal(img=img)
    ra_centre_solve = solve_result[0]
    dec_centre_solve = solve_result[1]

    print(f"Found a starfield at ra:{ra_centre_solve:.2f}, dec:{dec_centre_solve:.2f} jd:{jd_now}")


    # Compute az and el
    az, el = raDec2AltAz(ra_centre_solve, dec_centre_solve, jd_now, pp.lat, pp.lon)
    print(f"Measured as local azimuth and elevation of {az:.2f},{el:.2f}")
    print(f"Station az and el is {pp.az_centre:.2f},{pp.alt_centre:.2f}")

    az_el_angle_error = angularSeparationDeg(az, el, pp.az_centre, pp.alt_centre)

    print(f"Computed azimuth elevation angle error of {az_el_angle_error:.2f}")


    fps = 25
    speed_up = 255
    duration_seconds = 60 * 60 * 0.5

    video = cv2.VideoWriter(
        "synthetic_sky.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pp.X_res, pp.Y_res),
        isColor=False
    )

    jd_image = jd_now

    for frame_num in range(int(fps * duration_seconds / speed_up)):
        if frame_num % 10 == 0:
            print(f"Frame {frame_num} / {int(fps * duration_seconds / speed_up)}")
        pp, ra_array, dec_array, x_array, y_array, mag_array, names_array = \
            cat.queryPlatepar(
                pp,
                jd_image,
                az=175,
                el=-0.5
            )

        img = renderStarsGaussian(
            x_array,
            y_array,
            mag_array,
            names_array,
            (pp.Y_res, pp.X_res), label=False
        )

        #hd_108903_mask = (names_array == "HD 108903")
        #hd_108903_ra = ra_array[hd_108903_mask]
        #hd_108903_dec = dec_array[hd_108903_mask]
        #hd_108903_mag = mag_array[hd_108903_mask]
        #hd_108903_x = x_array[hd_108903_mask]
        #hd_108903_y = y_array[hd_108903_mask]

        #print(hd_108903_ra, hd_108903_dec, hd_108903_mag, hd_108903_x, hd_108903_y)

        video.write(img)

        #
        # Advance by one frame time
        #
        jd_image += speed_up * (1.0 / fps) / 86400.0

    video.release()

    pass