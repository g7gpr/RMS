import numpy as np

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats.StarCatalog import Catalog
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import datetime2JD
from datetime import datetime, UTC
from RMS.Formats.CALSTARS import renderStars
from RMS.Astrometry.AstrometryNet import novaAstrometryNetSolve
from RMS.Astrometry.AstrometryNet import astrometryNetSolve

import matplotlib.pyplot as plt
import os

jd_now = datetime2JD(datetime.now(UTC))
config = loadConfigFromDirectory(".config",  "/home/david/source/RMS/")
cat = Catalog(config)


ra_list, dec_list, mag_list, jd_list, name_list = [], [], [], [], []



pp = Platepar()
pp.read(os.path.join(config.rms_root_dir, config.platepar_name))
centre_x, centre_y = pp.X_res / 2, pp.Y_res/2
_, ra_centre, dec_centre, _ = xyToRaDecPP([jd_now], [centre_x], [centre_y], [1], pp, jd_time=True)

print(f"Generating a starfield at ra:{ra_centre[0]}, dec:{dec_centre[0]} jd:{jd_now}")
stars_list = cat.queryRaDec(ra_centre, dec_centre,40, n_brightest=500)[0]

for star in stars_list:
    name_list.append(star[0])
    ra_list.append(star[1])
    dec_list.append(star[2])
    mag_list.append(star[3])
    jd_list.append(jd_now)

ra_array = np.array(ra_list)
dec_array = np.array(dec_list)
mag_array = np.array(mag_list)
jd_array = np.array(jd_list)


x_array, y_array = raDecToXYPP(ra_array, dec_array, jd_now, pp)
pass

coords_list = []
intensity_array = np.power(10, -mag_array / 2.5)
brightest= min(mag_array)
intensity_array = brightest  - mag_array
i_min, i_max = min(intensity_array), max(intensity_array)
for x,y,i in zip (x_array, y_array, intensity_array):
    if 0 < x < pp.X_res and 0 < y < pp.Y_res:
        coords_list.append((x, y, i, 2))
bitmap = renderStars(np.array(coords_list),(pp.Y_res, pp.X_res), gaussian=True)

max_val = bitmap.max()
max_intensity = None

if max_intensity is not None and max_intensity > 0:
    max_intensity = max_intensity * 0.8
    bitmap = np.minimum(bitmap, max_intensity)
    grey = ((bitmap * 255 / max_intensity)).astype(np.uint8)
elif max_val > 0 and max_intensity is None:
    grey = ((bitmap * 255 / max_val)).astype(np.uint8)
else:
    grey = np.minimum(bitmap, 255).astype(np.uint8)

image =  np.minimum(grey, 255)
plt.figure(figsize=(8,8), frameon=False)
plt.imshow(image, cmap='gray', origin="upper")
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(f"/home/david/output.png")

results = astrometryNetSolve(img=image)
identified_ra, identified_dec = results[0], results[1]
print(f"Image created at ra:{ra_centre[0]:.3f} dec:{dec_centre[0]:.3f}")
print(f"Image identified at ra:{identified_ra:.3f} dec:{identified_dec:.3f}")
pass


def renderStars(coords, shape, gaussian=True):
    """
    Render circular blobs with per-star radii into a bitmap.

    coords: array of [x, y, intensity, radius]
    shape: (height, width)
    gaussian: if True, use Gaussian falloff instead of flat circle
    """
    H, W = shape
    bitmap = np.zeros((H, W), dtype=np.float32)

    # Cache stencils so repeated radii don't recompute
    stencil_cache = {}

    for x, y, I, R in coords:
        y = int(y)
        x = int(x)
        I = float(I)
        if not gaussian:
            R = int(R * 0.8)
        else:
            R = int(R * 0.8)

        if R <= 0:
            continue

        # Build or retrieve stencil
        if R not in stencil_cache:
            yy, xx = np.ogrid[-R:R+1, -R:R+1]

            if gaussian:
                sigma = max(R, 1)

                K = int(np.ceil(3 * sigma))

                yy, xx = np.ogrid[-K:K + 1, -K:K + 1]

                stencil = np.exp(
                    -(xx * xx + yy * yy) /
                    (2 * sigma * sigma)
                )
            else:
                stencil = (xx*xx + yy*yy) <= R*R
                stencil = stencil.astype(np.float32)

            stencil_cache[R] = stencil

        stencil = stencil_cache[R]

        # Bounds in output image
        x0, y0 = max(0, y - R), max(0, x - R)
        x1, y1 = min(W, x + R + 1), min(H, y + R + 1)

        # Bounds in stencil
        sy0, sx0 = x0 - (x - R), y0 - (y - R)
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

        # Add scaled stencil
        bitmap[y0:y1, x0:x1] += 3 * I * stencil[sy0:sy1, sx0:sx1]

    return bitmap

def renderSky(ra=None, dec=None, pp=None, jd=None, show_constellation=False, generate_name_overlay=False, label_image=False):

    pass
