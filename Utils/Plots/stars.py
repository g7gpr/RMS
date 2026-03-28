import numpy as np

import numpy as np

def magnitudeToSize(mag, mag0=2.0, size0=20.0,
                    min_size=0.5, max_size=40.0):
    """
    Convert stellar magnitude to a visually pleasing point size.
    Accepts lists, tuples, scalars, or NumPy arrays.
    """

    # Ensure mag is a NumPy array
    mag = np.asarray(mag, dtype=float)

    # Logarithmic brightness scaling (Pogson relation)
    sizes = size0 * 10 ** (-0.4 * (mag - mag0))

    # Clamp extremes to keep the atlas elegant
    return np.clip(sizes, min_size, max_size)




def plotStars(ax, theta, r, names_list=None, ra_deg=None, dec_deg=None, mag=None, color="#66aaff", alpha=0.6, size=None):
    """Scatter-plot stars in polar coordinates."""

    sizes = magnitudeToSize(mag)
    mag = np.array(mag)
    alpha = np.clip(1.2 - 0.15 * mag, 0.2, 1.0)

    # Mask bright vs faint
    bright = mag < 2.0
    faint  = ~bright

    # Faint stars first

    if np.any(faint):
        ax.scatter(
            theta[faint],
            r[faint],
            s=sizes[faint],
            c=color,
            alpha=alpha[faint],
            zorder=5
        )

    # Bright stars on top
    if np.any(bright):
        ax.scatter(
            theta[bright],
            r[bright],
            s=sizes[bright],
            c=color,
            alpha=alpha[bright],
            zorder=6
)
    if names_list is not None and ra_deg is not None and dec_deg is not None:
        for th, rr, nm, ra, dec in zip(theta, r, names_list, ra_deg, dec_deg):
            label = f"{nm}" #\nRA={ra:.1f}°\nDec={dec:.1f}°"
            ax.text(th, rr, label,
                    fontsize=7,
                    ha="left", va="bottom",
                    color="black",
                    zorder=2001)

    #ax.scatter(theta, r, s=size, c=color, alpha=alpha)


