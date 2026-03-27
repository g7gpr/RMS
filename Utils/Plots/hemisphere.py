import numpy as np
import matplotlib.pyplot as plt

from transforms import radecToPolar
from stars import plotStars
from constellations import plotConstellationLines
from grids import (
    addRaTicks,
    addDecTicks,
    addRaGrid,
    addDecGrid,
    addEquator,
    addEcliptic,
)



def plotHemisphere(fig, ax, ra_deg, dec_deg, constellations, mag=None, names_list=None, hemisphere="south"):
    """
    Plot a single hemisphere (north or south) as a polar sky chart.
    Returns (fig, ax).
    """


    theta, r = radecToPolar(ra_deg, dec_deg, hemisphere=hemisphere)
    print("theta size:", theta.size)
    print("r size:", r.size)
    print("theta range:", theta.min(), theta.max())
    print("r range:", r.min(), r.max())


    plotStars(ax, theta, r, ra_deg=ra_deg, dec_deg=dec_deg, mag=mag, names_list=names_list, size=1.2, alpha=0.5)

    plotConstellationLines(ax, constellations)


    addRaTicks(ax, hemisphere=hemisphere)
    addDecTicks(ax)
    addRaGrid(ax)
    addDecGrid(ax)
    addEquator(ax)
    addEcliptic(ax, hemisphere=hemisphere)

    ax.set_ylim(0.0, np.deg2rad(90.0))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fig.suptitle(
        f"Global Meteor Network — {hemisphere.capitalize()} Hemisphere",
        y=0.05,
        color="#0044aa",
    )

    return fig, ax
