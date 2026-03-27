import csv
import os
import numpy as np

from transforms import radecToPolar

def loadConstellationLines(csv_path):
    """Load constellation line segments from a 4-column CSV."""
    lines = []
    ra1_col, dec1_col, ra2_col, dec2_col = 0, 1, 2, 3

    csv_path = os.path.expanduser(csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 4:
                continue
            ra1 = float(row[ra1_col])
            dec1 = float(row[dec1_col])
            ra2 = float(row[ra2_col])
            dec2 = float(row[dec2_col])
            lines.append((ra1, dec1, ra2, dec2))
    return lines


def filterSouthernConstellations(lines):
    """Keep only segments fully in the southern hemisphere."""
    south_lines = []
    for ra1, dec1, ra2, dec2 in lines:
        if dec1 < 0.0 and dec2 < 0.0:
            south_lines.append((ra1, dec1, ra2, dec2))
    return south_lines


def filterNorthernConstellations(lines):
    """Keep only segments fully in the northern hemisphere."""
    north_lines = []
    for ra1, dec1, ra2, dec2 in lines:
        if dec1 >= 0.0 and dec2 >= 0.0:
            north_lines.append((ra1, dec1, ra2, dec2))
    return north_lines


def plotConstellationLines(ax, lines, color="#0044aa", alpha=0.22, lw=0.35, hemisphere="north"):
    """Draw constellation line segments."""
    for ra1, dec1, ra2, dec2 in lines:
        theta1, r1 = radecToPolar(ra1, dec1)
        theta2, r2 = radecToPolar(ra2, dec2)


        if dec1 < 0.0 and dec2 < 0.0:
            theta1, r1 = theta1, r1
            theta2, r2 = theta2, r2
        elif dec1 > 0.0 and dec2 > 0.0:
            theta1, r1 = np.mod(np.pi - theta1, 2*np.pi), r1
            theta2, r2 = np.mod(np.pi - theta2, 2*np.pi), r2


        ax.plot([theta1, theta2], [r1, r2], color=color, alpha=alpha, lw=lw)


def plotConstellationLabels(ax, lines, color="#6699cc", fontsize=10):
    """
    Placeholder: label constellations at segment midpoints.
    (You can later map segments to actual constellation names.)
    """
    ras = [(ra1 + ra2) / 2.0 for ra1, _, ra2, _ in lines]
    decs = [(dec1 + dec2) / 2.0 for _, dec1, _, dec2 in lines]

    for ra_deg, dec_deg in zip(ras, decs):
        theta, r = radecToPolar(ra_deg, dec_deg)
        ax.text(theta, r, "", color=color, fontsize=fontsize)
