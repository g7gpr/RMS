import matplotlib.pyplot as plt
import numpy as np
import os
from hemisphere import plotHemisphere


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



def readIAUCSN(catalog_path, catalog_name, additional_fields = []):
    """

    Arguments:
        catalog_path: [str] path to catalog
        catalog_name: [str] name of catalog file
        additional_fields: [list] names of additional fields to return

    Returns:
        star_data: [list] of star data [ra, dec. m]
        mag_band: [list] of mag band data per star
        mag_band_ratios: [list] each entry zero, not supported for this catalog
        extra_values_dict:[dict] dictionary indexed of extra values
    """

    extra_values_dict = {}

    for field in additional_fields:
        extra_values_dict[field] = []


    star_data, name_list, mag_band, mag_band_ratios = [], [], [], []
    with open(os.path.join(os.path.expanduser(catalog_path), catalog_name), 'r') as catalog_file:

        for line in catalog_file:
            line_counter = 0
            if line.startswith('#') or line.startswith('$') or line == '\n':
                pass
            else:
                break
            header_line = line
        names, positions = inferColumnPositions(header_line)

        for line in catalog_file:
            if line.startswith('#') or line.startswith('$'):
                continue
            if not line[0].isalnum():
                continue
            f = []
            for col in range(0,len(positions)):
                f.append(line[positions[col][0]:positions[col][1]].strip())
            try:
                r, d, m, mb = float(f[12]), float(f[13]), float(f[8]), f[9]
                star_data.append(np.array([float(r), float(d), float(m)]))
                mag_band.append(mb)
                mag_band_ratios.append(0)
                for additional_field in additional_fields:
                    additional_field_column = names.index(additional_field)
                    value = line[positions[additional_field_column][0]:positions[additional_field_column][1]].strip()
                    if "name" in additional_field.lower():
                        extra_values_dict[additional_field].append(value.encode('utf-8'))
                    else:
                        extra_values_dict[additional_field].append(value)
            except:
                print("Error loading IAU-CSN line {}".format(line))

        star_data = np.array(star_data)

        for field in additional_fields:
            extra_values_dict[field] = np.array(extra_values_dict[field])


    return star_data, mag_band, mag_band_ratios, extra_values_dict


def addCurvedHemisphereTitle(ax, text, hemisphere, radius_deg=98, spread_deg=45):
    """
    Draw a fully curved title around the rim of a polar hemisphere plot.
    Each character is positioned and rotated individually.
    """

    # Convert radius to radians for polar coordinates
    r = np.deg2rad(radius_deg)

    # Base angles:
    #   north = π (top)
    #   south = 0 (bottom)
    #
    # Apply required 90° shifts:
    #   south: move clockwise  by 90° → -π/2
    #   north: move anticlockwise by 90° → +π/2
    if hemisphere == "north":
        theta_center = np.pi + np.pi/2      # top → rotate anticlockwise 90°
    else:
        theta_center = 0 - np.pi/2          # bottom → rotate clockwise 90°

    # Angular positions for each character
    n = len(text)
    if n > 1:
        dtheta = np.deg2rad(spread_deg) / (n - 1)
    else:
        dtheta = 0

    theta_start = theta_center - 0.5 * (n - 1) * dtheta

    for i, char in enumerate(text):
        theta = theta_start + i * dtheta

        # Tangent rotation
        rotation = np.rad2deg(theta) + 90

        ax.text(
            theta,
            r,
            char,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=14,
            color="#444444",
            fontweight="light"
        )


def plotAtlasTwoPanel(
    ra_south_deg,
    dec_south_deg,
    mag_south,
    ra_north_deg,
    dec_north_deg,
    mag_north,
    constellations_south,
    constellations_north,
    names_list_north=None,
    names_list_south=None
):
    """
    Plot a two-panel atlas: south and north hemispheres side by side.
    Returns fig.
    """
    fig = plt.figure(figsize=(28, 14), dpi=400)

    ax_south = fig.add_subplot(121, projection="polar")
    ax_north = fig.add_subplot(122, projection="polar")

    plotHemisphere(fig, ax_south, ra_south_deg, dec_south_deg, constellations_south, mag=mag_south, names_list=names_list_south, hemisphere="south")
    plotHemisphere(fig, ax_north, ra_north_deg, dec_north_deg, constellations_north, mag=mag_north,names_list=names_list_north, hemisphere="north")

    addCurvedHemisphereTitle(ax_south, "SOUTHERN HEMISPHERE", "south")
    addCurvedHemisphereTitle(ax_north, "NORTHERN HEMISPHERE", "north")

    fig.suptitle(
        "The Global Meteor Network All Sky Survey",
        y=0.92,
        color="#444444",
        fontweight="light"
    )

    return fig
