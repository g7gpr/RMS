# RPi Meteor Station
# Copyright (C) 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

import copy
import os
import cv2
import datetime
import glob
from curses.ascii import isalnum

from numpy.ma.core import equal
from RMS.Astrometry.ApplyAstrometry import correctVignetting

from RMS.Formats.StarCatalog import readStarCatalog
import numpy as np
import RMS.Formats.CALSTARS as calstars
import RMS.Formats.FFfits as FFfits
import json
import ephem
import itertools

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDec2AltAz, raDecToXYPP, angularSeparation
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import date2JD, jd2Date,  altAz2RADec

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz, pyRefractionApparentToTrue



def sphericalDomainWrapping(ra_min, ra_max, dec_min, dec_max,
                            ra_range_min=0, ra_range_max=360, dec_range_min=-90, dec_range_max=+90):
    """ For a query in a spherical domain, compute the queries required to cover a search space.

    Arguments:
        ra_min: [float] right ascension in degrees.
        ra_max: [float] right ascension in degrees.

    Keyword arguments:
        ra_range_min: [float] optional, default 0 right ascension in degrees domain minimum.
        ra_range_max: [float] optional, default 360 right ascension in degrees domain maximum.
        dec_range_min: [float] optional, default -90 declination in degrees domain minimum.
        dec_range_max: [float] optional, default 90 declination in degrees domain maximum.

    Return:
        [list] of queries required to cover a spherical search space.
    """

    query = []
    query.append([ra_min, ra_max, ra_range_min, ra_range_max])
    query.append([dec_min, dec_max, dec_range_min, dec_range_max])
    return nDimensionDomainSplit(query)


def domainWrapping(query_min, query_max, range_min, range_max):
    """ For a query in an arbitrarily ranged domain, return a query, or a pair of queries.

    For example, for degrees of longitude 0-360.
    340, 380, 0, 360 returns [[340, 360], [0, 20]]

    For degrees of latitude -90, + 90.
    70, 120, -90, 90 returns [[70, 90], [-90, -60]]

    Arguments:
        query_min: [float] Lower bound of query range.
        query_max: [float] Upper bound of query range.
        range_min: [float] Lower bound of domain range.
        range_max: [float] Upper bound of domain range.

    Return:
        [list]: a query, or a pair of queries.
    """

    if query_max < query_min:
        return [[query_min, query_max]]

    # Compensate for range_mins
    span = range_max - range_min
    query_min, query_max = (query_min - range_min) % span, (query_max - range_min) % span

    # No work to do
    if query_min < query_max:
        # Decompensate and return
        return [[range_min + query_min, range_min + query_max]]

    # Case where query_high was on the maximum range limit, avoid returning two overlapping queries
    if query_min > query_max and query_max == range_min:
        # Decompensate and return
        return [[range_min + query_min, range_max]]

    # Case where query_high was on the minimum range limit, avoid returning two overlapping queries
    if query_min > query_max and query_max == 0:
        # Decompensate and return
        return [[range_min + query_min, range_max]]

    # Case where one end was outside of range, return two queries
    elif query_min > query_max:
        q1 = [range_min + query_min, range_max]
        q2 = [range_min, range_min + query_max]
        return [q1, q2]

    # Finally return query range for the case where the full range was specified
    else:
        return [[range_min, range_max]]


def nDimensionDomainSplit(query_list):
    """For an arbitrary number of dimensions, return a cartesian product of domain wrap queries.

    For example a query between:
        160 and 200 degrees of longitude (-180 +180),
        80 and 110 degrees of latitude (-90 + 90) and
        2300 and 0100 the next day would be passed as

        [[160, 200, -180, +180], [80, 110, -90, +90], [23, 25, 0, 24]]

    and would return

        [[[160, 180], [80, 90], [23, 24]],
         [[160, 180], [80, 90], [0, 1]],
         [[160, 180], [-90, -70], [23, 24]],
         [[160, 180], [-90, -70], [0, 1]],
         [[-180, -160], [80, 90], [23, 24]],
         [[-180, -160], [80, 90], [0, 1]],
         [[-180, -160], [-90, -70], [23, 24]],
         [[-180, -160], [-90, -70], [0, 1]]]

    Only the minimum number of wrapped_queries are returned. In this case the degrees of longitude
    does not wrap around, so only 6 wrapped_queries are required for complete coverage.

        [[340, 350, 0, 360], [80, 110, -90, +90], [23, 25, 0, 24]]


        [[340, 350], [80, 90], [23, 24]]
        [[340, 350], [80, 90], [0, 1]]
        [[340, 350], [-90, -70], [23, 24]]
        [[340, 350], [-90, -70], [0, 1]]
        [[340, 360], [0, 20]]
        [[70, 90], [-90, -60]]

    Arguments:
        query_list: List of wrapped_queries in the form [[query_min, query_max], [range_min, range_max], ... ]

    Return:
        list: Cartesian product of split wrapped_queries as a list.
    """
    # Compute the split wrapped_queries for each dimension
    wrapped_queries = []
    for query in query_list:
        query_min, query_max = query[0], query[1]
        domain_low, domain_high = query[2], query[3]
        wrapped_queries.append(domainWrapping(query_min, query_max, domain_low, domain_high))

    # Produce the cartesian product across all result dimensions
    wrapped_query_list = list(itertools.product(*wrapped_queries))

    # Convert to a list
    wrapped_queries = []
    for result in wrapped_query_list:
        wrapped_queries.append(list(result))
    return wrapped_queries


def getImageIntensityCutoff(star_list, brightest=10):
    """ Given a list of stars find the cutoff magnitude to give
        brightest number stars.

    Arguments:
        star_list: [list] list of stars with the intensity in [2].


    Keyword arguments:
        brightest: [int] number of stars to return.

    Return:
        cut_off: [float] the brightness of the star which has
                brightness stars brighter than it.
    """
    intensity_list = []
    for star in sorted(star_list):
        intensity = star[2]
        intensity_list.append(intensity)

    intensity_list.sort(reverse=True)
    if len(intensity_list) > brightest:
        cut_off = intensity_list[brightest]
    else:
        cut_off = 0
    return cut_off


def generateAzEL(pp, jd, x_data, y_data, ra, dec,  mag):
    """
    Given arrays of data return add azimuth and elevation for each point.

    Arguments:
        pp: [object] platepar
        jd: [array] array of julian dates
        x_data: [array] array of x_data
        y_data: [drray] array of y_data
        ra: [array] array of right ascension in degrees
        dec: [array] array of declination in degrees
        mag: [array] array of stellar magnitudes

    Return:
        fits_radec_ma_ax_el_list: [list] list of stars for a fits file
                                    each star is [x, y, r, d, m, az, el, 0]
    """


    fits_radec_m_az_el_list = []
    for x, y, r, d, m, jd in zip(x_data, y_data, ra, dec, mag, jd):
        az, el = raDec2AltAz(r, d, jd, pp.lat, pp.lon)
        fits_radec_m_az_el_list.append([x, y, r, d, m, az, el, 0])
    return fits_radec_m_az_el_list

def platepar2AltAz(pp):
    """
    Get the Alt and Az of a platepar
    Arguments:
        pp: Platepar

    Return:
        Ra_d : [degrees] Ra of the platepar at its creation date
        dec_d : [degrees] Dec of the platepar at its creation date
        JD : [float] JD of the platepar creation
        lat : [float] lat of the station
        lon : [float] lon of the station

    """

    RA_d, dec_d = np.radians(pp.RA_d), np.radians(pp.dec_d)
    lat, lon = np.radians(pp.lat), np.radians(pp.lon)

    return np.degrees(cyTrueRaDec2ApparentAltAz(RA_d, dec_d, pp.JD, lat, lon))

def fitsToJd(ff_name):
    """
    Convert a fits file name to a julian date.

    Arguments:
        ff_name:[str] name of the fits file.

    Return:
        jd [float] JD time of the file

    """
    fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff_name), "%Y-%m-%d %H:%M:%S.%f")
    return date2JD(fits_date.year, fits_date.month, fits_date.day, fits_date.hour, fits_date.minute,
                      fits_date.second, fits_date.microsecond / 1000)

def getPlateparsCalstar(full_path, use_calstar=True, platepar_path=None):
    """
    Get the platepars and calstar file from the directory at full_path.

    If use_calstar is False, generate a calstar file from each .fits file in the directory.
    This means that stars that were not detected can be marked from the catalog, and generally
    produces a cleaner output.

    Arguments:
        full_path: [str] Captured files directory.

    Keyword Arguments:
        use_calstar: [bool] optional default True. If False, create a synthetic calstar file
                        from the list of all the fits files in the directory without any
                        star information. If True, get the calstar from the directory.

    Returns:
        pp : [object] Platepar object.
        recalibrated_platepars : [object] Platepar object.
        recalibrated_platepars_loaded : [bool] True if a recalibrated platepars dict was found.
        calstar: [object] Calstar object, either synthesized from filenames, or loaded from capture directory.
    """
    pp = Platepar()
    expanded_full_path = os.path.expanduser(full_path)
    if os.path.exists(os.path.join(expanded_full_path, "platepar_cmn2010.cal")):
        full_path_platepar = os.path.join(expanded_full_path, "platepar_cmn2010.cal")
        # read in
        pp.read(full_path_platepar)
    else:
        if not platepar_path is None:
            pp.read(platepar_path)

    if os.path.exists(os.path.join(expanded_full_path, "platepars_all_recalibrated.json")):
        full_path_recal_platepar = os.path.join(expanded_full_path, "platepars_all_recalibrated.json")

        with open(full_path_recal_platepar) as f:
            recalibrated_platepars = json.load(f)
            recalibrated_platepars_loaded = True
    else:
        recalibrated_platepars = None
        recalibrated_platepars_loaded = False

    if use_calstar:
        if len(glob.glob(os.path.join(expanded_full_path, "*CALSTARS*"))):
            full_path_calstars = glob.glob(os.path.join(expanded_full_path, "*CALSTARS*"))[0]
            calstars_path = os.path.dirname(full_path_calstars)
            calstars_name = os.path.basename(full_path_calstars)
            calstar, chunk_frames = calstars.readCALSTARS(calstars_path, calstars_name)
        else:
            calstar = []
    else:
        all_files = os.listdir(full_path)
        calstar = []
        try:
            station_code_from_dir = os.path.basename(full_path).split("_")[0]
            for f in sorted(all_files):
                if f.startswith("FF_{}_".format(station_code_from_dir)) and f.endswith(".fits"):
                    calstar.append([f, []])
        except:
            pass


    return pp, recalibrated_platepars, recalibrated_platepars_loaded, calstar

def filterByMag(catalog, magnitude=6):
    """
    Return a filtered catalog.

    Arguments:
        catalog: [catalog] Star catalog object.

    Keyword Arguments:
        magnitude: [float] optional default 5 - drop stars fainter than this from catalog.

    Returns:
        catalog: [catalog] Filtered catalog object with only stars over a lower (brighter) magnitude.
    """


    star_data, mag_band, mag_band_ratios, extra_values_dict = catalog
    if 'preferred_name' in extra_values_dict:
        name_list = extra_values_dict['preferred_name']
    if 'Name/ASCII' in extra_values_dict:
        name_list = extra_values_dict['Name/ASCII']

    cat_mask = np.array(star_data[:, 2] < magnitude).T
    star_data = star_data[cat_mask]
    name_list = name_list[cat_mask]

    return star_data, name_list

def queryStarData(query_list, star_data, name_list):
    """
    Take multiple queries and apply to star data.

    Arguments:
        query_list: [list] each item is a list [ra_min, ra_max, dec_min, dec_max]
        star_data: [array] data set of stars [ra, dec]
        name_list: [list] list of names, same order as star_data.

    Returns:
        stars: [list] stars matching the query.
        names: [list] names matching the query.
    """

    stars, names = [], []
    for query in sorted(query_list):
        ra_min, ra_max = query[0][0], query[0][1]
        dec_min, dec_max = query[1][0], query[1][1]
        cat_mask = np.array((star_data[:, 0] >= ra_min) &
                            (star_data[:, 0] <= ra_max) &
                            (star_data[:, 1] >= dec_min) &
                            (star_data[:, 1] <= dec_max)).T

        stars.extend(star_data[cat_mask])
        names.extend(name_list[cat_mask])

    return stars, names

def annotateImage(file_name, calstar_radec_name, img, font):
    """Annotates an image of stars with information.

    Arguments:
        file_name: [str] fits file name
        calstar_radec_name: [dict]
        img: [array] open cv2 image
        font: [str] font name

    Returns:
        img: [array] open cv2 image
    """

    if file_name in calstar_radec_name:
        for star in sorted(calstar_radec_name[file_name]):
            star_name, x, y = star[0], int(star[1]), int(star[2])
            r, d = float(star[3]), float(star[4])
            az, el = float(star[6]), float(star[7])
            star_name = "{}".format(star_name)
            radec = "r:{:.1f}  d:{:.1f}".format(r, d)
            azel = "az:{:.1f} el:{:.1f}".format(az, el)
            if type(star[10]) == str:
                cat_m = float(star[5])
                mag = "cat_mag:{:.1f}".format(cat_m)
            else:
                cat_m, obs_m = float(star[5]), float(star[10])
                mag = "cat_mag:{:.1f} obs_mag:{:.1f}".format(cat_m, obs_m)

            color = 150
            cv2.putText(img, star_name, (x + 30, y - 35), font, 0.5, (color, color, color), 1, cv2.LINE_AA)
            cv2.putText(img, radec, (x + 30, y - 20), font, 0.4, (color, color, color), 1, cv2.LINE_AA)
            cv2.putText(img, azel, (x + 30, y - 5), font, 0.4, (color, color, color), 1, cv2.LINE_AA)
            cv2.putText(img, mag, (x + 30, y + 10), font, 0.4, (color, color, color), 1, cv2.LINE_AA)
            x += 2
            y += 2
            inner = 10
            outer = 20
            cv2.line(img, (x + outer, y), (x + inner, y), (0, 0, 255), 1)
            cv2.line(img, (x - outer, y), (x - inner, y), (0, 0, 255), 1)
            cv2.line(img, (x, y + outer), (x, y + inner), (0, 0, 255), 1)
            cv2.line(img, (x, y - outer), (x, y - inner), (0, 0, 255), 1)

    return img

def generateFITSStarData(stars, names, jd, pp, stretch_fov=1.1, persistance=100):
    """

    Arguments:
        stars: [list] list of stars [ra, dec, mag]
        names: [list] list of names, same order as stars.
        jd: [float] Julian date of the fits file.
        pp: [platepar] Platepar.
        stretch_fov: [float] ratio to increase the size of the field of view to allow stars to drift out nicely.
        persistance: [int] Number of frames that a star can be missing from CALSTAR before it is removed.

    Return:
        fits_star_data: [list] list of stars sorted by magnitude increasing (brightest first)
    """
    fits_star_data = []
    stretch_fov = 1 + (stretch_fov - 1) / 2

    for star, name in zip(stars, names):
        r, d, m = float(star[0]), float(star[1]), float(star[2])
        az, el = raDec2AltAz([r], [d], [jd], pp.lat, pp.lon)

        x, y = raDecToXYPP(np.array([r]), np.array([d]), jd, pp)
        x, y = x[0], y[0]
        if ((0 - pp.X_res * (stretch_fov - 1)) < x < pp.X_res * stretch_fov and
                (0 - pp.Y_res * (stretch_fov - 1)) < y < pp.Y_res * stretch_fov):
            image_star_data = [name.decode('utf-8'), x, y, r, d, m, az[0], el[0], persistance]
            fits_star_data.append(image_star_data)
    fits_star_data.sort(key=lambda x: x[5])
    return fits_star_data

def getClosest(x, y, reference_list, x_pos=1, y_pos=0, closeness=1):
    """

    Arguments:
        x: [float] x coordinate
        y: [float] y coordinate
        reference_list: [list] reference list of objets

    Keyword Arguments:
        x_pos: [int] position of x coordinate in reference list optional default 1
        y_pos: [int] position of y coordinate in reference list optional default 0
        closeness: [float] Cartesian pixel separation less than which is close

    Returns:

    """

    dist = np.inf
    for p in reference_list:
        x_, y_ = p[x_pos], p[y_pos]
        if np.hypot(x_ - x, y_ - y) < dist:
            dist = np.hypot(x_ - x, y_ - y)
            closest = p

    return closest, dist, dist < closeness

def updateCalstarRadecName(fits_file, fits_star_data, pp, calstar_radec_name,
                           star_list, pixel_separation=10, use_calstar=True):

    """

    Arguments:
        fits_file: [str] Name of the fits file.
        fits_star_data: [list] List of star data for this fits file
        pp: [object] Platepar
        calstar_radec_name: [list]
        star_list: [list]
        brightest: [int] Number of stars to add, ordered by decreasing magnitude
        pixel_separation: [float] Separation between pixels before adding a star
        use_calstar: [bool] Optional, default True. If False Ignore calstar file and use the list of fits files in the night directory

    Return:
        calstar_radec_name: [list] Calstar file entries, with ra and dec and name of the object
    """

    if fits_file in calstar_radec_name:
        stars_to_annotate = calstar_radec_name[fits_file]
    else:
        stars_to_annotate = []

    separated = True
    stars_added = len(stars_to_annotate)
    for star in fits_star_data:
        x, y = star[1], star[2]
        separated = True
        if use_calstar:
            p, dist, is_close = getClosest(x, y, star_list, closeness=3)
            if not is_close:
                continue
        for check_star in stars_to_annotate:
            x_, y_ = check_star[1], check_star[2]
            if np.hypot((x - x_),(y - y_)) < pixel_separation:
                separated = False
                break

        if separated:

            stars_added += 1

            if use_calstar:
                vignetting, offset = pp.vignetting_coeff, pp.mag_lev
                radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
                intensity = p[2]
                observed_corrected_magnitude = 0 - 2.5 * np.log10(correctVignetting(intensity, radius, vignetting)) + offset
                star.append(p[4])
                star.append(observed_corrected_magnitude)

            else:
                star.append(0)
                star.append("N/A")

            stars_to_annotate.append(star)

    stars_to_annotate.sort(key=lambda x: x[5])

    calstar_radec_name[fits_file] = stars_to_annotate

    return calstar_radec_name

def processCatalog(calstar_radec_name, calstar, star_data, name_list, pp,
                   recalibrated_platepars, recalibrated_platepars_loaded, brightest=20, pixel_separation=10,
                   use_calstar=True):
    """

    Arguments:
        calstar_radec_name: [list] Calstar file entries, with ra and dec and name of the object.
        calstar: [list] Calstar data.
        star_data: [list] List of object data for this catalog.
        name_list: [list] List of names for the objects in this catalog.
        pp: [object] Platepar for the station.
        recalibrated_platepars: [dict] Dictionary of platepars indexed by fits file and recalibrated per fits file.
        recalibrated_platepars_loaded: [bool] Boolean, True if reclibrated platepar file was found and loaded.
        brightest: [int] Number of objects to add, listed by brightness
        pixel_separation:[int] Pixel separation between objects to ad
        use_calstar: [bool] Optional, detault True. If False, generate a synthetic calstar as a list of FITS files, and annotate from catalog

    Returns:
        calstar_radec_name: [dict] Dictionary of fits files, and the objects they contain with name and radec

    """

    diagonal_fov = np.sqrt(pp.fov_v ** 2 + pp.fov_h ** 2)




    for fits_file, star_list in sorted(calstar):

        if len(calstar_radec_name[fits_file]) >= brightest:
            continue

        if recalibrated_platepars_loaded and fits_file in recalibrated_platepars:
            pp.loadFromDict(recalibrated_platepars[fits_file])

        jd  = fitsToJd(fits_file)
        az, alt =  platepar2AltAz(pp)
        fov_ra, fov_dec = altAz2RADec(az, alt, jd, pp.lat, pp.lon)
        query_list = sphericalDomainWrapping(fov_ra - diagonal_fov / 2, fov_ra + diagonal_fov / 2,
                                             fov_dec - diagonal_fov / 2, fov_dec + diagonal_fov / 2)
        stars, names = queryStarData(query_list, star_data, name_list)
        fits_star_data = generateFITSStarData(stars, names, jd, pp)
        calstar_radec_name = updateCalstarRadecName(fits_file, fits_star_data, pp, calstar_radec_name, star_list,
                                                    pixel_separation=pixel_separation,
                                                    use_calstar=use_calstar)

    return calstar_radec_name

def mapFITS2Objects(full_path, catalogs_list, brightest=20, pixel_separation=150, use_calstar=True, magnitude=4, platepar_path=None):
    """

    Arguments:
        full_path: [path] Full path to the captured directory
        catalogs_list: [list] List of all the catalog information to use. Catalogs earlier in the list will be preferred.
    Keyword arguments:
        brightest: [int] The number of stars, by brightness to add to the imag
        pixel_separation: [int] Opt, default 150, separation between pixesls of objects to add
        use_calstar: [bool] default True, if False generate the list of objecst from catalogs, not from observed objects

    Returns:
        calstar_radec_name: [dict] Indexed by fits file, data to be annotated

        e.g.

        ['Miaplacidus', 39.31237808055539, 628.6591358573708, 138.299906, -69.717208, 1.67, 200.84000501718765, 44.092690102140494, 99, 5.48, 3.1681016357828877]

        ['Name', x, y, ra, dec, catalog magnitude, azimuth, elevation, persistance, intensity sum ,compensated observed magnitude]
    """


    pp, recalibrated_platepars, recalibrated_platepars_loaded, calstar = getPlateparsCalstar(full_path, use_calstar=use_calstar, platepar_path=platepar_path)

    if calstar is None:
        return []

    calstar_radec_name = {}

    for fits_file, star_list in calstar:
        fits_star_data = addSolarSystemObjects(fits_file, pp, recalibrated_platepars, recalibrated_platepars_loaded)
        calstar_radec_name = updateCalstarRadecName(fits_file, fits_star_data, pp, calstar_radec_name, star_list,
                                                    pixel_separation=pixel_separation,
                                                    use_calstar=use_calstar)

    for catalog in catalogs_list:
        star_data, name_list = filterByMag(catalog, magnitude)
        calstar_radec_name = processCatalog(calstar_radec_name, calstar, star_data, name_list,
                                            pp, recalibrated_platepars, recalibrated_platepars_loaded,
                                            brightest=brightest, pixel_separation=pixel_separation,
                                            use_calstar=use_calstar)

    calstar_radec_name = preventFlickeringStars(pp, calstar_radec_name, pixel_separation=pixel_separation)


    return calstar_radec_name

def addSolarSystemObjects(fits_file, pp, recalibrated_platepars, recalibrated_platepars_loaded):
    """Return solar system object data per fits file

    Arguments:
        fits_file: [str] Name of fits file
        pp: [object] Platepar
        recalibrated_platepars: [dict] Dictionary of platepars indexed by fits file
        recalibrated_platepars_loaded: [bool] True if recalibrated platepars were loaded

    Return:
        fits_object_data: [list] List of solar system objects for this fits file

    """

    if recalibrated_platepars_loaded and fits_file in recalibrated_platepars:
        pp.loadFromDict(recalibrated_platepars[fits_file])
    jd = fitsToJd(fits_file)
    az, alt = platepar2AltAz(pp)
    fov_ra, fov_dec = altAz2RADec(az, alt, jd, pp.lat, pp.lon)
    names, solar_system_objects = solarSystemLibrary(jd)
    solar_system_objects_to_add, names_to_add = [], []
    for solar_system_object, name in zip(solar_system_objects, names):
        ra, dec = solar_system_object[0], solar_system_object[1]
        if angularSeparation(np.radians(ra), np.radians(dec),
                             np.radians(fov_ra), np.radians(fov_dec)) < np.radians(np.hypot(pp.fov_h, pp.fov_v)) / 1.8:
            solar_system_objects_to_add.append(solar_system_object)
            names_to_add.append(name)
    fits_object_data = generateFITSStarData(solar_system_objects_to_add, names_to_add, jd, pp)
    return fits_object_data

def ffname2JD(ff):
    """Return the julian date of a fits file name
    
    Arguments:
        ff: [str] Fits file name.

    Return:
        [float] Julian date
    """

    fits_date = datetime.datetime.strptime(FFfits.filenameToDatetimeStr(ff), "%Y-%m-%d %H:%M:%S.%f")
    return date2JD(fits_date.year, fits_date.month, fits_date.day, fits_date.hour, fits_date.minute,
                             fits_date.second, fits_date.microsecond / 1000)

def timeShift(star, pp, source_ff, dest_ff, use_calstars=True):
    """Shift a star for the change in time between source and destination fits file
    
    Arguments:
        star: [list] list of star data 
        pp: [object] Platepar
        source_ff: [str] Name of the source fits file
        dest_ff: [str] Name of the destination fits file

    Return:
        star: [list] Star with positioning data updated for time delta
    """

    source_jd, dest_jd = ffname2JD(source_ff), ffname2JD(dest_ff)
    jd, _r, _d, mag = xyToRaDecPP([source_jd],[star[1]], [star[2]], [star[5]], pp, jd_time=True)
    x, y = raDecToXYPP(_r, _d, dest_jd, pp)
    star[1], star[2], star[8] = x[0], y[0], star[8] - 1
    return star

def isSeperated(test_object, objects_list, pixel_separation=100):
    """Is test_object separated from all other objects in objects_list
    
    Arguments:
        test_object: [list] list of object data.
        objects_list: [list] list of lists of objects.
        
    Keyword arguments:
        pixel_separation: Optional.

    Return:
        is_separated: [bool] True if test object is separated from other objects.
    """


    x, y = test_object[1], test_object[2]

    separated = True
    for object in objects_list:
        x_, y_ = object[1], object[2]
        if ((x - x_) ** 2 + (y - y_) ** 2) ** 0.5 < pixel_separation:
            separated = False
            break
    return separated

def preventFlickeringStars(pp, calstar_radec_name, pixel_separation=100, brightest=20):
    """Makr.

    Arguments:
        pp: [object] Platepar
        calstar_radec_name: [list] list of star data per fits file
        pixel_separation:

    Return:

    """

    first_file = True
    new_calstar_radec_name = {}
    for ff in sorted(calstar_radec_name):

        if not first_file:
            _calstar_radec_name_ff = copy.deepcopy(new_calstar_radec_name[_ff])
            calstar_radec_name_ff = copy.deepcopy(calstar_radec_name[ff])
            calstar_radec_name_ff.sort(key=lambda x: x[5])
            new_list = []

            for _star in _calstar_radec_name_ff:
                _star_missing = True
                for star in calstar_radec_name_ff:
                    if star[0] == _star[0]:
                        _star_missing = False
                        break
                if _star_missing:
                    shifted_star = timeShift(_star, pp, _ff, ff)
                    if shifted_star[8] > 0 and isSeperated(shifted_star, new_list, pixel_separation=pixel_separation):
                        new_list.append(shifted_star)


            for star in calstar_radec_name_ff:
                if isSeperated(star, new_list, pixel_separation=pixel_separation):
                    new_list.append(star)

        else:
            first_file = False
            new_list = calstar_radec_name[ff]

        new_list.sort(key=lambda x: x[5])
        new_calstar_radec_name[ff] = new_list
        _ff = ff
    return new_calstar_radec_name

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

def objectRaDec(solar_system_object, observation_time_object):
    """Return the ra, dec and mag of an object

    Arguments:
        solar_system_object:
        observation_time_object:

    Return:
        ra_deg: [float] RA in degrees
        dec_deg: [float] Dec in degrees
        mag: [float] Magnitude
    """
    solar_system_object.compute(observation_time_object)
    obs_date = ephem.Date(observation_time_object)
    eq_initial = ephem.Equatorial(solar_system_object.ra, solar_system_object.dec, epoch=obs_date)
    eq_j2000 = ephem.Equatorial(eq_initial, epoch=ephem.J2000)
    ra_deg, dec_deg= pyEphem2DecimalDegrees(eq_j2000.ra, eq_j2000.dec)
    mag = solar_system_object.mag
    return ra_deg, dec_deg, mag

def pyEphem2DecimalDegrees(ra, dec):
    """Convert pyEphem ra dec format to decimal degrees

    Arguments:
        ra: [str] RA in h:m:s format
        dec: [float] Dec in d.m.s.

    Returns:
        ra_deg: [float] RA in degrees
        dec_deg: [float] dec in degrees
    """

    ra = str(ra).split(':')
    ra_deg = float(ra[0]) * 360 / 24
    ra_deg += float(ra[1]) * 360 / (24 * 60)
    ra_deg += float(ra[2]) * 360 / (24 * 60 * 60)
    dec_deg = np.degrees(float(repr(dec)))
    return ra_deg, dec_deg


def sort2ListsOnFieldInList2(list_1, list_2, list_2_field_to_sort):
    """

    Arguments:
        list_1: [list] List of data to be sorted based on a list in field 2
        list_2: [list] List of data to be sorted based on a field in list 2
        list_2_field_to_sort: [int] Field number to sort on

    Returns:
        sorted_list_1: [list] list_1 sorted on field in list 2
        sorted_list_2: [list] list_2 sorted on field in list 2
    """


    sorting_list = []
    for list_1_item, list_2_item in zip(list_1, list_2):
        sorting_list_entry = []
        sorting_list_entry.append(list_1_item)
        sorting_list_entry.append(list_2_item)
        sorting_list.append(sorting_list_entry)

    sorting_list.sort(key=lambda x: x[1][list_2_field_to_sort])

    sorted_list_1, sorted_list_2 = [], []
    for list_1_item, list_2_item in sorting_list:
        sorted_list_1.append(list_1_item)
        sorted_list_2.append(list_2_item)


    return sorted_list_1, sorted_list_2




def solarSystemLibrary(jd, sort_by_magnitude=True):
    """
    For given jd returns RaDec of solar system objects

    Arguments:
        jd: Julian Date.

    Keyword Arguments:
        sort_by_magnitude: [bool] If True sort by magnitude, brightest objects first

    Returns:
        object_data: [list] List of objects [r, d, m]
        names: [list] Names of objects.
    """

    observation_time = jd2Date(jd)
    observation_time_object = datetime.datetime(
                            year=observation_time[0], month=observation_time[1], day=observation_time[2],
                            hour=observation_time[3], minute=observation_time[4], second=observation_time[5],
                            microsecond=round(observation_time[6] * 1000))

    object_data, names = [], []

    object = ephem.Mercury(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Mercury".encode('utf-8'))

    object = ephem.Venus(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Venus".encode('utf-8'))

    object = ephem.Mars(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Mars".encode('utf-8'))

    object = ephem.Jupiter(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Jupiter".encode('utf-8'))

    object = ephem.Saturn(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Saturn".encode('utf-8'))

    object = ephem.Uranus(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Uranus".encode('utf-8'))

    object = ephem.Neptune(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Neptune".encode('utf-8'))

    object = ephem.Pluto(observation_time_object)
    r, d, m = objectRaDec(object, observation_time_object)
    object_data.append(np.array([r, d, m]))
    names.append("Pluto".encode('utf-8'))

    if sort_by_magnitude:
        names, object_data = sort2ListsOnFieldInList2(names, object_data, 2)

    return names, object_data


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
            if not isalnum(line[0]):
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

def getObjectNames(captured_directory, brightest=30, use_calstar=True, pixel_separation=200, annotate='with_calstar', platepar_path=None):
    """

    Arguments:
        captured_directory: [str] Path to a captured files directory.
    Keyword arguments:
        brightest: [int] Count of the brightest number of stars to add, optional, default 30
        use_calstar:  [bool] Optional, deafult True.  If False, uses the fits file names and catalog data to name objects.
                                If True, then only names objects recorded in calstar.


    Returns:
        fits_to_objects_dict: [dict] Dictionary indexed by fits file names of annotation data

        each dictionary item of the form

        ['Name', x, y, ra, dec, catalog magnitude, azimuth, elevation,
                                        persistance, intensity sum, compensated observed magnitude]



    """
    if annotate == 'with_calstar':
        use_calstar = True
    elif annotate == 'without_calstar':
        use_calstar = False
    else:
        return []


    catalogs_list = []
    catalog_full_path = os.path.expanduser("~/source/RMS/Catalogs/IAU-CSN.txt")
    catalog_path = os.path.dirname(catalog_full_path)
    catalog_name = os.path.basename(catalog_full_path)

    catalog_stars = readIAUCSN(catalog_path, catalog_name, additional_fields=['Name/ASCII'])
    catalogs_list.append(catalog_stars)

    catalog_full_path = os.path.expanduser("~/source/RMS/Catalogs/GMN_StarCatalog_LM9.0.bin")
    catalog_path = os.path.dirname(catalog_full_path)
    catalog_name = os.path.basename(catalog_full_path)
    catalog_stars = readStarCatalog(catalog_path, catalog_name, additional_fields=['preferred_name'])
    catalogs_list.append(catalog_stars)

    fits_to_objects_dict = mapFITS2Objects(captured_directory, catalogs_list, pixel_separation=200, brightest=brightest, use_calstar=use_calstar, platepar_path=platepar_path)

    return fits_to_objects_dict
