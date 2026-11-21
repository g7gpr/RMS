# RPi Meteor Station
# Copyright (C) 2026 David Rollinson
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

"""ADS-B aircraft reception and logging for RMS.

Receives SBS messages from dump1090, converts aircraft locations to
azimuth/elevation, projects them into image coordinates using the station
platepar, and records hourly aircraft positions, velocities and ICAO to callsign mapping
for later upload and analysis.
"""



import argparse
import socket
import datetime
import os
import json
import multiprocessing
import time
import tarfile
import glob
import bisect

import numpy as np


from RMS.Logger import LoggingManager, getLogger
from RMS.Formats.StarCatalogQueries import getPlateparSkyParameters
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import datetime2JD, latLonAlt2ECEF, ECEF2AltAz, altAz2RADec, ecef2LatLonAlt
from RMS.Math import angularSeparationDeg
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP


import RMS.ConfigReader as cr

# Set verbose
VERBOSE = False
VVERBOSE = False
LOG_ESCAPING = False
# Length of slice for time stripping

# 11 - hourly
# 12 - ten minutes
# 13 - one minute

SLICE_LENGTH = 12

# Constants

# Acceleration due to gravity
g = 9.80665

# Unit conversion constant
FT_TO_M = 0.3048
KT_TO_MS = 0.514444

# SBS message type
SBS_MESSAGE_TYPE = "MSG"

# Dump1090 ports
DUMP1090_SBS_MODE_PORT = 30003
DUMP1090_BEAST_MODE_PORT = 30005

# Tick
TICKS_PER_SECOND = 12e6

# SBS message keys
SBS_MSG_IDENTIFICATION      = "1"
SBS_MSG_SURFACE_POSITION    = "2"
SBS_MSG_AIRBORNE_POSITION   = "3"
SBS_MSG_AIRBORNE_VELOCITY   = "4"
SBS_MSG_SURVEILLANCE_ALT    = "5"
SBS_MSG_SURVEILLANCE_ID     = "6"
SBS_MSG_AIR_TO_AIR          = "7"
SBS_MSG_ALL_CALL_REPLY      = "8"


# SBS fields
FIELD_MSG_TYPE          = 0
FIELD_TX_TYPE           = 1
FIELD_SESSION_ID        = 2
FIELD_AIRCRAFT_ID       = 3
FIELD_ICAO              = 4
FIELD_FLIGHT_ID         = 5
FIELD_GENERATED_DATE    = 6
FIELD_GENERATED_TIME    = 7
FIELD_LOGGED_DATE       = 8
FIELD_LOGGED_TIME       = 9
FIELD_CALLSIGN          = 10
FIELD_ALTITUDE          = 11
FIELD_GROUNDSPEED       = 12
FIELD_TRACK             = 13
FIELD_LATITUDE          = 14
FIELD_LONGITUDE         = 15
FIELD_VERTICAL_RATE     = 16
FIELD_SQUAWK            = 17
FIELD_ALERT             = 18
FIELD_EMERGENCY         = 19
FIELD_SPI               = 20
FIELD_ON_GROUND         = 21


# ADS-B downlink formats
DF_ADSB_EXTENDED_SQUITTER = 17


# ADS-B type codes
TC_AIRCRAFT_IDENTIFICATION_MIN = 1
TC_AIRCRAFT_IDENTIFICATION_MAX = 4

TC_SURFACE_POSITION_MIN = 5
TC_SURFACE_POSITION_MAX = 8

TC_AIRBORNE_POSITION_BARO_MIN = 9
TC_AIRBORNE_POSITION_BARO_MAX = 18

TC_AIRBORNE_VELOCITY = 19

TC_AIRBORNE_POSITION_GNSS_MIN = 20
TC_AIRBORNE_POSITION_GNSS_MAX = 22

TC_AIRCRAFT_STATUS = 28
TC_AIRCRAFT_OPERATIONAL_STATUS = 31


BEAST_MODE_SHORT_FRAME = 0x32
BEAST_MODE_LONG_FRAME  = 0x33

CALLSIGN_TABLE = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######"

log = getLogger("rmslogger")


def isoAgeSeconds(newer, older):


    newer_dt = datetime.datetime.fromisoformat(newer)
    older_dt = datetime.datetime.fromisoformat(older)

    return (newer_dt - older_dt).total_seconds()


def isAircraftSurfacePosition(tc):
    return TC_SURFACE_POSITION_MIN <= tc <= TC_SURFACE_POSITION_MAX


def isAircraftIdentification(tc):
    return TC_AIRCRAFT_IDENTIFICATION_MIN <= tc <= TC_AIRCRAFT_IDENTIFICATION_MAX

def isAirbornePosition(tc):
    return TC_AIRBORNE_POSITION_BARO_MIN <= tc <= TC_AIRBORNE_POSITION_BARO_MAX

def isGNSSPosition(tc):
    return TC_AIRBORNE_POSITION_GNSS_MIN <= tc <= TC_AIRBORNE_POSITION_GNSS_MAX

def isAirborneVelocity(tc):

    return tc==TC_AIRBORNE_VELOCITY

def getbits(payload, first, last):
    value = int.from_bytes(payload, byteorder="big")

    total_bits = len(payload) * 8

    shift = total_bits - last

    mask = (1 << (last - first + 1)) - 1

    return (value >> shift) & mask


def formatTimeStamp(dt):
    """ Format a datetime object as a timestamp string.

    Converts a datetime object into the format
    YYYY-MM-DD HH:MM:SS.sss, retaining millisecond precision.

    Arguments:
        dt: [datetime] Datetime object to format.

    Return:
        time
    """

    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]




def filterVisible(time_series_data):

    """ Return the complete time series for aircraft which were
    visible at least once within the field of view.

    All observations are retained so that pre-entry and
    post-exit history remains available for rendering.

    Arguments:
        time_series_data: [dict] Timeseries entries

    Return:
        filtered: [dict] Timeseries entries where ICAO was visible for at least one frame

    """

    filtered = {}

    for icao, aircraft_time_series in time_series_data.items():

        visible = False

        for point in aircraft_time_series:

            if "x" in point and "y" in point:
                visible = True
                break

        if visible:
            filtered[icao] = aircraft_time_series

    return filtered


def extractStreams(derived_data, icao):

    position_data = derived_data[icao]["positions"]
    velocity_data = derived_data[icao]["velocity"]
    callsign_data = derived_data[icao]["callsigns"]

    return position_data, velocity_data, callsign_data

def extractTimestamps(derived_data, icao):



    # Get the data streams from the derived data
    position_data, velocity_data, callsign_data = extractStreams(derived_data, icao)

    # Get every timestamp for this icao
    ts_list = []

    for entry in position_data:
        ts_list.append(entry["ts"])
    for entry in velocity_data:
        ts_list.append(entry["ts"])
    for entry in callsign_data:
        ts_list.append(entry["ts"])

    # Produce a sorted list of time stamps for this ICAO
    ts_list = sorted(set(ts_list))

    return ts_list

def buildICAOTimeSeries(derived_data, icao_iterator, ts_list, ts_data, clean_data=False):


    if VERBOSE:
        log.info(f"Processing {icao_iterator} with {len(ts_list)} timestamps")
    position_data, velocity_data, callsign_data = extractStreams(derived_data, icao_iterator)

    icao_received, callsign, lat, lon, alt, groundspeed_m_s, vertical_m_s, track = None, None, None, None, None, None, None, None
    previous_dt_ts, d_dt_ts = None, None
    x, y = None, None
    position_ts, velocity_ts, callsign_ts = None, None, None

    position_data_pointer = 0
    velocity_data_pointer = 0
    callsign_data_pointer = 0

    if icao_iterator not in ts_data:
        if VERBOSE:
            log.info(f"Adding {icao_iterator} to ts_data")
        ts_data[icao_iterator] = []

    previous_total_specific_energy = None
    for ts in ts_list:

        if position_data:
            if position_data[position_data_pointer]["ts"] == ts:
                position = position_data[position_data_pointer]
                lat, lon, alt = position["lat"], position["lon"], position["alt"]
                if "x" in position and "y" in position:
                    x, y = position["x"], position["y"]
                else:
                    x, y, = None, None
                position_ts = ts
                if position_data_pointer < len(position_data) - 1:
                    position_data_pointer += 1

        if velocity_data:
            if velocity_data[velocity_data_pointer]["ts"] == ts:
                velocity = velocity_data[velocity_data_pointer]
                groundspeed_m_s, vertical_m_s = velocity["groundspeed"], velocity["vertical_rate"]
                track = velocity["track"]
                velocity_ts = ts
                if velocity_data_pointer < len(velocity_data) - 1:
                    velocity_data_pointer += 1

        if callsign_data:
            if callsign_data[callsign_data_pointer]["ts"] == ts:
                icao_callsign = callsign_data[callsign_data_pointer]
                callsign, icao_received = icao_callsign["callsign"], icao_callsign["icao"]
                callsign_ts = ts
                if callsign_data_pointer < len(callsign_data) - 1:
                    callsign_data_pointer += 1


        if clean_data:
            if ts is None or icao_received is None or callsign is None or lat is None or lon is None or alt is None or track is None:
                continue

            if groundspeed_m_s is None or vertical_m_s is None or position_ts is None or callsign_ts is None:
                continue

        dt_ts = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


        if groundspeed_m_s is not None and vertical_m_s is not None and alt is not None:
            aircraft_specific_kinetic_energy = 0.5 * (groundspeed_m_s ** 2 + vertical_m_s ** 2)
            aircraft_specific_potential_energy = alt * g
            aircraft_total_specific_energy = aircraft_specific_kinetic_energy + aircraft_specific_potential_energy

            if previous_total_specific_energy is None:
                d_specific_energy_s = 0

            else:
                # Compute time change
                d_dt_ts = (dt_ts - previous_dt_ts).total_seconds()

                if d_dt_ts > 0.1:
                    # Estimate a new specific energy rate of change only when enough time has elapsed.
                    # Otherwise retain the previous value to avoid noisy derivatives from very short intervals.
                    d_specific_energy_s = (aircraft_total_specific_energy - previous_total_specific_energy) / d_dt_ts


        else:
            d_specific_energy_s = None
            aircraft_total_specific_energy = None




        if x == -1 or y == -1 or x is None or y is None:



            ts_data[icao_iterator].append({"ts": ts,
                                           "icao": icao_received,
                                           "callsign": callsign,
                                           "lat": lat,
                                           "lon": lon,
                                           "alt": alt,
                                           "groundspeed": groundspeed_m_s,
                                           "vertical": vertical_m_s,
                                           "track": track,
                                           "specific_energy": aircraft_total_specific_energy,
                                           "d_specific_energy_s": d_specific_energy_s,
                                           "position_ts": position_ts,
                                           "velocity_ts": velocity_ts,
                                           "callsign_ts": callsign_ts})

        else:

            log.info(f"Position visible on image at {x}, {y}")
            ts_data[icao_iterator].append({"ts": ts,
                                           "icao": icao_received,
                                           "callsign": callsign,
                                           "lat": lat,
                                           "lon": lon,
                                           "alt": alt,
                                           "groundspeed": groundspeed_m_s,
                                           "vertical": vertical_m_s,
                                           "track": track,
                                           "specific_energy": aircraft_total_specific_energy,
                                           "d_specific_energy_s": d_specific_energy_s,
                                           "x": x,
                                           "y": y,
                                           "position_ts": position_ts,
                                           "velocity_ts": velocity_ts,
                                           "callsign_ts": callsign_ts})

        if aircraft_total_specific_energy is not None:
            previous_total_specific_energy = aircraft_total_specific_energy
            previous_dt_ts = dt_ts

    return ts_data

def reconstructFixedInformation(ts_data, icao_iterator, variable):

    if VERBOSE:
        log.info(f"Reconstructing fixed data for {icao_iterator}")
    aircraft_time_series = ts_data[icao_iterator]

    data = None
    for data_point in aircraft_time_series:
        if data_point[variable] is not None:
            data = data_point[variable]
            break

    for data_point in aircraft_time_series:
        if data_point[variable] is None and data is not None:
            data_point[variable] = data


    return ts_data

def interpolateLatLonAlt(ts_data, icao_iterator):


    if VERBOSE:
        log.info(f"Interpolating lat lon alt for {icao_iterator}")
    received_position_data_times = []



    aircraft_ts_data = ts_data[icao_iterator]

    for data_point in aircraft_ts_data:
        if data_point["position_ts"] == data_point["ts"]:
            received_position_data_times.append(data_point["ts"])





    for data_point in aircraft_ts_data:
        # Detect interpolated data
        position_ts = data_point["position_ts"]
        message_ts = data_point["ts"]
        if position_ts != message_ts:
            i = bisect.bisect_left(received_position_data_times, message_ts)
            earlier_index = i - 1 if i > 0 else None
            later_index = i if i < len(received_position_data_times) else None

            if earlier_index is None or later_index is None:
                continue

            earlier_ts = received_position_data_times[earlier_index]
            this_ts = data_point['ts']
            later_ts = received_position_data_times[later_index]

            lat, lon, alt = interpolateLatLonAltPoint(aircraft_ts_data, earlier_ts, this_ts, later_ts)
            data_point["lat"], data_point["lon"], data_point["alt"] = lat, lon, alt




    return ts_data

def interpolateLatLonAltPoint(ts_data, earlier_ts, this_ts, later_ts):

    earlier_ecef, later_ecef = None, None
    lat_earlier, lon_earlier, alt_earlier = None, None, None
    lat_later, lon_later, alt_later = None, None, None

    # Retrieve coordinates
    for point in ts_data:
        if point["ts"] == earlier_ts:
            lat_earlier, lon_earlier, alt_earlier = point["lat"], point["lon"], point["alt"]
            earlier_ecef = latLonAlt2ECEF(np.radians(lat_earlier), np.radians(lon_earlier), alt_earlier)
        if point["ts"] == later_ts:
            lat_later, lon_later, alt_later = point["lat"], point["lon"], point["alt"]
            later_ecef = latLonAlt2ECEF(np.radians(lat_later), np.radians(lon_later), alt_later)
            break


    vector = np.asarray(later_ecef) - np.asarray(earlier_ecef)
    time_delta_earlier_later = (datetime.datetime.fromisoformat(later_ts) - datetime.datetime.fromisoformat(earlier_ts)).total_seconds()
    time_delta_earlier_this = (datetime.datetime.fromisoformat(this_ts) - datetime.datetime.fromisoformat(earlier_ts)).total_seconds()

    x, y, z = earlier_ecef + (time_delta_earlier_this / time_delta_earlier_later) * vector

    lat_rads, lon_rads, alt = ecef2LatLonAlt(x, y, z)
    lat_deg, lon_deg = np.degrees(lat_rads), np.degrees(lon_rads)

    if VERBOSE:
        log.info(f"Interpolation")
        log.info(f"Earlier {earlier_ts} {lat_earlier:8.5f} {lon_earlier:8.5f} {alt_earlier:6.0f} ")
        log.info(f"This    {this_ts} {lat_deg:8.5f} {lon_deg:8.5f} {alt:6.0f} ")
        log.info(f"Later   {later_ts} {lat_later:8.5f} {lon_later:8.5f} {alt_later:6.0f} ")



    return lat_deg, lon_deg, alt

def interpolateVelocity(ts_data, icao_iterator):

    if VERBOSE:
        log.info(f"Interpolating velocity for {icao_iterator}")

    received_velocity_data_times = []



    aircraft_ts_data = ts_data[icao_iterator]

    for data_point in aircraft_ts_data:
        if data_point["velocity_ts"] == data_point["ts"]:
            received_velocity_data_times.append(data_point["ts"])





    for data_point in aircraft_ts_data:
        # Detect interpolated data
        velocity_ts = data_point["velocity_ts"]
        message_ts = data_point["ts"]
        if velocity_ts != message_ts:
            i = bisect.bisect_left(received_velocity_data_times, message_ts)
            earlier_index = i - 1 if i > 0 else None
            later_index = i if i < len(received_velocity_data_times) else None

            if earlier_index is None or later_index is None:
                continue

            earlier_ts = received_velocity_data_times[earlier_index]
            this_ts = data_point['ts']
            later_ts = received_velocity_data_times[later_index]

            groundspeed, vertical, track = interpolateVelocityPoint(aircraft_ts_data, earlier_ts, this_ts, later_ts)
            data_point["groundspeed"], data_point["vertical"], data_point["track"] = groundspeed, vertical, track


    return ts_data

def interpolateVelocityPoint(ts_data, earlier_ts, this_ts, later_ts):


    groundspeed_earlier, groundspeed_later = None, None
    vertical_earlier, vertical_later = None, None
    track_earlier, track_later = None, None


    # Retrieve velocities
    for point in ts_data:
        if point["ts"] == earlier_ts:
            groundspeed_earlier, vertical_earlier = point["groundspeed"], point["vertical"]
            track_earlier = np.radians(point["track"])
        if point["ts"] == later_ts:
            groundspeed_later, vertical_later = point["groundspeed"], point["vertical"]
            track_later = np.radians(point["track"])
            break

    d_groundspeed = groundspeed_later - groundspeed_earlier
    d_vertical = vertical_later - vertical_earlier

    # Convert ground tracks to vectors
    earlier_x, earlier_y = np.sin(track_earlier), np.cos(track_earlier)
    later_x, later_y = np.sin(track_later), np.cos(track_later)

    time_delta_later_earlier = (datetime.datetime.fromisoformat(later_ts) - datetime.datetime.fromisoformat(earlier_ts)).total_seconds()
    time_delta_this_earlier = (datetime.datetime.fromisoformat(this_ts) - datetime.datetime.fromisoformat(earlier_ts)).total_seconds()
    fraction = time_delta_this_earlier / time_delta_later_earlier

    groundspeed  = groundspeed_earlier + d_groundspeed * fraction
    vertical = vertical_earlier + d_vertical * fraction

    vx = earlier_x + fraction * (later_x - earlier_x)
    vy = earlier_y + fraction * (later_y - earlier_y)

    mag = np.sqrt(vx * vx + vy * vy)
    vx /= mag
    vy /= mag
    track = np.degrees(np.arctan2(vx,vy))

    if track < 0:
        track += 360.0

    if VERBOSE:
        log.info(f"Velocity interpolation")
        log.info(f"{earlier_ts} horizontal {groundspeed_earlier:6.2f} vertical {vertical_earlier:6.2f} track {np.degrees(track_earlier):6.2f}")
        log.info(f"{this_ts} horizontal {groundspeed:6.2f} vertical {vertical:6.2f} track {track:6.2f}")
        log.info(f"{later_ts} horizontal {groundspeed_later:6.2f} vertical {vertical_later:6.2f} track {np.degrees(track_later):6.2f}")


    return groundspeed, vertical, track

def computeTimeSeries(derived_data, clean_data=False):


    ts_data = {}

    # Initialise a dictionary for each aircraft's time series data
    for icao_iterator in derived_data:

        ts_list = extractTimestamps(derived_data, icao_iterator)
        ts_data = buildICAOTimeSeries(derived_data, icao_iterator, ts_list, ts_data, clean_data=clean_data)

    # Reconstruct missing data

    for icao_iterator in ts_data:

        reconstructFixedInformation(ts_data, icao_iterator, "callsign")
        reconstructFixedInformation(ts_data, icao_iterator, "icao")
        interpolateLatLonAlt(ts_data, icao_iterator)
        interpolateVelocity(ts_data, icao_iterator)


        pass

    return ts_data









def cprNLFunction(lat):

    lat = abs(lat)

    if lat < 10.47047130: return 59
    if lat < 14.82817437: return 58
    if lat < 18.18626357: return 57
    if lat < 21.02939493: return 56
    if lat < 23.54504487: return 55
    if lat < 25.82924707: return 54
    if lat < 27.93898710: return 53
    if lat < 29.91135686: return 52
    if lat < 31.77209708: return 51
    if lat < 33.53993436: return 50
    if lat < 35.22899598: return 49
    if lat < 36.85025108: return 48
    if lat < 38.41241892: return 47
    if lat < 39.92256684: return 46
    if lat < 41.38651832: return 45
    if lat < 42.80914012: return 44
    if lat < 44.19454951: return 43
    if lat < 45.54626723: return 42
    if lat < 46.86733252: return 41
    if lat < 48.16039128: return 40
    if lat < 49.42776439: return 39
    if lat < 50.67150166: return 38
    if lat < 51.89342469: return 37
    if lat < 53.09516153: return 36
    if lat < 54.27817472: return 35
    if lat < 55.44378444: return 34
    if lat < 56.59318756: return 33
    if lat < 57.72747354: return 32
    if lat < 58.84763776: return 31
    if lat < 59.95459277: return 30
    if lat < 61.04917774: return 29
    if lat < 62.13216659: return 28
    if lat < 63.20427479: return 27
    if lat < 64.26616523: return 26
    if lat < 65.31845310: return 25
    if lat < 66.36171008: return 24
    if lat < 67.39646774: return 23
    if lat < 68.42322022: return 22
    if lat < 69.44242631: return 21
    if lat < 70.45451075: return 20
    if lat < 71.45986473: return 19
    if lat < 72.45884545: return 18
    if lat < 73.45177442: return 17
    if lat < 74.43893416: return 16
    if lat < 75.42056257: return 15
    if lat < 76.39684391: return 14
    if lat < 77.36789461: return 13
    if lat < 78.33374083: return 12
    if lat < 79.29428225: return 11
    if lat < 80.24923213: return 10
    if lat < 81.19801349: return 9
    if lat < 82.13956981: return 8
    if lat < 83.07199445: return 7
    if lat < 83.99173563: return 6
    if lat < 84.89166191: return 5
    if lat < 85.75541621: return 4
    if lat < 86.53536998: return 3
    if lat < 87.00000000: return 2

    return 1


def cprNFunction(lat, isodd):
    nl = cprNLFunction(lat) - isodd

    if nl < 1:
        nl = 1

    return nl


def cprDlonFunction(lat, isodd, surface=0):

    if surface:
        return 90.0 / cprNFunction(lat, isodd)

    return 360.0 / cprNFunction(lat, isodd)


def decodeCPRAirborne(even_cprlat,
                        even_cprlon,
                        odd_cprlat,
                        odd_cprlon,
                        fflag):

    AIR_DLAT0 = 360.0 / 60.0
    AIR_DLAT1 = 360.0 / 59.0

    lat0 = float(even_cprlat)
    lat1 = float(odd_cprlat)

    lon0 = float(even_cprlon)
    lon1 = float(odd_cprlon)

    #
    # Compute latitude index j
    #
    j = int(np.floor(((59 * lat0 - 60 * lat1) / 131072.0) + 0.5))

    rlat0 = AIR_DLAT0 * ((j % 60) + lat0 / 131072.0)
    rlat1 = AIR_DLAT1 * ((j % 59) + lat1 / 131072.0)

    if rlat0 >= 270:
        rlat0 -= 360

    if rlat1 >= 270:
        rlat1 -= 360

    #
    # Latitude validity check
    #
    if (
        rlat0 < -90 or rlat0 > 90 or
        rlat1 < -90 or rlat1 > 90
    ):
        return -2, None, None

    #
    # Latitude zone check
    #
    if cprNLFunction(rlat0) != cprNLFunction(rlat1):
        return -1, None, None

    #
    # Compute longitude
    #
    if fflag:

        ni = cprNFunction(rlat1, 1)

        m = int(
            np.floor(
                (
                    (lon0 * (cprNLFunction(rlat1) - 1))
                    -
                    (lon1 * cprNLFunction(rlat1))
                ) / 131072.0
                + 0.5
            )
        )

        rlon = cprDlonFunction(rlat1, 1, 0) * (
            (m % ni) + lon1 / 131072.0
        )

        rlat = rlat1

    else:

        ni = cprNFunction(rlat0, 0)

        m = int(
            np.floor(
                (
                    (lon0 * (cprNLFunction(rlat0) - 1))
                    -
                    (lon1 * cprNLFunction(rlat0))
                ) / 131072.0
                + 0.5
            )
        )

        rlon = cprDlonFunction(rlat0, 0, 0) * (
            (m % ni) + lon0 / 131072.0
        )

        rlat = rlat0

    #
    # Normalize longitude to -180..180
    #
    rlon -= np.floor((rlon + 180.0) / 360.0) * 360.0

    return 0, rlat, rlon


class ADSBRecorder(multiprocessing.Process):

    """ Record ADS-B aircraft information from dump1090.

        Receives SBS messages from dump1090, converts aircraft positions into
        RMS sky coordinates, records aircraft positions, velocities and ICAO/callsign
        observations, and periodically writes ADS-B data products to disk for
        later upload and processing.
        """

    def __init__(self, config):
        """ Initialise the ADS-B recorder.

           Arguments:
               config: [Config] RMS configuration object.
           """

        # Do this first for inheritance handling
        super().__init__()

        self.exit = multiprocessing.Event()

        self.config = config                    # Config object
        self.positions = {}                        # Store for received location data
        self.icao_callsign_mapping = []         # Store for received icao to callsign information
        self.velocity = {}                      # Store for received velocity information
        self.s = None                           # Connection object
        self.buffer = b""                       # Buffer
        self.tc_counts = {}                     # Statistics counter
        self.position_cache = {}                # Cache for aircraft information
        self.field_length_list = []
        self.latest_timestamp = None
        self.last_log_line = ""
        self.history = {}

        # Initalise and read the platepar
        platepar_path = os.path.join(config.config_file_path, config.platepar_name)
        if os.path.exists(platepar_path):
            self.pp = Platepar()
            self.pp.read(platepar_path, use_flat=config.use_flat)


        self.recording_start = None             # Start time for the recording


    def __getstate__(self):

        """ Return a picklable representation of the ADS_B recorder.

        Required under the 'forkserver'/'spawn' start methods (the default on Linux from
        Python 3.14), where .start() pickles self and sends it to the child.
        """

        state = self.__dict__.copy()
        return state

    def getNow(self):

        return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def toTable(self):


        header = ""
        message_fields, message_headers = [], []
        log_line = "\n"


        for aircraft_counter, (iterator, aircraft) in enumerate(self.position_cache.items()):
            message_fields = []
            message_headers = []
            message_headers.append("ICAO")
            message_fields.append(iterator)


            callsign = ""
            message_headers.append("Callsign")
            if 'ident' in self.position_cache[iterator]:
                if 'callsign' in self.position_cache[iterator]['ident']:
                    callsign = self.position_cache[iterator]['ident']['callsign']
            message_fields.append(f"{callsign}")

            message_headers.append("Baro(ft)")
            alt_baro_message = ""
            if 'position' in self.position_cache[iterator]:
                if 'alt_baro' in self.position_cache[iterator]['position']:
                    alt_baro = self.position_cache[iterator]['position']['alt_baro']
                    if alt_baro is not None:
                        alt_baro_message = f"{alt_baro}"

            message_fields.append(alt_baro_message)

            message_headers.append("Delta(ft)")
            geom_delta_message = ""
            if 'velocity' in self.position_cache[iterator]:
                if 'geom_delta' in self.position_cache[iterator]['velocity']:
                    geo_delta = self.position_cache[iterator]['velocity']['geom_delta']
                    if geo_delta is not None:
                        if geo_delta > 0:
                            sign = "+"
                        else:
                            sign = "-"
                            geo_delta = - geo_delta
                        geom_delta_message = f"{sign}{geo_delta}"
            message_fields.append(geom_delta_message)

            message_headers.append("GNSS(ft)")
            gnss_message = ""
            if 'position' in self.position_cache[iterator]:
                if 'alt_gnss' in self.position_cache[iterator]['position']:
                    alt_gnss = self.position_cache[iterator]['position']['alt_gnss']
                    if alt_gnss is not None:
                        gnss_message = f"{alt_gnss:.0f}"
            message_fields.append(gnss_message)

            message_headers.append("Spd(kts)")
            groundspeed_message  = ""
            if 'velocity' in self.position_cache[iterator]:
                if 'groundspeed' in self.position_cache[iterator]['velocity']:
                    groundspeed = self.position_cache[iterator]['velocity']['groundspeed']
                    if groundspeed is not None:
                        groundspeed_message = f"{groundspeed:.1f}"
            message_fields.append(groundspeed_message)

            message_headers.append("Track(deg)")
            groundspeed_message  = ""
            if 'velocity' in self.position_cache[iterator]:
                if 'track' in self.position_cache[iterator]['velocity']:
                    groundspeed = self.position_cache[iterator]['velocity']['track']
                    if groundspeed is not None:
                        groundspeed_message = f"{groundspeed:.1f}"
            message_fields.append(groundspeed_message)


            message_headers.append("Lat(deg)")
            lat_message = ""
            if 'position' in self.position_cache[iterator]:
                if 'lat' in self.position_cache[iterator]['position'] and "lon" in self.position_cache[iterator]['position']:
                    lat, lon = self.position_cache[iterator]['position']['lat'], self.position_cache[iterator]['position']['lon']
                    if lat is not None and lon is not None:
                        lat = self.position_cache[iterator]['position']['lat']
                        lat_message += f"{lat:8.6f}"
            message_fields.append(lat_message)


            message_headers.append("Lon(deg)")
            lon_message = ""
            if 'position' in self.position_cache[iterator]:
                if 'lat' in self.position_cache[iterator]['position'] and 'lon' in self.position_cache[iterator]['position']:
                    lat, lon = self.position_cache[iterator]['position']['lat'], self.position_cache[iterator]['position']['lon']
                    if lat is not None and lon is not None:
                        lon = self.position_cache[iterator]['position']['lon']
                        lon_message += f"{lon:8.6f}"
            message_fields.append(lon_message)


            message, header = "", ""

            for i, field in enumerate(message_fields):

                field_length = len(field)
                header_length = len(message_headers[i])

                if i >= len(self.field_length_list):
                    self.field_length_list.append(0)
                if self.field_length_list[i] < field_length:
                    self.field_length_list[i] = field_length
                if self.field_length_list[i] < header_length:
                    self.field_length_list[i] = header_length

                message += " " * ((self.field_length_list[i]) - field_length) + field + " "
                header += message_headers[i] + " " * ((self.field_length_list[i]) - header_length) + " "
                separator = "-" * len(header)

            if aircraft_counter == 0:
                log_line += f"\n{header}\n"
                log_line += f"{separator}\n"

            log_line += f"{message}\n"

            if aircraft_counter == len(self.position_cache) - 1:
                log_line += f"{separator}\n"

        data_changed = (self.last_log_line != log_line)
        self.last_log_line = log_line

        return log_line, data_changed

    def store(self, icao, frame_ts, write_to_log=True):


        data_table, data_changed = self.toTable()
        if write_to_log and data_changed:
            log.info(data_table)


            if icao not in self.history:
                self.history[icao] = {}

            with open(os.path.join(self.config.data_dir, "adsb.txt"), "w") as f:

                f.write(f"{self.getNow()}\n{data_table}")


            self.history[icao][frame_ts] = self.position_cache[icao]

        pass

    def writeOutADSBData(self, pp=None, force_flush=True):

        """ Write ADS-B data products to disk.

        Arguments:
            config: [Config] RMS configuration object.
            recording_start_time: [str] Recording start time in the format
                YYYYMMDD_HHMMSS.
            positions: [dict] Aircraft positon data keyed by ICAO code.
            icao_call_mapping:/ [list] List of observed ICAO to callsign mappings.

        Return:
            file_paths: [list] Paths of the generated JSON files.
        """

        adsb_directory = os.path.expanduser(os.path.join(self.config.data_dir, self.config.adsb_dir))
        os.makedirs(adsb_directory, exist_ok=True)

        adsb_positions_filename = f"{self.config.stationID}_{self.recording_start}_ads-b.json"
        adsb_path = os.path.join(adsb_directory, adsb_positions_filename)

        if self.history:

            with open(adsb_path, "w") as fp:
                json.dump(self.history, fp, indent=4)
                if force_flush:
                    fp.flush()
                    os.fsync(fp.fileno())
        else:
            log.info("No ADS-B position data to write")

        return adsb_path

    def deriveFromADSB(positions, icao_callsign_mapping, velocity):

        derived_data = {}

        all_icaos = set(positions.keys())
        all_icaos.update(velocity.keys())

        for icao in sorted(all_icaos):
            derived_data[icao] = {
                "positions": positions.get(icao, []),
                "velocity": velocity.get(icao, []),
                "callsigns": [obs for obs in icao_callsign_mapping if obs["icao"] == icao]
            }

        time_series_by_icao = computeTimeSeries(derived_data)

        return derived_data, time_series_by_icao



    def parseBeastMode(self):

        tx_type = 0

        # Sync to frame
        i = self.buffer.find(b"\x1a")

        if i == -1:
            # Consume the buffer
            self.buffer = b""
            return tx_type

        # Discard garbage before first frame
        if i > 0:
            self.buffer = self.buffer[i:]



        if len(self.buffer) < 2:
            return tx_type

        frame_type = self.buffer[1]

        #log.info(f"Frame type: {frame_type}")
        if frame_type == BEAST_MODE_SHORT_FRAME:
            frame_length = 1 + 1 + 6 + 1 + 7

        elif frame_type == BEAST_MODE_LONG_FRAME:
            frame_length = 1 + 1 + 6 + 1 + 14

        else:
            self.buffer = self.buffer[1:]
            return tx_type



        if len(self.buffer) < frame_length:
            return tx_type

        decoded = bytearray()
        ptr = 0

        while ptr < len(self.buffer) and len(decoded) < frame_length:

            b = self.buffer[ptr]

            if b == 0x1a:

                if ptr + 1 < len(self.buffer) and self.buffer[ptr + 1] == 0x1a:
                    decoded.append(0x1a)
                    ptr += 2
                    continue

                if ptr > 0:
                    break

            decoded.append(b)
            ptr += 1
        if LOG_ESCAPING and ptr != len(decoded):
            log.info(f"Escaping activated wire={ptr} decoded={len(decoded)}")
            log.info(f"Buffer            {self.buffer[:ptr]}")
            log.info(f"Decoded {decoded}")
        frame = decoded[:frame_length]
        frame_ts = int.from_bytes(frame[2:8], byteorder='big') / TICKS_PER_SECOND

        if VVERBOSE:
            log.info(f"Buffer before splitting {len(self.buffer)} {self.buffer}")
        self.buffer = self.buffer[ptr:]
        if VVERBOSE:
            log.info(f"Buffer after  splitting {len(self.buffer)} {self.buffer}")
        #log.info(f"frame  {frame}")
        #log.info(f"buffer {self.buffer}")



        if frame_length == 16:

            signal = frame[8]
            payload = frame[9:16]

        elif frame_length == 23:

            signal = frame[8]
            payload = frame[9:23]

        else:
            log.info(f"Why are we here?")
            pass



        df = payload[0] >> 3

        #log.info(f"df={df} {frame}")

        if df == DF_ADSB_EXTENDED_SQUITTER:
            tc = (payload[4] >> 3) & 0x1F
            if tc not in self.tc_counts:
                self.tc_counts[tc] = 0
            self.tc_counts[tc] += 1

            #log.info(f"TC {self.tc_counts}")
            #log.info(f"tc={tc}")
            icao = payload[1:4].hex().upper()

            if icao not in self.position_cache:
                self.position_cache[icao] = {}



            if isAircraftIdentification(tc):

                chars = []

                bits = int.from_bytes(payload[5:11], byteorder='big')

                for i in range(8):
                    shift = (7 - i) * 6

                    code = (bits >> shift) & 0x3F

                    chars.append(code)

                    callsign = ""


                for code in chars:

                    if code < len(CALLSIGN_TABLE):
                        callsign += CALLSIGN_TABLE[code]
                    else:
                        callsign += "?"


                callsign = callsign.replace("#", "").replace("_", "").strip()



                if icao not in self.position_cache:
                    self.position_cache[icao] = {}

                if 'ident' not in self.position_cache[icao]:
                    self.position_cache[icao]['ident'] = {}

                dt = self.getNow()
                self.position_cache[icao]['ident']['callsign'] = callsign
                self.position_cache[icao]['ident']['callsign_ts'] = frame_ts
                self.position_cache[icao]['ident']['callsign_dt'] = dt
                self.position_cache[icao]['ts'] = self.position_cache[icao]['ident']['callsign_ts']
                self.position_cache[icao]['dt'] = dt
                self.store(icao, dt)

            elif isAirbornePosition(tc):

                if icao not in self.position_cache:
                    self.position_cache[icao] = {}


                # Decode altitude
                alt_field = ((payload[5] << 4) | (payload[6] >> 4)) & 0x0FFF
                q_bit = (alt_field >> 4) & 1
                if q_bit == 1:
                    alt_ft = (((alt_field & 0x0FE0) >> 1) | (alt_field & 0x000F)) * 25 - 1000
                    #log.info(f"{icao} altitude updated to {alt_ft}")
                else:
                    alt_ft = None


                f_flag = getbits(payload, 54, 54)

                cpr_lat = getbits(payload, 55, 71)

                cpr_lon = getbits(payload, 72, 88)




                if 'position' not in self.position_cache[icao]:
                    self.position_cache[icao]['position'] = {}



                if f_flag not in self.position_cache[icao]:
                    self.position_cache[icao]['position'][f_flag] = {}

                self.position_cache[icao]['position'][f_flag]["lat"] = cpr_lat
                self.position_cache[icao]['position'][f_flag]["lon"] = cpr_lon
                self.position_cache[icao]['position'][f_flag]["ts"] = frame_ts
                self.position_cache[icao]['position'][f_flag]["dt"] = self.getNow()

                if 0 in self.position_cache[icao]['position'] and 1 in self.position_cache[icao]['position']:

                    bit_0_age = isoAgeSeconds(self.getNow(), self.position_cache[icao]['position'][0]["dt"])
                    bit_1_age = isoAgeSeconds(self.getNow(), self.position_cache[icao]['position'][1]["dt"])

                    if max(bit_0_age, bit_1_age) < 180:

                        aircraft = self.position_cache[icao]['position']
                        rc, lat, lon = decodeCPRAirborne(aircraft[0]["lat"], aircraft[0]["lon"], aircraft[1]["lat"], aircraft[1]["lon"], 0)

                        self.position_cache[icao]['position']['lat'] = lat
                        self.position_cache[icao]['position']['lon'] = lon

                if 'velocity' in self.position_cache[icao]:
                    if "geom_delta" in self.position_cache[icao]['velocity']:
                        if alt_ft is not None and self.position_cache[icao]['velocity']['geom_delta'] is not None:
                            self.position_cache[icao]['position']["alt_gnss"] = alt_ft + self.position_cache[icao]['velocity']["geom_delta"]

                self.position_cache[icao]['position']["alt_baro"] = alt_ft
                self.position_cache[icao]['position']["ts"] = frame_ts
                self.position_cache[icao]['position']["dt"] = self.getNow()
                self.position_cache[icao]['ts'] = frame_ts
                dt = self.position_cache[icao]['position']['dt']
                self.position_cache[icao]['dt'] = dt
                self.store(icao, dt)

            elif isAirborneVelocity(tc):

                subtype = payload[4] & 0x07
                ew_dir = (payload[5] >> 2) & 0x01

                ew_raw = ((payload[5] & 0x03) << 8) | payload[6]
                ns_dir = (payload[7] >> 7) & 0x01
                ns_raw = ((payload[7] & 0x7F) << 3) | (payload[8] >> 5)

                ew_vel, ns_vel = ew_raw - 1, ns_raw - 1

                if ew_dir:
                    ew_vel = -ew_vel

                if ns_dir:
                    ns_vel = -ns_vel

                groundspeed = np.sqrt(ew_vel ** 2 + ns_vel ** 2)

                track = np.degrees(np.arctan2(ew_vel, ns_vel))

                if track < 0:
                    track += 360

                vert_rate_source = (payload[8] >> 3) & 0x01
                vert_rate_sign = (payload[8] >> 2) & 0x01

                vert_rate_raw = (
                        ((payload[8] & 0x03) << 7)
                        | (payload[9] >> 1)
                )

                if vert_rate_raw:
                    vert_rate = (vert_rate_raw - 1) * 64

                    if vert_rate_sign:
                        vert_rate = -vert_rate
                else:
                    vert_rate = None

                geom_delta_sign = (payload[10] >> 7) & 0x01

                geom_delta_raw = payload[10] & 0x7F

                if geom_delta_raw:
                    geom_delta = (geom_delta_raw - 1) * 25

                    if geom_delta_sign:
                        geom_delta = -geom_delta
                else:
                    geom_delta = None


                if icao not in self.position_cache:
                    self.position_cache[icao] = {}

                if 'velocity' not in self.position_cache[icao]:
                    self.position_cache[icao]['velocity'] = {}


                dt = self.getNow()
                self.position_cache[icao]['velocity']['geom_delta'] = geom_delta
                self.position_cache[icao]['velocity']['groundspeed'] = groundspeed
                self.position_cache[icao]['velocity']['track'] = track
                self.position_cache[icao]['velocity']['vert_rate'] = vert_rate
                self.position_cache[icao]['velocity']['dt'] = dt
                self.position_cache[icao]['velocity']['ts'] = frame_ts
                self.position_cache[icao]['ts'] = frame_ts
                self.position_cache[icao]['dt'] = dt
                self.store(icao, dt)

            elif isGNSSPosition(tc):
                log.info(f"ICAO={payload[1:4].hex().upper()} tc={tc} GNSS pl={payload.hex()}")
                gnss_alt = ((payload[5] << 4) | (payload[6] >> 4)) & 0x0FFF
                log.info(f"Altitude from GNSS message {gnss_alt}")

            elif isAircraftSurfacePosition(tc):
                log.info(f"ICAO={payload[1:4].hex().upper()} tc={tc} Position pl={payload.hex()}")

            else:
                pass



        return tx_type

    def doIteration(self):
        """ Process incoming SBS messages from dump1090.

        Reads data from the dump1090 SBS socket, decodes received messages,
        records aircraft positions, velocity and ICAO/callsign observations, and
        writes accumulated data products to disk at hourly intervals.
        """

        # Current UTC time used for recording block management
        # and output filename generation.
        time_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Write out cached data each hour
        if self.recording_start is None:

            # If no start time is passed in, then create a new time reference
            self.recording_start = time_now
            log.info(f"Starting first recording block at {self.recording_start}")

        else:

            # Extract hours from times
            current_hour = time_now[:SLICE_LENGTH]
            recording_start_hour = self.recording_start[:SLICE_LENGTH]

            # Create a new recording block each hour
            if current_hour != recording_start_hour:
                next_start_time = time_now

                # Write to disk
                log.info(f"Writing out ADS-B data between {self.recording_start} and {next_start_time}")
                self.writeOutADSBData(pp=self.pp)

                # After writing out, reset the history
                self.history = {}


                # Reset the recording start time variables
                self.recording_start = next_start_time
                log.info(f"Starting a new block at {self.recording_start}")

                # Clear the position cache


        # Pull in data from the socket, avoiding getting stuck waiting for data
        data = None
        try:
            data = self.s.recv(4096)

        except Exception as e:
            log.info(f"Creating new connection to {self.config.dump1090_host}:{self.config.dump1090_port}")

            if self.s is not None:
                self.s.close()
                self.s = None

            while self.s is None:
                try:
                    self.s = socket.create_connection((self.config.dump1090_host, self.config.dump1090_port), timeout=60)
                    log.info(f"Connection to {self.config.dump1090_host}:{self.config.dump1090_port} opened.")

                except OSError as e:
                    log.info(
                        f"Connection to {self.config.dump1090_host}:{self.config.dump1090_port} failed "
                        f"({e}). Waiting for retry."
                    )
                    time.sleep(60)




        if not data:
            # Add a sleep to avoid busy waits - this only executes when there is no data

            return self.buffer, self.recording_start, self.positions, self.icao_callsign_mapping, self.velocity

        # Add data to the buffer
        self.buffer +=  data


        if int(self.config.dump1090_port) == DUMP1090_BEAST_MODE_PORT:
            if VVERBOSE:
                log.info(f"Buffer {len(self.buffer)} {self.buffer}")
            while self.buffer:
                self.parseBeastMode()

        else:
            log.info(f"Unrecognised port number {self.config.dump1090_port}")
            tx_type = 0
            self.buffer == ""

        # Route and decode received data





        return self.buffer, self.recording_start, self.positions, self.icao_callsign_mapping, self.velocity

    def start(self):

        """Starts the ADS-B recorder"""
        super().start()
        log.info("ADS-B recorder has started")


    def stop(self):

        """ Stops the ADS-B recorder. """
        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("ADS-B recorder has stopped")




    def run(self,):

        """Run the ADS-B processing loop"""

        # Try to connect until exit
        self.s = None
        while not self.exit.is_set():
            try:
                self.s = socket.create_connection((self.config.dump1090_host, self.config.dump1090_port), timeout=120)
                log.info(f"Connection to {self.config.dump1090_host}:{self.config.dump1090_port} opened.")
                break
            except OSError as e:

                log.info(
                    f"Connection to "
                    f"{self.config.dump1090_host}:"
                    f"{self.config.dump1090_port} failed "
                    f"({e}). Waiting for retry."
                )

                time.sleep(50)



        # Run until exit
        while True:

            self.doIteration()

            if self.exit.is_set():
                log.info("ADS-B loop is exiting")
                break

        log.info("ADS-B loop has exited")

        try:
            self.s.close()


            log.info("Closed dump1090 connection")
        except Exception:
            log.info("Could not close dump1090 connection")
            pass

        log.info("ADS-B recorder has finished - writing out cached data")
        self.writeOutADSBData(force_flush=True)

    def flush(self):
        """Write all cached ADS-B data to disk.

        Cached parsed ads-b messages are written
        immediately without affecting the current recording session.
        """

        log.info("Flushing cached ADS-B data")
        self.writeOutADSBData(force_flush=True)

    def archiveData(self, captured_path, remove_files=True):
        """ Archive all ADS-B data files, and optionally the files

        Arguments:
            captured_path: [str] Path to the capture directory for the night

        Keyword arguments:
            remove_files: [bool] Optional, default true, remove files from ADS-B directory after archiving

        Return:
            archive_name: [str] Name of the generated archive.
        """

        archive_name = f"{os.path.basename(captured_path)}_adsb.tar.bz2"
        archive_path = os.path.join(captured_path, archive_name)
        adsb_path = os.path.expanduser(os.path.join(self.config.data_dir, self.config.adsb_dir))
        log.info(f"Saving ADS-B data from {adsb_path} to an archive to {archive_path}")


        self.flush()

        adsb_files_list = os.listdir(adsb_path)
        log.info(f"Collecting {len(adsb_files_list)} ADS-B data files for archiving")
        with tarfile.open(archive_path, "w:bz2") as tar:
            adsb_files = []

            for filename in adsb_files_list:
                adsb_full_path = os.path.join(adsb_path, filename)

                if os.path.isfile(adsb_full_path):
                    tar.add(adsb_full_path, arcname=filename)
                    adsb_files.append(adsb_full_path)



        if remove_files:
            for adsb_full_path in adsb_files:
                os.remove(adsb_full_path)

        return archive_name


### Test code / command line launch code below

def loopADSB(config, pp, host=None, port=None):

    """ Run the ADS-B processing loop without multiprocessing.

    Connects to a dump1090 SBS data source, receives and processes
    incoming ADS-B messages, records aircraft positions, velocities and ICAO/callsign
    observations, and periodically writes data products to disk.

    This function is primarily intended for testing and debugging the
    ADS-B processing pipeline without using the ADSBRecorder process
    wrapper.

    Arguments:
        config: [Config] RMS configuration object.
        pp: [Platepar] Station platepar object.

    Keyword arguments:
        host: [str] dump1090 host. If None, the value from the RMS
            configuration is used.
        port: [int] dump1090 port. If None, the value from the RMS
            configuration is used.

    Return:
        None
    """



    adsb_recorder = ADSBRecorder(config)

    # Loop forever
    while True:
        adsb_recorder.doIteration()








if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Receive, process and store ADS-B data from dump1090""")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('--host', metavar='HOST', help="""dump1090 host """)

    arg_parser.add_argument('--port', metavar='PORT', help="""dump1090 port """)

    arg_parser.add_argument( '--testfile',metavar='FILE', help='Read SBS messages from a file instead of a dump1090 socket.')

    arg_parser.add_argument("--testprocess", metavar='SECONDS', type=int, help="Run the ADS-B process for the given number of seconds")

    arg_parser.add_argument("--testtimeseries", metavar='DERIVEDPATH', type=str, help="Create a time series from a json containing derived data")

    arg_parser.add_argument('-v', '--verbose', action="store_true", help="""Run in verbose mode for debugging""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # set verbose mode
    VERBOSE = cml_args.verbose

    # Load the config
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    platepar_path = os.path.join(config.config_file_path, config.platepar_name)

    pp = Platepar()
    pp.read(config.platepar_name)

    if cml_args.testfile:
        with open(os.path.expanduser(cml_args.testfile)) as f:
            replay_data = [line.strip() for line in f if line.strip()]

    host, port = cml_args.host, cml_args.port

    # If no command line arguments given, then use settings in the config file
    config.dump1090_host = config.dump1090_host if host is None else host
    config.dump1090_port = config.dump1090_port if port is None else port


    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    log.info(f"For {config.stationID} starting ADS-B logging with {host}:{port}.")
    log.info(f"Logging to {config.data_dir}")

    if cml_args.testprocess:
        if config.adsb_enabled:
            log.info(f"ADS-B enabled in config, so starting as a process for {cml_args.testprocess} seconds.")
            adsb_recorder = ADSBRecorder(config)
            adsb_recorder.start()
            time.sleep(cml_args.testprocess)
            adsb_recorder.stop()
        else:
            log.info("ADSB not enabled, not starting")

    else:
        loopADSB(config, pp, host=host, port=port)