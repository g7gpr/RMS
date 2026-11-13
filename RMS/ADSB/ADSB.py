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

"""ADSB aircraft tracking support for RMS.

Receives SBS messages from dump1090, converts aircraft locations to
azimuth/elevation, projects them into image coordinates using the station
platepar, and records hourly aircraft track files and ICAO to callsign mapping
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
# Length of slice for time stripping

# 11 - hourly
# 12 - ten minutes
# 13 - one minute

SLICE_LENGTH = 11

# Constants

# Acceleration due to gravity
g = 9.80665

# Unit conversion constant
FT_TO_M = 0.3048
KT_TO_MS = 0.514444

# SBS message type
SBS_MESSAGE_TYPE = "MSG"

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




log = getLogger("rmslogger")


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

def writeOutADSBData(config,
                     recording_start_time,
                     tracks,
                     icao_call_mapping,
                     velocity,
                     force_flush=False):

    """ Write ADS-B data products to disk.

    Arguments:
        config: [Config] RMS configuration object.
        recording_start_time: [str] Recording start time in the format
            YYYYMMDD_HHMMSS.
        tracks: [dict] Aircraft track data keyed by ICAO code.
        icao_call_mapping:/ [list] List of observed ICAO to callsign mappings.

    Return:
        file_paths: [list] Paths of the generated JSON files.
    """

    adsb_directory = os.path.expanduser(os.path.join(config.data_dir, config.adsb_dir))
    os.makedirs(adsb_directory, exist_ok=True)

    adsb_filename = f"{config.stationID}_{recording_start_time}_adsb.json"
    adsb_path = os.path.join(adsb_directory, adsb_filename)

    if tracks:

        with open(adsb_path, "w") as fp:
            json.dump(tracks, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())
    else:
        log.info("No ADSB data to write")

    icao_call_filename = f"{config.stationID}_{recording_start_time}_icao_call.json"
    icao_call_path = os.path.join(adsb_directory, icao_call_filename)

    if icao_call_mapping:

        with open(icao_call_path, "w") as fp:
            json.dump(icao_call_mapping, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())
    else:
        log.info("No ICAO to callsign mapping data to write")

    velocity_filename = f"{config.stationID}_{recording_start_time}_velocity.json"
    velocity_path = os.path.join(adsb_directory, velocity_filename)

    if velocity:

        with open(velocity_path, "w") as fp:
            json.dump(velocity, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())
    else:
        log.info("No velocity data to write")

    adsb_derived_filename = f"{config.stationID}_{recording_start_time}_adsb_derived.json"
    adsb_derived_path = os.path.join(adsb_directory, adsb_derived_filename)

    time_series_filename = f"{config.stationID}_{recording_start_time}_adsb_time_series.json"
    time_series_path = os.path.join(adsb_directory, time_series_filename)

    # Produce derived data
    adsb_derived_data, time_series_data = deriveFromADSB(tracks, icao_call_mapping, velocity)

    if adsb_derived_data:

        with open(adsb_derived_path, "w") as fp:
            json.dump(adsb_derived_data, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())

    if time_series_data:
        with open(time_series_path, "w") as fp:
            json.dump(time_series_data, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())

    else:
        log.info("No adsb derived data to write")


    return [adsb_path, icao_call_path, velocity_path]

def deriveFromADSB(tracks, icao_callsign_mapping, velocity):

    derived_data = {}

    all_icaos = set(tracks.keys())
    all_icaos.update(velocity.keys())


    for icao in sorted(all_icaos):

        derived_data[icao] = {
            "heading": tracks.get(icao, []),
            "velocity": velocity.get(icao, []),
            "callsigns": [obs for obs in icao_callsign_mapping if obs["icao"] == icao]
        }

    time_series_by_icao = computeTimeSeries(derived_data)


    return derived_data, time_series_by_icao

def extractStreams(derived_data, icao):

    track_data = derived_data[icao]["heading"]
    velocity_data = derived_data[icao]["velocity"]
    callsign_data = derived_data[icao]["callsigns"]

    return track_data, velocity_data, callsign_data

def extractTimestamps(derived_data, icao):



    # Get the data streams from the derived data
    track_data, velocity_data, callsign_data = extractStreams(derived_data, icao)

    # Get every timestamp for this icao
    ts_list = []

    for entry in track_data:
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
    track_data, velocity_data, callsign_data = extractStreams(derived_data, icao_iterator)

    icao_received, callsign, lat, lon, alt, groundspeed_m_s, vertical_m_s, heading = None, None, None, None, None, None, None, None
    previous_dt_ts = None
    x, y = None, None
    track_ts, velocity_ts, callsign_ts = None, None, None

    track_data_pointer = 0
    velocity_data_pointer = 0
    callsign_data_pointer = 0

    if icao_iterator not in ts_data:
        if VERBOSE:
            log.info(f"Adding {icao_iterator} to ts_data")
        ts_data[icao_iterator] = []

    previous_total_specific_energy = None
    for ts in ts_list:

        if track_data:
            if track_data[track_data_pointer]["ts"] == ts:
                track = track_data[track_data_pointer]
                lat, lon, alt = track["lat"], track["lon"], track["alt"]
                if "x" and "y" in track:
                    x, y = track["x"], track["y"]
                else:
                    x, y, = -1, -1
                track_ts = ts
                if track_data_pointer < len(track_data) - 1:
                    track_data_pointer += 1

        if velocity_data:
            if velocity_data[velocity_data_pointer]["ts"] == ts:
                velocity = velocity_data[velocity_data_pointer]
                groundspeed_m_s, vertical_m_s = velocity["groundspeed"], velocity["vertical_rate"]
                heading = velocity["heading"]
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
            if ts is None or icao_received is None or callsign is None or lat is None or lon is None or alt is None or heading is None:
                continue

            if groundspeed_m_s is None or vertical_m_s is None or track_ts is None or callsign_ts is None:
                continue

        dt_ts = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        if groundspeed_m_s is not None and alt is not None and previous_dt_ts is not None:
            aircraft_specific_kinetic_energy = 0.5 * (groundspeed_m_s ** 2 + vertical_m_s ** 2)
            aircraft_specific_potential_energy = alt * g
            aircraft_total_specific_energy = aircraft_specific_kinetic_energy + aircraft_specific_potential_energy
            d_dt_ts = (dt_ts - previous_dt_ts).total_seconds()


            if previous_total_specific_energy is None:
                previous_total_specific_energy = aircraft_total_specific_energy
                d_specific_energy_s = 0
            else:
                d_specific_energy = aircraft_total_specific_energy - previous_total_specific_energy
                d_specific_energy_s = d_specific_energy / d_dt_ts

        else:
            d_specific_energy_s = None
            aircraft_total_specific_energy = None




        if x == -1 and y == -1:



            ts_data[icao_iterator].append({"ts": ts,
                                           "icao": icao_received,
                                           "callsign": callsign,
                                           "lat": lat,
                                           "lon": lon,
                                           "alt": alt,
                                           "groundspeed": groundspeed_m_s,
                                           "vertical": vertical_m_s,
                                           "heading": heading,
                                           "specific_energy": aircraft_total_specific_energy,
                                           "d_specific_energy_s": d_specific_energy_s,
                                           "track_ts": track_ts,
                                           "velocity_ts": velocity_ts,
                                           "callsign_ts": callsign_ts})

        else:

            ts_data[icao_iterator].append({"ts": ts,
                                           "icao": icao_received,
                                           "callsign": callsign,
                                           "lat": lat,
                                           "lon": lon,
                                           "alt": alt,
                                           "groundspeed": groundspeed_m_s,
                                           "vertical": vertical_m_s,
                                           "heading": heading,
                                           "specific_energy": aircraft_total_specific_energy,
                                           "d_specific_energy_s": d_specific_energy_s,
                                           "x": x,
                                           "y": y,
                                           "track_ts": track_ts,
                                           "velocity_ts": velocity_ts,
                                           "callsign_ts": callsign_ts})

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
    received_track_data_times = []



    aircraft_ts_data = ts_data[icao_iterator]

    for data_point in aircraft_ts_data:
        if data_point["track_ts"] == data_point["ts"]:
            received_track_data_times.append(data_point["ts"])





    for data_point in aircraft_ts_data:
        # Detect interpolated data
        track_ts = data_point["track_ts"]
        message_ts = data_point["ts"]
        if track_ts != message_ts:
            i = bisect.bisect_left(received_track_data_times, message_ts)
            earlier_index = i - 1 if i > 0 else None
            later_index = i if i < len(received_track_data_times) else None

            if earlier_index is None or later_index is None:
                continue

            earlier_ts = received_track_data_times[earlier_index]
            this_ts = data_point['ts']
            later_ts = received_track_data_times[later_index]

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

            groundspeed, vertical, heading = interpolateVelocityPoint(aircraft_ts_data, earlier_ts, this_ts, later_ts)
            data_point["groundspeed"], data_point["vertical"], data_point["heading"] = groundspeed, vertical, heading


    return ts_data

def interpolateVelocityPoint(ts_data, earlier_ts, this_ts, later_ts):


    groundspeed_earlier, groundspeed_later = None, None
    vertical_earlier, vertical_later = None, None
    heading_earlier, heading_later = None, None


    # Retrieve velocities
    for point in ts_data:
        if point["ts"] == earlier_ts:
            groundspeed_earlier, vertical_earlier = point["groundspeed"], point["vertical"]
            heading_earlier = np.radians(point["heading"])
        if point["ts"] == later_ts:
            groundspeed_later, vertical_later = point["groundspeed"], point["vertical"]
            heading_later = np.radians(point["heading"])
            break

    d_groundspeed = groundspeed_later - groundspeed_earlier
    d_vertical = vertical_later - vertical_earlier

    # Convert headings to vectors
    earlier_x, earlier_y = np.sin(heading_earlier), np.cos(heading_earlier)
    later_x, later_y = np.sin(heading_later), np.cos(heading_later)

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
    heading = np.degrees(np.arctan2(vx,vy))

    if heading < 0:
        heading += 360.0

    if VERBOSE:
        log.info(f"Velocity interpolation")
        log.info(f"{earlier_ts} horizontal {groundspeed_earlier:6.2f} vertical {vertical_earlier:6.2f} heading {np.degrees(heading_earlier):6.2f}")
        log.info(f"{this_ts} horizontal {groundspeed:6.2f} vertical {vertical:6.2f} heading {heading:6.2f}")
        log.info(f"{later_ts} horizontal {groundspeed_later:6.2f} vertical {vertical_later:6.2f} heading {np.degrees(heading_later):6.2f}")


    return groundspeed, vertical, heading

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


# To do - move this inside the adsb_recorder class



def parseSBS(fields, pp):
    """ Parse an SBS message received from dump1090.

       Decodes SBS messages and converts aircraft positions into RMS sky
       coordinates. Airborne position messages are transformed into
       azimuth/elevation, right ascension/declination, and image
       coordinates when the aircraft falls within the station field of view.

       Identification messages are converted into ICAO/callsign associations.

       Arguments:
           fields: [list] SBS message split into fields.
           pp: [Platepar] Station platepar object.

       Return:
           tx_type: [str] SBS transmission type.
           icao: [str] Aircraft ICAO identifier.
           d: [dict] Dictionary containing decoded message information.
       """


    # Discard incomplete messages
    if len(fields) < 22:
        return None, None, None

    # Discard messages which are not an SBS message
    if fields[FIELD_MSG_TYPE] != SBS_MESSAGE_TYPE:
        return None, None, None

    # Create date objects from received messages
    generated_dt = datetime.datetime.strptime(
        f"{fields[FIELD_GENERATED_DATE]} {fields[FIELD_GENERATED_TIME]}",
                "%Y/%m/%d %H:%M:%S.%f")

    # Create a timestamp and jd
    ts = formatTimeStamp(generated_dt)
    jd_message = datetime2JD(generated_dt)

    # Extract key fields for reuse
    tx_type = fields[FIELD_TX_TYPE]
    icao = fields[FIELD_ICAO]


    # Parse an airborne position message
    if tx_type == SBS_MSG_AIRBORNE_POSITION:

        # Check all position fields are populated
        if fields[FIELD_LATITUDE] != "" and fields[FIELD_LONGITUDE] != "" and fields[FIELD_ALTITUDE] != "":
            latitude = float(fields[FIELD_LATITUDE])
            longitude = float(fields[FIELD_LONGITUDE])
            altitude = float(fields[FIELD_ALTITUDE]) * FT_TO_M
        else:
            return tx_type, None, None

        # Compute vectors in ECEF
        vect_aircraft = latLonAlt2ECEF(np.radians(latitude), np.radians(longitude), altitude)
        vect_station = latLonAlt2ECEF(np.radians(pp.lat), np.radians(pp.lon), pp.elev)

        # Compute az and alt from the station to the aircraft
        az, alt = ECEF2AltAz(vect_station, vect_aircraft)

        # Convert the aircraft's position into sky coordinates
        ra_aircraft, dec_aircraft = altAz2RADec(az, alt, jd_message, pp.lat, pp.lon)

        # Compute the angular distance from the centre of the FoV
        ra_c, dec_c, max_sep, _, _ = getPlateparSkyParameters(pp, jd_message)
        ang_sep_deg = angularSeparationDeg(ra_aircraft, dec_aircraft, ra_c, dec_c)[0]

        # Prepare a dictionary of key parameters
        d = {"ts": ts,
             "lat": latitude, "lon": longitude, "alt": altitude,
             "az": az, "ele": alt,
             "ang_sep_deg": ang_sep_deg, "jd": jd_message, "ra": ra_aircraft, "dec": dec_aircraft}

        # Only compute image coordinates when within the sky cone
        if ang_sep_deg < max_sep:
            # Compute image coordinates
            x_arr, y_arr = raDecToXYPP(np.array([ra_aircraft]), np.array([dec_aircraft]), jd_message, pp)
            x, y = int(x_arr[0]), int(y_arr[0])

            # Test against image range, and add to dictionary if within range
            if 0 < x < pp.X_res and 0 < y < pp.Y_res:
                d["x"], d["y"] = x, y
                log.info(f"{icao} at {x},{y}")

        if VERBOSE:
            log.info(f"Received a position message from {icao} at lat:{latitude:8.5f} deg lon:{longitude:8.5f} deg alt:{altitude:5.0f} metres")


        return tx_type, icao, d

    # Parse an identification message
    elif tx_type == SBS_MSG_IDENTIFICATION:

        callsign = fields[FIELD_CALLSIGN].strip()

        # Record the relationship between ICAO and callsign from this message
        d = {"ts": ts,
             "icao": icao,
             "callsign": callsign}

        if VERBOSE:
            log.info(f"Received an identification message from {icao} using {callsign}")

        return tx_type, icao, d

    # Parse a velocity message
    elif tx_type == SBS_MSG_AIRBORNE_VELOCITY:

        groundspeed, v_rate = fields[FIELD_GROUNDSPEED], fields[FIELD_VERTICAL_RATE]
        track = fields[FIELD_TRACK]

        if groundspeed == "" or v_rate == "" or track == "":
            return None, None, None

        # Convert to floats in SI units
        groundspeed_m_s = float(groundspeed) * KT_TO_MS
        vertical_rate_m_s  = float(v_rate) * FT_TO_M / 60
        track_deg = float(track)

        # Create a dictionary of values
        d = {"ts": ts,
             "groundspeed": groundspeed_m_s,
             "vertical_rate": vertical_rate_m_s,
             "heading": track_deg,
             "jd": jd_message}

        if VERBOSE:
            log.info(f"Recording a velocity message from {icao} groundspeed:{groundspeed_m_s:.0f}m/s vertical:{vertical_rate_m_s:.0f}m/s")

        return tx_type, icao, d


    # If nothing has been parsed, return
    return tx_type, None, None


class ADSBRecorder(multiprocessing.Process):

    """ Record ADS-B aircraft information from dump1090.

        Receives SBS messages from dump1090, converts aircraft positions into
        RMS sky coordinates, records aircraft tracks and ICAO/callsign
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
        self.tracks = {}                        # Store for received location data
        self.icao_callsign_mapping = []         # Store for received icao to callsign information
        self.velocity = {}                      # Store for received velocity information
        self.s = None                           # Connection object
        self.buffer = b""                       # Buffer


        # Initalise and read the platepar
        platepar_path = os.path.join(config.config_file_path, config.platepar_name)
        if os.path.exists(platepar_path):
            self.pp = Platepar()
            self.pp.read(platepar_path, use_flat=config.use_flat)


        self.recording_start = None             # Start time for the recording


    def __getstate__(self):

        """ Return a picklable representation of the ADSB recorder.

        Required under the 'forkserver'/'spawn' start methods (the default on Linux from
        Python 3.14), where .start() pickles self and sends it to the child.
        """

        state = self.__dict__.copy()
        return state

    def doIteration(self):
        """ Process incoming SBS messages from dump1090.

        Reads data from the dump1090 SBS socket, decodes received messages,
        records aircraft position tracks and ICAO/callsign observations, and
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
                log.info(f"Writing out ADSB data between {self.recording_start} and {next_start_time}")
                writeOutADSBData(self.config, self.recording_start, self.tracks, self.icao_callsign_mapping, self.velocity)

                # After writing out, reset the cache
                self.tracks = {}
                self.icao_callsign_mapping = []

                # Reset the recording start time variables
                self.recording_start = next_start_time
                log.info(f"Starting a new block at {self.recording_start}")

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
                    self.s = socket.create_connection((host, port), timeout=60)
                    log.info(f"Connection to {host}:{port} opened.")

                except OSError as e:
                    log.info(
                        f"Connection to {host}:{port} failed "
                        f"({e}). Waiting for retry."
                    )
                    time.sleep(60)


            # return self.buffer, self.recording_start, self.tracks, self.icao_callsign_mapping, self.velocity

        if not data:
            # Add a sleep to avoid busy waits - this only executes when there is no data
            time.sleep(1)
            return self.buffer, self.recording_start, self.tracks, self.icao_callsign_mapping, self.velocity

        # Add data to the buffer
        self.buffer +=  data

        # If there is a \n in the buffer, then a whole message is available
        while b"\n" in self.buffer:

            # Break up the buffer into lines, and let the rest of the buffer carry over
            line, self.buffer = self.buffer.split(b"\n", 1)

            # Extract the fields
            fields = line.decode().strip().split(",")

            # Parse the SBS message
            tx_type, icao, d = parseSBS(fields, self.pp)

            # Ignore packets which produce no useful output
            if icao is None or d is None:
                if VVERBOSE:
                    log.info(f"Message of type {tx_type} produced no output {line}")

                continue

            # Route and decode received data

            if tx_type == SBS_MSG_AIRBORNE_POSITION:

                # Create a new list in the tracks dictionary if an airborne position is received for an unrecognised ICAO
                if icao not in self.tracks:
                    self.tracks[icao] = []

                # Record the airborne position against the ICAO
                self.tracks[icao].append(d)

            elif tx_type == SBS_MSG_AIRBORNE_VELOCITY:

                # Create a new list in the velocity dictionary if an airborne position is for an unrecognised ICAO
                if icao not in self.velocity:
                    self.velocity[icao] = []

                self.velocity[icao].append(d)

            elif tx_type == SBS_MSG_IDENTIFICATION:

                # Record all identification messages, deduplication is later in the pipeline
                self.icao_callsign_mapping.append(d)

        return self.buffer, self.recording_start, self.tracks, self.icao_callsign_mapping, self.velocity

    def start(self):

        """Starts the ADSB recorder"""
        super().start()
        log.info("ADSB recorder has started")


    def stop(self):

        """ Stops the ADSB recorder. """
        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("ADSB recorder has stopped")




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
                log.info("ADSB loop is exiting")
                break

        log.info("ADSB loop has exited")

        try:
            self.s.close()


            log.info("Closed dump1090 connection")
        except Exception:
            log.info("Could not close dump1090 connection")
            pass

        log.info("ADSB recorder has finished - writing out cached data")
        writeOutADSBData(self.config, self.recording_start, self.tracks, self.icao_callsign_mapping, force_flush=True)

    def flush(self):
        """Write all cached ADS-B data to disk.

        Cached aircraft tracks and ICAO/callsign observations are written
        immediately without affecting the current recording session.
        """

        log.info("Flushing cached ADS-B data")

        writeOutADSBData(
            self.config,
            self.recording_start,
            self.tracks,
            self.icao_callsign_mapping,
            force_flush=True
        )

    def archiveData(self, captured_path, remove_files=True):
        """ Archive all ADS-B data files, and optionally the files

        Arguments:
            captured_path: [str] Path to the capture directory for the night

        Keyword arguments:
            remove_files: [bool] Optional, default true, remove files from ADSB directory after archiving

        Return:
            archive_name: [str] Name of the generated archive.
        """

        archive_name = f"{os.path.basename(captured_path)}_adsb.tar.bz2"
        archive_path = os.path.join(captured_path, archive_name)
        adsb_path = os.path.expanduser(os.path.join(self.config.data_dir, self.config.adsb_dir))
        log.info(f"Saving ADSB data from {adsb_path} to an archive to {archive_path}")


        self.flush()

        with tarfile.open(archive_path, "w:bz2") as tar:
            adsb_files = []

            for filename in os.listdir(adsb_path):
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
    incoming ADS-B messages, records aircraft tracks and ICAO/callsign
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


def generateDerivedData(adsb_directory):

    # Test code

    groups = {}

    for filename in glob.glob(os.path.join(adsb_directory, "*.json")):

        basename = os.path.basename(filename)

        parts = basename.split("_")

        # AU0004_20261112_010000_adsb.json
        timestamp = f"{parts[1]}_{parts[2]}"

        if timestamp not in groups:
            groups[timestamp] = {}

        if basename.endswith("_adsb.json"):
            groups[timestamp]["adsb"] = filename

        elif basename.endswith("_icao_call.json"):
            groups[timestamp]["icao_call"] = filename

        elif basename.endswith("_velocity.json"):
            groups[timestamp]["velocity"] = filename

    for timestamp in sorted(groups):

        group = groups[timestamp]

        if not (
                "adsb" in group
                and "icao_call" in group
                and "velocity" in group
        ):
            log.warning(f"Incomplete ADSB dataset for {timestamp}")
            continue

        log.info(f"Processing {timestamp}")

        with open(group["adsb"]) as fp:
            tracks = json.load(fp)

        with open(group["icao_call"]) as fp:
            icao_callsign_mapping = json.load(fp)

        with open(group["velocity"]) as fp:
            velocity = json.load(fp)

        adsb_derived_data = deriveFromADSB(tracks, icao_callsign_mapping, velocity)


        if adsb_derived_data:

            adsb_derived_filename = f"{config.stationID}_{timestamp}_adsb_derived.json"
            adsb_derived_path = os.path.join(adsb_directory, adsb_derived_filename)

            with open(adsb_derived_path, "w") as fp:
                json.dump(adsb_derived_data, fp, indent=4)


        else:
            log.info("No adsb derived data to write")


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
    host = config.dump1090_host if host is None else host
    port = config.dump1090_port if port is None else port

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    log.info(f"For {config.stationID} starting adsb logging with {host}:{port}.")
    log.info(f"Logging to {config.data_dir}")

    if cml_args.testprocess:
        if config.adsb_enabled:
            log.info(f"ADSB enabled in config, so starting as a process for {cml_args.testprocess} seconds.")
            adsb_recorder = ADSBRecorder(config)
            adsb_recorder.start()
            time.sleep(cml_args.testprocess)
            adsb_recorder.stop()
        else:
            log.info("ADSB not enabled, not starting")

    elif cml_args.testfile:
        log.info(
            f"Replaying {len(replay_data)} SBS messages "
            f"from {cml_args.testfile}"
        )
        for line in replay_data:
            fields = line.strip().split(",")
            parseSBS(fields, pp)

    elif cml_args.testtimeseries:

        time_series_path = os.path.expanduser(cml_args.testtimeseries)

        log.info(f"Reading derived data from {time_series_path}")

        derived_data = {}

        for filename in sorted(glob.glob(os.path.join(time_series_path, "*_adsb_derived.json"))):

            log.info(f"Loading {filename}")

            with open(filename) as fp:
                imported_data = json.load(fp)
                log.info(f"Importing {filename}")


            for icao, aircraft in imported_data.items():

                if icao not in derived_data:
                    derived_data[icao] = aircraft

                else:
                    if VERBOSE:
                        log.info(f"Merging duplicate ICAO {icao}")

                    derived_data[icao]["heading"].extend(aircraft.get("head", []))

                    derived_data[icao]["velocity"].extend(aircraft.get("velocity", []))

                    derived_data[icao]["callsigns"].extend(aircraft.get("callsigns", []))

        log.info(f"Loaded {len(derived_data)} aircraft")

        interpolated_time_series = computeTimeSeries(derived_data)
        pass



    else:
        loopADSB(config, pp, host=host, port=port)