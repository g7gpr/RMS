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


import numpy as np

from RMS.Logger import LoggingManager, getLogger
from RMS.Formats.StarCatalogQueries import getPlateparSkyParameters
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import datetime2JD, latLonAlt2ECEF, ECEF2AltAz, altAz2RADec
from RMS.Math import angularSeparationDeg
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

import RMS.ConfigReader as cr



# Unit conversion constant
FT_TO_M = 0.3048

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


def writeOutADSBData(config, recording_start_time, tracks, icao_call_mapping, force_flush=False):
    """ Write ADS-B data products to disk.

    Arguments:
        config: [Config] RMS configuration object.
        recording_start_time: [str] Recording start time in the format
            YYYYMMDD_HHMMSS.
        tracks: [dict] Aircraft track data keyed by ICAO code.
        icao_call_mapping: [list] List of observed ICAO to callsign mappings.

    Return:
        file_paths: [list] Paths of the generated JSON files.
    """

    adsb_directory = os.path.expanduser(os.path.join(config.data_dir, config.adsb_dir))
    os.makedirs(adsb_directory, exist_ok=True)

    adsb_filename = f"{config.stationID}_{recording_start_time}_adsb.json"
    adsb_path = os.path.join(adsb_directory, adsb_filename)
    icao_call_filename = f"{config.stationID}_{recording_start_time}_icao_call.json"
    icao_call_path = os.path.join(adsb_directory, icao_call_filename)

    if tracks:
        log.info(f"Writing out ADSB data to {adsb_filename}")
        with open(adsb_path, "w") as fp:
            json.dump(tracks, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())

    if icao_call_mapping:
        log.info(f"Writing out ICAO to callsign mapping data to {icao_call_filename}")
        with open(icao_call_path, "w") as fp:
            json.dump(icao_call_mapping, fp, indent=4)
            if force_flush:
                fp.flush()
                os.fsync(fp.fileno())

    return [adsb_path, icao_call_path]

def doIteration(config, pp, s, buffer, recording_start_time, tracks, icao_callsign_mapping):
    """ Process incoming SBS messages from dump1090.

    Reads data from the dump1090 SBS socket, decodes received messages,
    records aircraft position tracks and ICAO/callsign observations, and
    writes accumulated data products to disk at hourly intervals.

    Arguments:
        config: [Config] RMS configuration object.
        pp: [Platepar] Platepar used for image coordinate projection.
        s: [socket.socket] Connected dump1090 SBS socket.
        buffer: [bytes] Input buffer used to accumulate partial socket
            reads between iterations.
        recording_start_time: [str] Start time of the current recording
            block in the format YYYYMMDD_HHMMSS.
        tracks: [dict] Aircraft track data keyed by ICAO code.
        icao_callsign_mapping: [list] List of ICAO/callsign observations.


    Return:
        buffer: [bytes] Updated socket buffer.
        recording_start_time: [str] Updated recording block start time.
        tracks: [dict] Updated aircraft track data.
        icao_callsign_mapping: [list] Updated ICAO/callsign observations.
    """


    # Current UTC time used for recording block management
    # and output filename generation.
    time_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


    # Write out cached data each hour
    if recording_start_time is None:

        # If no start time is passed in, then create a new time reference
        recording_start_time = time_now
        log.info(f"Starting first recording block at {recording_start_time}")

    else:

        # Extract hours from times
        current_hour = time_now[:11]
        recording_start_hour = recording_start_time[:11]

        # Create a new recording block each hour
        if current_hour != recording_start_hour:
            next_start_time = time_now

            # Write to disk
            log.info(f"Writing out ADSB data between {recording_start_time} and {next_start_time}")
            writeOutADSBData(config, recording_start_time, tracks, icao_callsign_mapping)

            # After writing out, reset the cache
            tracks = {}
            icao_callsign_mapping = []

            # Reset the recording start time variables
            recording_start_time = next_start_time



    # Pull in data from the socket
    data = s.recv(4096)

    if not data:
        # Add a sleep to avoid busy waits - this only executes when there is no data
        time.sleep(1)
        return buffer, recording_start_time, tracks, icao_callsign_mapping

    # Add data to the buffer
    buffer += data

    # If there is a \n in the buffer, then a whole message is available
    while b"\n" in buffer:

        # Break up the buffer into lines, and let the rest of the buffer carry over
        line, buffer = buffer.split(b"\n", 1)

        # Extract the fields
        fields = line.decode().strip().split(",")

        # Parse the SBS message
        tx_type, icao, d = parseSBS(fields, pp)

        # Ignore packets which produce no useful output
        if icao is None or d is None:
            continue

        # Route and decode received data

        if tx_type == SBS_MSG_AIRBORNE_POSITION:

            # Create a new list in the tracks dictionary if an airborne position is received for an unrecognised track
            if icao not in tracks:
                tracks[icao] = []

            # Record the airborne position against the ICAO
            tracks[icao].append(d)

        elif tx_type == SBS_MSG_IDENTIFICATION:

            # Record all identification messages, deduplication is later in the pipeline
            icao_callsign_mapping.append(d)

    return buffer, recording_start_time, tracks, icao_callsign_mapping



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

    # Create a timestamp
    ts = formatTimeStamp(generated_dt)

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
            return None, None, None


        # Create a JD
        jd_message = datetime2JD(generated_dt)

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
        d = {
            "ts": ts,
            "lat": latitude, "lon": longitude, "alt": altitude,
            "az": az, "ele": alt, "ang_sep_deg": ang_sep_deg,
            "jd": jd_message, "ra": ra_aircraft, "dec": dec_aircraft}

        # Only compute image coordinates when within the sky cone
        if ang_sep_deg < max_sep:
            # Compute image coordinates
            x_arr, y_arr = raDecToXYPP(np.array([ra_aircraft]), np.array([dec_aircraft]), jd_message, pp)
            x, y, = int(x_arr[0]), int(y_arr[0])

            # Test against image range, and add to dictionary if within range

            if 0 < x < pp.X_res and 0 < y < pp.Y_res:
                d["x"], d["y"] = x, y
                log.info(f"{icao} at {x},{y}")

        return tx_type, icao, d

    # Parse an identification message
    elif tx_type == SBS_MSG_IDENTIFICATION:

        # Record the relationship between ICAO and callsign from this message
        d = {"ts": ts, "icao": icao, "callsign": fields[FIELD_CALLSIGN].strip()}

        return tx_type, icao, d

    # If nothing has been parsed, return
    return None, None, None


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
        s = None
        while not self.exit.is_set():
            try:
                s = socket.create_connection((self.config.dump1090_host, self.config.dump1090_port), timeout=10)
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

        buffer = b""

        # Run until exit
        while True:

            buffer, self.recording_start, self.tracks, self.icao_callsign_mapping = (
                doIteration(self.config, self.pp, s, buffer, self.recording_start, self.tracks, self.icao_callsign_mapping))
            if self.exit.is_set():
                log.info("ADSB loop is exiting")
                break

        log.info("ADSB loop has exited")

        try:
            s.close()
            log.info("Closed dump1090 connection")
        except Exception:
            log.info("Could not close dump1090 connection")
            pass

        log.info("ADSB recorder has finished - writing out cached data")
        writeOutADSBData(self.config, self.recording_start, self.tracks, self.icao_callsign_mapping, force_flush=True)

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

    host = config.adsb_host if host is None else host
    port = config.adsb_port if port is None else port


    # Open a connecton to dump1090
    s = None
    while s is None:
        try:
            s = socket.create_connection((host, port), timeout=10)
            log.info(f"Connection to {host}:{port} opened.")

        except OSError as e:
            log.info(
                f"Connection to {host}:{port} failed "
                f"({e}). Waiting for retry."
            )
            time.sleep(50)

    buffer = b""
    tracks = {}
    icao_callsign_mapping = []

    recording_start_time = None

    # Loop forever
    while True:
        buffer, recording_start_time, tracks, icao_callsign_mapping = \
            doIteration(config, pp, s, buffer, recording_start_time, tracks, icao_callsign_mapping)



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

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

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

    else:
        if cml_args.testfile:
            log.info(
                f"Replaying {len(replay_data)} SBS messages "
                f"from {cml_args.testfile}"
            )
            for line in replay_data:
                fields = line.strip().split(",")
                parseSBS(fields, pp)
        else:
            loopADSB(config, pp, host=host, port=port)