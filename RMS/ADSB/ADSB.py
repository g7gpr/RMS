import argparse
import socket
import datetime
import os
import json
import multiprocessing
import time

import numpy as np

import RMS.ConfigReader as cr

from RMS.Logger import LoggingManager, getLogger
from RMS.Formats.StarCatalogQueries import getPlateparSkyParameters
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import datetime2JD, latLonAlt2ECEF, ECEF2AltAz, altAz2RADec
from RMS.Math import angularSeparationDeg
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

log = getLogger("rmslogger")

#Constants
FT_TO_M = 0.3048
KT_TO_MS = 0.514444

class ADSBRecorder(multiprocessing.Process):

    def __init__(self, config):


        """
            config: [config] RMS config instance.
        """

        self.config = config
        self.pp = Platepar()
        self.exit = multiprocessing.Event()

        platepar_path = os.path.join(config.config_file_path, config.platepar_name)
        if os.path.exists(platepar_path):
            self.pp = Platepar()
            self.pp.read(platepar_path, use_flat=config.use_flat)

        super(ADSBRecorder, self).__init__()


    def __getstate__(self):

        """ Return a picklable representation of the ADSB recorder.

        Required under the 'forkserver'/'spawn' start methods (the default on Linux from
        Python 3.14), where .start() pickles self and sends it to the child.
        """

        state = self.__dict__.copy()
        return state


    def start(self):

        """Starts the ADSB recorder"""

        log.info("ADSB recorder has started")
        super(ADSBRecorder, self).start()


    def stop(self):
        """ Stops the ADSB recorder. """

        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("ADSB recorder has stopped")

    def run(self,):
        """Start the loop"""



        s = socket.create_connection((self.config.dump1090_host, self.config.dump1090_port))
        buffer = b""
        tracks = {}
        last_hour = None
        print("Loop running")
        while not self.exit.is_set():
            last_hour, tracks = doIteration(self.config, self.pp, s, buffer, tracks, last_hour)




def writeOutTracks(config, last_hour, tracks):

    filename = f"{config.stationID}_{last_hour}_adsb_data.json"
    filepath = os.path.expanduser(os.path.join(config.data_dir, config.adsb_dir))
    filename_path = os.path.join(filepath, filename)
    log.info(f"Writing out ADSB data to {filename}")
    os.makedirs(filepath, exist_ok=True)

    with open(filename_path, "w") as fp:
        json.dump(tracks, fp, indent=4)



def loopADSB(config, pp, host=None, port=None, metric=None):


    host = config.adsb_host if host is None else host
    port = config.adsb_port if port is None else port
    metric = True if metric is None else metric

    s = socket.create_connection((host, port))



    buffer = b""
    tracks = {}
    last_hour = None

    while True:
        last_hour, tracks = doIteration(config, pp, s, buffer, tracks, last_hour)

def doIteration(config, pp, s, buffer, tracks, last_hour, metric=True):

    hour = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H0000")

    if last_hour is None:
        last_hour = hour
    else:
        if last_hour != hour:
            log.info("Writing out ADSB data")
            if tracks:
                writeOutTracks(config, last_hour, tracks)
            last_hour = hour
            tracks = {}
    data = s.recv(4096)
    if not data:
        return last_hour, tracks

    buffer += data

    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        fields = line.decode().strip().split(",")
        icao, d = parse_sbs(fields, metric, pp)

        if icao is None or d is None:
            continue

        try:

            jd, ts = d["jd"], d["ts"]
            latitude, longitude, altitude = d["lat"], d["lon"], d["alt"]
            az, el = d["az"], d["ele"]
            ang_sep_deg = d["ang_sep_deg"]

            log_message = f"{ts} {icao} lat:{latitude:+08.3f}, lon:{longitude:+08.3f},  alt:{altitude:05.0f} metres, az:{az:05.1f}, ele:{el:04.1f} angle deviation:{ang_sep_deg:03.0f} degrees"

            if "x" in d and "y" in d:
                x, y = d["x"], d["y"]
                log_message += f" - image {x}, {y}"

            log.info(log_message)

            if icao not in tracks:
                tracks[icao] = []

            tracks[icao].append(d)

        except Exception as e:
            print(f"Parse error: {e}")

    return last_hour, tracks

def parse_sbs(fields, metric, pp):

    if len(fields) < 22:
        return None, None



    message_type = fields[0]
    transmission_type = fields[1]
    session_id = fields[2]
    aircraft_id = fields[3]
    icao = fields[4]
    flight_id = fields[5]

    generated_date = fields[6]
    generated_time = fields[7]

    logged_date = fields[8]
    logged_time = fields[9]

    callsign = fields[10]


    track = fields[13]  # degrees



    vertical_rate = fields[16]  # ft/min

    squawk = fields[17]

    alert = fields[18]
    emergency = fields[19]
    spi = fields[20]

    on_ground = fields[21]


    if message_type != "MSG":
        return None, None

    # create date objects

    generated_dt = datetime.datetime.strptime(f"{fields[6]} {fields[7]}", "%Y/%m/%d %H:%M:%S.%f")
    logged_dt = datetime.datetime.strptime(f"{fields[8]} {fields[9]}", "%Y/%m/%d %H:%M:%S.%f")

    if transmission_type == "3":

        if fields[14] != "" and fields[15] != "" and fields[11] != "":
            latitude = float(fields[14])
            longitude = float(fields[15])
            altitude = float(fields[11]) * FT_TO_M
        else:
            return None, None
        if metric:
            #log.info(f"Generated:{generated_dt} Logged:{logged_dt} {callsign} lat:{latitude}, lon:{longitude}, altitude:{altitude}m")
            pass
        else:
            alt_ft= altitude / FT_TO_M
            #log.info(f"Generated:{generated_dt} Logged:{logged_dt} {callsign} lat:{latitude}, lon:{longitude}, altitude:{alt_ft}ft")
            pass


        jd_message = datetime2JD(generated_dt)

        vect_aircraft = latLonAlt2ECEF(np.radians(latitude), np.radians(longitude), altitude)
        vect_station = latLonAlt2ECEF(np.radians(pp.lat), np.radians(pp.lon), pp.elev)
        az, alt = ECEF2AltAz(vect_station, vect_aircraft)


        ra_c, dec_c, max_sep, _, _ = getPlateparSkyParameters(pp, jd_message)
        ra_aircraft, dec_aircraft = altAz2RADec(az, alt, jd_message, pp.lat, pp.lon)
        ang_sep_deg = angularSeparationDeg(ra_aircraft, dec_aircraft, ra_c, dec_c)[0]
        ts = generated_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


        inferred_data = {"jd": jd_message, "ts": ts,
                 "lat": latitude, "lon": longitude, "alt": altitude,
                 "az": az, "ele": alt, "ang_sep_deg": ang_sep_deg}

        if ang_sep_deg < max_sep:
            x, y = raDecToXYPP(np.array([ra_aircraft]), np.array([dec_aircraft]), jd_message, pp)
            if 0 < x < pp.X_res and 0 < y < pp.Y_res:
                inferred_data["x"], inferred_data["y"] = int(x[0]), int(y[0])

        return icao, inferred_data

    return None, None





if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture and compression.
        """)



    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('--host', metavar='HOST', help="""dump1090 host """)

    arg_parser.add_argument('--port', metavar='PORT', help="""dump1090 port """)

    arg_parser.add_argument("--av", metavar='UNITS', help="""Display in aviation units (ft / kts)""")

    arg_parser.add_argument( '--testfile',metavar='FILE', help='Read SBS messages from a file instead of a dump1090 socket.')

    arg_parser.add_argument("--testprocess", action="store_true", help="Test spawing as a process")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('../../adsb'))
    pp = Platepar()
    pp.read(config.platepar_name)

    if cml_args.testfile:
        with open(os.path.expanduser(cml_args.testfile)) as f:
            replay_data = [line.strip() for line in f if line.strip()]

    host, port = cml_args.host, cml_args.port
    metric = not cml_args.av

    units = "metric" if metric else "aviation"

    # If no command line arguments given, then use settings in the config file
    host = config.dump1090_host if host is None else host
    port = config.dump1090_port if port is None else port

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    log = getLogger("rmslogger")

    log.info(f"For {config.stationID} starting adsb logging with {host}:{port} with {units} units.")
    log.info(f"Logging to {config.data_dir}")

    if cml_args.testprocess:
        if config.adsb_enabled:
            log.info("ADSB enabled in config, so starting")
            adsb_recorder = ADSBRecorder(config)
            adsb_recorder.start()
            time.sleep(60)
            adsb_recorder.stop()
        else:
            log.info("ADSB not enabled, not starting")

    else:
        if cml_args.testfile:
            for line in replay_data:
                fields = line.strip().split(",")
                parse_sbs(fields, metric, pp)
        else:
            loopADSB(config, pp, host=host, port=port, metric=metric)