import argparse
import socket
import datetime
import os

import numpy as np

import RMS.ConfigReader as cr

from RMS.Logger import LoggingManager, getLogger, checkLoggingHealth
from RMS.Formats.StarCatalogQueries import getPlateparSkyParameters
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.Conversions import date2JD, datetime2JD, raDec2AltAz, vector2RaDec, latLonAlt2ECEF, ECEF2AltAz, \
    getECEFVectorBetweenGeoPoints, altAz2RADec, geo2Cartesian, J2000_JD
from RMS.Math import angularSeparationDeg
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP

#Constants

FT_TO_M = 0.3048
KT_TO_MS = 0.514444


def loop(host, port, metric, pp):

    s = socket.create_connection((host, port))



    buffer = b""

    while True:
        data = s.recv(4096)
        if not data:
            break

        buffer += data

        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)


            try:
                #log.info(f"Received : {line}")
                fields = line.decode().strip().split(",")
                parse_sbs(fields, metric, pp)
            except Exception as e:
                print(f"Parse error: {e}")




def parse_sbs(fields, metric, pp):
    if len(fields) < 22:
        return



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
        return

    # create date objects

    generated_dt = datetime.datetime.strptime(f"{fields[6]} {fields[7]}", "%Y/%m/%d %H:%M:%S.%f")
    logged_dt = datetime.datetime.strptime(f"{fields[8]} {fields[9]}", "%Y/%m/%d %H:%M:%S.%f")

    if transmission_type == "3":

        if fields[14] != "" and fields[15] != "" and fields[11] != "":
            latitude = float(fields[14])
            longitude = float(fields[15])
            altitude = float(fields[11]) * FT_TO_M
        else:
            return
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
        log.info(
            f"{generated_dt} {aircraft_id} {icao} observed at Az:{az:.2f}, Alt:{alt:.2f} lat:{latitude:.3f}, lon:{longitude:.3f}, alt:{altitude:.0f}metres angle:{ang_sep_deg:.0f} degrees")
        if ang_sep_deg > max_sep:
            #log.info(f"Angular separation from camera boresight is {ang_sep_deg} - cannot be in fov of camera")
            return
        x, y = raDecToXYPP(np.array([ra_aircraft]), np.array([dec_aircraft]), jd_message, pp)
        x, y = x[0], y[0]
        log.info(f"{aircraft_id} {icao} at image coordinates of {x},{y}")
        pass

if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture and compression.
        """)



    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('--host', metavar='HOST', help="""dump1090 host """)

    arg_parser.add_argument('--port', metavar='PORT', help="""dump1090 port """)

    arg_parser.add_argument("--av", metavar='PORT', help="""Display in aviation units (ft / kts)""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))
    pp = Platepar()
    pp.read(config.platepar_name)

    host, port = cml_args.host, cml_args.port
    metric = not cml_args.av

    units = "metric" if metric else "aviation"

    # Allow defaults of local machine and the SBS-1/Basestation port - most friendly
    host = "127.0.0.1" if host is None else host
    port = 30003 if port is None else port

    log = getLogger("rmslogger")

    log.info(f"For {config.stationID} starting adsb logging with {host}:{port} with {units} units.")
    # Test data
    test = "MSG,3,1,1,7C4A07,1,2026/11/09,21:14:11.828,2026/11/09,21:14:11.861,,650,,,-32.12770,115.86243,,,0,,0,0"

    parse_sbs(test.strip().split(","), metric, pp)

    loop(host, port, metric, pp)