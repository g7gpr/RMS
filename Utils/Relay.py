# rsync based uploader for RMS
# Copyright (C) 2026 David Rollinson
#
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



import os
import subprocess

from RMS.Formats.Platepar import stationData
from RMS.Logger import getLogger
import argparse
import RMS.ConfigReader as cr
import time
import datetime
import getpass

LOG_FILE_PREFIX = "Relay"

log = getLogger("rmslogger", stdout=False)


if __name__ == '__main__':

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Upload files using sftp.""")


    arg_parser.add_argument('-t', '--time', metavar='TIME', type=int, \
        help="Time between starts of the relay, in minutes")

    arg_parser.add_argument('-v', '--verbose', action="store_true",
                            help="""Increase verbosity level""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    if cml_args.time is None:
        cycle_time_minutes = 15
    else:
        cycle_time_minutes = round(cml_args.time,0)


    log.info("Uploader relay starting")
    potential_station_paths_list = []
    fs_root = "/home/"
    users_list = os.listdir(fs_root)
    station_files_paths_list = []
    for p in users_list:
        p = os.path.join(fs_root, p)
        if os.path.exists(p):
            if os.path.isdir(p):
                station_files_paths_list.append(p)

    station_files_paths_list.sort()
    log.info("Found following potential station files paths")
    for p in station_files_paths_list:
        log.info(p)

    log.info("Gather remote file information")



    config_paths_list, station_list = [], []

    potential_station_paths_list.sort()
    for potential_station_path in sorted(potential_station_paths_list):
        potential_config_path = os.path.join(potential_station_path, ".config")
        if os.path.exists(potential_config_path):
            config_paths_list.append(potential_config_path)
    config_dict = {}

    for config_path in config_paths_list:
        config = cr.loadConfigFromDirectory([config_path], os.path.abspath('../Tests'))
        remote_host_path = os.path.join(config.data_dir,"rsync_remote_host.txt")
        if os.path.exists(remote_host_path):
            if cml_args.verbose:
                log.info(f"Found {remote_host_path}")
            if os.path.isfile(remote_host_path):
                config_dict[config.stationID] = config
                if cml_args.verbose:
                    log.info(f"Adding config for station {config.stationID} from {config_path}")
        else:
            if cml_args.verbose:
                log.info(f"Excluding {config_path} because no {remote_host_path} was found")

    start_time = datetime.datetime.now()
    cycle_time_seconds = 60 * cycle_time_minutes
    log.info(f"Uploader process initialised at {start_time}")
    log.info(f"Cycle time is {str(datetime.timedelta(seconds =cycle_time_seconds))}")
    while True:

        wait_time = (start_time - datetime.datetime.now())
        # If the uploader is more than one cycle late
        while wait_time.total_seconds() < (0 - cycle_time_seconds):
            # Add a cycle time and check again
            wait_time += datetime.timedelta(seconds=cycle_time_seconds)
            if cml_args.verbose:
                log.info("Skipping an upload cycle, because more than a whole cycle late")

        if wait_time.total_seconds() > 1:
            log.info(f"Waiting {str(wait_time).split('.')[0]} before restarting upload process at {start_time.strftime('%H:%M:%S')}")
            time.sleep(wait_time.total_seconds())
        else:
            if wait_time.total_seconds() > -3:
                pass
                log.info(f"Starting upload process immediately, as due at {start_time.strftime('%H:%M:%S')} and time now is {datetime.datetime.now().strftime('%H:%M:%S')}.")
            else:
                pass
                log.info(f"Starting upload process immediately, start time was {start_time.strftime('%H:%M:%S')}, "
                         f"time now is {datetime.datetime.now().strftime('%H:%M:%S')}, overdue by {0 - round(wait_time.total_seconds() / 60)} minutes")


        makeUpload(config_dict, verbose=cml_args.verbose)
        start_time = start_time + datetime.timedelta(seconds = cycle_time_seconds)
