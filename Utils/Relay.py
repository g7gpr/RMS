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
import paramiko

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

    hostname = 'gmn.uwo.ca'

    remote_file_dict_of_lists = {}
    for p in station_files_paths_list:
        log.info(f"Working on {p}")
        username = os.path.basename(p)
        key_path = os.path.join(p, ".ssh", "id_rsa")
        try:
            key = paramiko.RSAKey.from_private_key_file(key_path)
            log.info(f"Found key for {username}")
        except:
            log.info("No key for {}".format(username))
            continue
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(hostname=hostname, username=username, pkey=key)

        try:
            sftp = ssh.open_sftp()
        except:
            log.info(f"Unable to open sftp connection for {username}")
            continue

        remote_processed_files = sftp.listdir(os.path.join("files","processed"))
        remote_processed_files.sort()
        remote_file_dict_of_lists[username] = remote_processed_files


    config_paths_list, station_list = [], []

    potential_station_paths_list.sort()
    for potential_station_path in sorted(potential_station_paths_list):
        potential_config_path = os.path.join(potential_station_path, ".config")
        if os.path.exists(potential_config_path):
            config_paths_list.append(potential_config_path)
    config_dict = {}


    start_time = datetime.datetime.now()
    cycle_time_seconds = 60 * cycle_time_minutes
    log.info("Uploader process initialised at {}".format(start_time))
    log.info("Cycle time is {}".format(str(datetime.timedelta(seconds =cycle_time_seconds))))
    while True:

        wait_time = (start_time - datetime.datetime.now())
        # If the uploader is more than one cycle late
        while wait_time.total_seconds() < (0 - cycle_time_seconds):
            # Add a cycle time and check again
            wait_time += datetime.timedelta(seconds=cycle_time_seconds)
            if cml_args.verbose:
                log.info("Skipping an upload cycle, because more than a whole cycle late")

        if wait_time.total_seconds() > 1:
            time.sleep(wait_time.total_seconds())
        else:
            if wait_time.total_seconds() > -3:
                pass

            else:
                pass


        start_time = start_time + datetime.timedelta(seconds = cycle_time_seconds)
