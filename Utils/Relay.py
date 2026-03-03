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
import json


LOG_FILE_PREFIX = "Relay"
log = getLogger("rmslogger", stdout=False)
REMOTE_FILES_DICT_PATH = "/home/gmn/relay/remotefiles.json"

def getRemoteStationsPathsList(fs_root="/home/"):

    users_list = os.listdir(fs_root)
    station_files_paths_list = []
    for p in users_list:
        if len(p) != 6:
            continue
        p = os.path.join(fs_root, p)
        if os.path.exists(p):
            if os.path.isdir(p):
                files_dir_path = os.path.join(p, "files")
                if os.path.isdir(files_dir_path):
                    station_files_paths_list.append(p)

    station_files_paths_list.sort()

    return station_files_paths_list


def getRemoteFilesDict(station_files_paths_list, hostname="gmn.uwo.ca"):

    log.info("Gather remote file information")

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
        log.info(f"Attempting connection to {username}@{hostname} using key from {key_path}")
        ssh.connect(hostname=hostname, username=username, pkey=key)

        try:
            sftp = ssh.open_sftp()
        except:
            log.info(f"Unable to open sftp connection for {username}")
            continue

        remote_processed_files = sftp.listdir(os.path.join("files", "processed"))
        remote_processed_files.sort()
        log.info(f"Adding {len(remote_processed_files)} files to {username}")
        remote_file_dict_of_lists[username] = remote_processed_files

    return remote_file_dict_of_lists


if __name__ == '__main__':


    start_time = datetime.datetime.
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

    cycle_time_seconds = cycle_time_minutes * 60


    log.info("Uploader relay starting")
    stations_paths_list = getRemoteStationsPathsList()
    remote_files_dict = getRemoteFilesDict(stations_paths_list)

    remote_files_dict_dir = os.path.dirname(REMOTE_FILES_DICT_PATH)

    log.info(f"Making directory for {remote_files_dict_dir}")
    if not os.path.exists(remote_files_dict_dir):
        os.makedirs(remote_files_dict_dir)
    with open(REMOTE_FILES_DICT_PATH, "w") as file_handle:
        json.dump(remote_files_dict, file_handle, indent=4, sort_keys=True)

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

        for station in remote_files_dict:
            log.info(f"Working on {station}")