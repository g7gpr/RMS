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
import shutil
import random

from cv2 import connectedComponentsWithStats

from RMS.Formats.Platepar import stationData
from RMS.Logger import getLogger
import argparse
import RMS.ConfigReader as cr
import time
import datetime
import paramiko
import json
import logging

LOG_FILE_PREFIX = "Relay"
log = getLogger("rmslogger", stdout=False)
REMOTE_FILES_DICT_PATH = "/home/gmn/relay/remotefiles.json"
FS_ROOT = "/home/"
HOSTNAME = "gmn.uwo.ca"
MAX_TIME_PER_STATION = 120



for name in ("paramiko", "paramiko.transport", "paramiko.hostkeys"):
    logger = logging.getLogger(name)
    logger.handlers.clear()      # remove any handlers Paramiko attached
    logger.propagate = False     # stop messages bubbling to root
    logger.setLevel(logging.CRITICAL)





def sortByPriority(f_list, order=None):

    if order is None:
        order = ["metadata", "detected", "imgdata", "frames_timelapse"]

    def fileRank(f):

        rank = {word: i for i, word in enumerate(order)}
        for word in order:
            if word in f:
                return rank[word]
        return len(order)

    order = ["metadata", "detected", "imgdata", "frames_timelapse"]
    return sorted(sorted(f_list), key=fileRank, reverse=False)




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
    remote_timelapse_files = []
    for station_path in station_files_paths_list:
        log.info(f"Working on {station_path}")
        username = os.path.basename(station_path)
        key_path = os.path.join(station_path, ".ssh", "id_rsa")
        try:
            key = paramiko.RSAKey.from_private_key_file(key_path)
            if cml_args.verbose:
                log.info(f"Found key for {username}")
        except:
            log.info("No key for {}".format(username))
            continue
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if cml_args.verbose:
            log.info(f"Attempting connection to {username}@{hostname} using key from {key_path}")
        ssh.connect(hostname=hostname, username=username, pkey=key)

        try:
            sftp = ssh.open_sftp()
        except:
            log.info(f"Unable to open sftp connection for {username}")
            continue

        remote_processed_files = sftp.listdir(os.path.join("files", "processed"))
        remote_unprocessed_files = sftp.listdir(os.path.join("files"))
        for f in remote_unprocessed_files:
            if f.startswith(username.upper()) and f.endswith("_frames_timelapse.tar"):
                remote_timelapse_files.append(f)
        remote_files = remote_processed_files + remote_timelapse_files
        remote_files.sort()

        log.info(f"Adding {len(remote_files)} files to {username}")
        remote_file_dict_of_lists[username] = remote_files
        pass

    return remote_file_dict_of_lists


def uploadFile(station, f, sftp, hostname=HOSTNAME, test=False, counter=None):

    if test:
        return True, 0

    local_file_path = os.path.join(FS_ROOT, station.lower(),"files",f)
    remote_file_path = os.path.join("files",f)
    if test:
        log.info(f"Simulating good upload of {local_file_path} to {remote_file_path} for station {station}")
        return True, 0
    upload_start_time = datetime.datetime.now()
    success = sftp.put(local_file_path, remote_file_path, confirm=True)
    upload_end_time = datetime.datetime.now()
    time_elapsed_seconds = (upload_end_time - upload_start_time).total_seconds()
    size = os.path.getsize(local_file_path)  / (1000 * 1000)
    data_rate = size / time_elapsed_seconds
    if counter is None:
        log.info(f"Uploaded {os.path.basename(local_file_path)} to {station}@{hostname}:{remote_file_path} at {data_rate:6.2f}MB/s")
    else:
        log.info(
            f"Uploaded {size:6.1f}MB to {station}@{hostname}:{remote_file_path} in {time_elapsed_seconds:.0f} seconds at {data_rate:3.2f}MB/s ({counter})")
    return success, size

def doMaintenance(stations_paths_list):

    for station_path in stations_paths_list:
        if cml_args.verbose:
            log.info(f"Performing maintenance of {os.path.basename(station_path)}")
        pass
        incoming_directory_path = os.path.join(station_path,"files","incoming")
        files_directory_path = os.path.join(station_path, "files")
        if os.path.exists(incoming_directory_path):
            if os.path.isdir(incoming_directory_path):
                for f in os.listdir(incoming_directory_path):
                    source_file = os.path.join(incoming_directory_path, f)
                    if os.path.isfile(source_file):
                        if f.endswith(".confirmed"):
                            os.unlink(source_file)
                            continue
                        if f.endswith(".tar") or f.endswith(".tar.bz2"):
                            destination_file = os.path.join(files_directory_path, f)
                            shutil.move(source_file, destination_file)
            log.info(f"Removing {incoming_directory_path}")
            try:
                shutil.rmtree(incoming_directory_path)
            except:
                log.info("Failed")

if __name__ == '__main__':


    start_time = datetime.datetime.now().replace(microsecond=0)
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

    log.info(f"Uploader relay starting at {start_time}")
    stations_paths_list = getRemoteStationsPathsList(fs_root=FS_ROOT)
    doMaintenance(stations_paths_list)

    if os.path.exists(REMOTE_FILES_DICT_PATH):
        try:
            with open(REMOTE_FILES_DICT_PATH, "r") as file_handle:
                remote_files_dict = json.load(file_handle)

        except:
            log.info("Unable to load remote files dictionary, removing")
            os.unlink(REMOTE_FILES_DICT_PATH)

    if not os.path.exists(REMOTE_FILES_DICT_PATH):
        remote_files_dict = getRemoteFilesDict(stations_paths_list)
        remote_files_dict_dir = os.path.dirname(REMOTE_FILES_DICT_PATH)

        if not os.path.exists(remote_files_dict_dir):
            log.info(f"Making directory for {remote_files_dict_dir}")
            os.makedirs(remote_files_dict_dir)
        with open(REMOTE_FILES_DICT_PATH, "w") as file_handle:
            json.dump(remote_files_dict, file_handle, indent=4, sort_keys=True)
            file_handle.flush()

    missing_stations = []
    for station_path in stations_paths_list:
        station = os.path.basename(station_path)
        if not station in remote_files_dict:
            log.info(f"Station {station} is missing from the remote files dict.")
            remote_files_dict[station] = getRemoteFilesDict([station_path])[station]
            if station in remote_files_dict:
                log.info(f"Station {station} has been added")
                with open(REMOTE_FILES_DICT_PATH, "w") as file_handle:
                    json.dump(remote_files_dict, file_handle, indent=4, sort_keys=True)
                    file_handle.flush()

    out_of_time = False
    while True:

        wait_time = (start_time - datetime.datetime.now())
        # If the uploader is more than one cycle late
        while wait_time.total_seconds() < (0 - cycle_time_seconds):
            # Add a cycle time and check again
            wait_time += datetime.timedelta(seconds=cycle_time_seconds)
            if cml_args.verbose:
                log.info("Skipping an upload cycle, because more than a whole cycle late")

        if out_of_time:
            log.info("Not waiting - some stations did not fully upload on last pass")
            out_of_time = False
        else:
            if wait_time.total_seconds() > 1:
                log.info(f"Waiting {str(wait_time).split('.')[0]} before restarting upload process at {start_time.strftime('%H:%M:%S')}")
                time.sleep(wait_time.total_seconds())
            else:
                if wait_time.total_seconds() > -3:
                    pass

                else:
                    pass

            start_time = start_time + datetime.timedelta(seconds = cycle_time_seconds)


        for station in remote_files_dict:
            if cml_args.verbose:
                log.info(f"Working on {station}")
            remote_files_set = set(remote_files_dict[station])
            local_files = set(os.listdir(os.path.join(FS_ROOT,station,"files")))
            local_data_files = []
            for f in local_files:
                if f.startswith(station.upper()) and f.endswith(".tar.bz2"):
                    local_data_files.append(f)
                if f.startswith(station.upper()) and f.endswith("_frames_timelapse.tar"):
                    local_data_files.append(f)
            local_files_set = set(local_data_files)
            files_to_upload = sorted(list(local_files_set - remote_files_set))

            total_data = 0
            for f in files_to_upload:
                total_data += os.path.getsize(os.path.join(FS_ROOT, station.lower(), "files", f))  / (1000 * 1000)
            if total_data > 0 or cml_args.verbose:
                log.info(f"For station {station} {total_data:.0f}MB to upload")

            if len(files_to_upload):
                if cml_args.verbose:
                    log.info(f"Files to upload for {station}")

                files_to_upload = sortByPriority(files_to_upload)
                username = station.lower()
                station_path = os.path.join(FS_ROOT, username)
                key_path = os.path.join(station_path, ".ssh", "id_rsa")
                try:
                    key = paramiko.RSAKey.from_private_key_file(key_path)
                    if cml_args.verbose:
                        log.info(f"Found key for {username}")
                except:
                    log.info("No key for {}".format(username))
                    continue

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                connected = False
                while not connected:
                    try:
                        if cml_args.verbose:
                            log.info(f"Attempting connection to {username}@{HOSTNAME} using key from {key_path}")
                        ssh.connect(hostname=HOSTNAME, username=username, pkey=key)
                        connected = True
                    except:
                        delay_seconds = random.randint(1200, 2400)
                        delay_minutes = delay_seconds / 60
                        restart_time = (datetime.datetime.now() + datetime.timedelta(seconds = delay_seconds)).replace(microsecond=0)
                        log.info(f"Connection refused - waiting {delay_minutes:.1f} minutes - continuing at {restart_time}")
                        time.sleep(60 * 20)
                        time.sleep(delay_seconds)

                try:
                    sftp = ssh.open_sftp()
                    if cml_args.verbose:
                        log.info(f"Opened connection {username}@{HOSTNAME}")
                except:
                    log.info(f"Unable to open sftp connection for {username}")
                    continue

                # Record a start time for this station
                start_station_time =  datetime.datetime.now()
                if len(files_to_upload) > 0 and cml_args.verbose:
                    if len(files_to_upload) == 1:
                        log.info(f"For station {station}, 1 file to upload")
                    else:
                        log.info(f"For station {station}, {len(files_to_upload)} files to upload")
                i, data_sent, time_elapsed_on_this_station_seconds, data_rate = 0, 0, None, 0
                for f in files_to_upload:
                    time_elapsed_on_this_station_seconds = (datetime.datetime.now() - start_station_time).total_seconds()
                    # If we have been here too long. then break this loop and start on the next station
                    if time_elapsed_on_this_station_seconds > MAX_TIME_PER_STATION:
                        log.info(f"Spent {time_elapsed_on_this_station_seconds:.0f} seconds, moving onto the next station")
                        if not out_of_time:
                            log.info(f"{station} ran out of time - setting out_of_time to True")
                            out_of_time = True
                        break
                    i += 1
                    upload_success, mb_sent = uploadFile(station, f, sftp, test=False, counter=f"{i}/{len(files_to_upload)}")
                    data_sent += mb_sent
                    if upload_success:
                        if cml_args.verbose:
                            log.info(f"File {f} was uploaded successfully")
                        remote_files_set = set(remote_files_dict[station])
                        remote_files_set.add(f)
                        remote_files_dict[station] = list(remote_files_set)
                    time_elapsed_on_this_station_seconds = (datetime.datetime.now() - start_station_time).total_seconds()
                if time_elapsed_on_this_station_seconds is not None:
                    data_rate = data_sent / time_elapsed_on_this_station_seconds
                log.info(f"{data_sent:4.0f}MB were uploaded for station {station} at {data_rate:3.2f}MB/s")
                # Write out the updated json file - do this once per station to reduce the chance of corruption
                with open(REMOTE_FILES_DICT_PATH, "w") as file_handle:
                    json.dump(remote_files_dict, file_handle, indent=4, sort_keys=True)
                    file_handle.flush()