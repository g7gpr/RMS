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
import bz2
import tarfile

from cv2 import connectedComponentsWithStats

from RMS.Formats.Platepar import stationData
from RMS.Logger import getLogger
from pathlib import Path

import argparse
import RMS.ConfigReader as cr
import time
import datetime
import paramiko
import json
import logging
import traceback

LOG_FILE_PREFIX = "Relay"
log = getLogger("rmslogger", stdout=False)
REMOTE_FILES_DICT_PATH = "/home/gmn/relay/remotefiles.json"
FS_ROOT = "/home/"
HOSTNAME = "gmn.uwo.ca"
MAX_TIME_PER_STATION = 120
LAG_WARNING_THRESHOLD = datetime.timedelta(hours=4)
LAG_WARNING_DEADBAND = datetime.timedelta(minutes=15)

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


def attr2Dict(file_attributes_list):

    file_attributes_list_of_dicts = []
    for attr in file_attributes_list:
        file_attributes_list_of_dicts.append({
        "filename": attr.filename,
        "size": attr.st_size,
        "mtime": attr.st_mtime,
        "mode": attr.st_mode,
        "uid": attr.st_uid,
        "gid": attr.st_gid,
        })

    return file_attributes_list_of_dicts


def getRemoteFilesDict(station_files_paths_list, hostname="gmn.uwo.ca", verbose=False):

    log.info("Gather remote file information")

    remote_file_dict_of_lists = {}
    remote_timelapse_files = []
    for station_path in station_files_paths_list:
        if verbose:
            log.info(f"Working on {station_path}")
        username = os.path.basename(station_path)
        key_path = os.path.join(station_path, ".ssh", "id_rsa")
        try:
            key = paramiko.RSAKey.from_private_key_file(key_path)
            if verbose:
                log.info(f"Found key for {username}")
        except:
            log.info("No key for {}".format(username))
            continue
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if verbose:
            log.info(f"Attempting connection to {username}@{hostname} using key from {key_path}")
        ssh.connect(hostname=hostname, username=username, pkey=key)

        try:
            with ssh.open_sftp() as sftp:
                remote_processed_files_list = attr2Dict(sftp.listdir_attr(os.path.join("files", "processed")))
                remote_unprocessed_files_list = attr2Dict(sftp.listdir_attr(os.path.join("files")))
        except:
            log.info(f"Unable to open sftp connection for {username}")
            continue




        ssh.close()
        for f in remote_unprocessed_files_list:
            if f['filename'].startswith(username.upper()) and f['filename'].endswith("_frames_timelapse.tar"):
                remote_timelapse_files.append(f)
        remote_files = remote_processed_files_list + remote_timelapse_files
        remote_files.sort(key=lambda remote_files: remote_files['filename'].lower())

        log.info(f"Adding {len(remote_files)} files to {username}")
        remote_file_dict_of_lists[username] = remote_files
        pass

    return remote_file_dict_of_lists

def isValidBz2(path):

    try:
        with bz2.BZ2File(path, 'rb') as f:
            for _ in iter(lambda: f.read(1024 * 1024), b''):
                pass
        return True
    except (OSError, EOFError):
        return False


def isValidTar(path):

    try:
        with tarfile.open(path, 'r:*') as tf:
            for _ in tf:
                pass
        return True
    except (tarfile.ReadError, OSError, EOFError):
        return False


def testArchive(file_path, verbose=False):

    file_name = os.path.basename(file_path)
    file_type = Path(file_name).suffix.lower()

    if file_type == ".tar":
        if verbose:
            log.info(f"Testing tar integrity for file {file_name}")
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                if isValidTar(file_path):
                    if verbose:
                        log.info(f"{file_name} is a valid archive")
                    return True

    elif file_type == ".bz2":
        if verbose:
            log.info(f"Testing bz2 integrity for file {file_name}")
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                if isValidBz2(file_path):
                    if verbose:
                        log.info(f"{file_name} is a valid archive")
                    return True

    # All other cases
    log.warning(f"{file_name} is not a path to a valid archive")
    return False

def uploadFile(station, f, sftp, test=False):

    if test:
        return True, 0

    local_file_path = os.path.join(FS_ROOT, station.lower(),"files",f)
    if not testArchive(local_file_path, verbose=False):
        log.info(f"{os.path.basename(local_file_path)} archive is not valid")
        return False, 0
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
    int_ts = int(time_elapsed_seconds)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    filetime_utc = datetime.datetime.fromtimestamp(os.path.getmtime(local_file_path), tz=datetime.timezone.utc)
    lag_time = now_utc - filetime_utc
    lag_time_str = f"{lag_time.days}d "
    lag_time_str += (datetime.datetime(1970,1,1, tzinfo=datetime.timezone.utc) + lag_time).strftime("%H:%M:%S")


    return success, size, lag_time, lag_time_str, remote_file_path, data_rate, int_ts

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

    archive_to_test = "/home/david/tmp/au001w/AU001W_20260306_112352_036009_metadata_invalid.tar.bz2"
    if os.path.exists(archive_to_test):
        log.info(f"Testing {os.path.basename(archive_to_test)} - testArchive returns {testArchive(archive_to_test)}")

    archive_to_test = "/home/david/tmp/au001w/AU001W_20260306_112352_036009_metadata.tar.bz2"
    if os.path.exists(archive_to_test):
        log.info(f"Testing {os.path.basename(archive_to_test)} - testArchive returns {testArchive(archive_to_test)}")

    start_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
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


    # Log start of process
    log.info(f"Uploader relay starting at {start_time}")
    stations_paths_list = getRemoteStationsPathsList(fs_root=FS_ROOT)
    doMaintenance(stations_paths_list)

    # Try and load the state of the remote file system
    if os.path.exists(REMOTE_FILES_DICT_PATH):
        try:
            with open(REMOTE_FILES_DICT_PATH, "r") as file_handle:
                remote_files_dict = json.load(file_handle)

        except:
            log.info("Unable to load remote files dictionary, removing")
            os.unlink(REMOTE_FILES_DICT_PATH)

    # Otherwise pull in the state of the remote file system into a dictionary
    if not os.path.exists(REMOTE_FILES_DICT_PATH):
        remote_files_dict = getRemoteFilesDict(stations_paths_list)
        remote_files_dict_dir = os.path.dirname(REMOTE_FILES_DICT_PATH)

        # Save this state as a json
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
    first_iteration = True

    while True:

        wait_time = (start_time - datetime.datetime.now(datetime.timezone.utc))
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

        max_lag_time_across_stations = datetime.timedelta(seconds=0)

        previous_max_lag_time_across_stations = max_lag_time_across_stations
        data_sent_this_iteration = 0
        total_data_to_be_sent = 0
        station_loop_start_time = datetime.datetime.now(datetime.timezone.utc)
        for station in remote_files_dict:
            if cml_args.verbose:
                log.info(f"Working on {station}")
            remote_files_list_of_dict = remote_files_dict[station]


            local_file_dir = Path(os.path.join(FS_ROOT,station,"files"))

            local_files_list_of_dict = []
            for p in local_file_dir.iterdir():
                if p.is_file():
                    st = p.stat()
                    local_files_list_of_dict.append({
                        "filename": p.name,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                        "mode": st.st_mode,
                        "uid": st.st_uid,
                        "gid": st.st_gid,
                    })

            local_data_files_list_of_dict = []
            for f in local_files_list_of_dict:
                if f['filename'].startswith(station.upper()) and f['filename'].endswith(".tar.bz2"):
                    local_data_files_list_of_dict.append(f)
                if f['filename'].startswith(station.upper()) and f['filename'].endswith("_frames_timelapse.tar"):
                    local_data_files_list_of_dict.append(f)

            files_to_upload = []
            remote_filenames = {f['filename']: f for f in remote_files_list_of_dict}
            local_filenames = {f['filename']: f for f in local_files_list_of_dict}

            for l in local_data_files_list_of_dict:
                local_name, local_size = l["filename"],  l["size"]


                if local_name not in remote_filenames:
                    files_to_upload.append(local_name)
                    continue

                remote_size = remote_filenames[local_name]['size']

                if local_size != remote_size:
                    local_size_mb, remote_size_mb = local_size / (1024 ** 2), remote_size / (1024 ** 2)
                    log.warning(f"Adding {local_name} because local size of {local_size / (1024 **2)}MB did not match remote size of {remote_size / (1024 ** 2)}MB")
                    files_to_upload.append(local_name)



            total_data = 0
            for f in files_to_upload:
                total_data += os.path.getsize(os.path.join(FS_ROOT, station.lower(), "files", f))  / (1000 * 1000)
            if total_data > 0 or cml_args.verbose:
                if len(files_to_upload) > 1:
                    log.info(f"For station {station} {total_data:.0f}MB to upload in {len(files_to_upload)} files")
                else:
                    log.info(f"For station {station} {total_data:.0f}MB to upload in 1 file")
            total_data_to_be_sent += total_data
            if len(files_to_upload):

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
                        time.sleep(delay_seconds)

                try:
                    with ssh.open_sftp() as sftp:
                        if cml_args.verbose:
                            log.info(f"Opened connection {username}@{HOSTNAME}")
                        # Record a start time for this station
                        start_station_time = datetime.datetime.now()
                        if len(files_to_upload) > 0 and cml_args.verbose:
                            if len(files_to_upload) == 1:
                                log.info(f"For station {station}, 1 file to upload")
                            else:
                                log.info(f"For station {station}, {len(files_to_upload)} files to upload")
                        i, data_sent, time_elapsed_on_this_station_seconds, data_rate = 0, 0, None, 0
                        for f in files_to_upload:
                            time_elapsed_on_this_station_seconds = (
                                        datetime.datetime.now() - start_station_time).total_seconds()
                            # If we have been here too long. then break this loop and start on the next station
                            if time_elapsed_on_this_station_seconds > MAX_TIME_PER_STATION:
                                if not out_of_time:
                                    if cml_args.verbose:
                                        log.info(f" {station} ran out of time - setting out_of_time to True")
                                    out_of_time = True
                                break
                            i += 1
                            upload_success, mb_sent, lag_time, lag_time_str, remote_file_path, data_rate, int_ts = uploadFile(station, f, sftp, test=False)

                            log_line = f"{mb_sent:6.1f}MB to {station}@{HOSTNAME}:{remote_file_path} in {int_ts:03d} seconds at {data_rate:01.2f}MB/s - delay {lag_time_str}"
                            data_sent += mb_sent
                            this_station_seconds = int((datetime.datetime.now() - start_station_time).total_seconds())
                            data_rate_so_far_this_station = data_sent / this_station_seconds
                            log_line += f" ({i}/{len(files_to_upload)}) {this_station_seconds:03d} seconds cumulative to send {data_sent:04.1f}MB at {data_rate_so_far_this_station:01.2f}MB/s"

                            log.info(log_line)

                            if lag_time > max_lag_time_across_stations:
                                log.info(f"   Got a new max_lag_time of {lag_time}")
                                max_lag_time_across_stations = lag_time

                            if upload_success:
                                if cml_args.verbose:
                                    log.info(f"File {f} was uploaded successfully")
                                    log.info(f"Adding {local_filenames[f]} to uploaded files dict")

                                remote_file_list = remote_files_dict[station]

                                # Remove all entries with this filename - work in place
                                initial_length = len(remote_file_list)
                                if cml_args.verbose:
                                    log.info(f"Before removal of {f} list length is {initial_length}")
                                remote_file_list[:] = [d for d in remote_file_list if d["filename"] != f]
                                subsequent_length = len(remote_file_list)
                                if cml_args.verbose:
                                    log.info(f"After removal of {f} list length is {subsequent_length}")

                                if initial_length != subsequent_length:
                                    log.info(f"{f} had multiple entries")

                                # Add the new one
                                remote_file_list.append(local_filenames[f])
                                if cml_args.verbose:
                                    log.info(f"After append of {f} list length is {len(remote_file_list)}")


                            time_elapsed_on_this_station_seconds = (datetime.datetime.now() - start_station_time).total_seconds()
                            int_ts = int(time_elapsed_on_this_station_seconds)

                        if time_elapsed_on_this_station_seconds is not None:
                            data_rate = data_sent / time_elapsed_on_this_station_seconds

                        log.info(f" For station {station} {data_sent:.0f}MB were uploaded in {int_ts:03d} seconds at {data_rate:3.2f}MB/s")
                        data_sent_this_iteration += data_sent
                        ssh.close()
                        log.info(f" Closed connection for {station}")



                except Exception as e:
                    ssh.close()
                    log.info(f" Closed connection for {station}")
                    log.info(f" Unable to upload {f} \n {e}")
                    log.info(traceback.format_exc())
                    continue

        # Write out the updated json file - do this once per iteration to reduce the chance of corruption
        with open(REMOTE_FILES_DICT_PATH, "w") as file_handle:
            log.info("Writing remote files status")
            json.dump(remote_files_dict, file_handle, indent=4, sort_keys=True)
            file_handle.flush()
            log.info("Writing remote files status - completed")

        lag_time_log_text = f"Maximum lag time is {max_lag_time_across_stations}"


        log.info(f"Lag warning threshold  / deadband {LAG_WARNING_THRESHOLD} / {LAG_WARNING_DEADBAND}")

        if max_lag_time_across_stations > LAG_WARNING_THRESHOLD and not first_iteration:

            log.info(f"Maximum / difference / previous lag time {max_lag_time_across_stations} / {previous_max_lag_time_across_stations} /  {max_lag_time_across_stations - previous_max_lag_time_across_stations}")

            if max_lag_time_across_stations > previous_max_lag_time_across_stations + LAG_WARNING_DEADBAND:
                lag_increase = (max_lag_time_across_stations - previous_max_lag_time_across_stations).total_seconds()
                lag_increase_minutes = round(lag_increase / 60)
                lag_time_log_text += f" and has increased by {lag_increase_minutes} minutes"

            elif max_lag_time_across_stations < previous_max_lag_time_across_stations - LAG_WARNING_DEADBAND:
                lag_reduction = (previous_max_lag_time_across_stations - max_lag_time_across_stations).total_seconds()
                lag_reduction_minutes = round(lag_reduction / 60)
                lag_time_log_text += f" and has reduced by {lag_reduction_minutes} minutes"

        log.info(lag_time_log_text)
        previous_max_lag_time_across_stations = max_lag_time_across_stations
        first_iteration = False
        log.info(f"Total data sent this iteration {data_sent_this_iteration:.1f} MB")
        total_data_to_be_sent -= data_sent_this_iteration
        log.info(f"Total data to be sent {total_data_to_be_sent:.1f} MB")
        time_taken_this_iteration_seconds = ((datetime.datetime.now(datetime.timezone.utc) - station_loop_start_time).total_seconds())

        if time_taken_this_iteration_seconds > 0:
            data_rate_mb_per_second = data_sent_this_iteration / time_taken_this_iteration_seconds
            seconds_to_completion = total_data_to_be_sent / data_rate_mb_per_second
            estimated_completion_time = (datetime.datetime.now() + datetime.timedelta(seconds=seconds_to_completion)).replace(microsecond=0)
            log.info(f"Time this iteration {time_taken_this_iteration_seconds:.0f} seconds Data rate {data_rate_mb_per_second:.2f} MB/s Completion time {estimated_completion_time}")
