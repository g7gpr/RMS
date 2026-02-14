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

LOG_FILE_PREFIX = "EXTERNAL"

log = getLogger("rmslogger", stdout=False)


def createLock(config, log):
    """ If no file config.reboot_lock_file exists in config.data_dir, create one

    Arguments:
        config: [config] RMS config instance
        log: [logger] logger instance

    Returns:
        Nothing
    """


    log.info("Applying reboot lock")
    lockfile = os.path.join(os.path.expanduser(config.data_dir), config.reboot_lock_file)
    with open(lockfile, 'w') as _:
        pass

    pass

def removeLock(config, log):
    """ If the file config.reboot_lock_file exists in config.data_dir, remove it

    Arguments:
        config: [config] RMS config instance
        log: [logger] logger instance

    Returns:
        Nothing
    """

    log.info("Removing reboot lock")
    lockfile = os.path.join(os.path.expanduser(config.data_dir), config.reboot_lock_file)
    if os.path.exists(lockfile):
        os.remove(lockfile)
    else:
        log.warning("No reboot lock file found at {}".format(lockfile))

def uploadMade(rsync_stdout, log_uploaded_files=False):
    """If stdout from rsync shows that a file was changed on the remote, return True.

    Arguments:
        rsync_stdout: [string] stdout from rsync

    Keyword arguments:
        log_uploaded_files: [bool] whether to log uploaded files

    Return:
        True if a file was uploaded, otherwise false
    """

    rsync_stdout = rsync_stdout.decode('utf-8')
    rsync_stdout_lines = rsync_stdout.splitlines()

    changed_files = []
    for line in rsync_stdout_lines:
        if line.startswith("<"):
            changed_files.append(line.split(" ")[1])

    changed_files.sort()
    if log_uploaded_files:
        if len(changed_files):
            log.info("Uploaded files:")
        for f in changed_files:
            log.info(f"\t{f}")

    if len(changed_files):
        return True
    else:
        return False

def makeUpload(config_dict):

    """using rsync, make an upload

    Arguments:
        config: [config] RMS config instance

    Keyword arguments:
        return_after_each_upload: [bool] After each file has been uploaded, return [default false]

    Upload files in priority order to a remote server using rsync

    """



    local_path_modifier_list = ["*_metadata.tar.bz2",
                                "*_detected.tar.bz2",
                                "*_imgdata.tar.bz2",
                                "*.tar.bz2",
                                "*.tar"]

    modifier_descriptors_list = ["metadata", "detected", "imgdata", "all other tar.bz2 files", "frames files"]

    target_dir_list = ['archive', 'archive', 'archive', 'archive', 'frames']


    # Strategy is to set upload_mode to True, and only allow the while loop to end
    # once all the stations and priorities have been iterated, with no upload
    upload_made = True

    while upload_made:
        upload_made = False
        for local_path_modifier, descriptor, target_dir in zip(local_path_modifier_list, modifier_descriptors_list, target_dir_list):
            #log.info(f"Uploading {descriptor}")
            if upload_made:
                if local_path_modifier_list.index(local_path_modifier) != 0:
                    break

            for station in config_dict:
                config = config_dict[station]
                station_id = config.stationID
                station_id_lower = station_id.lower()
                remote_host_address_path = os.path.expanduser(os.path.join(config.data_dir, "rsync_remote_host.txt"))

                key_path = os.path.expanduser(config.rsa_private_key)

                if not os.path.exists(remote_host_address_path):
                    log.info(f"\t\tRemote host path not found at {remote_host_address_path}")
                    continue
                if not os.path.isfile(remote_host_address_path):
                    continue

                remote_path = os.path.join("/", "home", station_id_lower, "files", "incoming")
                if target_dir == "archive":
                    target_dir_from_config = config.archived_dir
                elif target_dir == "frames":
                    target_dir_from_config = config.frame_dir
                else:
                    continue
                local_path = os.path.join(config.data_dir, target_dir_from_config)
                with open(remote_host_address_path) as f:
                    rsync_remote_host = f.readline()
                    user_host = f"{station_id_lower}@{rsync_remote_host}:".replace("\n", "")
                # modify the local path to send files in the right order
                local_path_modified = os.path.join(local_path, local_path_modifier)
                # build rsync command
                command_string = f"rsync --progress -av --itemize-changes --bwlimit=512 --partial-dir=partial/ -e  'ssh  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {key_path}'  {local_path_modified} {user_host}{remote_path}"
                result = subprocess.run(command_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # If return after each upload is selected, then return, so that a check is made for all the highest
                # priority files again
                upload_made = uploadMade(result.stdout, log_uploaded_files=True)
                if upload_made:
                    if local_path_modifier_list.index(local_path_modifier) != 0:
                        #log.info("Made an upload, which was not highest priority, so restarting at highest priority of upload")
                        break
                    else:
                        #log.info("Made an upload, of the highest priority; continuing files from other stations at this priority level")
                        pass

            if upload_made:
                #log.info("Made an upload, restarting with highest priority uploads")
                break


if __name__ == '__main__':

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Upload files using rsync.
        """)



    arg_parser.add_argument('-c', '--config', metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-t', '--time', metavar='TIME', type=int, \
        help="Time between starts of the uploader, in minutes")




    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    if cml_args.config is not None:
        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))
        log.info(f"Loaded config file for station {config.stationID}")
        config_dict = {}
        config_dict[config.stationID] = config
        makeUpload(None, None, config)
    else:
        config = None

    if cml_args.time is None:
        cycle_time_minutes = 15
    else:
        cycle_time_minutes = round(cml_args.time,0)


    log.info("Uploader daemon starting")
    potential_station_paths_list = []
    stations_dir = f"/home/{getpass.getuser()}/source/Stations"
    if os.path.exists(stations_dir):
        for p in os.listdir(stations_dir):
            p = os.path.join(stations_dir, p)
            if os.path.exists(p):
                if os.path.isdir(p):
                    potential_station_paths_list.append(p)

    home_dir = "/home/"
    for p in os.listdir(home_dir):
        p = os.path.join(home_dir, p)
        if os.path.exists(p):
            if os.path.isdir(p):
                try:
                    potential_station_paths_list.append(os.path.join(p, "source/RMS"))
                except:
                    pass





    config_paths_list, station_list = [], []

    potential_station_paths_list.sort()
    for potential_station_path in sorted(potential_station_paths_list):
        potential_config_path = os.path.join(potential_station_path, ".config")
        if os.path.exists(potential_config_path):
            config_paths_list.append(potential_config_path)
    config_dict = {}

    for config_path in config_paths_list:
        config = cr.loadConfigFromDirectory([config_path], os.path.abspath('.'))
        remote_host_path = os.path.join(config.data_dir,"rsync_remote_host.txt")
        if os.path.exists(remote_host_path):
            if os.path.isfile(remote_host_path):
                config_dict[config.stationID] = config
        else:
            log.info(f"Excluding {config_path} because no remote_host_path was found")

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
                         f"time now is {datetime.datetime.now().strftime('%H:%M:%S')}, overdue by {str(datetime.timedelta(seconds = wait_time.total_seconds()))}")

        makeUpload(config_dict)
        start_time = start_time + datetime.timedelta(seconds = cycle_time_seconds)
