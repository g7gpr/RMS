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

    changed_files = [line for line in rsync_stdout.splitlines() if rsync_stdout.startswith((">", 'c', '*'))]

    if log_uploaded_files:
        log.info("Uploaded files:")
        for f in changed_files:
            log.info(f"\t{f}")

    if len(changed_files):
        return True
    else:
        return False

def makeUpload(config, return_after_each_upload=False):

    """using rsync, make an upload

    Arguments:
        config: [config] RMS config instance

    Keyword arguments:
        return_after_each_upload: [bool] After each file has been uploaded, return [default false]

    Upload files in priority order to a remote server using rsync

    """


    station_id = config.stationID
    station_id_lower = station_id.lower()
    print(config.rsa_private_key)

    key_path = os.path.expanduser(config.rsa_private_key)

    remote_path = os.path.join("/", "home",station_id_lower,"files","incoming")
    local_path = os.path.join(config.data_dir, config.archived_dir)
    with open(os.path.expanduser(os.path.join(config.data_dir, "rsync_remote_host.txt"))) as f:
        rsync_remote_host = f.readline()
        user_host = f"{station_id_lower}@{rsync_remote_host}:".replace("\n","")

    log.info(f"Using key from {key_path}")
    log.info(f"To copy files from {local_path} to {user_host}{remote_path}")

    local_path_modifier_list = ["*_metadata.tar.bz2",
                                "*_detected.tar.bz2",
                                "*_imgdata.tar.bz2",
                                "*.tar.bz2"]


    for local_path_modifier in local_path_modifier_list:


        # modify the local path to send files in the right order
        local_path_modified = os.path.join(local_path, local_path_modifier)
        log.info(f"Sending {local_path_modified}")
        # build rsync command
        command_string = f"rsync --progress -av --itemize-changes -e 'ssh -i {key_path}'  {local_path_modified} {user_host}{remote_path}"
        result = subprocess.run(command_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # If return after each upload is selected, then return, so that a check is made for all the highest
        # priority files again
        if return_after_each_upload and uploadMade(result.stdout, log_uploaded_files=True):
            return True


    # Now send the frame_dir

    local_path = os.path.join(config.data_dir, config.frame_dir, "*.tar")
    command_string = f"rsync --progress -av -e 'ssh -i {key_path}' {local_path} {user_host}{remote_path}"
    result = subprocess.run(command_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if uploadMade(result.stdout, log_uploaded_files=True):
        return True
    else:
        return False


if __name__ == '__main__':

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Upload files using rsync.
        """)



    arg_parser.add_argument('-c', '--config', metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    if cml_args.config is not None:
        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))
        log.info(f"Loaded config file for station {config.stationID}")
        makeUpload(None, None, config)
    else:
        config = None

    potential_stations_list = os.listdir("/home/rms/source/Stations")

    station_list = []

    for potential_station in sorted(potential_stations_list):
        if len(potential_station) == 6 and potential_station[:2].isalpha():
            station_list.append(potential_station)


    config_dict = {}

    for station in station_list:
        config_path_list = [os.path.join("/home/rms/source/Stations/", station,".config")]
        if os.path.exists(config_path_list[0]):
            if os.path.isfile(config_path_list[0]):
                log.info(f"Loading config for {station} from {config_path_list[0]}")
                config_dict[station] = cr.loadConfigFromDirectory(config_path_list, os.path.abspath('.'))

    while True:
        for station in config_dict:
            log.info(f"Making uploads for {station}")
            if makeUpload(config_dict[station], return_after_each_upload=True):
                break
