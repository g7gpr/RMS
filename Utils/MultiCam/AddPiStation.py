# This software is part of the Linux port of RMS
# Copyright (C) 2023  Ed Harman
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
#
# This version history no longer maintained following migration into git
#
# Version 1.7   - changed desktop launcher display names and added conkyrc1 mod
#
# Version 1.6   - added support for RPi-5 platforms that can run max 4 cameras
#
# Version 1.5   - added support for non English locales where user user directories may not include a directory named Desktop
#		 i.e. this enables support of RMS on a non English distro install
#
# Version 1.4   - moved codebase into RMS/Scripts/MultiCamLinux
#
# Version 1.3	- fixed path to CMN desktop shortcut
#
# Version 1.2	- added a  change to the flag reboot_after_processing from true to false
#
# Version 1.1
# Changes	- added station arguments to  Launch scripts
#		- changed desktop links  for StartCapture to symbolic links of the scripts within .config/autostart
#
# Version 1.2	- Peter E. took over dev of this script and any blame going forward
#		- white space and indents added
#
# Translated to Python and revision history managed in git

import subprocess
import os
import urllib.request
import time

import RMS.ConfigReader as cr
from RMS.Misc import sanitise, mkdirP
from Utils.MultiCam.Common import createAutoStartEntry, createShowLiveStreamEntry, setTimeZone, setQuotas
from Utils.MultiCam.Common import checkUserDesktopDirectoryEnvironment, uncomment, computeQuotas
from Utils.MultiCam.Common import moveIfExists, copyIfExists, changeOptionValue
from Utils.MultiCam.Common import getStationsToAdd, customiseConfig, makeKeys
import argparse
import RMS.StartCapture


MAX_STATION = 4	# maximum number of stations allowed on a Pi5

def lsCPU():

    """

    Returns: [str] CPU information

    """

    return subprocess.check_output("lscpu").decode('ascii')


def piModel(read_from = "/proc/cpuinfo"):
    """

    Args:
        read_from: [str] optional, default "/proc/cpuinfo".

    Returns:
        Model of Pi
    """


    if os.path.exists(read_from):
        with open(read_from, 'r') as f:
            line_list = list(filter(lambda x: 'Model' in x, f.readlines()))

            if len(line_list):
                model_line_list = line_list[0].split(":")
            else:
                return None

            if len(model_line_list) == 2:
                return sanitise((model_line_list[1]), space_substitution=" ").strip()
            else:
                return None

    else:
        return "Unable to determine"

def isPi5():
    """

    Returns:[bool] True if hardware is identifed as Pi5

    """

    if piModel() is None:
        return False
    else:
        return "Pi 5" in piModel()

def firstStationConfigured(config_path = "~/source/RMS/.config"):
    """
    Examines the .config file in RMS path to discover if station is still set as XX0001
    and there is nothing in ~/RMS/Stations
    This is a good indicator that no configuration work has been carried out
    Args:
        config_path: [path] optional default ~/source/RMS/.config

    Returns:[bool] True if stationID is no longer XX0001

    """
    config_path = os.path.expanduser(config_path)
    stations_path = os.path.expanduser("~/source/Stations")
    if os.path.exists(config_path):
        config = cr.parse(config_path)
    else:
        print("Returning because could not find config")
        return False
    if os.path.exists(stations_path):
        print("Found station path")
        if len(os.listdir(stations_path)) == 0:
            print("No station yet configured")
            return False
        else:
            return True
    if config.stationID == "XX0001":
        print("Returning because could not find stations_path")
        return False
    else:
        return True

def computeNewStationPaths(config, new_station_id = None, stations_folder = "Stations"):
    """

    Args:
        config: [config] Config instance
        new_station_id: [str] New station id e.g. AU0004
        stations_folder: [str] optional, default Stations, name of the folder in ~/RMS to be created

    Returns:
        [tuple](path,path) new_station_config_location, new_station_data_dir
    """

    existing_station_config_path = os.path.expanduser("~/source/RMS/.config")
    if new_station_id is None:
        new_station_id = config.stationID

    new_station_id =new_station_id.lower()

    if os.path.exists(existing_station_config_path):
        stations_config_directory = os.path.dirname(os.path.expanduser("~/source/RMS"))
        new_station_config_location = os.path.join(stations_config_directory, stations_folder, new_station_id)
        new_station_data_dir = os.path.join(config.data_dir, new_station_id)

    return new_station_config_location, new_station_data_dir

def configureFirstStation(path_to_config = "~/source/RMS/.config"):
    """
    Gets operator input to configure the first station.
    Latitude, longitude and elevation will be propagated to all other cameras configured for this station
    station id and ip address will only be used for the first station to be configured

    Args:
        path_to_config: [path] optional, default "~/source/RMS/.config
    Returns:
        Nothing
    """
    path_to_config = os.path.expanduser(path_to_config)
    print("First station not yet configured")
    print("Please provide a stationID, location of the station, and the camera IP address")
    print("These values can be edited later")
    station_id = input("first station id:").upper()
    latitude = input("latitude (wgs84):")
    longitude = input("longitude (wgs84):")
    elevation = input("Elevation in metres above mean sea level:")
    ip_address = input("First station sensor ip address:")

    if os.path.exists(path_to_config):
        pass
    else:
        getConfigFromMaster(dest_path=os.path.dirname(path_to_config))


    fh = open(path_to_config, "r")
    config_lines = []
    for line in fh:
        config_lines.append(line)
    fh.close()

    config_lines = changeOptionValue(config_lines, "stationID", station_id)
    config_lines = changeOptionValue(config_lines, "latitude", latitude)
    config_lines = changeOptionValue(config_lines, "longitude", longitude)
    config_lines = changeOptionValue(config_lines, "elevation", elevation)
    config_lines = changeOptionValue(config_lines, "ip_address", ip_address)

    fh = open(path_to_config, "w")
    fh.writelines(config_lines)
    fh.close()


def getMaskFromMaster(source_url = "https://raw.githubusercontent.com/CroatianMeteorNetwork/RMS/prerelease/mask.bmp"
                                                                                        , dest_path = "~/source/RMS"):
    """
    Download mask file from github
    Args:
        source_url: [str] source URL for .config file
        dest_path: [path] destination for .config ilfe

    Returns: [bool] True if mask download succeeded

    """


    dest_path = os.path.expanduser(os.path.join(dest_path, os.path.basename(source_url)))
    mkdirP(os.path.dirname(dest_path))
    urllib.request.urlretrieve(source_url, dest_path)

    if os.path.exists(dest_path):
        return True
    else:
        print("Mask download from github failed")
        return False

def getConfigFromMaster(source_url = "https://raw.githubusercontent.com/CroatianMeteorNetwork/RMS/prerelease`/.config"
                                                                                        , dest_path = "~/source/RMS"):

    """
    Get the .config file from github
    Args:
        source_url: [str] URL to download config from
        dest_path: [path] location to save the .config file

    Returns: [bool] True if succeeded

    """

    dest_path = os.path.expanduser(os.path.join(dest_path, os.path.basename(source_url)))

    urllib.request.urlretrieve(source_url, dest_path)

    if os.path.exists(dest_path):
        return True
    else:
        print(".config download from github failed")
        return False



def createDesktopShortcuts(station):
    """
    Create the desktop shortcuts
    Args:
        station: [str] stationID

    Returns:
        Nothing
    """

    desktop_start_capture_path = os.path.expanduser("~/Desktop/{}_StartCapture.sh".format(station))
    if os.path.exists(desktop_start_capture_path):
        os.unlink(desktop_start_capture_path)
    fh = open(desktop_start_capture_path, "w")
    fh.writelines(createAutoStartEntry(station))
    fh.close

    desktop_show_live_stream_path = os.path.expanduser("~/Desktop/Show_LiveStream-{}.desktop".format(station))
    if os.path.exists(desktop_show_live_stream_path):
        os.unlink(desktop_show_live_stream_path)

    fh = open(desktop_show_live_stream_path, "w")
    fh.writelines(createShowLiveStreamEntry(station))
    fh.close()
    os.chmod(desktop_show_live_stream_path, 0o644)

def cleanDesktop():
    """
        Remove disused files from Desktop
    Returns:
        Nothing
    """

    files_to_delete = ["~/Desktop/CMNbinViewer.sh","~/Desktop/RMS_ShowLiveStream.sh","~/Desktop/RMS_StartCapture.sh",
	                   "~/Desktop/RMS_config.txt", "~/Desktop/TunnelIPCamera.sh", "~/Desktop/DownloadOpenVPNconfig.sh"]

    for f in files_to_delete:
        full_path = os.path.expanduser(f)
        if os.path.exists(full_path):
            os.unlink(full_path)



def copyPiStation(config_path ="~/source/RMS/.config", first_station = False, new_station_id = None, ip=None, debug=False):

    """
    Copies a station from ~/source/RMS to ~/source/Stations, and if it is the first station, migrates ~/RMS_data/etc
    to ~/RMS_data/$Station/etc

    Generally safe to run multiple times, as can detect if a station has already been migrated

    Args:
        config_path:[path] path to config file
        first_station: [bool] is this the first station
        new_station_id: [str] id of new station
        ip: [str] ip address for sensor of new station
        debug: [bool] prints debugging information

    Returns:
        nothing
    """

    # Get the path to the config file
    config_path = os.path.expanduser(config_path)

    # Do not copy the default station



    # Check the config path exists and then use to create stations
    if os.path.exists(config_path):
        config = cr.parse(config_path)

        if "XX" in config.stationID:
            mkdirP(os.path.expanduser("~/source/Stations"))
            return

        # If we are just migrating the first station, then use the stationID from the config file
        new_station_id = config.stationID if new_station_id is None else new_station_id

        # Work out the existing paths

        platepar_path = os.path.expanduser(os.path.join(os.path.dirname(config_path), config.platepar_name))
        mask_path = os.path.expanduser(os.path.join(os.path.dirname(config_path),config.mask_file))

        if debug:
            print("Platepar path {}, Mask path {}".format(platepar_path, mask_path))

        # Get the new station paths
        new_station_config_path, new_station_data_dir = computeNewStationPaths(config, new_station_id=new_station_id)

        # If both of the new paths exist, then stop working on this station
        if os.path.exists(new_station_config_path) and os.path.exists(new_station_data_dir):
            if debug:
                print("Station {} appears to have been migrated to multicam already, not doing any work".format(new_station_id))
            return

        # If first station is set then print some helpful information
        if first_station:
            print("It appears you have already configured your first station {}".format(new_station_id))
            print("so its config files will now be relocated to {:s}".format(new_station_config_path))
            print("Captured data from your original station is stored in {}".format(config.data_dir))
            print("This data will be moved to {:s}".format(new_station_data_dir))

            # Get rid of old icons
            cleanDesktop()

            # Has the data_dir already been created
            if os.path.exists(new_station_data_dir):
                print("Data directory for {} has already been migrated".format(new_station_id))
            else:
                # If it has not been migrated then work out where the existing archived and captured dir should be
                expected_path_of_captured_dir = os.path.join(config.data_dir, config.captured_dir)
                expected_path_of_archived_dir = os.path.join(config.data_dir, config.archived_dir)

                # Check that everything is where it should be before we start work
                if os.path.exists(expected_path_of_archived_dir) and os.path.exists(expected_path_of_captured_dir):

                    # For the first station only, we have to rename the whole RMS_data directory
                    # Strategy is to rename RMS_data to the name of the new station
                    # Then create a new RMS_data directory
                    # Then move the renamed directory into RMS_data

                    temp_rms_data_path = os.path.join(os.path.dirname(config.data_dir), config.stationID.lower())
                    if debug:
                        print("Temporary data path for first station {}".format(temp_rms_data_path))
                    moveIfExists(config.data_dir, temp_rms_data_path, debug=False)
                    if debug:
                        print("Final destination for first station {}".format(new_station_data_dir))
                    moveIfExists(temp_rms_data_path, new_station_data_dir, debug=False)
                else:
                    #
                    if debug:
                        print("Cannot find expected directories for {} in {}".format(new_station_id, config.data_dir))

            if os.path.exists(os.path.join(new_station_config_path,".config")):
                if debug:
                    print("Config for {} has already been migrated".format(new_station_id))
            else:
                mkdirP(new_station_config_path)
                copyIfExists(config_path, new_station_config_path, debug=False)


        else:
            # This is not the first station
            # Migrate the .config file
            if os.path.exists(new_station_config_path):
                if debug:
                    print("Config for {} has already been migrated".format(new_station_id))
            else:
                mkdirP(new_station_config_path)
                copyIfExists(config_path, new_station_config_path, debug=False)

        # Move the platepar
        copyIfExists(platepar_path, os.path.join(new_station_config_path, config.platepar_name), debug=False)

        # Move the mask
        if os.path.exists(os.path.join(new_station_config_path,os.path.basename(mask_path))):
            if debug:
                print("Mask already migrated to {}".format(new_station_config_path))
        else:
            if moveIfExists(mask_path, new_station_config_path):
                pass
            else:
                getMaskFromMaster(dest_path=new_station_config_path)

        # Finish off by creating desktop shortcuts
        createDesktopShortcuts(config.stationID)
        desktop_path = checkUserDesktopDirectoryEnvironment()
        if desktop_path != "":
            if debug:
                print("Desktop environment variable found as {}".format(desktop_path))
        else:
            if debug:
                print("No desktop environment variable found")
        # whilst adding stations set extra_space as 40GB
        extra_space_gb = 40

        customiseConfig(new_station_config_path, new_station_id.upper(), new_station_data_dir,
                                                    extra_space_gb, ip, reboot_after_processing=False)

        makeKeys()

def configureAutoStart(config_path, mode="autostart"):


    mode = mode.lower()
    config_path = os.path.expanduser(config_path)

    ### Delete this code eventually - this not the preferred way
    if mode=="wayfire":

        configuration_line = "rms = ~/source/RMS/Scripts/MultiCamLinux/Pi/RMS_StartCapture_MCP.sh"
        with open(config_path, 'r') as f:

            lines_list = f.readlines()

        auto_start_section_exists = False
        already_configured = False
        in_autostart_section = True
        for line in lines_list:
            if line == "[autostart]":
                autostart_section_exists = True
                in_autostart_section = True
            if in_autostart_section:
                if line == configuration_line:
                    already_configured = True

        if already_configured:
            return
        else:
            if not auto_start_section_exists:
                lines_list.append("\n[autostart]\n")
            lines_list.append("\n{}".format(configuration_line))

        with open(config_path, 'w') as f:

            for line in lines_list:
                f.write(line)

    if mode=="autostart":

        configuration_line = ""
        configuration_line += "[Desktop Entry]\n"
        configuration_line += "Type=Application\n"
        configuration_line += "Name=RMS_FirstRun\n"
        configuration_line += "Comment=Start RMS\n"
        configuration_line += "NoDisplay=true\n"
        configuration_line += "Exec=/usr/bin/lxterminal -e '~/source/RMS/Scripts/MultiCamLinux/Pi/RMS_StartCapture_MCP.sh'"

        with open(config_path, 'w') as f:
            f.writelines(configuration_line)




def removeExistingAutoStart(autostart_path):

    autostart_path = os.path.expanduser(autostart_path)

    dest = os.path.join(autostart_path, "hide")
    mkdirP(dest)
    moveIfExists(os.path.join(autostart_path, "RMS_FirstRun.desktop"), dest)

def rewireAutoStart():
    """
    This function converts the autostart system on the Pi to be suitable for MultiCamera operation

    The existing system uses ~/.config/autostart - and the design philosophy for MultiCamera is to change this
    as little as possible

    ~/.config/autostart/RMS_FirstRun.desktop

    Returns:

    """





if __name__ == "__main__":

    debug = False
    arg_parser = argparse.ArgumentParser(description=""" Deleting old observations.""")
    arg_parser.add_argument('-s', '--stations', nargs='*', metavar='STATIONS_TO_ADD', type=str, help="Station to run")
    arg_parser.add_argument('-a', '--addresses', nargs='*', metavar='IP_ADDRESSES', type=str, help="Camera ip addresses")
    arg_parser.add_argument('-l', '--launch', action='store_true', help="Launch stations")
    cml_args = arg_parser.parse_args()



    ignore_hardware = True

    stations_list = []
    if cml_args.stations is not None:
        stations_list = cml_args.stations
    else:
        stations_path = os.path.expanduser("~/source/Stations/")
        if not os.path.exists(stations_path):
            mkdirP(stations_path)
        if cml_args.launch:
            stations_list = os.listdir(stations_path)

    if cml_args.addresses is not None:
        ip_list = cml_args.addresses
    else:
        ip_list = []

    if isPi5() or ignore_hardware:
        if debug:
            print("Hardware reported as {}, continuing".format(piModel()))
    else:
        print("Pi 5 or similar required for multicamera operation, quitting")
        exit(1)

    # Is a station already configured
    if not firstStationConfigured() and not len(stations_list):
        # Get operator input to configure first station

        configureFirstStation()
        # Copy the first station to new location
        copyPiStation(first_station=True)

    # If no stations were configured at first run or not trying to launch ask for more stations

    if stations_list == [] or not cml_args.launch:
        stations_list, ip_list = getStationsToAdd(stations_list, ip_list)

    # Work through the list of stations
    for entry, ip in zip(stations_list, ip_list):
        s=sanitise(entry.lower())
        copyPiStation(new_station_id=s, first_station=False, ip=ip, debug=False)

    # This prevents gui from placing windows directly on top of each other
    uncomment("~/.config/wayfire.ini", "mode")

    # Create .ssh keys if they do not exist
    print(makeKeys(copy_pub_to="~/Desktop"))

    # Set timezone to UTC
    setTimeZone()

    # Compute disc use quotas
    quotas = computeQuotas()

    # And apply disc use quotas
    for entry in stations_list:
        path_to_config = os.path.join(os.path.expanduser("~/source/Stations"),entry.lower())
        setQuotas(path_to_config, quotas)

    rewireAutoStart()


    cameras = os.listdir(os.path.expanduser("~/source/Stations/"))
    cameras.sort()
    if cml_args.launch:
        for entry in cameras:
            entry = sanitise(entry)
            path_to_config = os.path.expanduser(os.path.join("~/source/Stations/",entry.lower(),".config"))
            launch_command = "lxterminal --title {} --command ".format(entry)
            launch_command += "'source ~/vRMS/bin/activate; python -m RMS.StartCapture -c {}; sleep 10'".format(path_to_config)
            print("Launching station {}".format(sanitise(entry).lower()))
            os.system(launch_command)
            time.sleep(60)
