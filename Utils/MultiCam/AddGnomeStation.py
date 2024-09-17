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

from RMS.Misc import isRaspberryPi
import os
import subprocess
import argparse
from datetime import datetime
from RMS.Misc import mkdirP
from shutil import copy
from Utils.MultiCam.Common import createAutoStartEntry, createShowLiveStreamEntry
from Utils.MultiCam.Common import checkUserDesktopDirectoryEnvironment, customiseConfig, makeKeys


def detectPi():

	"""
	Discover if this is Raspberry Pi platform, and terminate if it is

	Returns:
		nothing
	"""

	if isRaspberryPi():
		print("The add_GStation.sh script should not be used on Raspberry Pi.")
		print("Please use add_Pi_Station.sh to add cameras on a Pi5.")
		exit(1)


def install(package, purge_first = False, ask = False):
	"""
	Call apt-get to install a package
	Args:
		package (): package to be installed
		purge_first (): optional, default false, remove the package before installation
		ask (): option, default false, ask terminal before installing

	Returns:
		Nothing
	"""


	if purge_first:
		subprocess.run(['sudo', '-S', 'apt-get', 'purge', package, "-y"])

	if subprocess.run(['dpkg', '-L', package], capture_output=True).returncode != 0:
		if ask:
			response = input("Do you wish to install {}?".format(package)).lower()
			if response == 'y':
				do_install = True
			else:
				do_install = False
		else:
			do_install = True
	else:
		do_install = False
	if do_install:
		subprocess.run(['sudo', '-S', 'apt-get', 'install', package, "-y"])

	else:
		pass

def getLxSettings():

	"""
	Function to return settings to be applied to the LX Terminal settings file
	Returns:
		list of settings to be applied
	"""


	settings = []
	settings.append(["scrollback","10000"])
	settings.append(["geometry_columns", "120"])
	settings.append(["geometry_rows", "25"])

	return settings




def configureLxTerminal(settings_list, config_file_path_and_name):

	"""
	Edit the LX Terminal settings file
	Args:
		settings_list : list of settings
		config_file_path_and_name : path and name to config file

	Returns:
		nothing
	"""


	subprocess.run(['lxterminal','-e','exit'])
	config_file_path_and_name = os.path.expanduser(config_file_path_and_name)

	fh = open(config_file_path_and_name, "r")
	config_as_list = []
	for line in fh:
		config_as_list.append(line)
	fh.close()

	output_line_list = []
	for line in config_as_list:
		written_line = False
		for setting, value in settings_list:
			if line[:len(setting)] == setting:
				output_line_list.append ("{}={}\n".format(setting,value))
				written_line = True
		if not written_line:
			output_line_list.append(line)

	fh = open(config_file_path_and_name, "w")
	fh.writelines(output_line_list)
	fh.close()

def getStationsToAdd():

	"""
	Request user input for which stations to add.
	Not compatible with Python 2.7 as input behaves very differently.
	Returns:
		list of stations to add
	"""

	stations_list = []

	while True:
		response = input("Enter station ID, <cr> to end: ")
		if response == "":
			break
		else:
			stations_list.append(response)


	return stations_list


def addStations(stations_list):

	"""
	Create and populate the folder structure for the new stations

	In brief this is

	~/source/Stations/xx0001/
					 /xx0002/
					 /xx0003/


	Args:
		stations_list (): a list of station to be added

	Returns:
		nothing
	"""

	source_dir = os.path.expanduser("~/source")
	stations_path = os.path.join(source_dir, "Stations")

	mkdirP(stations_path)
	mkdirP(os.path.expanduser("~/RMS_data"))
	mkdirP(os.path.expanduser("~/.config/autostart"))

	for station in stations_list:
		station = station.lower()
		new_station_config_path = os.path.join(stations_path, station)
		if os.path.exists(new_station_config_path):
			print("Station {} already exists".format(station.upper()))
		else:
			addStation(station, len(stations_list))


def createSoftLinksForStartCapture(station, free_space_allocation_per_station = 30):

	"""
	Create the softlinks required for a single station

	Args:
		station (): stationID
		free_space_allocation_per_station (float): optional default 30 GB free space to be allocated per station

	Returns:
		nothing
	"""

	src_path = os.path.expanduser("~/.config/autostart/{}_StartCap.desktop".format(station))
	dest_path = os.path.expanduser("~/Desktop/{}_StartCap.desktop".format(station))
	os.chmod(src_path, 0o755)
	if os.path.exists(dest_path):
		os.unlink(dest_path)
	os.symlink(src_path, dest_path)
	os.chmod(dest_path, 0o755)
	subprocess.run(['gio', 'set', src_path,"metadata::trusted", "true"])


def addStation(station, number_of_stations):

	"""
	Add the station, and customise the .config file as required

	Args:
		station (): stationID
		number_of_stations (): total_number_of_stations, used for configuring extra_space parameter

	Returns:

	"""

	# Create path variables
	source_dir = os.path.expanduser("~/source")
	stations_path = os.path.join(source_dir, "Stations")
	original_config_path = os.path.join(source_dir, "RMS", ".config")
	new_station_config_path = os.path.join(stations_path, station.lower())
	station_config_path_and_name = os.path.join(new_station_config_path, ".config")

	# Create the new directory for station config, mask, platepar etc
	mkdirP(new_station_config_path)
	copy(original_config_path, station_config_path_and_name)

	# Setup the start capture path
	desktop_start_capture_path = os.path.expanduser("~/.config/autostart/{}_StartCap.desktop".format(station))
	if os.path.exists(desktop_start_capture_path):
		os.unlink(desktop_start_capture_path)

	# Setup the auto start for this station
	fh = open(desktop_start_capture_path,"w")
	fh.writelines(createAutoStartEntry(station))
	fh.close()
	os.chmod(desktop_start_capture_path, 0o755)
	createSoftLinksForStartCapture(station)

	# Move the mask to the new location
	copy(os.path.join(source_dir,"RMS/mask.bmp"),new_station_config_path)

	# Create the path for the live_stream icon
	desktop_show_live_stream_path = os.path.expanduser("~/Desktop/Show_LiveStream-{}.desktop".format(station))
	if os.path.exists(desktop_show_live_stream_path):
		os.unlink(desktop_show_live_stream_path)

	# Create the icon
	fh = open(desktop_show_live_stream_path,"w")
	fh.writelines(createShowLiveStreamEntry(station))
	fh.close()
	os.chmod(desktop_show_live_stream_path, 0o644)

	subprocess.run(['gio', 'set', desktop_show_live_stream_path, "metadata::trusted", "true"])
	os.chmod(desktop_show_live_stream_path, 0o777)
	data_dir = os.path.join(os.path.expanduser("~/RMS_data"),station)
	extra_space = number_of_stations * free_space_allocation_per_station

	# Apply the new settings the .config file
	customiseConfig(new_station_config_path, station.upper(), data_dir, reboot_after_processing=False)


def applyGsettings(screen_timeout = 600, lock_screen_enabled = False):

	"""
	Apply appropriate settings to Gnome
	Args:
		screen_timeout (): optional, default 600, screen lock time_out
		lock_screen_enabled (): optional, default false, screen_lock

	Returns:
		nothing
	"""

	if lock_screen_enabled:
		lock_screen_setting = "true"
	else:
		lock_screen_setting = "false"
	subprocess.run(['gsettings', 'set', 'org.gnome.settings-daemon.plugins.power', 'sleep-inactive-ac-type', "'nothing'"])
	subprocess.run(['gsettings', 'set', 'org.gnome.settings-daemon.plugins.power', 'sleep-inactive-ac-timout', '0'])
	subprocess.run(['gsettings', 'set', 'org.gnome.desktop.session', 'idle-delay', str(screen_timeout)])
	subprocess.run(['gsettings', 'set', 'org.gnome.desktop.screensaver', 'lock_enabled', lock_screen_setting])


def addActivate(bashrc_path=os.path.expanduser("~/.bashrc")):

	"""
	Update bashrc with settings for RMS
	Args:
		bashrc_path (): optional, default ~/.bashrc, path to the .bashrc file

	Returns:
		nothing
	"""


	bashrc_path = os.path.expanduser("~/.bashrc")

	fh = open(bashrc_path,'r')

	for line in fh:
		if "cd ~/source/RMS" in line:
			return
	fh.close()

	fh = open(bashrc_path, 'a')
	fh.write("\n")
	fh.write("# added by RMS {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	fh.write("\n")
	fh.write("cd ~/source/RMS\n")
	fh.write("source ~/vRMS/bin/activate")
	fh.close()



if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description="Add a station to MultiCameraLinux.")

	arg_parser.add_argument("-d","--rms_data", nargs='?', default="~/RMS_data", help="Path to the data directory")

	arg_parser.add_argument("-s","--stations", default="",
							help="Stations to add")



	# Parse the args
	cml_args = arg_parser.parse_args()

	# Are we running on Pi, should not be - will exit if we are
	detectPi()
	# I am not sure what to do with this information?
	checkUserDesktopDirectoryEnvironment()

	install("lxterminal", purge_first=False)
	configureLxTerminal(getLxSettings(), "~/.config/lxterminal/lxterminal.conf" )

	if cml_args.stations == "":
		stations_list = getStationsToAdd()
	else:
		stations_list = cml_args.stations.split(",")

	addStations(stations_list)
	makeKeys()
	setTimeZone()
	applyGsettings()
	addActivate()
	install("pcmanfm", ask=True)
	install("mousepad", ask=True)