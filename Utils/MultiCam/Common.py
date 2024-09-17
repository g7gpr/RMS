import subprocess
import os
from shutil import move, copy
from glob import glob

from RMS.Astrometry.Conversions import latLonAlt2ECEF
from RMS.DeleteOldObservations import availableSpace, usedSpaceNoRecursion
from RMS.Misc import mkdirP

def moveIfExists(src, dest, debug=False):

	"""
	Wrapper for move, checks to see if files exist

	Args:
		src:[path] source file
		dest:[dest] destination path or file
		debug: [bool] optional, default false, print debugging information

	Returns:

	"""



	src, dest = os.path.expanduser(src),  os.path.expanduser(dest)
	if os.path.exists(src):
		move(src, dest)
		if debug:
			print("{} was found and moved to {}".format(src, dest))
		return True
	else:
		if debug:
			print("{} was not found, cannot move".format(src))
		return False

def copyIfExists(src, dest, debug=False):
	"""
	Wrapper for copy, checks to see if files exist

	Args:
		src:[path] source file
		dest:[dest] destination path or file
		debug: [bool] optional, default false, print debugging information

	Returns:

	"""
	src, dest = os.path.expanduser(src),  os.path.expanduser(dest)
	if os.path.exists(src):
		dest_name = os.path.join(dest, os.path.basename(src))
		if not os.path.exists(os.path.dirname(dest)):
			mkdirP(dest)
		if os.path.exists(dest_name):
			os.unlink(dest_name)
		copy(src, dest)
		if debug:
			print("{} was found and copied to {}".format(src, dest))
		return True
	else:
		if debug:
			print("{} was not found, cannot copy".format(src))
		return False


def checkUserDesktopDirectoryEnvironment():


	"""
	Discover which desktop is running

	Returns:
		path to the xdg_user_dir
	"""


	xdg_user_dir = subprocess.check_output(['xdg-user-dir', 'DESKTOP']).decode('utf-8')

	if os.path.exists(xdg_user_dir):
		return xdg_user_dir
	else:
		return ""

def createAutoStartEntry(station):

	"""
	Add auto start entry for one station
	Args:
		station (): stationID

	Returns:
		string with newlines to be added to configuration file
	"""


	entry = ""
	entry += "[Desktop Entry]\n"
	entry += "Name=${item}-Startcapture\n"
	entry += "Type=Application\n"
	entry += 'Exec=lxterminal --title={} -e "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh {}"\n'.format(station, station)
	entry += "Hidden=false\n"
	entry += "NoDisplay=false\n"
	entry += "Icon=lxterminal\n"

	return entry


def createShowLiveStreamEntry(station):

	"""
	Create show live stream desktop entry
	Args:
		station (): stationID

	Returns:
		newline delimited string to be used to create the showlive stream entry
	"""


	entry = ""
	entry += "[Desktop Entry]\n"
	entry += "Name=${item}-ShowLiveStream\n"
	entry += "Type=Application\n"
	entry += 'Exec=lxterminal --title=Stream-{} -e "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh {}"\n'.format(station, station)
	entry += "Hidden=false\n"
	entry += "NoDisplay=false\n"
	entry += "Icon=lxterminal\n"

	return entry

def uncomment(file_path, setting, comment_marker = "#", backup_file_ext = "backup"):
	"""
	Remove a comment from a line

	Args:
		file_path: [path] path to file to be worked on
		setting: [str] the setting to be uncommented
		comment_marker: [char] optional, default # comment marker characeter
		backup_file_ext: [string] backup file is created, this

	Returns:
		[bool] True if change was made, else false
	"""


	backup_file = os.path.join(os.path.dirname(file_path), "{}.{}".format(os.path.basename(file_path), backup_file_ext))
	backup_file = os.path.expanduser(backup_file)
	copyIfExists(file_path, backup_file)
	setting_commented = "{}{}".format(comment_marker, setting)
	file_path = os.path.expanduser(file_path)
	if not os.path.exists(file_path):
		return
	with open(file_path, 'r') as f:
		file_as_list = f.readlines()

	with open(file_path, 'w') as f:

		change_made = False
		for line in file_as_list:
			if setting_commented in line:
				print("Removing comment from {}".format(line))
				line = line.replace(setting_commented, setting)
				print("Written as            {}".format(line))
				change_made = True
			f.write(line)
	if os.path.exists(backup_file):
		os.unlink(backup_file)
	return change_made


def getStationsToAdd(stations_list=[], ip_list=[], debug=False):

	"""
	Request user input for which stations to add.
	Not compatible with Python 2.7 as input behaves very differently.
	Returns:
		list of stations to add
	"""



	while True:
		response = input("Enter station ID, <cr> to end: ")
		if response == "":
			break
		else:
			if validateStationData(response.upper(),0,0,0, "192.168.1.1"):
				stations_list.append(response.upper())
			else:
				print("Station ID not in expected format of two letters")
				print("followed by 4 alphanumeric characters, excluding")
				print("letters O and I")
				continue
		response = input("Enter sensor ip for {}: ".format(response.upper()))
		if response == "":
			break
		else:
			ip_list.append(response)



	return stations_list, ip_list

def changeOptionValue(lines_list, option, value, delimiter = ":"):

	"""
	Change an individual option in a .config file

	Args:
		lines_list (): list of lines read from a file
		option (): the option to be changed
		value (): value to be changed to
		delimiter (): optional, default : the delimiter between option and value
		delimiter (): optional, default : the delimiter between option and value

	Returns:
		output_list : entire file, with the lines changed referring to the option
	"""
	change_made = False
	modified_option = False
	if option == "ip_address":
		option = "device"
		modified_option = True


	output_list = []
	for line in lines_list:
		if len(line) > len(option):
			if line.lower()[:len(option)] == option.lower():
				if option == "device" and modified_option:
					protocol = line.split(":")[1].strip()
					tail = line.split(":")[3].strip()
					output_line = "{}: {}://{}:{}\n".format(option, protocol, value, tail)
				else:
					output_line = "{}{} {}\n".format(option, delimiter, value)
				change_made = True
			else:
				output_line = line
		else:
			output_line = line
		output_list.append(output_line)

	if not change_made:
		print("Warning, option {} was not changed to {} - is your .config file up to date?".format(option, value))

	return output_list




def customiseConfig(path_to_config, stationid, data_dir, extra_space, ip=None, reboot_after_processing=True):

	"""
	Make the changes to the .config file per station
	Args:
		path_to_config ():path to the .config file to be customised
		stationid (): the stationID to be used
		data_dir (): the data_directory to be used
		extra_space (): the extra_space to be allowed
		reboot_after_processing (): optional, default False, generally not desirable for multiple camera systems

	Returns:
		nothing
	"""


	fh = open(os.path.join(path_to_config,".config"),"r")
	config_lines = []
	for line in fh:
		config_lines.append(line)
	fh.close()

	config_lines = changeOptionValue(config_lines, "stationID", stationid)
	config_lines = changeOptionValue(config_lines, "data_dir", data_dir)
	config_lines = changeOptionValue(config_lines, "extra_space", extra_space)
	config_lines = changeOptionValue(config_lines, "reboot_after_procesing", reboot_after_processing)

	if ip is not None:
		config_lines = changeOptionValue(config_lines, "ip_address", ip)

	fh = open(os.path.join(path_to_config, ".config"), "w")
	fh.writelines(config_lines)
	fh.close()





def checkForKeys(key_path = "~/.ssh/id_rsa"):

	"""
	Discover if keys exist in a specific location
	Args:
		key_path (): optional, path to the private key

	Returns:
		boolean: True if the private key exists
	"""

	return os.path.exists(os.path.expanduser(key_path))


def makeKeys(key_path = "~/.ssh", copy_pub_to = None, permit_create = False):


	"""
	Make .ssh keys
	Args:
		key_path (): path where keys should be placed

	Returns:
		nothing
	"""
	message = ""

	public_key_path = getPublicKeyPath()
	if public_key_path is None and not permit_create:
		return "No key directory found, will continue to create stations, but keys must be created."

	if not checkForKeys() and not permit_create:
		return "No keys found in key directory, and not permitted to create new keys. Keys must be created to allow uploads."

	if not checkForKeys():
		if permit_create:
			message += "Generating keys"
			subprocess.run(['ssh-keygen', '-t', 'rsa', '-f', key_path, '-q', '-p'])
	else:
		message += "Keys already created"
	if copy_pub_to is None:
		return ""

	copy_pub_to = os.path.expanduser(copy_pub_to)


	if os.path.exists(copy_pub_to) and not os.path.exists(
												os.path.join(copy_pub_to, os.path.basename(public_key_path))):
		copyIfExists(getPublicKeyPath(), copy_pub_to)
		message += " in {}\n".format(key_path)
		message += "Your new id_rsa.pub public key file is at {}\n".format(copy_pub_to)
		message += "Be sure to send a copy of this file to Denis"

		return message
	else:
		return ""

def getPublicKeyPath(key_dir="~/.ssh"):
	"""

	Args:
		key_dir: [path] optional, deafault ~/.ssh

	Returns:
		path to the public key file (i.e. *.pub)
	"""
	key_dir = os.path.expanduser(key_dir)
	pub_keys = glob(os.path.join(key_dir, "*.pub"))
	if len(pub_keys):
		return pub_keys[0]
	else:
		return None

def computeQuotas(stations_path="~/source/Stations", debug=False, allowance_for_one_night = 18):
	"""
	Compute disc use quotas in a multicamera station
	Args:
		stations_path:[path] optional, default ~/source/Stations
		debug:[bool] optional, default False, print debugging information
		allowance_for_one_night: [float] optional, default 18, amount of space expected to consumed each session
	Returns:[tuple](rms_data_quota, arch_dir_quota, bz2_files_quota)

	"""

	stations_path = os.path.expanduser(stations_path)
	if not os.path.exists(stations_path):
		print("Cannot find existing stations in {}, so cannot compute quotas".format(stations_path))
		return 10, 1, 1

	# Get the available space on the drive
	available_space_gb = availableSpace("/") / (1024 ** 3)

	if debug:
		print("Space available on whole drive: {}GB".format(available_space_gb))


	# Find space already used for data
	space_used_by_existing_stations_gb = usedSpaceNoRecursion("~/RMS_data")

	# Add to this the available space on the drive, to give the ultimate amount available for storage
	total_space_available_for_data_gb = available_space_gb + space_used_by_existing_stations_gb

	if debug:
		print("Total space available for data is {}GB".format(total_space_available_for_data_gb))

	# Get the number of stations by counting the entries in ~/source/Stations
	number_of_stations = len(os.listdir(os.path.expanduser("~/source/Stations")))

	if debug:
		print("There are {} stations configured".format(number_of_stations))

	# Compute and round the total allowance for rms_data per station
	rms_data_quota = int((total_space_available_for_data_gb / number_of_stations) - allowance_for_one_night)
	if debug:
		print("Allowing {} per station".format(rms_data_quota))

	# Allocate 0.1 of this space for archived directories
	arch_dir_quota = int(rms_data_quota * 0.1)

	# Allocate 0.05 of this space for bz2 files
	bz2_files_quota = int(rms_data_quota * 0.05)
	if debug:
		print("Allowing {} for archived directories, and {} for bz2 files".format(arch_dir_quota, bz2_files_quota))

	return rms_data_quota, arch_dir_quota, bz2_files_quota

def setTimeZone(time_zone="UTC"):

	"""
	Set the time zone to UTC
	Args:
		time_zone (): optional, default UTC

	Returns:
		nothing
	"""

	subprocess.run(['timedatectl', 'set-timezone', time_zone])


def setQuotas(path_to_config, quotas_tuple, debug=False):
	"""

	Sets the quota information in the .config file
	Args:
		path_to_config:[path] path to config file
		quotas_tuple: [tuple](rms_data_quota, arch_dir_quota, bz2_files_quota)
		debug: [bool] print debugging information

	Returns:
		nothing
	"""

	if debug:
		print("Working on {}".format(path_to_config))
		print("Settings quotas as rms_data {}".format(quotas_tuple[0]))
		print("             arch_dir_quota {}".format(quotas_tuple[1]))
		print("            bz2_files_quota {}".format(quotas_tuple[2]))



	fh = open(os.path.join(path_to_config,".config"),"r")
	config_lines = []
	for line in fh:
		config_lines.append(line)
	fh.close()

	config_lines = changeOptionValue(config_lines, "rms_data_quota", quotas_tuple[0])
	config_lines = changeOptionValue(config_lines, "arch_dir_quota", quotas_tuple[1])
	config_lines = changeOptionValue(config_lines, "bz2_files_quota", quotas_tuple[2])

	fh = open(os.path.join(path_to_config, ".config"), "w")
	fh.writelines(config_lines)
	fh.close()


def validateStationData(station_id, lat, lon, elevation, ip_address):
	return station_id[0:2].isalpha() and station_id[2:6].isalnum() \
		and -90 < lat < 90 and -180 < lon < 360 \
		and -100 < elevation < 10000 \
		and not ("i" in ststion_id.lower()) \
		and not ("o" in station_id.lower())


def detectMostRecentLogAccess(config, time_window = 30):

	log_dir = os.path.join(config.data_dir, config.log_dir)
	latest_time_stamp = 0
	if os.path.exists(os.path.join(config.data_dir, config.log_dir)):

		for file_name in os.listdir(log_dir):
			time_stamp = os.path.getmtime(os.path.join(log_dir, file_name))
			if time_stamp > latest_time_stamp:
				latest_time_stamp = time_stamp

	return latest_time_stamp







