# RPi Meteor Station
# Copyright (C) 2026 David Rollinson
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

from __future__ import print_function, division, absolute_import

import os
import requests
import tempfile
import subprocess
import json
import tqdm
import datetime
import random
import time
import tarfile
import shutil

from pathlib import Path
from RMS.Logger import getLogger
from RMS.ConfigReader import parse

# Constants

# urls
STATION_COORDINATES_JSON_URL = "https://globalmeteornetwork.org/data/kml_fov/GMN_station_coordinates_public.json"

# Paths

# stationID will be substitued by the station to download
REMOTE_PATH_TEMPLATE = "analysis@gmn.uwo.ca:/home/stationID/files/processed"

LOCAL_PATH_TEMPLATE = "/archive/cache/gmn/data/date"
REMOTE_FILE_JSON_PATH = "/archive/cache/gmn/index/remote_files.json"

# Suffixes
MINIMAL_ARCHIVE_SUFFIX = "minimal.tar.bz2"

log = getLogger("rmslogger")


def parseFilename(file_name):
    """Parse a GMN archive filename into its component fields.

    Arguments:
        file_name: [str]
            GMN archive filename.

    Return:
        start: [str]
            Filename prefix before the station identifier.

        station_id: [str]
            Station identifier.

        date_field: [str]
            Observation date in YYYYMMDD format.

        time_field: [str]
            Observation time in HHMMSS format.

        sub_time_field: [str]
            Sub-second timestamp field.

        end: [str]
            Filename suffix following the timestamp fields.
    """

    part_list = file_name.split("_")
    first_numeric = None

    for p, part in enumerate(part_list):
        if part.isnumeric():
            first_numeric = p
            break

    if first_numeric is None:
        raise ValueError(f"Unable to parse filename: {file_name}")

    station_id = part_list[first_numeric - 1]
    date_field = part_list[first_numeric]
    time_field = part_list[first_numeric + 1]
    sub_time_field = part_list[first_numeric + 2]

    start = "_".join(part_list[0:first_numeric-1])
    end = "_".join(part_list[first_numeric+3:])

    return start, station_id, date_field, time_field, sub_time_field, end

def buildFilename(station_id, date_field, time_field, sub_time_field, start="", end="" ):

    """Construct a GMN archive filename from component fields.

    Arguments:
        station_id: [str]
            Station identifier.

        date_field: [str]
            Observation date in YYYYMMDD format.

        time_field: [str]
            Observation time in HHMMSS format.

        sub_time_field: [str]
            Sub-second timestamp field.

    Keyword arguments:
        start: [str]
            Optional filename prefix.

        end: [str]
            Optional filename suffix.

    Return:
        file_name: [str]
            Constructed filename.
    """

    file_name = f"{station_id}_{date_field}_{time_field}_{sub_time_field}"

    if start != "":
        file_name = f"{start}_{file_name}"
    if end != "":
        if end[0] == ".":
            file_name = f"{file_name}{end}"
        else:
            file_name = f"{file_name}_{end}"

    return file_name

def filenameToRemotePath(file_name, remote_path_template):
    """Convert a filename into a remote GMN archive path.

    Arguments:
        file_name: [str]
            GMN archive filename.

        remote_path_template: [str]
            Template used to construct the remote station path.

    Return:
        remote_path: [str]
            Full remote path to the archive file.
    """

    start, station_id, date_field, time_field, sub_time_field, end = parseFilename(file_name)
    file_name = buildFilename(station_id, date_field, time_field, sub_time_field, start=start, end=end)
    remote_path = os.path.join(remote_path_template.replace("stationID", station_id.lower()), file_name)

    return remote_path

def stationIDToRemotePath(station_id, remote_path_template):
    """Construct the remote archive directory path for a station.

    Arguments:
        station_id: [str]
            Station identifier.

        remote_path_template: [str]
            Template used to construct the remote station path.

    Return:
        remote_path: [str]
            Remote path for the station archive directory.
    """
    remote_path = remote_path_template.replace("stationID", station_id.lower())

    return remote_path

def filenameToLocalPath(file_name, local_path_template=None, minimal_archive_suffix=None):

    """Convert a GMN filename into a local cache path.

        Arguments:
            file_name: [str]
                GMN archive filename.

        Keyword arguments:
            local_path_template: [str]
                Template used to construct the local cache path.

            minimal_archive_suffix: [str]
                Suffix used for cached minimal archives.

        Return:
            local_path: [str]
                Local cache path for the archive.
        """


    local_path_template = local_path_template or LOCAL_PATH_TEMPLATE

    if minimal_archive_suffix is None:
        minimal_archive_suffix = MINIMAL_ARCHIVE_SUFFIX

    start, station_id, date_field, time_field, sub_time_field, end = parseFilename(file_name)
    file_name = buildFilename(station_id, date_field, time_field, sub_time_field, start=start, end=minimal_archive_suffix)

    local_path = local_path_template.replace("date", date_field)
    local_path = os.path.join(local_path, file_name)

    return local_path

def parseServerFileTimestamp(file_name):
    """Extract the observation timestamp from a GMN filename.

    Arguments:
        file_name: [str]
            GMN archive filename.

    Return:
        timestamp: [datetime]
            UTC timestamp extracted from the filename.

        None:
            Returned if the filename cannot be parsed.
    """
    parts = file_name.split("_")
    if len(parts) < 4:
        return None

    date_str = parts[1]
    time_str = parts[2]

    try:
        dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None

def sortFilesByTime(files):
    """Sort filenames by observation timestamp.

    Arguments:
        files: [list]
            Collection of GMN archive filenames.

    Return:
        sorted_files: [list]
            Filenames sorted by observation time.
    """
    return sorted(files, key=parseServerFileTimestamp)

def getStationSet(url=None, country_code=None):
    """Retrieve the list of GMN stations.

    Keyword arguments:
        url: [str]
            URL of the station coordinates JSON file.

        country_code: [str]
            Optional country code filter.

    Return:
        station_list: [list]
            Sorted list of station identifiers.
    """

    if url is None:
        url = STATION_COORDINATES_JSON_URL

    response = requests.get(url)
    response.raise_for_status()  # fail fast if the JSON is unreachable

    stations_dict = response.json()
    station_list = []

    for station in stations_dict:
        station_lower = station.lower()

        if country_code is None:
            station_list.append(station)
        else:
            if station_lower.startswith(country_code.lower()):
                station_list.append(station)

    return sorted(station_list)

def getLocalFilesSet(local_path=None):
    """Retrieve the set of cached archive files.

    Keyword arguments:
        local_path: [str]
            Path template used to locate the cache directory.

    Return:
        local_files_set: [set]
            Set of archive file paths found in the local cache.
    """

    if local_path is None:
        local_path = LOCAL_PATH_TEMPLATE

    local_path = os.path.dirname(local_path)
    local_files_set = {f for f in Path(local_path).rglob("*.tar.bz2") if f.is_file()}

    return local_files_set

def lsRemote(remote_path, port=22):
    """List files in a remote archive directory.

    Arguments:
        remote_path: [str]
            Remote directory path.

    Keyword arguments:
        port: [int]
            SSH port used for the rsync connection.

    Return:
        files_set: [set]
            Set of filenames present in the remote directory.

            An empty set is returned if the directory does not exist.
    """

    files_set = set()


    while not files_set:
        cmd = [
            "rsync",
            "--list-only",
            "-z",
            "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p {}".format(port),
            remote_path + "/",
            "."
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Directory does not exist
        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "no such file" in stderr or "failed to open" in stderr:
                return files_set

            elif "connection refused" in stderr:
                log.info("Connect refused, waiting")
                time.sleep(600)
                continue

        for line in result.stdout.splitlines():
            parts = line.split()
            if parts:
                files_set.add(parts[-1])


    return files_set

def getRemotePathsSet(stations_set, remote_path_template=None):
    """Build remote archive paths for a collection of stations.

    Arguments:
        stations_set: [set]
            Set of station identifiers.

    Keyword arguments:
        remote_path_template: [str]
            Template used to construct station archive paths.

    Return:
        remote_path_set: [set]
            Set of remote archive directory paths.
    """
    if remote_path_template is None:
        remote_path_template = REMOTE_PATH_TEMPLATE

    remote_path_set = set()
    for station_id in stations_set:
        remote_path = stationIDToRemotePath(station_id, remote_path_template)
        remote_path_set.add(remote_path)

    return remote_path_set

def filterRemoteFiles(files_set):
    """Filter valid GMN archive files from a filename collection.

    Arguments:
        files_set: [set]
            Collection of filenames.

    Return:
        valid_tars_set: [set]
            Set of valid GMN archive filenames.
    """
    valid_tars_set = set()
    for file_name in files_set:
        if file_name.endswith("tar.bz2") and len(file_name.split("_")) == 5 and "imgdata" not in file_name:
            valid_tars_set.add(file_name)

    return valid_tars_set

def getRemoteFilesSet(station_coordinates_json_url=None,
                      remote_path_template=None,
                      country_code=None):
    """Retrieve the set of archive files available on the GMN server.

    Keyword arguments:
        station_coordinates_json_url: [str]
            URL of the station coordinates JSON file.

        remote_path_template: [str]
            Template used to construct station archive paths.

        country_code: [str]
            Optional country code filter.

    Return:
        remote_files_set: [set]
            Set of remote archive filenames.
    """



    if station_coordinates_json_url is None:
        station_coordinates_json_url = STATION_COORDINATES_JSON_URL

    if remote_path_template is None:
        remote_path_template = REMOTE_PATH_TEMPLATE

    if country_code is None:
        country_code = ""

    log.info(f"Downloading station coordinates from {station_coordinates_json_url}")
    stations_set = getStationSet(station_coordinates_json_url, country_code=country_code)
    log.info(f"Retrieving file lists for {len(stations_set)} stations")

    remote_paths_set = getRemotePathsSet(stations_set, remote_path_template=remote_path_template)

    remote_files_set = set()
    for remote_path in tqdm.tqdm(remote_paths_set):
        remote_files_set = remote_files_set | filterRemoteFiles(lsRemote(remote_path))

    return remote_files_set

def loadRemoteFileJson(remote_file_json_path=None):
    """Load a cached remote archive manifest.

    Keyword arguments:
        remote_file_json_path: [str]
            Path to the cached remote file manifest.

    Return:
        remote_file_set: [set]
            Set of archive filenames contained in the manifest.
    """

    if remote_file_json_path is None:
        remote_file_json_path = REMOTE_FILE_JSON_PATH

    if not os.path.exists(remote_file_json_path):
        return set()

    with open(remote_file_json_path) as f:
        return set(json.load(f))

def buildRemoteFileJson(remote_file_json_path=None,
                        station_coordinates_json_url=None,
                        remote_path_template=None,
                        country_code=None):
    """Build a manifest of files available on the GMN archive server.

    Keyword arguments:
        remote_file_json_path: [str]
            Path where the generated manifest will be written.

        station_coordinates_json_url: [str]
            URL of the station coordinates JSON file.

        remote_path_template: [str]
            Template used to construct station archive paths.

        country_code: [str]
            Optional country code filter.

    Return:
        remote_file_set: [set]
            Set of remote archive filenames written to the manifest.
    """


    if remote_file_json_path is None:
        remote_file_json_path = REMOTE_FILE_JSON_PATH

    if station_coordinates_json_url is None:
        station_coordinates_json_url = STATION_COORDINATES_JSON_URL

    if remote_path_template is None:
        remote_path_template = REMOTE_PATH_TEMPLATE

    if country_code is None:
        country_code = ""


    remote_file_set = getRemoteFilesSet(
                            station_coordinates_json_url=station_coordinates_json_url,
                            remote_path_template=remote_path_template,
                            country_code=country_code)

    os.makedirs(os.path.dirname(remote_file_json_path), exist_ok=True)

    with open(remote_file_json_path, "w") as f:
        json.dump(sortFilesByTime(remote_file_set), f, indent=4 )

    return remote_file_set

def downloadFilenameTo(file_name, target_directory,
                       remote_path_template=None,
                       max_tries=3):
    """Download an archive from the GMN server.

    Arguments:
        file_name: [str]
            GMN archive filename.

        target_directory: [str]
            Directory where the archive will be downloaded.

    Keyword arguments:
        remote_path_template: [str]
            Template used to construct station archive paths.

        max_tries: [int]
            Maximum number of download attempts.

    Return:
        downloaded_path: [str]
            Path to the downloaded archive.

        None:
            Returned if the archive could not be downloaded.
    """
    remote_path_template = remote_path_template or REMOTE_PATH_TEMPLATE
    remote_path = filenameToRemotePath(file_name, remote_path_template=remote_path_template)
    local_path = os.path.join(target_directory, file_name)


    download_count = 0

    while not os.path.exists(local_path):
        try:
            # Check whether the file exists
            download_count += 1
            result = subprocess.run(['rsync', '-z', remote_path], capture_output=True, text=True)

            if "No such file or directory" in result.stderr:
                return None

            elif "connection refused" in result.stderr:
                log.info("Connection refused, retry counter set to 0 and waiting")
                time.sleep(600)
                download_count = 0

            else:
                cmd = [
                    'rsync',
                    '--partial',
                    '--partial-dir=.rsync-partial',
                ]

            cmd.extend([remote_path, target_directory])

            # Make the download
            result = subprocess.run(cmd, capture_output=True, text=True)

            if "connection refused" in result.stderr:
                log.info("Connection refused, retry counter set to 0 and waiting")
                time.sleep(600)
                download_count = 0

        except:

            delay = random.randint(600, 900)
            resume_time = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=delay)).isoformat()
            print(f"Waiting {delay/60:.1f} minutes until {resume_time} for {file_name}")
            time.sleep(delay)
            return None

        print("Retrying download")
        time.sleep(600)

    downloaded_path = os.path.join(target_directory, file_name)


    return downloaded_path

def extractTo(file_path, target_directory):
    """Extract a compressed archive.

    Arguments:
        file_path: [str]
            Path to the archive file.

        target_directory: [str]
            Directory where the archive contents will be extracted.

    Return:
        target_directory: [str]
            Directory containing the extracted archive contents.

        None:
            Returned if the input file path is None.
    """

    if file_path is None:
        return None

    file_name = os.path.basename(file_path)
    #log.info(f"Extracting {file_name} to {target_directory}")

    while not os.path.exists(file_path):
        log.info(f"{file_name} does not exist, waiting for file system to sync")
        time.sleep(20)

    extract_success = False

    while not extract_success:
        with tarfile.open(file_path, 'r:bz2') as tar:
            # Following try / except is to handle earlier library versions which do not have filter keyword
            try:
                tar.extractall(path=target_directory, filter="fully_trusted")
                extract_success = True
            except:
                try:
                    tar.extractall(path=target_directory)
                    extract_success = True
                except:
                    extract_success = False
        if not extract_success:
            log.info(f"Failed extract for {file_name}")
        time.sleep(10)

    return target_directory

def copyToMinimalArchiveDirectory(file_name,
                                  source_path,
                                  destination_path):
    """Copy files required for a minimal archive.

    Extract the calibration and metadata files required by the
    StarPipeline from a full GMN archive and copy them into a
    minimal archive directory.

    Arguments:
        file_name: [str]
            GMN archive filename.

        source_path: [str]
            Path to the extracted archive contents.

        destination_path: [str]
            Directory where the minimal archive files will be copied.

    Return:
        destination_path: [str]
            Directory containing the minimal archive contents.
    """

    config_file_path = os.path.join(source_path, ".config")
    archive_config = parse(config_file_path)
    start, station_id, date_field, time_field, sub_time_field, end = parseFilename(file_name)

    file_to_copy_set = set()
    file_to_copy_set.add(config_file_path)


    if hasattr(archive_config, "mask_file"):
        file_to_copy_set.add(archive_config.mask_file)

    if hasattr(archive_config, "platepar_name"):
        file_to_copy_set.add(archive_config.platepar_name)

    if hasattr(archive_config, "platepars_recalibrated_name"):
        file_to_copy_set.add(archive_config.platepars_recalibrated_name)

    if hasattr(archive_config, "platepars_flux_recalibrated_name"):
        file_to_copy_set.add(archive_config.platepars_flux_recalibrated_name)

    file_to_copy_set.add(buildFilename(station_id, date_field, time_field, sub_time_field, start="CALSTARS", end=".txt"))
    file_to_copy_set.add(buildFilename(station_id, date_field, time_field, sub_time_field, start="FTPdetectinfo", end=".txt"))

    archive_name = os.path.basename(filenameToLocalPath(file_name, minimal_archive_suffix=""))
    destination_path = os.path.join(destination_path, archive_name)
    #log.info(f"Destination path is {destination_path}")
    os.makedirs(destination_path, exist_ok=True)

    for file_to_copy in file_to_copy_set:
        path_to_copy = os.path.join(source_path, file_to_copy)
        if os.path.exists(path_to_copy):
            shutil.copy2(path_to_copy, destination_path)
        else:
            log.info(f"{path_to_copy} does not exist")

    return destination_path

def makeMinimalArchive(filename,
                       archive_preparation_directory,
                       minimal_archived_files_directory_path,
                       minimal_archive_suffix=None):
    """Create a compressed minimal archive.

    Arguments:
        filename: [str]
            GMN archive filename.

        archive_preparation_directory: [str]
            Directory where the archive will be created.

        minimal_archived_files_directory_path: [str]
            Directory containing the files to archive.

    Keyword arguments:
        minimal_archive_suffix: [str]
            Suffix used for minimal archive filenames.

    Return:
        archive_path: [str]
            Path to the generated minimal archive.
    """

    if minimal_archive_suffix is None:
        minimal_archive_suffix = MINIMAL_ARCHIVE_SUFFIX

    cached_archive_path = filenameToLocalPath(filename, minimal_archive_suffix=minimal_archive_suffix)

    archive_name = os.path.basename(cached_archive_path)
    #log.info(f"Making {archive_name}")
    archive_path = os.path.join(archive_preparation_directory, archive_name)

    #log.info(f"Making an archive from directory {minimal_archived_files_directory_path} in {archive_path}")
    with tarfile.open(archive_path, "w:bz2") as tar:
        cached_archive_name = os.path.basename(filenameToLocalPath(filename, minimal_archive_suffix=""))
        tar.add(minimal_archived_files_directory_path, arcname=cached_archive_name)

    return archive_path

def moveToCache(file_name, archive_path,
                local_path_template=None):
    """Move a minimal archive into the local cache.

    Arguments:
        file_name: [str]
            GMN archive filename.

        archive_path: [str]
            Path to the generated minimal archive.

    Keyword arguments:
        local_path_template: [str]
            Template used to construct local cache paths.

    Return:
        local_cache_path: [str]
            Final path of the cached archive.
    """
    if local_path_template is None:
        local_path_template = LOCAL_PATH_TEMPLATE

    local_cache_path = filenameToLocalPath(file_name, local_path_template=local_path_template)
    #log.info(f"Moving archive from {archive_path} to {local_cache_path}")
    local_cache_path_dir = os.path.dirname(local_cache_path)
    os.makedirs(local_cache_path_dir, exist_ok=True)
    os.chmod(local_cache_path_dir, 0o775)
    shutil.move(archive_path, local_cache_path)
    os.chmod(local_cache_path, 0o666)

    return local_cache_path

def createMinimalArchive(file_name,
                         local_path_template=None,
                         remote_path_template=None,
                         minimal_archive_suffix=None):
    """Create and cache a minimal archive.

    Download a full GMN archive, extract the files required by the
    StarPipeline, create a minimal archive, and store it in the
    local cache.

    Arguments:
        file_name: [str]
            GMN archive filename.

    Keyword arguments:
        local_path_template: [str]
            Template used to construct local cache paths.

        remote_path_template: [str]
            Template used to construct remote archive paths.

        minimal_archive_suffix: [str]
            Suffix used for minimal archive filenames.

    Return:
        cached_path: [str]
            Path to the cached minimal archive.

        None:
            Returned if archive creation fails.
    """
    local_path_template = local_path_template or LOCAL_PATH_TEMPLATE
    remote_path_template = remote_path_template or REMOTE_PATH_TEMPLATE
    minimal_archive_suffix = minimal_archive_suffix or MINIMAL_ARCHIVE_SUFFIX
    cached_path = None

    with tempfile.TemporaryDirectory() as temp_download_directory:
        temp_downloaded_path = downloadFilenameTo(file_name, temp_download_directory, remote_path_template=remote_path_template)
        if temp_downloaded_path is None:
            return None
        with tempfile.TemporaryDirectory() as temp_extract_directory:
            temp_extracted_path = extractTo(temp_downloaded_path, temp_extract_directory)
            with tempfile.TemporaryDirectory() as minimal_archive_files_directory:
                minimal_archived_files_directory_path = copyToMinimalArchiveDirectory(file_name, temp_extracted_path, minimal_archive_files_directory)
                with tempfile.TemporaryDirectory() as archive_preparation_directory:
                    archive_path = makeMinimalArchive(file_name, archive_preparation_directory, minimal_archived_files_directory_path, minimal_archive_suffix)
                    cached_path = moveToCache(file_name, archive_path, local_path_template=local_path_template)

    return cached_path

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Download minimal cache of GMN raw data \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('--remote-path-template', default=None, help="Remote path template")

    arg_parser.add_argument('--local-path-template', default=None, help="Local cache path template")

    arg_parser.add_argument('--minimal-archive-suffix', default=None, help="Archive filename suffix")

    arg_parser.add_argument('--station-coordinates-json-url', default=None, help="Station coordinates JSON URL")

    arg_parser.add_argument('--remote-file-json-path', default=None, help="Path to cached remote file manifest")

    arg_parser.add_argument('--country-code', default=None, help="Country code to retrieve")

    arg_parser.add_argument('-s', '--scan_remote_file_store', dest='scan_remote_file_store',
                            default=False, action='store_true', help='Scan the remote file store')

    cml_args = arg_parser.parse_args()


    country_code = cml_args.country_code or ""

    scan_remote_file_store = False

    if cml_args.scan_remote_file_store:
        log.info("Scanning remote file store because of command line argument.")
        scan_remote_file_store = True


    station_coordinates_json_url = cml_args.station_coordinates_json_url or STATION_COORDINATES_JSON_URL
    remote_file_json_path = cml_args.remote_file_json_path or REMOTE_FILE_JSON_PATH
    if not os.path.exists(remote_file_json_path):
        log.info(f"No local copy of files in {remote_file_json_path}, scanning remote file store.")
        scan_remote_file_store = True


    remote_path_template = cml_args.remote_path_template or REMOTE_PATH_TEMPLATE
    if scan_remote_file_store:
        remote_file_set = buildRemoteFileJson(remote_file_json_path=remote_file_json_path,
                            station_coordinates_json_url=station_coordinates_json_url,
                            remote_path_template=remote_path_template,
                            country_code=country_code)

    else:
        try:
            remote_file_set = loadRemoteFileJson(remote_file_json_path=remote_file_json_path)
        except:
            log.info(f"Unable to load {remote_file_json_path}, rebuilding.")
            remote_file_set = buildRemoteFileJson(remote_file_json_path=remote_file_json_path,
                                                  station_coordinates_json_url=station_coordinates_json_url,
                                                  remote_path_template=remote_path_template,
                                                  country_code=country_code)



    remote_file_list = sortFilesByTime(remote_file_set)
    files_to_download_set = set()

    local_path_template = cml_args.local_path_template or LOCAL_PATH_TEMPLATE
    minimal_archive_suffix = cml_args.minimal_archive_suffix or MINIMAL_ARCHIVE_SUFFIX

    files_found = 0

    for file_name in remote_file_list:
        local_path = filenameToLocalPath(file_name, local_path_template=local_path_template)
        if os.path.exists(local_path):
            files_found += 1
        else:
            files_to_download_set.add(file_name)

    if files_found:
        log.info(f"Found {files_found} archives already in cache, {len(files_to_download_set)} will be downloaded.")

    downloaded_count = 0
    total_files_to_download = len(files_to_download_set)
    download_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    time_per_download = 30
    forecast_completion_time = datetime.datetime.now(tz=datetime.timezone.utc) + (datetime.timedelta(seconds=total_files_to_download * time_per_download))

    for file_to_download in sortFilesByTime(list(files_to_download_set)):
        files_to_download = total_files_to_download - downloaded_count
        log.info(f"Downloading {file_to_download} archives forecast completion at {forecast_completion_time:%Y-%m-%d %H:%M:%S} {files_to_download} archives remaining at {time_per_download:.0f} seconds per archive.")
        local_path = createMinimalArchive(file_to_download, remote_path_template=remote_path_template, local_path_template=local_path_template)
        downloaded_count +=1
        time_now = datetime.datetime.now(tz=datetime.timezone.utc)
        elapsed_time = time_now - download_start_time
        time_per_download = elapsed_time.total_seconds() / downloaded_count

        forecast_completion_time = (datetime.datetime.now(tz=datetime.timezone.utc) +
                                    datetime.timedelta(seconds=time_per_download * files_to_download))