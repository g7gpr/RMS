# Copyright (C) 2024
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


"""
Routine to mirror the daily trajectory files from https://globalmeteornetwork.org/data/traj_summary_data/
and from them create a file with all the trajectories, optionally without the duplicates.

"""

import os
import sys
from html.parser import HTMLParser
from dateutil import parser
import time
import numpy as np
from numpy.ma.extras import column_stack

from RMS.Misc import mkdirP
import RMS.ConfigReader as cr
import random
import datetime
import sqlite3

if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os, ssl

        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request

traj_summary_all_filename = "traj_summary_all.txt"

def columnTranslate(column):

    column = column.replace("q AU", "Perihelion AU")
    column = column.replace("Q AU", "Aphelion AU")

    return column

def createDataBase(config):

    return sqlite3.connect(os.path.join(os.path.expanduser(config.data_dir), "trajectories.db"))

def tableExists(conn, table_name):

    query = "SELECT name FROM sqlite_master WHERE type='table' AND name='{}';".format(table_name)
    if conn.cursor().execute(query).fetchone() is None:
        return False

    return True


def createTable(conn, column_list):

    create_table_statement = ""
    create_table_statement += "CREATE TABLE 'Trajectories' (\n"
    column_list.append("source file")
    column_list.append("source file date")
    column_list.append("source file bytes")

    for column in column_list:
        # for SQLLite default to text
        column_type = "TEXT"

        if "deg" in column:
            column_type = "REAL"
        if "sigma" in column:
            column_type = "REAL"
        if "km" in column:
            column_type = "REAL"
        if "AU" in column:
            column_type = "REAL"
        if "AbsMag" in column:
            column_type = "REAL"
        if "sec" in column:
            column_type = "REAL"
        if "Mass" in column:
            column_type = "REAL"
        if "arcsec" in column:
            column_type = "REAL"
        if "No" in column:
            column_type = "INTEGER"
        if "Num" in column:
            column_type = "INTEGER"
        if "bytes" in column:
            column_type = "INTEGER"

        column = columnTranslate(column)

        create_table_statement += " '{}' '{}',".format(column, column_type) + "\n"

    create_table_statement = create_table_statement[:-2] + "\n );"

    print(create_table_statement)
    conn.execute(create_table_statement)
    pass


def createTrajectoryDataDirectoryStructure(config):
    """
    Creates a folder structure in line with the format used at https://globalmeteornetwork.org/data/traj_summary_data/
    :return: nothing
    """

    trajectory_summary_directory = os.path.join(os.path.expanduser(config.data_dir), "traj_summary_data")
    daily_directory = os.path.join(trajectory_summary_directory, "daily")
    monthly_directory = os.path.join(trajectory_summary_directory, "monthly")
    trajectory_summary_all_file = os.path.join(trajectory_summary_directory, traj_summary_all_filename)

    mkdirP(os.path.expanduser(config.data_dir))
    mkdirP(trajectory_summary_directory)
    mkdirP(daily_directory)
    mkdirP(monthly_directory)

    return trajectory_summary_all_file, daily_directory, monthly_directory

def lastFile(dir):

    """

    Args:
        dir: Directory

    Returns:
        full path to last alphnumeric file in dir
    """

    directory=os.listdir(dir)
    directory.sort()
    return os.path.join(dir, directory[-1])

def readTrajFileCol(trajectory_file, column, length=0, ignore_line_marker="#"):
    """

    Args:
        trajectory_file: trajectory file to be read
        column: column number to be read
        length: maximum length of list to return
        ignore_line_marker: marker which signifies a comment

    Returns:
        a list of values from a single column in a trajectory file

    """

    value_list = []
    col_no = getHdrNum(trajectory_file, column)

    with open(trajectory_file) as input_fh:
        for line in input_fh:
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue
            value = line.split(";")[col_no]
            value_list.append(value)

    if length != 0:
        del value_list[length:]

    return value_list


def readTrajFileMultiCol(trajectory_file, column_list, length=0, ignore_line_marker="#", convert_to_radians=True,
                         solar_lon_range=None):
    """

    Read a GMN format trajectory summary file into a list of lists, one list for each column name provided in
    column list.

    :param traj_summary_all_file:path to the file to hold the daily files combined
    :param column_list: list of column headers to read
    :param length: number of trajectories to remove from the start of the list
    :param ignore_line_marker: lines starting with this string will be ignored
    :param convert_to_radians: Whether the values with "deg" in the field should be converted to radians
    :return: list of values
    """

    value_list, col_no_list = [], []

    for column in column_list:
        col_no_list.append(getHeaders(trajectory_file).index(column))

    if solar_lon_range is not None:
        print("Seeking between solar longitudes {} and {}".format(solar_lon_range[0], solar_lon_range[1]))
    with open(trajectory_file) as input_fh:

        for line in input_fh:
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue

            if solar_lon_range is not None:
                if not float(solar_lon_range[0]) < float(line.split(";")[5]) < float(solar_lon_range[1]):
                    continue

            line_value_list = []
            for col_no, field in zip(col_no_list, column_list):
                value = line.split(";")[col_no]

                if "deg" in field and convert_to_radians:
                    line_value_list.append(np.radians(float(value)))
                else:
                    try:
                        line_value_list.append(float(value))
                    except:
                        line_value_list.append(value)

            value_list.append(line_value_list)

    if length != 0:
        del value_list[length:]

    return value_list


def getHeaders(trajectory_file):
    """

    Args:
        trajectory_file: full path to a trajectory file

    Returns:
        a list of the headers in a trajectory file
    """

    trajectory_file = lastFile(daily_directory)
    header_list = []
    traj_summary = open(trajectory_file, 'r')

    header_line_counter = 0
    for line in traj_summary:
        if line != "\n" and line[0] == '#' and ";" in line and not "---" in line:
            headers = line[1:].split(';')  # get rid of the hash at the front
            if header_line_counter == 0:
                header_list = [""] * len(headers)
            column_count = 0
            for header in headers:
                header_list[column_count] = str(header_list[column_count]).strip() + " " + str(header).strip()
                if header_list[column_count].strip() == "+/- sigma" and column_count > 1:
                    header_list[column_count] = header_list[column_count - 1].strip() + " " + header_list[
                        column_count].strip()
                else:
                    header_list[column_count] = header_list[column_count].strip()
                column_count += 1
            header_line_counter += 1

    return header_list


def getHdrNum(trajectory_file, header):

    return getHeaders(trajectory_file).index(header)


class HTMLStripper(HTMLParser):
    """
    Class to handle converting HTML to plain text
    """

    def __init__(self):
        self.buffer = ""
        HTMLParser.__init__(self)

    def handle_data(self, newdata):
        self.buffer += newdata


def getFieldsFromRow(line, field_list, table_header_list=[], delim=";"):


    """

    Args:
        line: line of fields
        field_list: list of fields to be recovered
        table_header_list: list of headers in the line
        delim: deliminted used

    Returns:
        list of values
    """

    value_list = []
    if table_header_list == []:
        for field in field_list:
            value_list.append(line.split(delim)[getHdrNum(trajectory_summary_all_file, field)])
    else:
        for field in field_list:
            value_list.append(line.split(delim)[table_header_list.index(field)])

    return (value_list)


def isDuplicate(line_1, line_2, table_header_list = None):
    """
    Detect if two trajectories, adjacent to each other, may be duplicate trajectories

    Args:
        line_1:first trajectory line
        line_2:second trajectory line

    Returns:
        boolean
    """




    if len(line_1.split(";")) < 71 or len(line_2.split(";")) < 71:
        return False


    lat1_s = float(getFieldsFromRow(line_1, ["LatBeg +N deg"], table_header_list)[0])
    lon1_s = float(getFieldsFromRow(line_1, ["LonBeg +E deg"], table_header_list)[0])
    lat1_e = float(getFieldsFromRow(line_1, ["LatEnd +N deg"], table_header_list)[0])
    lon1_e = float(getFieldsFromRow(line_1, ["LonEnd +E deg"], table_header_list)[0])
    time_1 = getFieldsFromRow(line_1, ["Beginning UTC Time"], table_header_list)[0]

    lat2_s = float(getFieldsFromRow(line_2, ["LatBeg +N deg"], table_header_list)[0])
    lon2_s = float(getFieldsFromRow(line_2, ["LonBeg +E deg"], table_header_list)[0])
    lat2_e = float(getFieldsFromRow(line_2, ["LatEnd +N deg"], table_header_list)[0])
    lon2_e = float(getFieldsFromRow(line_2, ["LonEnd +E deg"], table_header_list)[0])
    time_2 = getFieldsFromRow(line_2, ["Beginning UTC Time"], table_header_list)[0]



    if abs(lat1_s - lat2_s) < 0.1 and abs(lon1_s - lon2_s) < 0.1 and \
            abs(lat1_e - lat2_e) < 0.1 and abs(lon1_e - lon2_e) < 0.1 and \
            (abs(parser.parse(time_1) - parser.parse(time_2))).total_seconds() < 2:

        return True
    else:
        return False


def downloadStrippedPage(page):
    """

    Args:
        page: web page to be downloaded

    Returns:
        returned as a utf 8 byte sequence

    """

    if sys.version_info[0] < 3:
        web_page = urllib2.urlopen(page).read()
    else:
        web_page = urllib.request.urlopen(page).read().decode("utf-8")

    stripper = HTMLStripper()
    stripper.feed(web_page)
    return stripper.buffer


def getNamesDatesAndSizesFromURL(page):

    """

    Args:
        page:text version of apache file directory page

    Returns:
        page : the page
        file_name_list : list of file names
        date_list : list of dates
        size_list : list of filesizes in bytes
s
    """

    line_list = downloadStrippedPage(page).splitlines()
    file_name_list, date_list, size_list = [], [], []

    for this_line in line_list:

        if this_line == '':
            continue
        if this_line[0:8] == 'Index of':
            continue

        column_list = []
        column_list += this_line.split()
        file_name_list.append(column_list[0])
        date_list.append("{} {}".format(column_list[1], column_list[2]))
        file_size_text = column_list[3]
        file_size_final_character = file_size_text[-1]

        if file_size_final_character.isalpha():
            file_size_multiplier = file_size_final_character
            file_size = int(file_size_text[:-1]) - 1
        else:
            file_size_multiplier = ""
            file_size = int(file_size_text)

        file_size = round(file_size) * 1000 if file_size_multiplier == "K" else file_size
        file_size = round(file_size) * 1000 * 1000 if file_size_multiplier == "M" else file_size
        size_list.append(file_size)
    return page, file_name_list, date_list, size_list


def fileAppend(output_fh, file_to_append, ignore_line_marker, drop_duplicates):
    """

    Append a file, to output_fh without lines starting with ignore_line_marker

    :param output_fh: handle of file to be appended to
    :param file_to_append: file to be appended
    :param ignore_line_marker: marker of any line to be ignored
    :param drop_duplicates: allow drop_duplicates function to control append operation
    :return: output_fh, duplicate_count
    """


    table_header_list = getHeaders(file_to_append)

    duplicate_count = 0
    with open(file_to_append) as input_fh:
        previous_line = ""
        for line in input_fh:
            if line[0] == "\n" or line[0] == ignore_line_marker:
                continue
            if previous_line != "":
                if isDuplicate(previous_line, line, table_header_list):
                    duplicate_count += 1
                    if not drop_duplicates:
                        output_fh.write(line)
                else:
                    output_fh.write(line)
            previous_line = line


    return output_fh, duplicate_count


def createFileWithHeaders(destination_file_path, header_source_file):
    """

    Creates a file using the headers from another file

    :param destination_file_path: file to be created
    :param header_source_file: file whose headers are to be used
    :return: file handle of the newly created file
    """


    if os.path.exists(destination_file_path):
        os.remove(destination_file_path)
    mkdirP(os.path.dirname(destination_file_path))
    output_handle = open(destination_file_path, "w")

    with open(header_source_file) as input_handle:
        line_no = 0
        for line in input_handle:
            if line != "\n":
                line_no += 1
                if line[0] == "#":
                    output_handle.write(line)



    return output_handle


def createAllFile(traj_summary_all_file, drop_duplicates):
    """

    :param traj_summary_all_file:path to the file to hold the daily files combined
    :param drop_duplicates: [bool] detect and drop duplicates
    :return: duplicate_count
    """

    print("\n\n")
    print("Creating {}".format(traj_summary_all_file))
    directory = os.listdir(daily_directory)
    directory.sort()
    last_file = directory[-1]
    print("Using file {} to create headers".format(last_file))
    fh = createFileWithHeaders(traj_summary_all_file, os.path.join(daily_directory, last_file))

    directory_list = os.listdir(daily_directory)
    directory_list.sort()

    duplicate_count = 0
    for traj_file in directory_list:
        if traj_file[13:20].isnumeric():
            output_fh, duplicates = fileAppend(fh, os.path.join(daily_directory, traj_file), "#", drop_duplicates)
            duplicate_count += duplicates
        else:
            print("Not adding {} to the {} file".format(traj_file, traj_summary_all_file))

    print("Found {} suspected duplicate trajectories".format(duplicate_count))
    fh.close()
    return duplicate_count


def insertData(local_target_file, conn=None):

    if local_target_file == "traj_summary_yesterday.txt":
        return

    if local_target_file == "traj_summary_latest_daily.txt":
        return

    sql_command = "DELETE FROM Trajectories WHERE 'source files' LIKE '{}'".format(local_target_file)
    conn.execute(sql_command)
    conn.commit()

    column_list = getHeaders(local_target_file)

    data_to_insert  = readTrajFileMultiCol(local_target_file, column_list, convert_to_radians = False)


    if os.path.exists(local_target_file):
        local_mod_time, local_size = os.stat(local_target_file).st_mtime, round(os.stat(local_target_file).st_size)

    sql_command = ""

    sql_command += "INSERT INTO Trajectories \n"
    sql_command += " ( \n"
    for column in column_list:
        column = columnTranslate(column)
        sql_command += " '{}', ".format(column)

    sql_command += "'source file', 'source file date', 'source file bytes' ) \n"

    sql_command += "VALUES \n"
    for row_list in data_to_insert:
        sql_command += "("
        row_list.append(os.path.basename(local_target_file))
        row_list.append(local_mod_time)
        row_list.append(local_size)
        for data_item in row_list:
            sql_command += "'{}', ".format(data_item)
        sql_command = sql_command[:-2]
        sql_command += "),\n"

    sql_command = sql_command[:-2]
    sql_command += ";"

    conn.execute(sql_command)
    conn.commit()

    pass

def mirror(config=None, page="https://globalmeteornetwork.org/data/traj_summary_data/daily/",
                force_reload=False, max_downloads=10, daily_directory=None, conn=None):
    """
    Function to mirror the trajectories of the global meteor network, only downloading changed files, and from the
    changed files optionally constructing a file with all the trajectories

    Args:
        page: web page to be mirrored
        force_reload: ignore any previous downloads
        max_downloads: maximum number of files to be downloaded in any session

    Returns:
        nothing
    """

    print("Mirroring {}".format(page))
    print("Force reload is {}".format(force_reload))

    trajectory_summary_all_file, daily_directory, monthly_directory = createTrajectoryDataDirectoryStructure(config)

    page, file_name_list, date_list, size_list = getNamesDatesAndSizesFromURL(page)
    downloadFiles(daily_directory ,page, file_name_list, date_list, size_list, max_downloads=max_downloads, force_reload=force_reload, conn=conn)

def downloadFile(url,local_target_file, conn=None):

    urllib.request.urlretrieve(url, local_target_file)

    if not tableExists(conn, "Trajectories"):
        createTable(conn, getHeaders(local_target_file))

    insertData(local_target_file, conn)

def downloadFiles(daily_directory, page, file_name_list, date_list, size_list, max_downloads=10, force_reload=False, conn=None):
    files_downloaded = 0
    file_name_list.reverse()
    date_list.reverse()
    size_list.reverse()
    for file_name, remote_date, remote_size in zip(file_name_list, date_list, size_list):

        url = "{}{}".format(page, file_name)
        remote_mod_time = time.mktime(parser.parse(remote_date).timetuple())
        local_target_file = os.path.expanduser(os.path.join(daily_directory, file_name))

        if force_reload:
            print("Downloading {} because force reload selected".format(url))
            downloadFile(url, local_target_file, conn=conn)

        if os.path.exists(local_target_file):
            local_mod_time, local_size = os.stat(local_target_file).st_mtime, round(os.stat(local_target_file).st_size)

            if local_mod_time < remote_mod_time:
                print("Re-downloading {} because remote file is newer than local file".format(url))
                files_downloaded += 1
                downloadFile(url, local_target_file, conn=conn)


            if local_size < remote_size:
                print("Re-downloading {} because remote file is larger than local file".format(url))
                files_downloaded += 1
                downloadFile(url, local_target_file, conn=conn)

            with open(local_target_file) as input_handle:
                for line in input_handle:
                    if line[0] == "#" or line == "\n":
                        continue
                    value_count = len(line.split(";"))
                    if value_count != 86:
                        print("Re-downloading {} because file is not correct format".format(url))
                        downloadFile(url, local_target_file, conn=conn)
                        break


        else:
            print("Downloading new file {}".format(url))
            downloadFile(url, local_target_file, conn=conn)
            files_downloaded += 1

        if files_downloaded >= max_downloads:
            return


if __name__ == "__main__":

    import argparse



    arg_parser = argparse.ArgumentParser(description="""Download new or changed files from a web page of links \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-f', '--force_reload', dest='force_reload', default=False, action="store_true",
                            help="Force download of all files")

    arg_parser.add_argument('-a', '--all', dest='all', default=False, action="store_true",
                            help="Create a trajectory_summary_all file")

    arg_parser.add_argument('-m', '--max_downloads', dest='max_downloads', default=7, type=int,
                            help="Maximum number of files to download")

    arg_parser.add_argument('-w', '--webpage', dest='page',
                            default="https://globalmeteornetwork.org/data/traj_summary_data/daily/", type=str,
                            help="Webpage to use")

    arg_parser.add_argument('-n', '--no_download', dest='no_download', default=False, action="store_true",
                            help="Do not download anything")

    arg_parser.add_argument('-d', '--drop_dup', dest='drop_duplicates', default=False, action="store_true",
                            help="Detect duplicates and remove from traj_all file")

    arg_parser.add_argument('-r', '--repeat', dest='repeat', default=False, action="store_true",
                            help="Run indefinitely with a delay between runs")

    cml_args = arg_parser.parse_args()

    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    trajectory_summary_all_file, daily_directory, monthly_directory = (
        createTrajectoryDataDirectoryStructure(config))

    conn = createDataBase(config)
    if cml_args.repeat:
        print("Running indefinitely...")
    else:
        print("Running once...")

    while cml_args.repeat:

        if not cml_args.no_download:
            mirror(config = config, page = cml_args.page,
                    force_reload = cml_args.force_reload, max_downloads = cml_args.max_downloads, daily_directory=daily_directory, conn=conn)
        delay = random.randrange(120,360)
        print("Next run at {}".format(datetime.datetime.now() + datetime.timedelta(seconds=delay)))
        time.sleep(delay)

    if not cml_args.no_download:
        mirror(config=config, page=cml_args.page,
               force_reload=cml_args.force_reload, max_downloads=cml_args.max_downloads,
               daily_directory=daily_directory)

    if cml_args.all:
        duplicate_count = createAllFile(trajectory_summary_all_file, cml_args.drop_duplicates)
