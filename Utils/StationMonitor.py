# Copyright (c) 2025

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" Monitor stations and email once when a station goes unhealthy.
"""

from __future__ import print_function, division, absolute_import

import os.path
import smtplib
import email.mime.text
import sys
import sqlite3
import datetime
import time
from RMS.Misc import niceFormat, sanitise


import RMS.ConfigReader as cr

# Constants

global_meteor_network = "https://globalmeteornetwork.org"
last_upload_webpage = "weblog"
station_directory_static_template = "https://globalmeteornetwork.org/weblog/$CC/$STATION_ID/static/"
station_observation_summary_static_template = "$STATION_ID_observation_summary_static.txt"
station_monitor_filename = "station_monitor.db"
mail_configuration_file = os.path.expanduser("~/secrets/stationmonitor.conf")

# Colour printing

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

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

def addOperators(conn, add_operators):
    """ Add an operator to the database.

    Arguments:
        conn: [object] Database connection instance.
        add_operators: [str] Operators to add, separated by spaces - not a list.

    Return:
       Nothing.
    """

    operators = add_operators.split()
    for operator in operators:
        op = operator.strip()
        if verbosity > 1:
            print("Adding operator {}".format(op))

        sql_command = ""
        sql_command += "INSERT INTO stations (OperatorEmail)  \n"
        sql_command += "            VALUES   ('{}') \n".format(op)
        sql_command += "            ON CONFLICT(OperatorEmail) DO NOTHING \n"

        if verbosity > 2:
            print(sql_command)

        conn.execute(sql_command)
        conn.commit()

def listStatus(conn):
    """ List the status of all stations.

    Arguments:
        conn: [object] Database connection instance.

    Return:
       Formatted string of results.
    """
    sql_command = ""
    sql_command += "SELECT  \n"
    sql_command += "        StationID, \n"
    sql_command += "        FileTimeStamp,  \n"
    sql_command += "        WeblogUpdatedTimeOfLastChange, \n"
    sql_command += "        HoursSinceLastChange, \n"
    sql_command += "        LastCheckGood \n"

    sql_command += "        FROM station_status \n"
    sql_command += "        ORDER BY StationID \n"

    results = conn.execute(sql_command)

    output = HEADER
    output += " Station ID  | File Time Stamp           | Weblog Last Change        | Hours Since Last Change  | Status\n"
    for result in results:
        if result is not None:
            status = "good" if result[4] == 1 else "bad"
            if status == "good":
                output += OKBLUE
            if result[3] > 24:
                output += WARNING
            if status == "bad":
                output += FAIL
            output += " {:<11} | {:<10} | {:<10} | {:<24} | {:<10}\n".format(result[0], result[1], result[2], result[3], status)

    output += ENDC
    return output

def dropOperators(conn, drop_operators):
    """ Drop an operator and the association with station from the database.

    Arguments:
        conn: [object] Database connection instance.
        drop_operators: [str] Operators to drop, separated by spaces - not a list.

    Return:
        Nothing.
    """

    operators = drop_operators.split()
    for operator in operators:
        op = operator.strip()
        if verbosity > 1:
            print("Dropping operator {}".format(op))

        sql_command = ""
        sql_command += "DELETE FROM stations  \n"
        sql_command += "            WHERE OperatorEmail = ('{}') \n".format(op)
        if verbosity > 2:
            print(sql_command)
        conn.execute(sql_command)

    conn.commit()

def existsOperatorEmail(conn, operator_email):
    """ Check if an operator exists in the database.

    Arguments:
        conn: [object] Database connection instance.
        operator_email: [str] operator_email to search for.

    Return:
        [Boolean] True if an operator exists in the database, else False.
    """

    sql_command = ""
    sql_command += "SELECT OperatorEmail FROM stations WHERE OperatorEmail = '{}'\n".format(operator_email)
    result = conn.execute(sql_command).fetchone()
    if result is not None:
        return True
    else:
        return False

def getOperatorsStations(conn, operatorEmail):
    """ Check if an operator exists in the database.

    Arguments:
        conn: [object] Database connection instance.
        operator_email: [str] operator_email to search for.

    Return:
        [Boolean] True if an operator exists in the database, else False.
    """

    sql_command = ""
    sql_command += "SELECT StationId FROM stations WHERE OperatorEmail = '{}'\n".format(operatorEmail)
    result = conn.execute(sql_command).fetchone()
    if result is not None:
        return result[0].split(",")
    else:
        return ""

def listToCommaSeparatedList(items, sep =","):
    """ Create a comma separated list from a list of items.

    Arguments:
        items: [list] List of items.

    Keyword Arguments:
        sep: [str] Separator to use.

    Return:
        comma_separated_list: [str] Comma separated list.
    """

    comma_seperated_list = ""
    for item in items:
        comma_seperated_list += "{}{} ".format(item.strip(), sep)

    comma_seperated_list = comma_seperated_list[:-2]

    return comma_seperated_list

def listToUniqueCommaSeparatedList(items, drop=None):
    """ Create a comma separated list from a list of items each item unique.

    Arguments:
        items: [list] List of items.

    Keyword Arguments:
        drop: [str] An item to drop.

    Return:
        comma_separated_list: [str] Comma separated list.
    """

    unique_list = []
    for item in items:
        if item not in unique_list:
            if drop is not None:
                if item.strip().lower() != drop.strip().lower():
                    unique_list.append(item.strip())
            else:
                unique_list.append(item.strip())
    unique_list.sort()
    comma_separated_list = listToCommaSeparatedList(unique_list)
    return comma_separated_list

def validStationName(candidate_station_name):
    """ Check a station name is valid

    Criteria

    6 characters in total
    All characters alphanumeric
    First two Alpha

    Arguments:
        candidate_station_name: [str] Candidate station name.

    Return:
        valid: [bool] True if valid, else False.
    """

    # 6 characters
    if len(candidate_station_name) != 6:
        return False

    # All characters alphanumeric
    for c in candidate_station_name:
        if not c.isalnum():
            return False

    # First two letter alpha
    for c in candidate_station_name[:2]:
        if not c.isalpha():
            return False

    return True

def addStations(conn, parameter):
    """ Create a comma separated list from a list of items each item unique.

    Arguments:
        items: [list] List of items.

    Keyword Arguments:
        drop: [str] An item to drop.

    Return:
        comma_separated_list: [str] Comma separated list.
    """

    parameter_list = parameter.split()
    operator_email = parameter_list.pop(0)

    station_list = []
    for s in parameter_list:
        if validStationName(s):
            station_list.append(s)
        else:
            if verbosity > 0:

                print("Not adding station {}, does not conform to format of country_code followed by four characters.".format(s))
                return

    if existsOperatorEmail(conn, operator_email):
        existing_stations_list = getOperatorsStations(conn, operator_email)
        if verbosity > 1:
            print("Operator {} has stations {}".format(operator_email, existing_stations_list))
        station_list =  listToUniqueCommaSeparatedList(existing_stations_list + parameter_list)
        sql_command = ""
        sql_command += "UPDATE stations"
        sql_command += "    SET \n"
        sql_command += "             StationID  = '{}'\n".format(station_list)
        sql_command += "    WHERE OperatorEmail = '{}'\n".format(operator_email)

    else:
        if verbosity > 0:
            print("Adding operator {}".format(operator_email))
        station_list = listToUniqueCommaSeparatedList(parameter_list)
        sql_command = ""
        sql_command += "INSERT INTO stations                    \n"
        sql_command += "   (                                    \n"
        sql_command += "    OperatorEmail,                      \n"
        sql_command += "    StationID                           \n"
        sql_command += "                                    )   \n"
        sql_command += "   VALUES                               \n"
        sql_command += "   (                                    \n"
        sql_command += "    '{}',                               \n".format(operator_email)
        sql_command += "    '{}'                                \n".format(station_list)
        sql_command += "                                    )   \n"

    if verbosity > 2:
        print(sql_command)

    conn.execute(sql_command)
    conn.commit()

    pass

def dropStations(conn, parameter):
    """ Drop stations from the database.

    Arguments:
        parameter: [str] a string "operator@example.com xx0001 xx0002 xx0003" etc.
    Return:
        Nothing.
    """

    parameter_list = parameter.split()
    operator_email = parameter_list.pop(0)

    if not existsOperatorEmail(conn, operator_email):
        print("Operator {} does not exist".format(operator_email))
        return


    sql_command = ""
    sql_command += "SELECT OperatorEmail, StationID \n"
    sql_command += "       FROM stations \n"
    sql_command += "       WHERE OperatorEmail = '{}'".format(operator_email.strip())

    if verbosity > 2:
        print(sql_command)

    results = conn.execute(sql_command).fetchall()
    if results == []:
        print("No match found")
        return

    for record in results:
        stations = record[1]
        for station_to_drop in parameter_list:
            print("station to drop {}".format(station_to_drop))
            stations = listToUniqueCommaSeparatedList(stations.split(","), drop=station_to_drop)

    sql_command = ""
    sql_command += "UPDATE stations"
    sql_command += "    SET \n"
    sql_command += "             StationID  = '{}'\n".format(stations)
    sql_command += "    WHERE OperatorEmail = '{}'\n".format(operator_email)

    if verbosity > 2:
        print(sql_command)
    conn.execute(sql_command)
    conn.commit()
    return

def listConfiguration(conn, nice_format=True):
    """ List the configuration of the system.

    Arguments:
        conn: [object] Database connection instance.

    Keyword Arguments:
        nice_format: [bool] If True, return a nice formatted string.

    Return:
        output: [string] formatted output string.
        """

    sql_command = ""
    sql_command += "SELECT OperatorEmail , StationID FROM stations"
    configuration = conn.execute(sql_command).fetchall()

    configuration_as_string = HEADER + "Operator email:StationID\n"
    for entry in configuration:
        # Strip newlines
        configuration_as_string += OKCYAN + "{}:{}\n".format(entry[0].replace("\n",""), entry[1].replace("\n",""))


    output =  niceFormat(configuration_as_string) if nice_format else configuration_as_string + ENDC
    return output

def listDurations(conn, count=5):
    """ List the weblog run time durations.

    Arguments:
        conn: [object] Database connection instance.

    Keyword Arguments:
        count: [int] Number of durations to list, most recent first

    Return:
        output: [string] formatted output string.
        """

    sql_command = ""
    sql_command += "SELECT RunTime , RunDuration FROM server_run_times \n"
    sql_command += "                             ORDER BY RunTime DESC \n"
    sql_command += ("                             LIMIT '{}'\n".format(count))

    records = conn.execute(sql_command).fetchall()
    records.reverse()

    output = OKCYAN + "\n"
    for record in records:
        output += "{} | {}\n".format(record[0], record[1])
    output += ENDC
    return output

def nextHour(time_object):
    """ Return the time object of the start of the next hour.

    Arguments:
        time_object: [time object]

    Return:
        next_hour: [time object]
    """

    next_hour = time_object.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)

    return next_hour


    return next_hour

def getAppPassword():
    """ Return the application password address from the stationmonitor.conf file.
    Arguments:
        None.
    Return:
        password: [str] application password.
    """

    with open (os.path.expanduser(mail_configuration_file),"r") as f:
        password = f.readlines()[1].strip()
    return password

def getSenderAddress():
    """ Return the sender email address from the stationmonitor.conf file.

    Arguments:
        None
    Return:
        [str] sender email address.
    """

    with open(os.path.expanduser(mail_configuration_file), "r") as f:
        sender_address = f.readlines()[0].strip()
    return sender_address

def getMailConfig():
    """ This function gets the configuration for the gmail app sending client.

        Arguments:
            None
        Return:
            [str], [str] sender email address, application password.
        """


    return getSenderAddress(), getAppPassword()

def statusFallingEdge(op_email, station_id, changed_time):
    """ This function contains any actions to be taken when a station goes from healthy to unhealthy.

    Arguments:
        op_email: [str] email of the operator responsible for the station.
        station_id: [str] rms station_id.
        changed_time: [timeobject] the time to report to the operator.

    Return:
        Nothing.
    """

    print("Station {} went unhealthy, last upload was registered at {}".format(station_id, changed_time))
    sender = getSenderAddress()
    subject = "Station {} unhealthy since {}".format(station_id, changed_time)
    body = "Automated email from g7gpr@outlook.com"
    sendEmail(sender, op_email, subject, body)

def createStationStatus(conn):
    """Create the station status table.

    Holds   StationID - XX0001
            FileTimeStamp - The time stamp seen on the weblog - this is only used to detect changes.
            WeblogUpdatedTimeOfLastChange - The time of the webserver update when a change of FileTimeStamp was detected.
            HoursSinceLastChange - The estimated time of the last change - based on when the file timestamp was observed to change
                                    using the weblog last updated time as the time point.
    Arguments:
        conn: [object] Database connection instance.

    Return:
        Nothing
    """
    try:
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'station_status';""").fetchall()

        if tables:
            # upgradeDB(conn) put this in if needed
            return conn
    except:
        return None

    conn.execute("""CREATE TABLE station_status (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,   
                            StationID TEXT NOT NULL,
                            FileTimeStamp TEXT NOT NULL,
                            WeblogUpdatedTimeOfLastChange TEXT NOT NULL,
                            HoursSinceLastChange FLOAT NOT NULL,
                            LastCheckGood INTEGER NOT NULL
                            )""")

    # Commit the changes
    conn.commit()

def createStations(conn):
    """Create the stations table, holding Operator emails and comma separated list of stations.

        OperatorEmail       |   StationID
        g7gpr@outlook.com   |   au000a, au000c, au000d, etc

        Arguments:
            conn: [object] Database connection instance.

        Return:
            Nothing
        """

    try:
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'stations';""").fetchall()

        if tables:
            return conn
    except:
        return None

    conn.execute("""CREATE TABLE stations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,   
                            OperatorEmail TEXT NOT NULL UNIQUE,
                            StationID TEXT
                            )""")

    # Commit the changes
    conn.commit()

def createServerRunTimes(conn):
    """Create the server run times table, holding only the time when the weblog was updated.

        Arguments:
            conn: [object] Database connection instance.

        Return:
            nothing
        """

    try:
        sql_command = "SELECT name FROM sqlite_master WHERE type = 'table' and name = 'server_run_times'"
        tables = conn.cursor().execute(sql_command).fetchall()

        if tables:
            # upgradeDB(conn) put this in if needed
            return conn
    except:
        return None

    sql_command = ""
    sql_command += "CREATE TABLE server_run_times \n"
    sql_command += "( \n"
    sql_command += "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    sql_command += "    RunTime TEXT NOT NULL,\n"
    sql_command += "    RunDuration TEXT NOT NULL\n"
    sql_command += "                                        ) \n"

    # Execute and commit
    conn.execute(sql_command)
    conn.commit()

def createStationMonitorDB(db_path):
    """ Creates the StationMonitor database. Tries only once.

    Arguments:
        db_path: [path] path to the database.

    Returns:
        conn: [connection] connection to database if success else None.

    """

    if not os.path.exists(os.path.dirname(db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(os.path. db_path))

    try:
        conn = sqlite3.connect(db_path)

    except:
        print("Failed to create station monitor database")
        return None

    createStationStatus(conn)
    createStations(conn)
    createServerRunTimes(conn)

    # Return the connection
    return conn

def urlLines(url):
    """ Given a URL, return the content split by lines.

    Arguments:
        url: [url] URL of the page to be read.

    Return:
        web_page_list: [list] list of lines in web page.
    """

    try:
        if sys.version_info[0] < 3:
            web_page_lines = urllib2.urlopen(url).read().splitlines()
        else:
            web_page_lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    except:
        if verbosity > 1:
            print("{} was not found".format(url))
        return None

    return web_page_lines

def getFileTimeStampFromWeblog(station_code):
    """Get the file stamp time from the weblog.

    Args:
        station_code: [str] Station code - AU0004.

    Return:
        time: [object] python time aware object.
    """

    country_code = station_code[0:2].upper()
    static_url = station_directory_static_template.replace("$CC", country_code).replace("$STATION_ID", station_code.upper())
    static_file = station_observation_summary_static_template.replace("$STATION_ID", station_code.upper()).replace("$CC", country_code)

    if verbosity > 1:
        print("Country code {}".format(country_code))
        print("URL of directory {}".format(static_url))
        print("Filename {}".format(static_file))

    web_page = urlLines(static_url)

    if web_page is None:
        if verbosity > 1:
            print("No webpage found at {}".format(static_url))
        return None

    for line in web_page:
        if static_file in line:

            line_split = line.split()

            date_str = "{} {}".format(line_split[2], line_split[3])

            date_obj = datetime.datetime.strptime(date_str, "%d-%b-%Y %H:%M").replace(tzinfo=datetime.timezone.utc)
            break
    return date_obj

def getWeblogLastUpdatedFromWeb(url=None):
    """Get the weblog update from the observation summary file on the website.

    Arguments:

    Keyword Arguments:
        url: [url] URL of the page containing the date string.

    Return:
        time: [object] Python time zone aware object.
    """

    if url is None:
        url = os.path.join(global_meteor_network, last_upload_webpage)

    try:
        web_page = urlLines(url)
    except:
        return None

    if web_page is None:
        # This is probably Ctrl-C event
        return None
    for line in web_page:
        line = line.strip()
        if line.startswith("Last updated:"):
            last_updated_date_str = line.strip().replace("Last updated:", "").replace("UTC","").strip()
            return datetime.datetime.strptime(last_updated_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    return None

def getFileTimeStampFromDB(conn, station_code, fallback_to_weblog=False):
    """Get the time stamp of the observation summary file from the database.

    If not found in the database, then get from the weblog as a fallback.

    Arguments:
        conn: [object] Sqlite database connection instance.
        station_code: [string] Station code.

    Return:
        time: [object] Python time zone aware object.
    """

    sql_statement = ""
    sql_statement += "SELECT FileTimeStamp "
    sql_statement += "  FROM station_status "
    sql_statement += "  WHERE StationID = '{}' LIMIT 1".format(station_code)
    if verbosity > 2:
        print(sql_statement)

    result = conn.cursor().execute(sql_statement).fetchone()

    if result is None:
        if fallback_to_weblog:
            return getFileTimeStampFromWeblog(station_code)
        else:
            return result

    if len(result) == 1:
        return datetime.datetime.strptime(result[0].replace(":",""), "%Y-%m-%d %H%M%S%z")
    else:
        return None

def getLastWeblogUpdatedTimeFromDB(conn):
    """Get the last server run time from the database.

    Arguments:
        conn: [object] Sqlite database connection instance.

    Return:
        time: [object] Python time zone aware object.
    """

    last_update_time = None

    sql_statement = "SELECT RunTime FROM server_run_times ORDER BY RunTime DESC LIMIT 1"
    result = conn.cursor().execute(sql_statement).fetchone()

    if result is None:
        return result

    if len(result) == 1:
        return datetime.datetime.strptime(result[0].replace(":",""), "%Y-%m-%d %H%M%S%z")
    else:
        return None

    return result

def getLastChangedTimeFromDB(conn, station_id):
    """Get the time the the observation summary file was changed.

    Arguments:
        conn: [object] Sqlite database connection instance.
        station_id: [string] Station code.

    Return:
        time: [object] Python time zone aware object.
    """

    sql_statement = "SELECT WeblogUpdatedTimeOfLastChange FROM station_status WHERE StationID = '{}' LIMIT 1".format(station_id)
    result = conn.cursor().execute(sql_statement).fetchone()

    if result is None:
        return result
    # Make timezone aware
    last_update_time = datetime.datetime.strptime(result[0].replace(":",""), "%Y-%m-%d %H%M%S%z")
    return last_update_time

def getOperatorStationList(conn):
    """Get a list station operator emails, each one with a list of stations.

    Arguments:
        conn: [object] Sqlite database connection instance.
        station_id: [string] Station code.

    Return:
        operator_station_list: [operator, [stations]] list of operators each with a station list.
    """

    sql_statement = ""
    sql_statement += "SELECT OperatorEmail, StationID \n"
    sql_statement += "FROM stations"

    result = conn.cursor().execute(sql_statement).fetchall()
    if verbosity > 1:
        print("Operator station list {}".format(result))
    operator_station_list = []
    for entry in result:
        op_email = entry[0]
        if verbosity > 1:
            print("Operator email {}".format(op_email))
        stations_unstripped = entry[1].split(",")

        # Remove spaces from the station names
        stations = []
        for s in stations_unstripped:
            stations.append(s.strip())
        operator_station_list.append([op_email, stations])

    return operator_station_list

def getLastStatusFromDB(conn, stationID):
    """Get a list station operator emails, each one with a list of stations.

    Arguments:
        conn: [object] Sqlite database connection instance.
        station_id: [string] Station code.

    Return:
        result: [int] last status 0 is bad, 1 is good.
    """
    sql_statement = "SELECT LastCheckGood FROM station_status WHERE StationID = '{}' LIMIT 1".format(stationID)
    result = conn.cursor().execute(sql_statement).fetchone()
    if result is None:
        # always start new stations as unhealthy to avoid an email flood
        return 0
    return result[0]

def weblogUpdated(conn, run_time):
    """Has the weblog updated time changed from the previous run?

    run_time is passed into this function to avoid multiple calls to the webserver.

    Arguments:
        conn: [object] Sqlite database connection instance.
        run_time: [object] Python time object of the latest server run.

    Return:
        result: [bool] True if updated or no previous records.
    """

    last_run_time = getLastWeblogUpdatedTimeFromDB(conn)
    if last_run_time is None:
        return True

    if last_run_time < run_time:
        return True
    else:
        return False

    return False

def insertWeblogUpdatedTime(conn, weblog_updated_time):
    """Insert a server run time into the database, appends to the server_run_times table.

    Arguments:
        conn: [object] Sqlite database connection instance.
        weblog_updated_time: [object] Python time zone aware object.

    Return:
        result: [bool] True if server run time has been updated, else False.
    """

    last_weblog_updated_time_from_db = getLastWeblogUpdatedTimeFromDB(conn)
    if last_weblog_updated_time_from_db is None or last_weblog_updated_time_from_db < weblog_updated_time:
        if last_weblog_updated_time_from_db is None:
            run_duration = "No prior run"
        else:
            run_duration = weblog_updated_time - last_weblog_updated_time_from_db
        sql_statement = ""
        sql_statement += "INSERT INTO server_run_times ('RunTime', 'RunDuration') \n"
        sql_statement += "                      VALUES ('{}',       '{}')".format(weblog_updated_time, run_duration)
        if verbosity > 1:
            print(sql_statement)
        conn.execute(sql_statement)
        conn.commit()
        return True
    else:
        return False

def sendEmail(sender, recipient, subject, body):
    """Insert a server run time into the database, appends to the server_run_times table.

    Arguments:
        sender: [str] email address of the sender.
        recipient: [string] email address of the recipient.
        subject: [string] subject of the email.
        body: [string] body of the email.

    Return:
        Nothing
    """

    if verbosity > 0:
        print("\nSTART \n")
        print("To: {}".format(recipient))
        print("Subject: {}".format(subject))
        print("\n")
        print("{}".format(body))
        print("\nEND \n")

    # Create the email message
    msg = email.mime.text.MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    sender, password = getMailConfig()

    if not os.path.exists(os.path.expanduser(mail_configuration_file)):
        print("No configuration file found at {}".format(mail_configuration_file))
        print("Mail cannot be sent")
    else:
        # Send the email via Gmail's SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())

def checkUpdateStation(conn, station_id, server_run_time, warning_hours=36):
    """Insert a file_time stamp into the database, and take action.

    If the file_time stamp has changed, then webserver update time at which that change was made is recorded.
    If the file_time stamp has not changed, then if it has not changed in more than 36 hours, and the previous
    status was 1 (good) set status to 0, and run statusFallingEdge function.

    Arguments:
        conn: [object] Sqlite database connection instance.
        op_email: [string] email address of the operator email.
        station_id: [string] Station code.
        file_stamp_time: [object] Python time zone aware object.
        changed: [bool] True if changed, False otherwise.

    Return:
        rising_edge: [bool] True if station went from unhealthy to healthy.
        falling_edge: [bool] True if station went from healthy to unhealthy.
    """

    # Get initial values
    last_server_run_time = getLastWeblogUpdatedTimeFromDB(conn)
    time_of_last_change_from_db = getLastChangedTimeFromDB(conn, station_id)
    file_time_stamp_from_db = getFileTimeStampFromDB(conn, station_id)
    file_time_stamp_from_weblog = getFileTimeStampFromWeblog(station_id)
    last_status = getLastStatusFromDB(conn, station_id)

    if file_time_stamp_from_weblog is None:
        if verbosity > 1:
            print("Unable to process station {}".format(station_id))
        return False, False

    # Initialise rising, falling edge and new station
    rising_edge, falling_edge, new_station = False, False, False

    # If no previous server run time, then set to the current server run time
    if last_server_run_time is None:
        last_server_run_time = server_run_time


    if time_of_last_change_from_db is None:
        # If no time of last change then use the time stamp from the weblog
        old_file_time_stamp = file_time_stamp_from_weblog
        new_station = True
    else:
        # If it is a new station set old_file_time_stamp to the file_time_stamp_from_db
        old_file_time_stamp = file_time_stamp_from_db

    if verbosity > 1:
        print("Station                      : {}".format(station_id))
        print("Server run time              : {}".format(server_run_time))
        print("Last server run time         : {}".format(last_server_run_time))
        print("Time of last change from db  : {}".format(time_of_last_change_from_db))
        print("File time stamp from db      : {}".format(file_time_stamp_from_db))
        print("File time stamp from weblog  : {}".format(file_time_stamp_from_weblog))
        print("Last status                  : {}".format(last_status))

    # If it is a new station, then assume the time of last change is the filetime.
    # This is the only time we do this, normally the time of the last change is taken
    # from the previous weblog last updated time.
    if time_of_last_change_from_db is None:
        time_of_last_change_from_db = file_time_stamp_from_weblog

    # Has the file time stamp on the weblog changed
    if file_time_stamp_from_weblog > old_file_time_stamp:
        # If we know that it has changed set
        #               changed_time to the last server run time
        #               status to 1 (good)
        #               hours_since_last change is 0
        #
        # We can't compute hours_since_last change more accurately, because
        # the FileTimeStamp could be running late because of an upload backlog
        changed_time, status, time_of_last_change = last_server_run_time, 1, last_server_run_time
        hours_since_last_change = (server_run_time - time_of_last_change).total_seconds() / 3600
    else:
        # If it has not changed then we know that the hours_since_last_change is at least
        # the time between the last server run time and the WeblogTimeOfLastChange
        if not time_of_last_change_from_db is None:
            # If this is a normal run

            time_of_last_change = time_of_last_change_from_db
        else:
            # If it is a first run for this station
            time_of_last_change = last_server_run_time
        hours_since_last_change = (server_run_time - time_of_last_change).total_seconds() / 3600

        if verbosity > 1:
            print("Time since last change: {}".format(hours_since_last_change))

        # If time_since_last_change is greater than warning_hours
        if hours_since_last_change > warning_hours:
            last_status = getLastStatusFromDB(conn, station_id)
            if verbosity > 1:
                print("Last status for {} was {}".format(station_id.lower(), last_status))
            # If last status was high, then this is a new failure
            if last_status == 1:
                falling_edge = True
            # Set status to 0
            status = 0
        else:
            # If last status was low, then this is a station going healthy
            if last_status == 0:
                rising_edge = True
            status = 1


    # Develop the SQL Command
    sql_command = writeStatus(conn, new_station, station_id, file_time_stamp_from_weblog,
                              time_of_last_change, hours_since_last_change, status)

    if verbosity > 2:
        print(sql_command)
    conn.execute(sql_command)
    conn.commit()

    if new_station:
        rising_edge, falling_edge = False, False

    return rising_edge, falling_edge

def writeStatus(conn, new_station, station_id, file_time_stamp_from_weblog, time_of_last_change, hours_since_last_change, status):
    """ Formulate the SQL command to write the status of a station.

    Arguments:
        conn: [object] Sqlite database connection instance.
        new_station: [bool] True if station did not exist before.
        station_id: [str] Station ID XX0001 etc.
        file_time_stamp_from_weblog: [object] Python time zone aware object, the time stamp of the file being watched.
        time_of_last_change: [object] Python time zone aware object, the weblog last updated time when the change was recorded
        hours_since_last_change: [float] The number of hours since a change was last recorded
        status: 0 is bad, 1 is good

    Returns:
        sql_command: [string] SQL command
    """


    if new_station:
        # Delete any record that is here for this station - should not be possible
        sql_command = "DELETE FROM 'station_status' WHERE StationID = '{}'".format(station_id)
        if verbosity > 2:
            print(sql_command)
        conn.execute(sql_command)
        conn.commit()
        # Create a new record
        sql_command = ""
        sql_command += "INSERT INTO station_status \n"
        sql_command += "            (StationID, FileTimeStamp, WeblogUpdatedTimeOfLastChange, HoursSinceLastChange, LastCheckGood) \n"
        sql_command += "            VALUES \n"
        sql_command += ("            ('{}', '{}', '{}', '{:.1f}', '{}') \n"
                        .format(station_id, file_time_stamp_from_weblog, time_of_last_change, hours_since_last_change,
                                status))

    else:
        # Update an existing record
        sql_command = ""
        sql_command += "UPDATE station_status \n"
        sql_command += '        SET FileTimeStamp = "{}", \n'.format(file_time_stamp_from_weblog)
        sql_command += '            WeblogUpdatedTimeOfLastChange = "{}", \n'.format(time_of_last_change)
        sql_command += '            HoursSinceLastChange = "{:.1f}", \n'.format(hours_since_last_change)
        sql_command += '            LastCheckGood = "{}" \n'.format(status)
        sql_command += '        WHERE StationID = "{}"; \n'.format(station_id)
    return sql_command

def getStationsOperators(conn, station_id):
    """ Get a list of email addresses of operators associated with a station

        Arguments:
            conn: [object] Sqlite database connection instance.
            station_id: [str] Station ID XX0001 etc.

        Returns:
            operators: [list] List of email addresses of operators associated with a station
        """

    sql_command = ""
    sql_command += "SELECT OperatorEmail FROM stations WHERE StationID LIKE '%{}%'".format(station_id)

    results = conn.execute(sql_command).fetchall()

    operators = []
    for result in results:
        operators.append(result[0])

    return operators

def handleResults(conn, result_list):
    """ Handle the results from the station checks

        Arguments:
            conn: [object] Sqlite database connection instance.
            result_list: [station_ID, [rising_edge, falling_edge]]
        Returns:
            Nothing
        """

    for result in result_list:
        station_id, rising_edge, falling_edge = result[0], result[1][0], result[1][1]
        if falling_edge:
            operators_to_email = getStationsOperators(conn, station_id)
            for operator in operators_to_email:
                if verbosity > 1:
                    print("Emailing operator {}".format(operator))
                statusFallingEdge(operator, station_id, getLastChangedTimeFromDB(conn, station_id))

def processStation(conn, station_id, server_run_time, warning_hours=36):
    """Read information about a station, determine if new information has been presented and update the database.

    Arguments:
        conn: [object] Sqlite database connection instance.
        op_email: [string] email address of the operator email.
        station_id: [string] Station code.

    Return:
        Nothing
    """

    # Remove spaces from station
    station_id = station_id.strip()

    # rising_edge true, means a station has gone from unhealthy to healthy
    # falling_edge true, means a station has gone from healthy to unhealthy
    rising_edge, falling_edge = checkUpdateStation(conn, station_id, server_run_time, warning_hours=warning_hours)

    if verbosity > 1:
        print("For station                  : {}".format(station_id))
        print("              rising edge is : {}".format(rising_edge))
        print("             falling edge is : {}".format(falling_edge))

    return rising_edge, falling_edge

def stationMonitor(syscon, repeat=False, delay_minutes=60, verbosity=1, warning_hours=36):
    """Main function of station monitor.

    Call this externally to launch station monitor.

    Arguments:
        syscon: [object] RMS configuration instance.

    Keyword Arguments:
        repeat: [bool] If true, then run forever, else run once.
        delay_minutes: [int] Number of minutes to wait between checks.
        verbosity:[int] Verbosity of output. 0 - nothing, 1 - email traffic, 2 - + logic, 3 + SQL.
        warning_hours: [int] Default 36 Hours of inactivity before an email is sent.

    Return:
        Nothing.
    """

    station_monitor_db_path = os.path.join(syscon.data_dir, station_monitor_filename)
    conn = createStationMonitorDB(station_monitor_db_path)
    run_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    while True:
        station_count = 0
        iteration_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        weblog_last_updated_from_web = getWeblogLastUpdatedFromWeb()
        if verbosity > 0:
            print(OKCYAN)
            print("===========================================================")
            print(" Station monitor started at     : {}".format(iteration_start_time.replace(microsecond=0)))
            print(" Weblog last updated            : {}".format(weblog_last_updated_from_web.replace(microsecond=0)))
        if weblogUpdated(conn, weblog_last_updated_from_web):

            # Initialise working lists
            processed_station_list, result_list, operator_station_list = [], [], getOperatorStationList(conn)

            # Iterate through the operators
            for operator in operator_station_list:
                op_email, station_list = operator[0], operator[1]
                for station in station_list:
                    # If this station has already been processed, then skip it
                    if station in processed_station_list:
                        continue
                    station_count += 1
                    rising_edge, falling_edge = processStation(conn, station, weblog_last_updated_from_web, warning_hours = warning_hours )
                    result_list.append([station.strip(), [rising_edge, falling_edge]])

            handleResults(conn, result_list)

        else:
            last_server_updated_time = getLastWeblogUpdatedTimeFromDB(conn)
            weblog_update_duration = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0) \
                                                                    - last_server_updated_time.replace(microsecond=0)
            if verbosity > 0:
                print(" No weblog update for           : {}".format(weblog_update_duration))

        iteration_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
        iteration_duration = (iteration_end_time - iteration_start_time).total_seconds()

        # Finally write the server run time into the database
        insertWeblogUpdatedTime(conn, weblog_last_updated_from_web)
        if verbosity > 0:
            print(" Station monitor completed at   : {}".format(datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)))

        if repeat:
            # Compute when the next run should be
            next_run_start_time = run_start_time + datetime.timedelta(minutes=delay_minutes)
            run_start_time = next_run_start_time
            if verbosity > 0:
                print("                      Processed : {:d} stations".format(station_count))
                print("                             in : {:.0f} seconds".format(iteration_duration))
                print()
                print(" Next run at                    : {}".format(next_run_start_time.replace(microsecond=0)))
                print("===========================================================\n\n" + ENDC)

            sleep_time = (run_start_time - datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)).total_seconds()
            # Avoid negative sleep times
            sleep_time = max(sleep_time, 0)
            # Wait for next iteration compensating for the previous delay
            time.sleep(sleep_time)
        else:
            if verbosity > 0:
                print("                      Processed : {:d} stations".format(station_count))
                print("                             in : {:.0f} seconds".format(iteration_duration))

                print("===========================================================\n\n" +ENDC)
            break
        print(ENDC)

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Monitor the health of Global Meteor Network stations. \
            """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-r', '--repeat', dest='repeat', type=int, default=0,
                            help="Loop time for continuous running in minutes. If not specified, runs once.")

    arg_parser.add_argument('-v', '--verbosity', metavar='VERBOSITY', type=int,
                           help="""0 - silent, 1 - (default) show emails and basic info, 2 - and logic, 3 and SQL  """)

    arg_parser.add_argument('-w', '--warning_hours', metavar='WARNING_HOURS', type=int,
                            help="""Hours late before sending email default 36""")

    arg_parser.add_argument('--add_operators', dest='add_operators', type=str,
                            help="Add operators to the database.")

    arg_parser.add_argument('--drop_operators', dest='drop_operators', type=str,
                            help="Remove operators from the database.")

    arg_parser.add_argument('--add_stations', dest='add_stations', type=str,
                            help="Add stations to the database.")

    arg_parser.add_argument('--drop_stations', dest='drop_stations', type=str,
                            help="Drop stations from the database.")

    arg_parser.add_argument('--list_configuration', dest='list_configuration', default=False, action="store_true",
                            help="Show the configuration.")

    arg_parser.add_argument('--list_status', dest='list_status', default=False, action="store_true",
                            help="Show the status of all the stations.")

    arg_parser.add_argument('--list_durations', dest='list_durations', type=int,
                            help="Show the durations of the past number of runs.")

    cml_args = arg_parser.parse_args()


    repeat = True if cml_args.repeat >= 1 else False
    delay = int(cml_args.repeat) if repeat else 0
    warning_hours = 36 if cml_args.warning_hours is None else int(cml_args.warning_hours)
    verbosity = 1 if cml_args.verbosity is None else int(cml_args.verbosity)

    quit_after_config_changes = False

    # Load the config file
    syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))
    conn = createStationMonitorDB(os.path.join(syscon.data_dir, station_monitor_filename))


    if cml_args.add_operators:
        quit_after_config_changes = True
        addOperators(conn, cml_args.add_operators)

    if cml_args.drop_operators:
        quit_after_config_changes = True
        dropOperators(conn, cml_args.drop_operators)

    if cml_args.add_stations:
        quit_after_config_changes = True
        addStations(conn, cml_args.add_stations)

    if cml_args.drop_stations:
        quit_after_config_changes = True
        dropStations(conn, cml_args.drop_stations)

    if cml_args.list_configuration:
        quit_after_config_changes = True
        print(listConfiguration(conn))

    if cml_args.list_status:
        quit_after_config_changes = True
        print(listStatus(conn))

    if cml_args.list_durations is None:
        pass
    else:
        quit_after_config_changes = True
        print(listDurations(conn, int(cml_args.list_durations)))

    if quit_after_config_changes:
        quit()

    if repeat:
        if verbosity > 0:
            print("\nStation monitor repeating every {:.2f} hours.\n".format(delay / 60))
    else:
        if verbosity > 0:
            print("\nStation monitor running once.\n")

    if not os.path.exists(os.path.expanduser(mail_configuration_file)) and verbosity > 0:
        print(WARNING + "No configuration file found at {}".format(mail_configuration_file))
        print("Mail cannot be sent")
        print("Mail configuration file first line is the email address")
        print("Second line is the API key")
        print("For example \n")
        print("station.operator@gmail.com \nabcd efgh ijkl mnop \n" + ENDC)

    if len(getOperatorStationList(conn)) == 0:
        print(WARNING + "No stations or operators have been configured")
        print(WARNING + "To add stations use --add_stations flag ")
        print(WARNING + "For example \n \n")
        print(OKGREEN + "python -m Utils.StationMonitor --add_station 'operator@example.com xx0001 xx0002 xx0003'  \n")
        print(WARNING + "To list configuration use --list_configuration")
        print(WARNING + "To run continuously use -r or --repeat followed by an integer numbet of minutes")
        print(WARNING + "For example \n \n")
        print(OKGREEN + "python -m Utils.StationMonitor -r 120 \n")

        print(ENDC)

    # Run the main function
    stationMonitor(syscon, repeat=repeat, delay_minutes=delay, verbosity=verbosity, warning_hours=warning_hours)
